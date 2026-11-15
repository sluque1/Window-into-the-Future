"""Where does a predicted voxel get its evidence? (spec §7)

The reason the voxel-query decoder is worth building at all: because every voxel
attends for itself, the attention map answers the question ablation deltas can
only circle around -- when the model correctly predicts an interaction at some
place, is it reading *recent observation* (anticipation) or *what is sitting
there* (a co-occurrence prior)?

For each true-positive voxel this records the attention distribution over the
sequence and reports:

  * mass by segment -- video / location / gaze / obb / trajectory / env / sep,
    fractions summing to 1;
  * within the video segment, mass by Δt, counted back from the END of the
    observation window (the anchor itself is a further `anticipation_time`
    later, so Δt=0 is the last second the model actually saw, not the anchor).
    Mass concentrating as Δt -> 0 is evidence of anticipation from recent
    observation; a flat profile says the video tokens are a bag of scene context;
  * within the obb segment, whether the top-attended box sits in or near the
    queried voxel. High co-location reads as "there is a kettle here, kettles get
    used". Low co-location with high attention on a *different* region is the
    interesting case and the harder one to explain away as co-occurrence.

The same three are computed for false positives and true negatives, because "the
model looks at recent video" is only informative if it does so *differently* when
it is right.

Requires a trained --decoder voxel_query checkpoint. Attention weights force
nn.MultiheadAttention off its scaled_dot_product_attention fast path and
materialise a [B, heads, 4096, L] tensor (~1.1 GB at B=16 on the obb branch,
~8.7 GB on env), so keep --batch_size small and prefer the obb branch.

Usage:
    python analysis/voxel_query_attention.py \
        --checkpoint results/<run>/checkpoint-1234 \
        --test_participants P07 --n_episodes 200 \
        --voxel_centers_dir ../new_voxel_centers

Writes analysis/out/voxel_query_attention/<run>/{summary.json,by_delta_t.csv,
raw_<episode>.npz}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import GRID, HDEpicInteractionDataset, NUM_VOXELS  # noqa: E402
from model import CustomConfig, CustomTransformerModel  # noqa: E402
from train import torch_default_data_collator  # noqa: E402

OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out', 'voxel_query_attention')


def load_checkpoint(path):
    """Load a voxel-query model, failing loudly on anything unexpected.

    Every failure mode here produces plausible-looking numbers if ignored: a
    linear checkpoint has no attention to report, a partially-matching state dict
    leaves a randomly initialised decoder that still emits a tidy attention map,
    and a config mismatch silently re-shapes the queries. So: strict load, no
    fallbacks, no strict=False.
    """
    if not os.path.isdir(path):
        raise SystemExit(f"No such checkpoint directory: {path}")
    cfg_path = os.path.join(path, 'config.json')
    if not os.path.exists(cfg_path):
        raise SystemExit(f"{path} has no config.json -- is it a checkpoint directory?")
    config = CustomConfig.from_pretrained(path)
    if getattr(config, 'decoder', 'linear') != 'voxel_query':
        raise SystemExit(
            f"{path} was trained with --decoder {getattr(config, 'decoder', 'linear')!r}. "
            "This probe reads the voxel-query decoder's attention; a linear readout "
            "has none to report.")

    weights = None
    for name in ('model.safetensors', 'pytorch_model.bin'):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            weights = candidate
            break
    if weights is None:
        raise SystemExit(f"{path} holds no model weights (looked for "
                         "model.safetensors / pytorch_model.bin)")

    model = CustomTransformerModel(config)
    if weights.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    # strict=False only to REPORT both lists; anything non-empty is fatal.
    if missing or unexpected:
        raise SystemExit(
            f"State dict does not match the model built from {cfg_path}.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}\n"
            "Refusing to run on a partially initialised model.")
    return model.eval(), config


def voxel_ijk_table():
    """(4096, 3) grid coordinates, C order -- flat = i*256 + j*16 + k."""
    return np.stack(np.unravel_index(np.arange(NUM_VOXELS), (GRID, GRID, GRID)), axis=-1)


def summarise(attn_row, spans, obs_time):
    """Segment fractions and the video segment's Δt profile for one voxel.

    attn_row: [L] attention over the sequence, already summing to ~1.
    Returns (dict segment -> fraction, np.ndarray [obs_time] video mass by Δt),
    where Δt counts seconds back from the last observed second (index 0 = the
    end of the observation window, `anticipation_time` before the anchor).
    """
    by_segment = {}
    per_delta = np.zeros(obs_time, dtype=np.float64)
    for name, start, end in spans:
        by_segment[name] = by_segment.get(name, 0.0) + float(attn_row[start:end].sum())
        if name == 'video':
            v = attn_row[start:end]
            # Visual tokens are end-aligned: the LAST token is the second nearest
            # the anchor, so index n counts back as Δt = (len - 1 - n) seconds.
            per_delta[:len(v)] += v[::-1]
    return by_segment, per_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n_episodes', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=4,
                    help="Small on purpose: attention is [B, heads, 4096, L].")
    ap.add_argument('--threshold', type=float, default=0.1,
                    help="Probability above which a voxel counts as predicted. The "
                         "default is the middle of the four thresholds train.py's "
                         "ComputeMetricsWrapper already reports at {0.01,0.05,0.1,0.2}.")
    ap.add_argument('--tn_stride', type=int, default=400,
                    help="Keep every Nth true negative. Roughly 4070 of the 4096 "
                         "voxels are true negatives in any episode, and they are only "
                         "a contrast baseline, so the default keeps ~10 per episode. "
                         "Set to 1 to keep all of them.")
    ap.add_argument('--n_raw_episodes', type=int, default=4,
                    help="Episodes whose full [n_voxels, L] slices are saved.")
    ap.add_argument('--test_participants', type=str, default='P07')
    ap.add_argument('--voxel_centers_dir', type=str, default='')
    ap.add_argument('--anticipation_time', type=float, default=5)
    ap.add_argument('--future_time', type=float, default=180)
    ap.add_argument('--time_anchor', type=str, default='gaze',
                    choices=['trajectory', 'gaze'])
    ap.add_argument('--out_dir', type=str, default='')
    args = ap.parse_args()

    model, config = load_checkpoint(args.checkpoint)
    obs_time = config.observation_time    # from the checkpoint, never re-specified here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[probe] {args.checkpoint}\n[probe] modality={config.input_modality} "
          f"query={config.voxel_query_type} bias={config.voxel_bias} "
          f"readout={config.readout} device={device}")

    # Loader options that change a TENSOR'S SHAPE must be taken from the
    # checkpoint's own config, not from this script's defaults. trajectory_simple
    # is the one that bites: model.forward maps trajectory_tokens whenever the
    # loader supplies them, regardless of input_modality, so a mismatch here is
    # an immediate shape error in trajectory_mapper even on an obb-only run.
    dataset = HDEpicInteractionDataset(
        'test', participants=args.test_participants.split(','),
        observation_time=config.observation_time,
        anticipation_time=args.anticipation_time,
        future_time=args.future_time,
        static_env=config.use_static_env,
        voxel_centers_dir=args.voxel_centers_dir or None,
        max_obb_tokens=config.max_obb_tokens,
        trajectory_simple=config.trajectory_simple,
        env_class_averaging=config.env_class_averaging,
        env_max_classes_per_voxel=config.env_max_classes_per_voxel,
        time_anchor=args.time_anchor)
    n = min(args.n_episodes, len(dataset))
    print(f"[probe] {n} of {len(dataset)} test episodes")

    ijk = voxel_ijk_table()
    groups = ('true_positive', 'false_positive', 'true_negative')
    seg_totals = {g: {} for g in groups}
    delta_totals = {g: np.zeros(obs_time) for g in groups}
    counts = {g: 0 for g in groups}
    obb_colocation = []      # (grid distance from queried voxel to top-attended box)
    out_dir = args.out_dir or os.path.join(
        OUT_BASE, os.path.basename(os.path.normpath(args.checkpoint)))
    os.makedirs(out_dir, exist_ok=True)

    saved_raw = 0
    for start in range(0, n, args.batch_size):
        rows = [dataset[i] for i in range(start, min(start + args.batch_size, n))]
        batch = torch_default_data_collator(rows)
        batch = {k: (v.to(device) if torch.is_tensor(v) else
                     {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch, need_attention=True)
        attn = out['attention']            # [B, 4096, L]
        if attn is None:
            raise SystemExit("The model returned no attention weights -- it is not "
                             "a voxel_query decoder.")
        spans = out['segments']
        probs = torch.sigmoid(out['interaction'])
        labels = batch['labels']['interaction']

        for b in range(attn.shape[0]):
            pred = probs[b] > args.threshold
            truth = labels[b] > 0
            selection = {
                'true_positive': (pred & truth).nonzero(as_tuple=True)[0],
                'false_positive': (pred & ~truth).nonzero(as_tuple=True)[0],
                # Sub-sampled: there are ~4070 of them per episode and they are
                # only a contrast baseline (--tn_stride).
                'true_negative': (~pred & ~truth).nonzero(as_tuple=True)[0][::args.tn_stride],
            }
            for group, idx in selection.items():
                for v in idx.tolist():
                    row = attn[b, v].float().cpu().numpy()
                    seg, per_delta = summarise(row, spans, obs_time)
                    for name, mass in seg.items():
                        seg_totals[group][name] = seg_totals[group].get(name, 0.0) + mass
                    delta_totals[group] += per_delta
                    counts[group] += 1
                    if group == 'true_positive':
                        for name, s, e in spans:
                            if name != 'obb':
                                continue
                            top = int(row[s:e].argmax())
                            if not bool(batch['obb_mask'][b, top]):
                                continue    # top slot is padding: no box to compare
                            box_ijk = batch['obb_voxels'][b, top].cpu().numpy() - 1
                            obb_colocation.append(
                                float(np.linalg.norm(box_ijk - ijk[v])))

            if saved_raw < args.n_raw_episodes:
                keep = torch.cat([selection['true_positive'],
                                  selection['false_positive']])[:256]
                np.savez_compressed(
                    os.path.join(out_dir, f"raw_{start + b:05d}.npz"),
                    attention=attn[b, keep].float().cpu().numpy(),
                    voxels=keep.cpu().numpy(),
                    labels=labels[b].cpu().numpy(),
                    probs=probs[b].float().cpu().numpy(),
                    segments=np.array([(nm, s, e) for nm, s, e in spans], dtype=object))
                saved_raw += 1

    summary = {'checkpoint': args.checkpoint,
               'n_episodes': n,
               'threshold': args.threshold,
               'modality': config.input_modality,
               'voxel_query_type': config.voxel_query_type,
               'voxel_bias': config.voxel_bias,
               'readout': config.readout,
               'counts': counts,
               'attention_by_segment': {}}
    print()
    for group in groups:
        if not counts[group]:
            print(f"[{group}] none found")
            continue
        frac = {k: v / counts[group] for k, v in sorted(seg_totals[group].items())}
        summary['attention_by_segment'][group] = frac
        print(f"[{group}] n={counts[group]}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in frac.items()))

    if obb_colocation:
        co = np.asarray(obb_colocation)
        summary['obb_colocation'] = {
            'mean_grid_distance': float(co.mean()),
            'median_grid_distance': float(np.median(co)),
            'frac_within_2_voxels': float((co <= 2).mean()),
            'n': int(co.size),
        }
        print(f"\n[true_positive] top-attended OBB is {co.mean():.2f} voxels from the "
              f"queried voxel on average; {100 * (co <= 2).mean():.1f}% within 2. "
              "High co-location = a co-occurrence prior rather than anticipation.")

    # Δt profile of the video segment: the anticipation signal, if there is one.
    with open(os.path.join(out_dir, 'by_delta_t.csv'), 'w') as f:
        f.write("delta_t_seconds," + ",".join(groups) + "\n")
        for dt in range(obs_time):
            vals = [delta_totals[g][dt] / counts[g] if counts[g] else float('nan')
                    for g in groups]
            f.write(f"{dt}," + ",".join(f"{v:.6f}" for v in vals) + "\n")

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir}/summary.json, by_delta_t.csv and {saved_raw} raw_*.npz")


if __name__ == '__main__':
    main()
