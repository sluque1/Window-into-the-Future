"""Is the CLS token an attention sink?

Under `--decoder voxel_query` nothing reads sequence seat 0 -- `transformer_output[0]`
is never indexed, the 4096 coordinate queries cross-attend to the whole sequence
instead. So `--readout cls` adds a token that no longer has the job it is named
after. The hypothesis this probe tests is that it acquired a different one.

Softmax has no null option: every attention row sums to exactly 1, so a voxel
query that finds nothing relevant anywhere in the 404 tokens must still emit a
convex combination of them. With ~21 of 4096 voxels positive, most queries are in
exactly that position. A token that is never masked and carries no content is the
natural destination for that mass -- "I looked and found nothing" becomes
representable instead of being smeared over whichever OBB tokens scored slightly
above chance. This is the attention-sink phenomenon (Xiao et al., ICLR 2024) and
the register phenomenon in ViTs (Darcet et al., ICLR 2024), where the damaging
case is the one WITHOUT a dedicated slot: the model recruits real content tokens,
their norms blow up and their features stop being usable as evidence.

Four measurements, in increasing order of how much they prove:

  1. MASS -- attention mass on CLS, per token, against the 1/L a uniform row
     would give and against what an average content token receives. Split by
     true-positive / true-negative voxels, because the sink story predicts
     negatives lean on it harder.
  2. TOP-1 -- how often CLS is a voxel's single most-attended token. Mass can
     accumulate from many small weights; this says whether it actually wins.
  3. NORM -- the per-token norm of the encoder output, by segment. The ViT
     registers signature is an anomalously large norm on whichever token got
     recruited for the role.
  4. ABLATION (the causal one) -- mask CLS out of the decoder's memory and
     re-decode. Mass is only suggestive; if removing the token barely moves the
     logits it was not load-bearing whatever its mass. Controls ablate a SEP, a
     random real OBB token and a random video token the same way, because
     removing any key perturbs a softmax and the question is whether CLS is
     special.

The ablation re-runs only the decoder, reusing the captured encoder output -- so
the encoder is not re-run 5 times and the control arms cost almost nothing.

Usage:
    python analysis/cls_sink_probe.py \
        --checkpoint /data/.../snapshots/<run>/results/<run>/checkpoint-1920 \
        --code_dir   /data/.../snapshots/<run> \
        --test_participants P09 --n_episodes 100 \
        --voxel_centers_dir ../new_voxel_centers

`--code_dir` points at the snapshot whose model.py built the checkpoint. Default
is this repo, which is right only for a checkpoint from the current working tree;
for a frozen snapshot the repo has usually drifted and the strict load below is
what will tell you so.

Writes analysis/out/cls_sink_probe/<checkpoint>/summary.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'cls_sink_probe')


def load_code(code_dir):
    """Import loader/model/train from `code_dir`, which may be a snapshot.

    Prepended rather than appended: a snapshot holds the same module NAMES as the
    repo, so appending would silently import the repo's versions and the strict
    state-dict load would then fail for a reason that has nothing to do with the
    checkpoint.
    """
    code_dir = os.path.abspath(code_dir)
    if not os.path.isfile(os.path.join(code_dir, 'model.py')):
        raise SystemExit(f"--code_dir {code_dir} has no model.py")
    sys.path.insert(0, code_dir)
    from loader import GRID, HDEpicInteractionDataset, NUM_VOXELS  # noqa: E402
    from model import CustomConfig, CustomTransformerModel        # noqa: E402
    from train import torch_default_data_collator                 # noqa: E402
    return dict(GRID=GRID, Dataset=HDEpicInteractionDataset, NUM_VOXELS=NUM_VOXELS,
                CustomConfig=CustomConfig, Model=CustomTransformerModel,
                collate=torch_default_data_collator, code_dir=code_dir)


def load_checkpoint(path, mod):
    """Strict load of a voxel_query checkpoint. No fallbacks: a partially
    initialised decoder still emits a perfectly tidy attention map."""
    if not os.path.isdir(path):
        raise SystemExit(f"No such checkpoint directory: {path}")
    config = mod['CustomConfig'].from_pretrained(path)
    if getattr(config, 'decoder', 'linear') != 'voxel_query':
        raise SystemExit(
            f"{path} was trained with --decoder {getattr(config, 'decoder', 'linear')!r}; "
            "this probe reads the voxel-query decoder's attention.")
    weights = next((os.path.join(path, n) for n in ('model.safetensors', 'pytorch_model.bin')
                    if os.path.exists(os.path.join(path, n))), None)
    if weights is None:
        raise SystemExit(f"{path} holds no model weights")
    model = mod['Model'](config)
    if weights.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"State dict does not match the model built from {path}/config.json.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}\n"
            "Point --code_dir at the snapshot that produced this checkpoint.")
    return model.eval(), config


class Accum:
    """Running mean of a scalar, kept per group so the printout is one table."""

    def __init__(self):
        self.total = {}
        self.n = {}

    def add(self, group, key, value, count=1):
        self.total.setdefault(group, {})
        self.n.setdefault(group, {})
        self.total[group][key] = self.total[group].get(key, 0.0) + float(value)
        self.n[group][key] = self.n[group].get(key, 0) + count

    def mean(self, group, key):
        if self.n.get(group, {}).get(key):
            return self.total[group][key] / self.n[group][key]
        return float('nan')

    def keys(self, group):
        return sorted(self.total.get(group, {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--code_dir', default=REPO,
                    help="Directory holding the model.py that built the checkpoint. "
                         "Default: this repo. Use the snapshot dir for a frozen run.")
    ap.add_argument('--n_episodes', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=4,
                    help="Small on purpose: attention is [B, 4096, L].")
    ap.add_argument('--threshold', type=float, default=0.1,
                    help="Probability above which a voxel counts as predicted.")
    ap.add_argument('--tn_stride', type=int, default=200,
                    help="Keep every Nth true negative; there are ~4070 per episode "
                         "and they are a contrast baseline. 1 keeps all.")
    ap.add_argument('--test_participants', type=str, default='P09')
    ap.add_argument('--voxel_centers_dir', type=str, default='')
    ap.add_argument('--anticipation_time', type=float, default=5)
    ap.add_argument('--future_time', type=float, default=180)
    ap.add_argument('--time_anchor', type=str, default='gaze',
                    choices=['trajectory', 'gaze'])
    ap.add_argument('--seed', type=int, default=0,
                    help="Picks the random control tokens for the ablation.")
    ap.add_argument('--out_dir', type=str, default='')
    args = ap.parse_args()

    mod = load_code(args.code_dir)
    model, config = load_checkpoint(args.checkpoint, mod)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    rng = np.random.default_rng(args.seed)
    print(f"[probe] {args.checkpoint}\n[probe] code={mod['code_dir']}\n"
          f"[probe] readout={config.readout} modality={config.input_modality} "
          f"query={config.voxel_query_type} token_layernorm="
          f"{getattr(config, 'token_layernorm', False)} device={device}")
    if config.readout != 'cls':
        print("[probe] NOTE: readout != 'cls', so there is no CLS seat. The SEP "
              "columns are still reported -- under this readout they are the only "
              "never-masked content-free slots, so they are where a sink would go.")

    dataset = mod['Dataset'](
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

    # The decoder's own inputs, captured on the way in. A pre-hook rather than a
    # hook on self.transformer because model.forward calls
    # self.decoder(transformer_output, attention_mask, ...) -- so one hook yields
    # BOTH the memory and the exact mask the decoder saw, and the ablation below
    # cannot drift from it by rebuilding the mask itself.
    captured = {}

    def grab(_module, inputs):
        captured['memory'] = inputs[0]
        captured['mask'] = inputs[1]

    model.decoder.register_forward_pre_hook(grab)

    mass = Accum()           # attention mass, by group and token role
    top1 = Accum()           # share of voxels whose argmax is that role
    norms = {}               # encoder-output token norm, by segment
    ablate = {}              # mean |delta logit| per ablated role
    cls_mass_vs_logit = []   # (mass on cls, logit) pairs, for the correlation
    seq_len = None
    spans_seen = None

    for start in range(0, n, args.batch_size):
        rows = [dataset[i] for i in range(start, min(start + args.batch_size, n))]
        batch = mod['collate'](rows)
        batch = {k: (v.to(device) if torch.is_tensor(v) else
                     {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch, need_attention=True)
        attn = out['attention']                      # [B, 4096, L], head-averaged
        if attn is None:
            raise SystemExit("No attention returned -- not a voxel_query decoder.")
        spans = out['segments']
        spans_seen = spans
        probs = torch.sigmoid(out['interaction'])
        labels = batch['labels']['interaction']
        memory, mask = captured['memory'], captured['mask']   # [L,B,H], [B,L]
        L = memory.shape[0]
        seq_len = L

        # --- (3) encoder-output norm by segment (the registers signature) ------
        tok_norm = memory.norm(dim=-1)               # [L, B]
        for name, s, e in spans:
            valid = ~mask[:, s:e].T                  # [e-s, B] True = real token
            if valid.any():
                v = tok_norm[s:e][valid]
                d = norms.setdefault(name, [0.0, 0])
                d[0] += float(v.sum())
                d[1] += int(v.numel())

        # --- (4) causal ablation, decoder only ---------------------------------
        # Re-decode with one key removed. Controls matter: removing ANY key
        # redistributes a softmax row, so the number only means something next to
        # what a content token of the same sequence costs.
        with torch.no_grad():
            base_logits, _ = model.decoder(memory, mask)
            targets = {}
            for name, s, e in spans:
                if name in ('cls', 'sep') and name not in targets:
                    targets[name] = s                # first cls / first sep
            for name, s, e in spans:
                if name in ('obb', 'video') and f'random_{name}' not in targets:
                    real = (~mask[0, s:e]).nonzero(as_tuple=True)[0]
                    if len(real):
                        targets[f'random_{name}'] = s + int(real[rng.integers(len(real))])
            for role, col in targets.items():
                m2 = mask.clone()
                m2[:, col] = True
                alt, _ = model.decoder(memory, m2)
                d = ablate.setdefault(role, [0.0, 0.0, 0])
                delta = (alt - base_logits).abs()
                d[0] += float(delta.sum())
                d[1] = max(d[1], float(delta.max()))
                d[2] += delta.numel()

        # --- (1) and (2), per voxel, split by group ----------------------------
        for b in range(attn.shape[0]):
            pred, truth = probs[b] > args.threshold, labels[b] > 0
            selection = {
                'true_positive': (pred & truth).nonzero(as_tuple=True)[0],
                'false_positive': (pred & ~truth).nonzero(as_tuple=True)[0],
                'false_negative': (~pred & truth).nonzero(as_tuple=True)[0],
                'true_negative': (~pred & ~truth).nonzero(as_tuple=True)[0][::args.tn_stride],
            }
            for group, idx in selection.items():
                if not len(idx):
                    continue
                rows_a = attn[b, idx].float()                   # [k, L]
                k = rows_a.shape[0]
                arg = rows_a.argmax(dim=1)                      # [k]
                for name, s, e in spans:
                    seg = rows_a[:, s:e]
                    # Total mass on the segment, and mass PER TOKEN -- the only
                    # form in which a 1-token CLS and a 250-token OBB segment are
                    # comparable at all.
                    mass.add(group, f'{name}_total', float(seg.sum()), k)
                    mass.add(group, f'{name}_per_token',
                             float(seg.sum()) / (e - s), k)
                    hit = ((arg >= s) & (arg < e)).sum()
                    top1.add(group, name, float(hit), k)
                if config.readout == 'cls':
                    for v, mcls in zip(idx.tolist(),
                                       rows_a[:, 0].tolist()):
                        cls_mass_vs_logit.append(
                            (mcls, float(out['interaction'][b, v])))

    # ---------------------------------------------------------------- report --
    groups = ['true_positive', 'false_positive', 'false_negative', 'true_negative']
    groups = [g for g in groups if mass.keys(g)]
    uniform = 1.0 / seq_len
    span_width = {name: e - s for name, s, e in spans_seen}
    summary = {'checkpoint': args.checkpoint, 'code_dir': mod['code_dir'],
               'n_episodes': n, 'seq_len': seq_len, 'readout': config.readout,
               'uniform_per_token': uniform,
               'segments': [(nm, s, e) for nm, s, e in spans_seen]}

    print(f"\n=== (1) attention mass PER TOKEN (uniform would be {uniform:.5f}) ===")
    names = sorted(span_width)
    head = f"{'segment':<12}{'n_tok':>7}" + "".join(f"{g.replace('_', '.'):>16}" for g in groups)
    print(head)
    per_token = {}
    for name in names:
        line = f"{name:<12}{span_width[name]:>7}"
        per_token[name] = {}
        for g in groups:
            v = mass.mean(g, f'{name}_per_token')
            per_token[name][g] = v
            line += f"{v:>15.5f}x" if False else f"{v:>16.5f}"
        print(line)
    summary['mass_per_token'] = per_token
    summary['mass_total'] = {name: {g: mass.mean(g, f'{name}_total') for g in groups}
                             for name in names}

    print(f"\n=== (2) share of voxels whose TOP-1 attended token is in segment ===")
    print(head)
    t1 = {}
    for name in names:
        t1[name] = {g: top1.mean(g, name) for g in groups}
        print(f"{name:<12}{span_width[name]:>7}" +
              "".join(f"{t1[name][g]:>16.4f}" for g in groups))
    summary['top1_share'] = t1

    print("\n=== (3) encoder-output token norm, by segment ===")
    summary['encoder_norm'] = {}
    for name in names:
        if name in norms and norms[name][1]:
            v = norms[name][0] / norms[name][1]
            summary['encoder_norm'][name] = v
            print(f"  {name:<12} {v:8.3f}")

    print("\n=== (4) ablation: mask one key out of the decoder's memory ===")
    print(f"  {'removed':<16}{'mean |dlogit|':>15}{'max |dlogit|':>15}")
    summary['ablation'] = {}
    for role in sorted(ablate):
        tot, mx, cnt = ablate[role]
        summary['ablation'][role] = {'mean_abs_delta': tot / cnt, 'max_abs_delta': mx}
        print(f"  {role:<16}{tot / cnt:>15.5f}{mx:>15.5f}")

    if cls_mass_vs_logit:
        arr = np.asarray(cls_mass_vs_logit)
        r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        summary['corr_cls_mass_vs_logit'] = r
        print(f"\n  corr(mass on CLS, logit) = {r:+.4f} over {len(arr)} voxels "
              "(negative = voxels leaning on CLS get scored DOWN, which is the "
              "sink doing its job)")

    # ------------------------------------------------------------- verdict ----
    print("\n=== reading it ===")
    if config.readout == 'cls' and 'cls' in per_token:
        c = per_token['cls'].get('true_negative', float('nan'))
        ratio = c / uniform
        content = max((per_token[n].get('true_negative', 0.0)
                       for n in names if n not in ('cls', 'sep')), default=0.0)
        print(f"  CLS carries {ratio:.1f}x a uniform token's mass on true negatives "
              f"({c:.4f} vs {uniform:.5f}), and {c / content:.1f}x the busiest "
              "content token." if content else "")
        a_cls = summary['ablation'].get('cls', {}).get('mean_abs_delta')
        a_ctl = [v['mean_abs_delta'] for k, v in summary['ablation'].items()
                 if k.startswith('random_')]
        if a_cls is not None and a_ctl:
            print(f"  Removing it moves the logits by {a_cls:.4f} on average "
                  f"against {max(a_ctl):.4f} for a random content token "
                  f"({a_cls / max(max(a_ctl), 1e-12):.1f}x).")
        print("  Sink if: per-token mass >> uniform AND higher on true negatives "
              "than true positives AND ablation moves logits much more than a "
              "content token does. Idle 768 parameters if all three are flat.")
        print("  If SEP outscores CLS on all three, the seps took the role and "
              "--readout cls is buying nothing here.")

    out_dir = args.out_dir or os.path.join(
        OUT_BASE, os.path.basename(os.path.normpath(args.checkpoint)))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir}/summary.json")


if __name__ == '__main__':
    main()
