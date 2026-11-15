"""How lopsided is the voxel-query decoder's residual, measured on real data?

`decoder.py` claims |q| ~ 55.7 against |attn_out| ~ 1.11 on a trained axis_sum
checkpoint -- a 50:1 split in favour of "which voxel am I" over "what did I
observe". That number has been quoted in the spec, in CLAUDE.md and in train.py's
help text, but nothing computed it reproducibly: `smoke_decoder.py` test 14
asserts the ratio holds on a RANDOMLY INITIALISED model, which guards the flag's
behaviour but says nothing about a trained one, and |attn_out| depends entirely on
the encoder output distribution, so it cannot be derived from the weights alone
the way |q| can.

This script computes both on a real batch:

    q         = decoder.queries()                  [4096, D]   -- weights only
    attn_out  = decoder.attn(ln_q(q), ln_m(memory), ln_m(memory))[0]
    h         = q + attn_out                       (--decoder_residual raw)

so `attn_out` is captured with a forward hook on the decoder's MultiheadAttention
rather than recomputed here -- recomputing it would risk quietly disagreeing with
what decoder.forward actually does, which is the whole thing being measured.

Reported per residual mode as the model was trained, since the flag changes which
two vectors are added:

    raw       h = q          + attn_out            ratio |q|      / |attn_out|
    normed    h = ln_q(q)    + attn_out            ratio sqrt(D)  / |attn_out|
    balanced  h = ln_q(q)    + ln_a(attn_out)      ratio ~1:1

Usage:
    python analysis/residual_balance.py \
        --checkpoint /data/.../checkpoint-18954 \
        --test_participants P01 \
        --voxel_centers_dir /home/.../new_voxel_centers \
        --n_episodes 64
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.data.data_collator import torch_default_data_collator  # noqa: E402

from loader import HDEpicInteractionDataset  # noqa: E402
from model import CustomConfig, CustomTransformerModel  # noqa: E402


def load_checkpoint(path):
    """Strict load of a voxel_query checkpoint -- same contract as
    voxel_query_attention.py: any missing/unexpected key is fatal, because a
    partially initialised decoder still produces perfectly plausible norms."""
    if not os.path.isdir(path):
        raise SystemExit(f"No such checkpoint directory: {path}")
    config = CustomConfig.from_pretrained(path)
    if getattr(config, 'decoder', 'linear') != 'voxel_query':
        raise SystemExit(
            f"{path} was trained with --decoder "
            f"{getattr(config, 'decoder', 'linear')!r}; there is no query residual "
            "to measure in a linear readout.")
    weights = None
    for name in ('model.safetensors', 'pytorch_model.bin'):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            weights = candidate
            break
    if weights is None:
        raise SystemExit(f"{path} holds no model weights")
    model = CustomTransformerModel(config)
    if weights.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"State dict does not match the model built from {path}/config.json\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}")
    return model.eval(), config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n_episodes', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--test_participants', type=str, default='P01')
    ap.add_argument('--voxel_centers_dir', type=str, default='')
    ap.add_argument('--anticipation_time', type=float, default=5)
    ap.add_argument('--future_time', type=float, default=180)
    ap.add_argument('--time_anchor', type=str, default='gaze')
    args = ap.parse_args()

    model, config = load_checkpoint(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    residual = getattr(config, 'decoder_residual', 'raw')
    print(f"[residual] {os.path.basename(args.checkpoint)}  "
          f"query_type={config.voxel_query_type}  residual={residual}  device={device}")

    # Shape-affecting loader options come from the checkpoint's own config, never
    # from this script's defaults -- same reasoning as voxel_query_attention.py.
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
    print(f"[residual] {n} of {len(dataset)} test episodes")

    dec = model.decoder
    captured = {}

    def hook(_module, _inputs, output):
        # nn.MultiheadAttention returns (attn_output, attn_weights); the decoder
        # runs batch_first=False, so this is [4096, B, D].
        captured['attn_out'] = output[0].detach()

    handle = dec.attn.register_forward_hook(hook)

    q_norms, a_norms, h_norms = [], [], []
    with torch.no_grad():
        # |q| is a property of the weights alone and identical every batch, but it
        # is recomputed inside the loop anyway so the three numbers always come
        # from the same forward pass.
        for start in range(0, n, args.batch_size):
            rows = [dataset[i] for i in range(start, min(start + args.batch_size, n))]
            # Drop 'labels' before collating: it is a nested dict, which
            # torch_default_data_collator cannot stack ("could not infer dtype of
            # dict"), and this measurement needs no targets -- only the encoder
            # output and the query the decoder builds from it.
            rows = [{k: v for k, v in r.items() if k != 'labels'} for r in rows]
            batch = torch_default_data_collator(rows)
            batch = {k: (v.to(device) if torch.is_tensor(v) else
                         {sk: sv.to(device) for sk, sv in v.items()}
                         if isinstance(v, dict) else v)
                     for k, v in batch.items()}
            model(**batch)
            attn_out = captured['attn_out']                       # [4096, B, D]
            b = attn_out.shape[1]
            q = dec.queries().unsqueeze(1).expand(-1, b, -1)      # [4096, B, D]
            lhs = q if residual == 'raw' else dec.ln_q(q)
            rhs = dec.ln_a(attn_out) if residual == 'balanced' else attn_out
            q_norms.append(lhs.norm(dim=-1).mean().item())
            a_norms.append(rhs.norm(dim=-1).mean().item())
            h_norms.append((lhs + rhs).norm(dim=-1).mean().item())

    handle.remove()
    qn, an, hn = np.mean(q_norms), np.mean(a_norms), np.mean(h_norms)
    print(f"\n  |query side|    = {qn:8.3f}   ({'q' if residual == 'raw' else 'ln_q(q)'})")
    print(f"  |evidence side| = {an:8.3f}   "
          f"({'ln_a(attn_out)' if residual == 'balanced' else 'attn_out'})")
    print(f"  |h| = |sum|     = {hn:8.3f}")
    print(f"\n  ratio           = {qn / an:8.2f} : 1   (query : evidence)")
    print(f"  evidence share  = {an / (qn + an) * 100:8.2f} %  of the summed magnitude")


if __name__ == '__main__':
    main()
