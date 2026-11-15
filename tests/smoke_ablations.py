"""Quick end-to-end smoke test for the 3 ablation configs: build a tiny
dataset (debug via train_n-like slicing isn't available on the raw Dataset,
so just take dataset[0:4]), collate, and run one forward pass through the
model for each config. Mirrors train.py's smoke test but standalone so it
can run fast against P07 only.

Scratch file -- safe to delete after use.
"""
import numpy as np
import torch

import os
import sys

# The modules under test live at the repository root, one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import HDEpicInteractionDataset
from model import CustomConfig, CustomTransformerModel
from object_class_vocab import NUM_OBJECT_CLASSES
from trajectory_tokens import trajectory_raw_dim_simple


def collate(features):
    first = features[0]
    batch = {}
    for k, v in first.items():
        if v is None or isinstance(v, str):
            continue
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        elif isinstance(v, np.ndarray):
            batch[k] = torch.tensor(np.stack([f[k] for f in features]))
        elif isinstance(v, dict):
            batch[k] = {}
            for sub_key in v:
                sub_vals = [f[k][sub_key] for f in features]
                if isinstance(sub_vals[0], np.ndarray):
                    batch[k][sub_key] = torch.tensor(np.stack(sub_vals))
                else:
                    batch[k][sub_key] = torch.stack(sub_vals)
        else:
            batch[k] = torch.tensor([f[k] for f in features])
    return batch


def run_config(name, modality, ds_kwargs, config_kwargs):
    print(f"\n=== {name} ===")
    ds = HDEpicInteractionDataset('val', participants=['P07'], verbose=False, **ds_kwargs)
    print(f"  dataset size: {len(ds)}")
    feats = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate(feats)

    config = CustomConfig(
        visual_feature_dim=4096, transformer_hidden_size=64,  # small for a fast smoke test
        num_transformer_layers=1, num_heads=4, output_dim=4096,
        num_object_classes=NUM_OBJECT_CLASSES, observation_time=30,
        use_static_env=True, input_modality=modality,
        trajectory_hidden_dim=32,
        **config_kwargs,
    )
    model = CustomTransformerModel(config)
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    print(f"  logits {tuple(out['interaction'].shape)} loss {out['loss'].item():.4f}")
    assert out['interaction'].shape == (len(feats), 4096)
    assert torch.isfinite(out['loss'])
    print("  OK")


run_config(
    "ablation2 trajsimple",
    "video,location,env,trajectory",
    dict(env_orphan_fallback=False, trajectory_simple=True),
    dict(trajectory_raw_dim=trajectory_raw_dim_simple(), trajectory_simple=True),
)

run_config(
    "ablation3 envpos",
    "video,location,env",
    dict(env_orphan_fallback=False),
    dict(env_positional_encoding=True),
)

run_config(
    "ablation4 envavg",
    "video,location,env",
    dict(env_orphan_fallback=False, env_class_averaging=True, env_max_classes_per_voxel=4),
    dict(env_class_averaging=True, env_max_classes_per_voxel=4),
)

# --- New: fallback averaging (scenario 1 = merge, scenario 2 = empty_only) ---
# Both feed the model the same [4096, K] class grid + mask as ablation4, so the
# model config is identical; only the loader's choice of which K classes differs.
for mode in ('merge', 'empty_only'):
    run_config(
        f"ablation5 env fallback averaging ({mode})",
        "video,location,env",
        dict(env_orphan_fallback=False, env_fallback_averaging=mode,
             env_max_classes_per_voxel=4),
        dict(env_class_averaging=True, env_max_classes_per_voxel=4),
    )

# --- New: gaze modality (one token per second of the observation window) ---
run_config(
    "ablation6 gaze",
    "video,location,env,gaze",
    dict(env_orphan_fallback=False),
    dict(),
)

print("\nALL SMOKE TESTS PASSED")
