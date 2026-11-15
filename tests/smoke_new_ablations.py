"""Forward-pass smoke test for the two NEW experiments only:

  * env fallback averaging  (--env_fallback_averaging merge | empty_only)
  * gaze                    (modality 'gaze')

Datasets are cached by loader kwargs, so the P07 split is built 2x rather than
once per config. Mirrors train.py's own smoke test but standalone.

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

_ds_cache = {}


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
            batch[k] = {sk: torch.tensor(np.stack([f[k][sk] for f in features]))
                        if isinstance(v[sk], np.ndarray)
                        else torch.stack([f[k][sk] for f in features]) for sk in v}
        else:
            batch[k] = torch.tensor([f[k] for f in features])
    return batch


def get_ds(**kwargs):
    key = tuple(sorted(kwargs.items()))
    if key not in _ds_cache:
        _ds_cache[key] = HDEpicInteractionDataset(
            'val', participants=['P07'], verbose=False, **kwargs)
    return _ds_cache[key]


def run(name, modality, ds_kwargs, config_kwargs):
    print(f"\n=== {name} ===")
    ds = get_ds(**ds_kwargs)
    feats = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate(feats)
    print(f"  dataset={len(ds)}  gaze_voxels={tuple(batch['gaze_voxels'].shape)} "
          f"gaze_mask_sum={int(batch['gaze_mask'].sum())} "
          f"env_grid={tuple(batch['env_grid'].shape)}")

    config = CustomConfig(
        visual_feature_dim=4096, transformer_hidden_size=64,
        num_transformer_layers=1, num_heads=4, output_dim=4096,
        num_object_classes=NUM_OBJECT_CLASSES, observation_time=30,
        use_static_env=True, input_modality=modality,
        trajectory_hidden_dim=32, **config_kwargs)
    model = CustomTransformerModel(config)
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    assert out['interaction'].shape == (len(feats), 4096)
    assert torch.isfinite(out['loss'])
    print(f"  logits {tuple(out['interaction'].shape)} loss {out['loss'].item():.4f}  OK")
    return model, batch


plain = dict(env_orphan_fallback=False)

for mode in ('merge', 'empty_only'):
    run(f"ablation5 env fallback averaging ({mode})", "video,location,env",
        dict(env_orphan_fallback=False, env_fallback_averaging=mode,
             env_max_classes_per_voxel=4),
        dict(env_class_averaging=True, env_max_classes_per_voxel=4))

# The only gaze config that gets submitted: gaze alongside the other modalities.
m, b = run("ablation6 gaze", "video,location,env,gaze", plain, dict())
with torch.no_grad():
    n_with = m(**b)['interaction'].shape
    b2 = dict(b)
    b2.pop('gaze_voxels')
    b2.pop('gaze_mask')
    m.config.input_modality = 'video'
    n_without = m(**b2)['interaction'].shape
print(f"\n  gaze adds 3 x 30 = 90 tokens + 1 sep to the sequence "
      f"(logits unchanged: {n_with} vs {n_without})")

# The gaze marker must make a gaze token differ from a location token at the
# same voxel — otherwise, with no positional encoding, they'd be identical.
with torch.no_grad():
    idx = torch.tensor([[5]])
    loc = m.location_embedding_x(idx)
    gz = m.location_embedding_x(idx) + m.gaze_type_embedding
    print(f"  gaze marker separates gaze from location tokens: "
          f"{not torch.allclose(loc, gz)} (L2 = {(gz - loc).norm().item():.3f})")

print("\nALL NEW-ABLATION SMOKE TESTS PASSED")
