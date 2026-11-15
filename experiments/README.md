# Experiment recipes

Each script here is a **thin wrapper**: it `export`s a set of environment
variables and calls `./submit_train.sh <tag>`, usually looping over
leave-one-participant-out folds. There is one training mechanism — see the
"Training" section of the top-level [README](../README.md) — and these are
configurations of it, recorded so a result can be traced back to the exact flags
that produced it.

All of them run from the repository root and `cd` there themselves, so you can
invoke them from anywhere:

```bash
./experiments/production/submit_loocv_fixture_obb_v2_gaze.sh          # all 9 folds
FOLDS="P07" ./experiments/production/submit_loocv_fixture_obb_v2_gaze.sh   # one fold
```

Every variable each script exports is still overridable from the outside, so a
one-off variation needs no new file:

```bash
FUTURE_TIME=60 ./experiments/production/submit_loocv_fixture_obb_v2_gaze.sh
```

Read each script's own header comment before running it — they record *why* each
flag is set, and which earlier run the configuration was recovered from.

---

## `production/` — the configurations behind the reported results

| Script | Modality | What it is |
|---|---|---|
| `submit_loocv_fixture_obb_v2_gaze.sh` | `video,location,obb,gaze` | The current best configuration (`vlog`). Identical to the `vlo` baseline below plus the gaze modality. 9-fold LOOCV. |
| `submit_loocv_fixture_obb_v2.sh` | `video,location,obb` | The `vlo` baseline the above derives from, with the token-scale fixes (`TOKEN_LAYERNORM`, `TYPE_EMBEDDINGS=unified`, `CLASS_EMBEDDING_GAIN`, `TIME_ENCODING`) that make it "v2". |

Both need the IoU-fused OBB cache (`FUSION=iou FUSED_CACHE=true`) — see
[preprocessing/README.md](../preprocessing/README.md) step 2 — and the
per-participant voxel grids in `new_voxel_centers/`, which they pick up
automatically now that `VOXEL_CENTERS_DIR` defaults to `$PWD/new_voxel_centers`.

The full flag set of the headline run is reproduced in the top-level README
under "Reproducing the headline configuration", so you can read it without
opening the script.

## `replication/` — the FIction baseline ladder

Six scripts stepping from the original FIction architecture to this work's, one
change at a time. Read in this order:

| # | Script | The one change |
|---|---|---|
| 1 | `submit_fiction_replication.sh` | The original [FIction](https://github.com/thechargedneutron/FIction) architecture and hyperparameters on HD-EPIC: a linear readout off one aggregate token. |
| 2 | `submit_fiction_replication_voxel_query_fourier.sh` | Swap the linear readout for the voxel-query decoder with Fourier queries — 4096 coordinate queries that each attend the sequence for themselves. |
| 3 | `submit_fiction_replication_vq_fourier_locjoint.sh` | Add `LOCATION_TOKENS=joint`: one location token per second instead of three per-axis tokens. |
| 4 | `submit_fiction_replication_vq_axis_sum_locjoint.sh` | Same as #3 but with `axis_sum` queries instead of `fourier` — isolates the query parameterisation. |
| 5 | `submit_fiction_replication_obb.sh` | Replace the `env` modality (4096 per-voxel tokens) with `obb` tokens (one per fused 3D box). |
| 6 | `submit_fiction_replication_vq_fourier_locjoint_obb.sh` | #3 and #5 together — the configuration the `production/` runs build on. |

---

## Scripts not in this repository

The thesis also ran ~50 further configurations: a numbered Q1–Q10 investigation
series (decoder decomposition, residual depth, focal loss, positional-encoding
sharing, time encoding, single-modality runs), per-configuration LOOCV sweeps,
modality-isolation and within-participant ablations, and output-bias and
encoding-scale sweeps. They are not published here because they are wrappers
around the same `submit_train.sh` with different flag values, and the flag
reference in the top-level README documents every knob they touch.

If you want to reconstruct one, the flags are the whole story: pick the values
from the README's configuration tables and pass them as environment variables.
