# Smoke tests

Seven scripts that check the parts of the model most likely to break silently.
They are assertion scripts, not a pytest suite: each prints a line per check and
exits non-zero on the first failure.

Run them from the repository root.

## Data-free — run these anywhere, in seconds

These build a tiny random batch in memory and need no HD-EPIC data, no GPU and
no checkpoint:

```bash
python tests/smoke_decoder.py            # voxel-query decoder: regression + correctness
python tests/smoke_position_encoding.py  # shared spatial Fourier code
python tests/smoke_time_encoding.py      # time-to-anchor encoding
python tests/smoke_token_scale.py        # --token_layernorm and --time_encoding_gain
python tests/smoke_obb.py                # does the OBB embedding carry its three signals?
```

`smoke_decoder.py` is the broadest of them. It covers the `--readout cls` seat
and its mask, every `--voxel_query_type`, permutation invariance, gradient
reachability for all 16 decoder parameters, the padding-mask contract, every
`--decoder_residual` arm, and the prior-init path — 15 test groups, ~200
assertions, about 100 seconds on CPU.

Its first check is a **bit-identical regression** against logits recorded in
`smoke_decoder_reference.pt`, asserting `max|delta| == 0`. That is a deliberately
absolute bar, and it is sensitive to the floating-point environment as well as to
the code: a different BLAS implementation, CPU, or torch build can shift float32
logits in the last bits without anything being wrong. If that one check fails
while everything below it passes, suspect the environment before the code, and
compare against `requirements.txt`. Re-record only if you intend to move the
baseline:

```bash
python tests/smoke_decoder.py --capture
```

## Needs the dataset

These two construct a real `HDEpicInteractionDataset`, so they need HD-EPIC and
the EgoVLPv2 features in place. Building a split takes several minutes — long
enough that a login-node reaper may kill it — so run them as a job:

```bash
sbatch --account=<acct> tests/smoke_sbatch.sh                                    # smoke_new_ablations.py
sbatch --account=<acct> --export=ALL,SMOKE_SCRIPT=tests/smoke_ablations.py tests/smoke_sbatch.sh
```

| Script | Covers |
|---|---|
| `smoke_ablations.py` | End-to-end forward pass for the trajectory / env-averaging / env-positional ablation configs |
| `smoke_new_ablations.py` | Forward pass for the two newest experiment configs |

`smoke_sbatch.sh` is CPU-only and uses `python -u`, because a buffered stdout
means a reaped job leaves an empty log and no clue why.
