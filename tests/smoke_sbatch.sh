#!/bin/bash
# CPU-only smoke test for the new ablations, run as a SLURM job.
#
# The loader takes several minutes per split, which is long enough that the
# login node's reaper kills it (and because python buffers stdout when
# redirected, it dies leaving an empty log -- hence `python -u` below).
#
# Usage: sbatch tests/smoke_sbatch.sh      (from the repository root)
#        sbatch --export=ALL,SMOKE_SCRIPT=tests/smoke_ablations.py tests/smoke_sbatch.sh

#SBATCH --job-name="fiction_v2_smoke"
# SLURM does not expand environment variables in #SBATCH lines, so there is no
# portable way to template this. Pass your account on the command line instead --
#   sbatch --account=<acct> ...   (submit_train.sh forwards $SLURM_ACCOUNT for you)
# -- or uncomment and hard-code it:
##SBATCH --account=<your-slurm-account>
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=01:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p out err

echo "Job started: $(date) on $(hostname)"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-egovlpv2-pip}"
echo "Python: $(which python)"

SMOKE_SCRIPT="${SMOKE_SCRIPT:-tests/smoke_new_ablations.py}"
echo "Running $SMOKE_SCRIPT"
echo ""

set +e
python -u "$SMOKE_SCRIPT"
RC=$?
set -e
echo ""
echo "Job finished: $(date) (exit=$RC)"
exit $RC
