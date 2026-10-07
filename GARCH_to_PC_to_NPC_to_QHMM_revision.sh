#!/bin/bash
#SBATCH -p general
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH -t 2-
#SBATCH --array=0-99
#SBATCH --mail-type=all
#SBATCH --mail-user=morganj@business.unc.edu
#SBATCH --output=Final/garch_T500/logs/sample_%a.out
#SBATCH --job-name=garch_T500

# One sample per array task. Sample s is seeded with (seed, s), so tasks are independent and a
# failed task can be resubmitted on its own, e.g. sbatch --array=17 GARCH_to_PC_to_NPC_to_QHMM_revision.sh
# Slurm does not create the log folder, so before submitting run: mkdir -p Final/garch_T500/logs
# Merge the per-sample files afterwards with merge_results.py.
#
# MAX_ITER, OUT_DIR and QHMM_METHOD (SLSQP or Nelder-Mead) can be overridden, e.g. for a smoke test
#   sbatch --array=0-3 -t 1:00:00 --output=Final/smoke/logs/sample_%a.out \
#          --export=ALL,MAX_ITER=5,OUT_DIR=Final/smoke GARCH_to_PC_to_NPC_to_QHMM_revision.sh

MAX_ITER=${MAX_ITER:-1000}
OUT_DIR=${OUT_DIR:-Final/garch_T500}
QHMM_METHOD=${QHMM_METHOD:-SLSQP}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python GARCH_to_PC_to_NPC_to_QHMM_revision.py --output_path "${OUT_DIR}/sample_${SLURM_ARRAY_TASK_ID}.json" --dgp=garch --heston_params=base --len_sequences=500 --n_samples=1 --start_sample=${SLURM_ARRAY_TASK_ID} --seed=0 --tol=1e-5 --max_iter=${MAX_ITER} --qhmm_method=${QHMM_METHOD} --k=2 --ncl=4
