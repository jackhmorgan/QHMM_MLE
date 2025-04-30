#!/bin/bash
#SBATCH -p general
#SBATCH --mem=64g
#SBATCH -t 4-
#SBATCH --mail-type=all
#SBATCH --mail-user=morganj@business.unc.edu
#SBATCH --output=spy_training/spy_training_test.out

python SPY_training.py --path "spy_training/final_results.json" --n_samples=100 --tol=1e-4 --max_iter=1000 --k=1 --ncl=4 --n_observations=4