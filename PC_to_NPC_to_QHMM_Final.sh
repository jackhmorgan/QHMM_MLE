#!/bin/bash
#SBATCH -p general
#SBATCH --mem=64g
#SBATCH -t 7-
#SBATCH --mail-type=all
#SBATCH --mail-user=morganj@business.unc.edu
#SBATCH --output=Final/final_16.out
#SBATCH --job-name=final_16

python PC_to_NPC_to_QHMM_Final.py --path "Final/final_4.json" --n_samples=100 --tol=1e-5 --max_iter=1000 --k=2 --ncl=4 --len_sequence=500