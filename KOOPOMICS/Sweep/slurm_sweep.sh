#!/bin/bash

#SBATCH --job-name=param_sweep                # Job name
#SBATCH --array=0-9                            # Array range (0-9 for 10 tasks; adjust as needed)
#SBATCH --ntasks=1                             # Number of tasks per job
#SBATCH --cpus-per-task=2                      # Reduced CPUs
#SBATCH --mem=2G                              # Keeping memory as is
#SBATCH --time=01:00:00                        # Time limit hrs:min:sec
#SBATCH --output=sweep_output_%A_%a.out        # Output file for each array job
#SBATCH --error=sweep_error_%A_%a.err          # Error file for each array job

# Load conda and activate your environment
module load conda
conda activate koopenv


python Sweep.py
