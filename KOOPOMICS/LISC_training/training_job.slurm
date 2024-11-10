#!/bin/bash
#SBATCH --job-name=train_koopman_model           # Name of the job
#SBATCH --output=logs/train_model_%j.log # Log file (with %j for job ID)
#SBATCH --error=logs/train_model_%j.err  # Error file
#SBATCH --partition=gpu                  # Partition (queue) name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks per node
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --gres=gpu:1                     # Number of GPUs (adjust if needed)
#SBATCH --time=5:00:00                  # Max time (HH:MM:SS)
#SBATCH --mem=4G                        # Memory per node (adjust if needed)
#SBATCH --mail-type=END,FAIL             # Notification options

module load conda

conda activate koopenv

python training_script.py


