# init_sweep.py
import wandb
import sys
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="W&B project name")
    args = parser.parse_args()

    # Read sweep config from stdin
    sweep_config = json.loads(sys.stdin.read())
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(sweep_id)
