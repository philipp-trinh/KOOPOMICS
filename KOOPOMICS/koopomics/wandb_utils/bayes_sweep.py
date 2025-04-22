import os
import wandb
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import torch
import numpy as np
import json
import yaml
from pathlib import Path
import time
import subprocess
from importlib.resources import files


from .base_sweep import BaseSweepManager, OuterCVExecutor

# Configure logging
logger = logging.getLogger(__name__)

class BayesSweepManager(BaseSweepManager):
    """
    Manager for wandb Single sweeps with cross-validation.
    
    Unlike GridSweepManager which creates separate sweeps for each max_Kstep-dl_structure combination,
    SingleSweepManager runs one sweep per outer_split that includes all max_Kstep and dl_structure
    combinations within a single config. This allows testing all combinations in a single slurm job .
    """
    
    def __init__(self, 
                 project_name: str, 
                 entity: Optional[str] = None,
                 CV_save_dir: Optional[str] = None,
                 data: Optional[Union[pd.DataFrame, Path]] = None,
                 condition_id: Optional[str] = None,
                 time_id: Optional[str] = None,
                 replicate_id: Optional[str] = None,
                 feature_list: Optional[List[str]] = None,
                 mask_value: Optional[float] = None,
                 parent_yaml: Optional[Union[str, Path]] = None):
        
        # Initialize parent class with all explicit parameters
        super().__init__(
            project_name=project_name,
            entity=entity,
            CV_save_dir=CV_save_dir,
            data=data,
            condition_id=condition_id,
            time_id=time_id,
            replicate_id=replicate_id,
            feature_list=feature_list,
            mask_value=mask_value,
            parent_yaml=parent_yaml
        )

        # Bayes-specific initialization
        self.results_manager.single_sweep = True

    def init_nested_cv(self, outer_num_folds: int = 5,
                      inner_num_folds: int = 4):
        """Initialize nested cross-validation with a single sweep per split"""
        # Create datasets directory only once
        datasets_dir = f"{self.CV_save_dir}/nested_cv_single_sweep"
        os.makedirs(datasets_dir, exist_ok=True)
        self.data_manager.cv_save_dir = datasets_dir

        # First prepare all the necessary data
        data_dir = self.data_manager.prepare_nested_cv_data(
            dl_structure='temporal', 
            max_Kstep=1,
            outer_num_folds=outer_num_folds, 
            inner_num_folds=inner_num_folds)

        max_Ksteps = [int(k) for k in sorted(self.data['time_id'].unique())[:-1]]
        dl_structures = ['random', 'temp_segm', 'temp_delay', 'temporal']

        # Now initialize a single sweep that includes all combinations
        self.init_sweep(dl_structures, max_Ksteps, data_dir, outer_num_folds, inner_num_folds)
    
    def init_sweep(self, dl_structures, max_Ksteps, data_dir, outer_num_folds, inner_num_folds):
        """Initialize a single sweep that includes all dl_structure and max_Kstep combinations"""
        # Create model dict save dir if needed
        if self.model_dict_save_dir is None:
            self.model_dict_save_dir = f"{self.CV_save_dir}/CV_single_sweep_model_dicts"
            os.makedirs(self.model_dict_save_dir, exist_ok=True)
            
        # Prepare base sweep config
        base_sweep_config = self.config_manager.config_dict
        
        # Update config to include all dl_structures and max_Ksteps as categorical options
        base_sweep_config["parameters"]["dl_structure"] = {
            "distribution": "categorical",
            "values": dl_structures
        }
        
        base_sweep_config["parameters"]["max_Kstep"] = {
            "distribution": "categorical", 
            "values": max_Ksteps[1:]
        }
        
        # Create sweep info dictionary
        sweep_info = {}

        init_script = files("koopomics.wandb_utils").joinpath("init_sweep.py")

        # Initialize one sweep per outer split (not per dl_structure/max_Kstep combination)
        for outer_split in range(outer_num_folds):
            print(f"Processing outer split {outer_split}...")
            
            # Create unique sweep config
            sweep_config = base_sweep_config.copy()
            sweep_name = f"{self.project_name}_{outer_split}_single_sweep"
            sweep_config["name"] = sweep_name
            sweep_config["parameters"]["sweep_name"] = {"value": outer_split}
            
            # Initialize sweep
            #sweep_id = wandb.sweep(sweep_config, project=self.project_name, entity=self.entity)
            
            process = subprocess.Popen(
                ["python", "-c", 
                f"import wandb,json,sys;print(wandb.sweep(json.load(sys.stdin),project='{self.project_name}'))"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            output, stderr = process.communicate(input=json.dumps(sweep_config))

            if process.returncode == 0:
                    sweep_id = output.strip().split('\n')[-1]  # Will contain only the sweep ID
                    print(f"Initialized sweep with ID: {sweep_id}")
            else:
                print(f"Error: {stderr}")

            # Force immediate cleanup of sweep resources
            time.sleep(1)  # Brief pause for resource release
            wandb.Api().flush()  # Ensure all network calls complete
            
                   
            # Create job name
            job_name = f"{self.project_name}_{outer_split}_single_sweep"
            
            # Save parameters - include all tensor paths
            sweep_info[job_name] = {
                "sweep_id": sweep_id,
                "train_config_path": f"{data_dir}/split_{outer_split}/train_config.yaml",  # To call all train sets
                "dl_structure": 'single_sweep',
                "max_Kstep": max_Ksteps,
                "outer_split": outer_split,
                "mask_value": self.data_manager.mask_value,
                "sweep_name": sweep_name,
                "inner_cv_num_folds": inner_num_folds,
                "num_inner_folds_to_use": self.num_inner_folds_to_use,
                "project_name": self.project_name,
            }
        
        # Save sweep info
        sweep_info_file = f"{data_dir}/{self.project_name}_single_sweep_info.json"
        with open(sweep_info_file, "w") as f:
            json.dump(sweep_info, f, indent=4)
        
        # Register sweep
        self.sweep_registry.save_sweep_info(
            sweep_info_file=sweep_info_file,
            dl_structure="single_sweep",  # Mark as single sweep
            max_Kstep=max(max_Ksteps),    # Use max Kstep for identification
            project_name=self.project_name
        )
        
        print(f"Single sweep information saved to {sweep_info_file}")

        # Final cleanup
        wandb.finish()
        time.sleep(2)  # Allow proper network teardown

        return sweep_info_file
    
    def run_complete_pipeline(self, outer_num_folds=5, inner_num_folds=4, num_sweep_replicates=4):
        """Run complete hyperparameter optimization pipeline with single sweeps"""
        # 1. Initialize CV data with single sweeps
        self.init_nested_cv(
            outer_num_folds=outer_num_folds,
            inner_num_folds=inner_num_folds
        )
        
        # 2. Submit all sweeps
        self.submit_all_sweeps(num_replicates=num_sweep_replicates)
        
        # 3. Monitor jobs
        self.monitor_jobs()
            
        # 4. Process results
        self.final_df, self.best_models = self.process_results()
        
        logger.info("Single sweep hyperparameter optimization pipeline completed successfully")

        return self.best_models

    def prepare_outer_cv(self):

        self.outer_cv_exec = OuterCVExecutor(self.CV_save_dir, self.results_manager.configs_list,
                                             data_config_path=self.parent_yaml)
        self.outer_cv_exec.single_sweep = True
        return self.outer_cv_exec
    
    def process_results(self):

        self.best_models = self.run_outer_cv()
        self.final_df = self.outer_cv_exec.result_csv()

        return self.final_df, self.best_models