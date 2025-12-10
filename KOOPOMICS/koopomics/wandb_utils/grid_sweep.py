import os
import logging
from koopomics.utils import torch, pd, np, wandb

from typing import Dict, List, Optional, Any, Union

import json
import yaml
from pathlib import Path

from .base_sweep import BaseSweepManager

# Configure logging
logger = logging.getLogger("koopomics")

class GridSweepManager(BaseSweepManager):
    """
    Manager for wandb Grid sweeps with cross-validation.
    
    Orchestrates the hyperparameter tuning process with grid search:
    - Data preparation with separate sweeps for each max_Kstep-dl_structure combination
    - Separate sweep configuration for each combination
    - CV execution
    - Results analysis
    """

    def init_nested_cv(self, dl_structures: List = ['random', 'temp_segm', 'temp_delay', 'temporal'], 
                      max_Ksteps: List = [1], outer_num_folds: int = 5,
                      inner_num_folds: int = 4):
        """Initialize nested cross-validation for multiple structures"""
        for Kstep in max_Ksteps:
            for dl_structure in dl_structures:
                datasets_dir = self.data_manager.prepare_nested_cv_data(
                    dl_structure=dl_structure, 
                    max_Kstep=Kstep,
                    outer_num_folds=outer_num_folds, 
                    inner_num_folds=inner_num_folds
                )
                self.init_sweep(dl_structure, Kstep, datasets_dir, outer_num_folds, inner_num_folds)
    
    def init_sweep(self, dl_structure, max_Kstep, datasets_dir, outer_num_folds, inner_num_folds):
        
        import wandb

        """Initialize sweeps for a specific data structure and Kstep"""
        # Create model dict save dir if needed
        if self.model_dict_save_dir is None:
            self.model_dict_save_dir = f"{self.CV_save_dir}/CV_{dl_structure}_model_dicts"
            os.makedirs(self.model_dict_save_dir, exist_ok=True)
            
        # Prepare base sweep config
        base_sweep_config = self.config_manager.config_dict
        base_sweep_config["parameters"]["dl_structure"] = {
            "distribution": "categorical",
            "values": [dl_structure]
        }
        base_sweep_config["parameters"]["max_Kstep"] = {
            "value": max_Kstep
        }
        
        # Load dataframes
        tensor_path = f"{datasets_dir}/saved_outer_cv_tensors.pth"
        outer_cv_tensors = torch.load(tensor_path)
        outer_splits = list(outer_cv_tensors.keys())
        
        # Create sweep info dictionary
        sweep_info = {}
        
        # Initialize sweeps for each outer split
        for outer_split in outer_splits:
            print(f"Processing {outer_split}...")
            
            # Create unique sweep config
            sweep_config = base_sweep_config.copy()
            sweep_name = f"{self.project_name}_{outer_split}_{dl_structure}_{max_Kstep}"
            sweep_config["name"] = sweep_name
            sweep_config["parameters"]["sweep_name"] = {"value": outer_split}
            
            # Initialize sweep
            sweep_id = wandb.sweep(sweep_config, project=self.project_name, entity=self.entity)
            
            # Create job name
            job_name = f"{self.project_name}_{outer_split}_{dl_structure}_{max_Kstep}"
            
            # Save parameters
            sweep_info[job_name] = {
                "sweep_id": sweep_id,
                "data_path": tensor_path,
                "dl_structure": dl_structure,
                "max_Kstep": max_Kstep,
                "outer_split": outer_split,
                "mask_value": self.data_manager.mask_value,
                "sweep_name": sweep_name,
                "inner_cv_num_folds": inner_num_folds,
                "num_inner_folds_to_use": self.num_inner_folds_to_use,
                "project_name": self.project_name,
            }
        
        # Save sweep info
        sweep_info_file = f"{datasets_dir}/{self.project_name}_{dl_structure}_{max_Kstep}_sweep_info.json"
        with open(sweep_info_file, "w") as f:
            json.dump(sweep_info, f, indent=4)
        
        # Register sweep
        self.sweep_registry.save_sweep_info(
            sweep_info_file=sweep_info_file,
            dl_structure=dl_structure,
            max_Kstep=max_Kstep,
            project_name=self.project_name
        )
        
        print(f"Sweep information saved to {sweep_info_file}")
        return sweep_info_file
    
    def run_complete_pipeline(self, dl_structures, max_Ksteps, outer_num_folds=5, inner_num_folds=4):
        """Run complete hyperparameter optimization pipeline"""
        # 1. Initialize CV data
        self.init_nested_cv(
            dl_structures=dl_structures,
            max_Ksteps=max_Ksteps,
            outer_num_folds=outer_num_folds,
            inner_num_folds=inner_num_folds
        )
        
        # 2. Submit all sweeps
        self.submit_all_sweeps(job_name_prefix=self.project_name)
        
        # 3. Monitor jobs
        self.monitor_jobs()
        
        # 4. Process results
        self.process_results()
        
        logger.info("Hyperparameter optimization pipeline completed successfully")
