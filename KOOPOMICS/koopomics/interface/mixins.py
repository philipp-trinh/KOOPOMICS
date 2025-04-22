import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Any, Tuple

from ..config import ConfigManager
from ..model import build_model_from_config, KoopmanModel
from ..training import create_trainer, OmicsDataloader
from ..test.test_utils import Evaluator, NaiveMeanPredictor
from ..interpret.interpret import KoopmanDynamics
from ..wandb_utils import GridSweepManager, BayesSweepManager
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============= MODEL MANAGEMENT MIXIN =============
class ModelManagementMixin:
    """
    Mixin providing model building, saving, and loading functionality.
    """
    
    def build_model(self) -> KoopmanModel:
        """
        Build the Koopman model based on configuration.
        
        Returns:
            KoopmanModel: Constructed Koopman model
        """
        logger.info("Building Koopman model...")
        self.model = build_model_from_config(self.config)
        logger.info(f"Model built with architecture: {self.model.__class__.__name__}")
        return self.model
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Parameters:
            path: Path to save the model
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or build a model.")
        
        logger.info(f"Saving model to {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
         
        # Save model
        torch.save(self.model.state_dict(), path)
        
        logger.info(f"Model saved successfully to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model.
        
        Parameters:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Loading model from {path}")
        
        # Build model if not already built
        if self.model is None:
            logger.info("Building model architecture")
            self.build_model()
        
        # Load model
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))
        
        logger.info("Model loaded successfully")


    def save_config(self, path: str) -> None:
        """
        Save the configuration.
        
        Parameters:
            path: Path to save the configuration
        """
        logger.info(f"Saving configuration to {path}")
        self.config.save_config(path)
        logger.info(f"Configuration saved successfully to {path}")

    def load_config(self, path: str) -> None:
        """
        Load configuration from file.
        
        Parameters:
            path: Path to load the configuration from
        """
        logger.info(f"Loading configuration from {path}.")
        self.config.load_config(path)
        logger.info(f"Configuration loaded successfully from {path}.")
        print(self.config.config)


# ============= TRAINING MIXIN =============
class TrainingMixin:
    """
    Mixin providing model training functionality.
    """
    
    def train(self, data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
             feature_list: Optional[List[str]] = None,
             replicate_id: Optional[str] = None,
             use_wandb: bool = False,
             model_dict_save_dir = None,
             group: Optional[str] = None,
             project_name: Optional[str] = 'KOOPOMICS') -> float:
        """
        Train the Koopman model.
        
        Parameters:
            data: Optional data to train on (if None, use previously loaded data)
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            use_wandb: Whether to use Weights & Biases for logging
            model_dict_save_dir: Directory to save model dictionaries
            group: Optional group name for W&B
            
        Returns:
            float: Best validation metric (baseline ratio)
        """
        # Load data if provided
        if data is not None:
            logger.info("Loading new data for training")
            self.load_data(data, feature_list, replicate_id)
        
        # Check if data is loaded
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("Data not loaded. Call load_data() first or provide data to train().")
        
        # Build model if not already built
        logger.info("Building model")
        self.build_model()
        
        # Create baseline model for comparison
        logger.info("Creating baseline model")
        baseline = NaiveMeanPredictor(self.train_loader, mask_value=self.mask_value)
        
        # Create trainer
        logger.info(f"Creating trainer with mode: {self.config.training_mode}")

        # If model_dict_save_dir is None, use the current working directory
        if model_dict_save_dir is None:
            model_dict_save_dir = os.getcwd()
            logger.info(f"No save directory provided. Using current working directory: {model_dict_save_dir}")

        # Create the trainer
        self.trainer = create_trainer(
            self.model,
            self.train_loader,
            self.test_loader,
            self.config,
            self.mask_value,
            baseline=baseline,
            use_wandb=use_wandb,
            model_dict_save_dir=model_dict_save_dir,
            group=group,
            project_name=project_name
        )

        # Log the save directory
        logger.info(f"Saving model_param_dicts into: {model_dict_save_dir}")

        # Train model
        logger.info("Starting training")
        best_metrics = self.trainer.train()
        logger.info(f"Training completed with best metric:")
        logger.info(best_metrics)


        return best_metrics
    
    def sweep_params(self, 
                    project_name: str, 
                    entity: Optional[str] = None,
                    CV_save_dir: Optional[str] = None,
                    sweep_method: str = 'bayes') -> Union[GridSweepManager, BayesSweepManager]:
        """
        Create a SweepManager for hyperparameter tuning with W&B.
        
        Parameters:
            project_name: Name of the W&B project
            entity: Optional W&B entity/organization
            CV_save_dir: Optional directory to save cross-validation results
            sweep_method: Type of sweep ('grid' or 'bayes')
            
        Returns:
            SweepManager: Manager for hyperparameter sweeps
        """
        logger.info(f"Creating Koopman SweepManager (method: {sweep_method})")

        # Common arguments
        sweep_args = {
            'project_name': project_name,
            'entity': entity,
            'CV_save_dir': CV_save_dir
        }

        # Add data-related args if loaders exist
        if self.train_loader is not None and self.test_loader is not None:
            sweep_args.update({
                'data': self.data,
                'condition_id': self.condition_id,
                'time_id': self.time_id,
                'replicate_id': self.replicate_id,
                'feature_list': self.feature_list,
                'mask_value': self.mask_value,
                'parent_yaml': self.yaml_path
            })

        # Instantiate the appropriate sweep manager
        if sweep_method == 'grid':
            sweep = GridSweepManager(**sweep_args)
        elif sweep_method == 'bayes':
            sweep = BayesSweepManager(**sweep_args)
        else:
            raise ValueError(f"Unknown sweep method: {sweep_method}. "
                        f"Supported methods are 'grid' and 'bayes'")

        logger.info(f"{type(sweep).__name__} created successfully")
        return sweep


# ============= PREDICTION & EVALUATION MIXIN =============
class PredictionEvaluationMixin:
    """
    Mixin providing prediction and evaluation functionality.
    """
    
    def predict(self, data: Union[pd.DataFrame, torch.Tensor],
               feature_list: Optional[List[str]] = None,
               replicate_id: Optional[str] = None,
               steps_forward: int = 1,
               steps_backward: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the trained model.
        
        Parameters:
            data: Data to predict on
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            steps_forward: Number of steps to predict forward
            steps_backward: Number of steps to predict backward
            
        Returns:
            Tuple of (backward_predictions, forward_predictions)
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Validate steps
        if steps_forward < 0 or steps_backward < 0:
            raise ValueError("steps_forward and steps_backward must be non-negative.")
        
        if steps_forward == 0 and steps_backward == 0:
            raise ValueError("At least one of steps_forward or steps_backward must be positive.")
        
        logger.info(f"Making predictions with steps_forward={steps_forward}, steps_backward={steps_backward}")
        
        # Convert data to tensor if needed
        if isinstance(data, pd.DataFrame):
            if feature_list is None or replicate_id is None:
                raise ValueError("feature_list and replicate_id are required for DataFrame input.")
            
            logger.info("Creating temporary data loader for prediction")
            # Create temporary data loader
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=1,
                max_Kstep=max(steps_forward, steps_backward),
                dl_structure='random',
                shuffle=False,
                mask_value=self.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed
            )
            
            # Get data loader
            data_loader = temp_loader.get_dataloaders()[0]  # Get only the first loader
            
            # Get first batch
            for batch in data_loader:
                input_data = batch[0].to(self.config.device)
                break
        else:
            input_data = data.to(self.config.device)
        
        # Make predictions
        logger.info("Running model prediction")
        self.model.eval()
        with torch.no_grad():
            backward_predictions, forward_predictions = self.model.predict(
                input_data,
                fwd=steps_forward,
                bwd=steps_backward
            )
        
        logger.info("Prediction completed")
        return backward_predictions, forward_predictions
    
    def evaluate(self, data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                feature_list: Optional[List[str]] = None,
                replicate_id: Optional[str] = None,
                compare_to_baseline: bool = True,
                feature_wise: bool = False,
                evaluate_embedding: bool = False) -> Dict[str, Any]:
        """
        Evaluate the trained model with various metrics.
        
        Parameters:
            data: Data to evaluate on (if None, use test data from load_data())
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            compare_to_baseline: Whether to compare model performance to baseline
            feature_wise: Whether to compute per-feature prediction errors
            evaluate_embedding: Whether to evaluate embedding quality
            
        Returns:
            Dict of evaluation metrics
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Starting model evaluation")
        
        # Use test data if no data provided
        if data is None:
            if self.test_loader is None:
                raise ValueError("Test data not loaded. Call load_data() first or provide data to evaluate().")
            
            logger.info("Using existing test data for evaluation")
            test_loader = self.test_loader
        else:
            logger.info("Creating temporary data loader for evaluation")
            # Create temporary data loader
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=self.config.batch_size,
                max_Kstep=self.config.max_Kstep,
                dl_structure=self.config.dl_structure,
                shuffle=False,
                mask_value=self.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed
            )
            
            # Get data loader
            test_loader = temp_loader.get_dataloaders()[0]  # Get only the first loader
        
        # Create baseline if needed
        baseline = None
        if compare_to_baseline:
            logger.info("Creating baseline model for comparison")
            baseline = NaiveMeanPredictor(self.train_loader, mask_value=self.mask_value)
        
        # Create evaluator
        logger.info("Creating evaluator")
        evaluator = Evaluator(
            self.model,
            self.train_loader,
            test_loader,
            mask_value=self.mask_value,
            max_Kstep=self.config.max_Kstep,
            baseline=baseline,
            model_name=self.model.__class__.__name__,
            criterion=None,
            loss_weights=self.config.loss_weights
        )
        
        # Evaluate model
        logger.info("Running evaluation")
        _, test_metrics, baseline_metrics = evaluator()
        
        # Combine metrics
        result_metrics = {**test_metrics}
        
        if compare_to_baseline and baseline_metrics:
            result_metrics.update({
                f"baseline_{k}": v for k, v in baseline_metrics.items()
            })
            
            # Calculate baseline ratio
            combined_test_loss = (test_metrics['forward_loss'] + test_metrics['backward_loss']) / 2
            combined_baseline_loss = (baseline_metrics['forward_loss'] + baseline_metrics['backward_loss']) / 2
            baseline_ratio = (combined_baseline_loss - combined_test_loss) / combined_baseline_loss
            result_metrics['baseline_ratio'] = baseline_ratio
            
            logger.info(f"Evaluation completed with baseline ratio: {baseline_ratio:.6f}")
        else:
            logger.info("Evaluation completed")
        
        # Compute feature-wise prediction errors if requested
        if feature_wise:
            logger.info("Computing feature-wise prediction errors")
            # Store feature errors as instance variable for easy access
            self.feature_errors = evaluator.compute_prediction_errors(test_loader)
            result_metrics.update({
                'feature_errors': self.feature_errors
            })
            # Add summary statistics for feature errors
            if len(self.feature_errors['fwd_feature_errors']) > 0:
                fwd_errors = self.feature_errors['fwd_feature_errors']
                best_feature = min(fwd_errors, key=fwd_errors.get)
                worst_feature = max(fwd_errors, key=fwd_errors.get)
                result_metrics.update({
                    'best_predicted_feature': best_feature,
                    'best_feature_error': fwd_errors[best_feature],
                    'worst_predicted_feature': worst_feature,
                    'worst_feature_error': fwd_errors[worst_feature]
                })
                
                # Create a mapping of feature names to errors if feature_list is available
                if hasattr(self, 'feature_list'):
                    # Forward prediction errors
                    self.feature_names_to_fwd_errors = {}
                    for idx, error in fwd_errors.items():
                        if idx < len(self.feature_list):
                            self.feature_names_to_fwd_errors[self.feature_list[idx]] = error
                    
                    # Backward prediction errors if available
                    if 'bwd_feature_errors' in self.feature_errors:
                        bwd_errors = self.feature_errors['bwd_feature_errors']
                        self.feature_names_to_bwd_errors = {}
                        for idx, error in bwd_errors.items():
                            if idx < len(self.feature_list):
                                self.feature_names_to_bwd_errors[self.feature_list[idx]] = error
                    
                    # Create sorted lists for easier analysis
                    self.sorted_features_by_fwd_error = sorted(
                        self.feature_names_to_fwd_errors.items(),
                        key=lambda x: x[1]
                    )
        
        # Evaluate embedding quality if requested
        if evaluate_embedding:
            logger.info("Evaluating embedding quality")
            # Store embedding metrics as instance variables
            self.embedding_metrics, baseline_embedding_metrics = evaluator.metrics_embedding()
            result_metrics.update({
                'embedding_metrics': self.embedding_metrics
            })
            
            if compare_to_baseline and baseline_embedding_metrics:
                self.baseline_embedding_metrics = baseline_embedding_metrics
                result_metrics.update({
                    'baseline_embedding_metrics': baseline_embedding_metrics
                })
                
                # Calculate embedding improvement ratio
                embedding_ratio = (baseline_embedding_metrics['identity_loss'] - self.embedding_metrics['identity_loss']) / baseline_embedding_metrics['identity_loss']
                result_metrics['embedding_improvement_ratio'] = embedding_ratio
        
        # Store last evaluation results for easy access
        self.last_evaluation_results = result_metrics
        
        return result_metrics
    
    def get_feature_errors(self, direction: str = 'forward', top_n: int = None, threshold: float = None,
                          sort_ascending: bool = True) -> Dict[str, float]:
        """
        Get feature prediction errors with convenient filtering and sorting options.
        
        Parameters:
            direction: Error direction, either 'forward' or 'backward' (default: 'forward')
            top_n: Return only the top N features by error (default: all features)
            threshold: Filter features with errors above/below this threshold
            sort_ascending: Sort by error in ascending order (default: True, best features first)
            
        Returns:
            Dict mapping feature names to their prediction errors
        """
        if self.feature_errors is None:
            raise ValueError("No feature errors available. Run evaluate() with feature_wise=True first.")
            
        # Get the appropriate error dictionary based on direction
        if direction.lower() == 'forward':
            if hasattr(self, 'feature_names_to_fwd_errors'):
                error_dict = self.feature_names_to_fwd_errors
            else:
                # Fall back to numeric indices if feature names aren't mapped
                error_dict = {str(idx): err for idx, err in self.feature_errors['fwd_feature_errors'].items()}
        elif direction.lower() == 'backward':
            if hasattr(self, 'feature_names_to_bwd_errors'):
                error_dict = self.feature_names_to_bwd_errors
            else:
                # Fall back to numeric indices if feature names aren't mapped
                error_dict = {str(idx): err for idx, err in self.feature_errors['bwd_feature_errors'].items()}
        else:
            raise ValueError("Direction must be either 'forward' or 'backward'")
            
        # Sort the errors
        sorted_items = sorted(error_dict.items(), key=lambda x: x[1], reverse=not sort_ascending)
        
        # Apply threshold filter if provided
        if threshold is not None:
            if sort_ascending:
                sorted_items = [(k, v) for k, v in sorted_items if v <= threshold]
            else:
                sorted_items = [(k, v) for k, v in sorted_items if v >= threshold]
                
        # Apply top_n limit if provided
        if top_n is not None:
            sorted_items = sorted_items[:top_n]
            
        # Return as dictionary
        return dict(sorted_items)


# ============= VISUALIZATION MIXIN =============
class VisualizationMixin:
    """
    Mixin providing visualization functionality.
    """
    
    def plot_feature_errors(self, n_features: int = 20, direction: str = 'forward',
                           show_feature_names: bool = True) -> None:
        """
        Plot feature prediction errors as a bar chart.
        
        Parameters:
            n_features: Number of features to show (default: 20)
            direction: Error direction, either 'forward', 'backward', or 'both' (default: 'forward')
            show_feature_names: Whether to show feature names on the x-axis (default: True)
        """
        import matplotlib.pyplot as plt
        
        if self.feature_errors is None:
            raise ValueError("No feature errors available. Run evaluate() with feature_wise=True first.")
            
        if direction.lower() not in ['forward', 'backward', 'both']:
            raise ValueError("Direction must be one of: 'forward', 'backward', 'both'")
        
        # Get feature errors
        if direction.lower() in ['forward', 'both']:
            fwd_errors = self.get_feature_errors('forward', top_n=n_features)
        
        if direction.lower() in ['backward', 'both']:
            bwd_errors = self.get_feature_errors('backward', top_n=n_features)
            
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        if direction.lower() == 'both':
            # Plot both forward and backward errors
            indices = range(len(fwd_errors))
            width = 0.35
            
            plt.bar([i - width/2 for i in indices], list(fwd_errors.values()), width=width,
                   label='Forward Prediction Error')
            plt.bar([i + width/2 for i in indices], list(bwd_errors.values()), width=width,
                   label='Backward Prediction Error')
            
            if show_feature_names:
                plt.xticks(indices, list(fwd_errors.keys()), rotation=90)
            else:
                plt.xticks(indices)
                
            plt.legend()
            
        else:
            # Plot single direction
            errors = fwd_errors if direction.lower() == 'forward' else bwd_errors
            plt.bar(range(len(errors)), list(errors.values()))
            
            if show_feature_names:
                plt.xticks(range(len(errors)), list(errors.keys()), rotation=90)
                
            plt.title(f"{direction.capitalize()} Prediction Errors by Feature")
            
        plt.xlabel("Features")
        plt.ylabel("Prediction Error (MSE)")
        plt.tight_layout()
        plt.show()


# ============= INTERPRETATION MIXIN =============
class InterpretationMixin:
    """
    Mixin providing model interpretation functionality.
    """
    
    def get_embeddings(self, data: Union[pd.DataFrame, torch.Tensor],
                      feature_list: Optional[List[str]] = None,
                      replicate_id: Optional[str] = None) -> torch.Tensor:
        """
        Get embeddings from the data using the trained model.
        
        Parameters:
            data: Data to get embeddings for
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            
        Returns:
            torch.Tensor: Embeddings
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Generating embeddings")
        
        # Convert data to tensor if needed
        if isinstance(data, pd.DataFrame):
            if feature_list is None or replicate_id is None:
                raise ValueError("feature_list and replicate_id are required for DataFrame input.")
            
            # Create temporary data loader
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=1,
                max_Kstep=1,
                dl_structure='random',
                shuffle=False,
                mask_value=self.config.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed
            )
            
            # Get data loader
            data_loader = temp_loader.get_dataloaders()[0]
            
            # Get first batch
            for batch in data_loader:
                input_data = batch[0].to(self.config.device)
                break
        else:
            input_data = data.to(self.config.device)
        
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.embedding.encode(input_data)
        
        logger.info("Embeddings generated")
        return embeddings
    
    def get_koopman_matrix(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the Koopman matrix (or matrices) from the trained model.
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Forward Koopman matrix, or tuple of (forward, backward) matrices
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Extracting Koopman matrix")
        
        # Get Koopman matrices
        kmatrices = self.model.kmatrix(detach=True)
        
        if isinstance(kmatrices, tuple):
            logger.info("Forward and backward Koopman matrices extracted")
            return kmatrices
        else:
            logger.info("Forward Koopman matrix extracted")
            return kmatrices
    
    def get_eigenvalues(self, plot: bool = False) -> Tuple:
        """
        Get the eigenvalues and eigenvectors of the Koopman matrix.
        
        Parameters:
            plot: Whether to plot the eigenvalues
            
        Returns:
            Tuple of eigenvalues and eigenvectors
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Computing eigendecomposition")
        
        return self.model.eigen(plot=plot)
        
    def get_dynamics(self, dataset_df: Optional[pd.DataFrame] = None,
                    test_set_df: Optional[pd.DataFrame] = None) -> KoopmanDynamics:
        """
        Create a KoopmanDynamics interpreter for analyzing and visualizing model dynamics.
        
        Parameters:
            dataset_df: Optional DataFrame for analysis (default: uses data from load_data())
            test_set_df: Optional test set DataFrame for evaluation
            
        Returns:
            KoopmanDynamics: An interpreter object for analyzing Koopman dynamics
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
            
        # Check if data is loaded
        if self.data_loader is None and dataset_df is None:
            raise ValueError("No data loaded. Call load_data() first or provide dataset_df")
        
        logger.info("Creating KoopmanDynamics interpreter")
        
        # Use data parameters from data_loader
        data_config = self.config.get_data_config()

        
        # Create and return KoopmanDynamics interpreter
        dynamics = KoopmanDynamics(
            model=self.model,
            dataset_df=dataset_df,
            feature_list=self.feature_list,
            replicate_id=self.replicate_id,
            time_id=self.time_id,
            condition_id=self.condition_id,
            mask_value=self.mask_value,
            device=self.config.device,
            test_set_df=test_set_df
        )
        
        logger.info("KoopmanDynamics interpreter created successfully")
        return dynamics


# ============= INITIALIZATION MIXIN =============
class InitializationMixin:
    """
    Mixin providing initialization and setup functionality.
    """

    def _init_from_run(self, run_id: str, model_dir: str):
        """Initialize from saved run"""
        self.model_dict_save_dir = Path(model_dir)
        param_path = self._get_param_file_path(run_id)
        state_path = self._get_state_file_path(run_id)
        
        self.config = ConfigManager(param_path)
        self.build_model()
        self.load_model(state_path)
        logger.info(f"Loaded model from run {run_id}")
        
    def _get_param_file_path(self, run_id: str, warn_multiple: bool = True) -> str:
        param_files = list(self.model_dict_save_dir.glob(f"{run_id}_*.json"))
        
        if not param_files:
            raise FileNotFoundError(f"No parameter files found for run {run_id} in {self.model_dict_save_dir}")
            
        if warn_multiple and len(param_files) > 1:
            print(f"Warning: Multiple parameter files found for {run_id}, using {param_files[0].name}")
            
        return str(param_files[0])
    
    def _get_state_file_path(self, run_id: str, warn_multiple: bool = True) -> str:
        state_files = list(self.model_dict_save_dir.glob(f"{run_id}_KoopmanModel*.pth"))
        
        if not state_files:
            raise FileNotFoundError(f"No state files found for run {run_id} in {self.model_dict_save_dir}")
            
        if warn_multiple and len(state_files) > 1:
            print(f"Warning: Multiple state files found for {run_id}, using {state_files[0].name}")
            
        return str(state_files[0])



    def _init_from_config(self, config):
        """Initialize fresh from config"""
        self.config = ConfigManager(config) if not isinstance(config, ConfigManager) else config
        self.build_model()
        logger.info("Initialized new model from config")
        
        
    def _init_components(self):
        """Initialize common components"""

        # Initialize model, data loader, and trainer to None
        self.model = None
        self.data_loader = None
        self.train_loader = None
        self.test_loader = None
        self.trainer = None
        
        # Initialize evaluation metrics storage
        self.feature_errors = None
        self.embedding_metrics = None
        self.last_evaluation_results = None


    def _set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"Random seed set to {seed}")
