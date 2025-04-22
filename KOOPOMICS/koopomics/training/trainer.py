import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any, Tuple

from .train_utils import Koop_Full_Trainer, Koop_Step_Trainer, Embedding_Trainer
from ..test.test_utils import NaiveMeanPredictor, Evaluator
from ..wandb_utils.wandb_utils import WandbManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Base class for all trainers.
    
    This class provides common functionality for all trainers.
    
    Attributes:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Testing data loader
        config (Dict[str, Any]): Training configuration
        device (torch.device): Device to use for training
        wandb_manager (WandbManager): Weights & Biases manager
    """
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, mask_value, 
                 use_wandb: bool = False, print_losses: bool = False,
                 model_dict_save_dir = None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the BaseTrainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.mask_value = mask_value
        self.device = config.device
        self.use_wandb = use_wandb
        self.group = group
        self.print_losses = print_losses
        self.model_dict_save_dir = model_dict_save_dir
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Get training configuration
        self.training_config = config.get_training_config()
        
        # Create baseline model
        self.baseline = NaiveMeanPredictor(train_loader, mask_value=self.mask_value)
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'test_loss': [],
            'baseline_ratio': []
        }
        
        # Initialize wandb manager if needed
        self.wandb_manager = None
        if self.use_wandb:
            self.wandb_manager = WandbManager(
                config=config.config,
                project_name=project_name,
                train_loader=train_loader,
                test_loader=test_loader,
                model_dict_save_dir=self.model_dict_save_dir,
                group=self.group
            )
    
    def train(self) -> float:
        """
        Train the model.
        
        Returns:
        --------
        float
            Best validation metric (baseline ratio)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                tags=[self.training_config['mode']]
            )
        
        try:
            # Implement training logic in subclasses
            raise NotImplementedError("Subclasses must implement train method")
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()
    
    def save_model(self, path: str) -> None:
        """
        Save the model.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        
        # Log model to wandb if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.log_model(self.model, self.model.__class__.__name__)
    
    def load_model(self, path: str) -> None:
        """
        Load the model.
        
        Parameters:
        -----------
        path : str
            Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Returns:
        --------
        Dict[str, Any]
            Evaluation metrics
        """
        # Create evaluator
        evaluator = Evaluator(
            self.model,
            self.train_loader,
            self.test_loader,
            mask_value=self.config.mask_value,
            max_Kstep=self.config.max_Kstep,
            baseline=self.baseline,
            model_name=self.model.__class__.__name__,
            criterion=None,
            loss_weights=self.config.loss_weights
        )
        
        # Evaluate model
        _, test_metrics, baseline_metrics = evaluator()
        
        # Calculate baseline ratio
        combined_test_loss = (test_metrics['forward_loss'] + test_metrics['backward_loss']) / 2
        combined_baseline_loss = (baseline_metrics['forward_loss'] + baseline_metrics['backward_loss']) / 2
        baseline_ratio = (combined_baseline_loss - combined_test_loss) / combined_baseline_loss
        
        # Add to metrics
        self.metrics['train_loss'].append(combined_test_loss)
        self.metrics['test_loss'].append(combined_test_loss)
        self.metrics['baseline_ratio'].append(baseline_ratio)
        
        # Create metrics dictionary
        metrics_dict = {
            'forward_loss': test_metrics['forward_loss'],
            'backward_loss': test_metrics['backward_loss'],
            'combined_loss': combined_test_loss,
            'baseline_forward_loss': baseline_metrics['forward_loss'],
            'baseline_backward_loss': baseline_metrics['backward_loss'],
            'baseline_combined_loss': combined_baseline_loss,
            'baseline_ratio': baseline_ratio
        }
        
        # Log metrics to wandb if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.log_metrics(metrics_dict)
        
        # Return metrics
        return metrics_dict
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics.
        
        Parameters:
        -----------
        save_path : Optional[str], default=None
            Path to save the plot. If None, the plot is displayed.
        """
        if not self.metrics['train_loss']:
            logger.warning("No metrics to plot")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.metrics['test_loss'], 'r-', label='Test Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot baseline ratio
        ax2.plot(epochs, self.metrics['baseline_ratio'], 'g-', label='Baseline Ratio')
        ax2.set_title('Baseline Ratio')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Log figure to wandb if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.log_figure(fig, 'training_metrics')
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()

class FullTrainer(BaseTrainer):
    """
    Trainer for full model training.
    
    This trainer trains the entire model (embedding and operator) at once.
    """
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = True,
                 model_dict_save_dir = None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the FullTrainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Wheter to print all losses per epoch
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
    
    def train(self) -> float:
        """
        Train the model.
        
        Returns:
        --------
        float
            Best validation metric (baseline ratio)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_full",
                tags=["full", self.training_config['backpropagation_mode']],
                group=self.group
            )
        
        try:
            # Get training parameters
            backpropagation_mode = self.training_config['backpropagation_mode']
            
            # Create trainer
            if backpropagation_mode == 'step':
                logger.info("Using step-wise backpropagation")
                trainer = Koop_Step_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.max_Kstep,
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    learning_rate_change=self.config.learning_rate_change,
                    num_epochs=self.config.num_epochs,
                    decayEpochs=self.config.decay_epochs,
                    loss_weights=self.config.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.verbose,
                    model_dict_save_dir = self.model_dict_save_dir

                )
            else:
                logger.info("Using full backpropagation")
                trainer = Koop_Full_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.max_Kstep,
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    learning_rate_change=self.config.learning_rate_change,
                    num_epochs=self.config.num_epochs,
                    decayEpochs=self.config.decay_epochs,
                    loss_weights=self.config.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.verbose,
                    model_dict_save_dir = self.model_dict_save_dir

                )
            
            # Train model
            best_baseline_ratio, best_fwd_loss, best_bwd_loss = trainer.train()
            
            # Save best model
            if hasattr(trainer, 'early_stopping') and hasattr(trainer.early_stopping, 'model_path'):
                self.load_model(trainer.early_stopping.model_path)
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model)
            
            return best_baseline_ratio, best_fwd_loss, best_bwd_loss
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

class ModularTrainer(BaseTrainer):
    """
    Trainer for modular model training.
    
    This trainer first trains the embedding module, then freezes it and trains the operator module.
    """
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = True,
                 model_dict_save_dir=None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the ModularTrainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Wheter to print all losses per epoch
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
        
        # Paths for saving intermediate models
        self.embedding_path = f"{self.model.__class__.__name__}_embedding.pth"
        self.operator_path = f"{self.model.__class__.__name__}_operator.pth"
    
    def train(self) -> float:
        """
        Train the model.
        
        Returns:
        --------
        float
            Best validation metric (baseline ratio)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_modular",
                tags=["modular", self.training_config['backpropagation_mode']],
                group=self.group
            )
        
        try:
            # Step 1: Train embedding
            logger.info("Training embedding module")
            embedding_trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                learning_rate_change=self.config.learning_rate_change,
                num_epochs=self.config.num_epochs,
                decayEpochs=self.config.decay_epochs,
                mask_value=self.mask_value,
                early_stop=self.config.early_stop,
                patience=self.config.patience,
                E_overfit_limit = self.config.E_overfit_limit,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                verbose=self.config.verbose,
                model_dict_save_dir = self.model_dict_save_dir

            )
            
            embedding_trainer.train()
            
            # Save embedding
            torch.save(self.model.embedding.state_dict(), self.embedding_path)
            logger.info(f"Embedding saved to {self.embedding_path}")
            
            # Log embedding to wandb if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.log_model(self.model)
            
            # Freeze embedding
            for param in self.model.embedding.parameters():
                param.requires_grad = False
            logger.info("Embedding parameters frozen")
            
            # Step 2: Train operator
            logger.info("Training operator module")
            backpropagation_mode = self.training_config['backpropagation_mode']
            
            # Create trainer
            if backpropagation_mode == 'step':
                logger.info("Using step-wise backpropagation")
                operator_trainer = Koop_Step_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.max_Kstep,
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    learning_rate_change=self.config.learning_rate_change,
                    num_epochs=self.config.num_epochs,
                    decayEpochs=self.config.decay_epochs,
                    loss_weights=self.config.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.verbose,
                    model_dict_save_dir = self.model_dict_save_dir

                )
            else:
                logger.info("Using full backpropagation")
                operator_trainer = Koop_Full_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.max_Kstep,
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    learning_rate_change=self.config.learning_rate_change,
                    num_epochs=self.config.num_epochs,
                    decayEpochs=self.config.decay_epochs,
                    loss_weights=self.config.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.verbose,
                    model_dict_save_dir = self.model_dict_save_dir

                )
            
            # Train operator
            best_baseline_ratio, best_fwd_loss, best_bwd_loss = operator_trainer.train()
            
            # Save best model
            if hasattr(operator_trainer, 'early_stopping') and hasattr(operator_trainer.early_stopping, 'model_path'):
                self.load_model(operator_trainer.early_stopping.model_path)
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model)
            
            return best_baseline_ratio, best_fwd_loss, best_bwd_loss
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

class ModularShiftTrainer(BaseTrainer):
    """
    Improved trainer for modular model training with sequential shift training.
    
    This trainer implements a sequential training approach:
    1. Train embedding module 
    2. Save parameters
    3. Load parameters and train shift 1 until early stop
    4. Save parameters
    5. Load parameters and train shift 2 until early stop
    6. Compare with shift 1 error and if worse, reset to shift 1 and don't continue
    7. If better, save parameters and train shift 3, and so on
    """
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = True,
                 model_dict_save_dir=None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the ModularShiftTrainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Whether to print all losses per epoch
        model_dict_save_dir : str, default=None
            Directory to save model dictionaries
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)

        self.num_shifts = self.config.max_Kstep
        
        # Paths for saving intermediate models
        self.embedding_path = f"{self.model.__class__.__name__}_embedding.pth"
        self.base_model_path = f"{self.model.__class__.__name__}_base.pth"
        self.shift_paths = [f"{self.model.__class__.__name__}_shift{i+1}.pth" for i in range(self.num_shifts)]
        
        # Track performance metrics for each shift
        self.shift_metrics = []
    
    def train_embedding(self):
        """
        Train the embedding module.
        
        Returns:
        --------
        float
            Best embedding identity loss
        """
        logger.info("Training embedding module")
        embedding_trainer = Embedding_Trainer(
            self.model,
            self.train_loader,
            self.test_loader,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            learning_rate_change=self.config.learning_rate_change,
            num_epochs=self.config.num_epochs,
            decayEpochs=self.config.decay_epochs,
            mask_value=self.mask_value,
            early_stop=self.config.early_stop,
            patience=self.config.patience,
            E_overfit_limit=self.config.E_overfit_limit,
            baseline=self.baseline,
            model_name=self.model.__class__.__name__,
            wandb_log=self.use_wandb,
            verbose=self.config.verbose,
            model_dict_save_dir=self.model_dict_save_dir
        )
        
        best_identity_loss = embedding_trainer.train()
        
        # Save embedding
        torch.save(self.model.state_dict(), self.embedding_path)
        logger.info(f"Embedding model saved to {self.embedding_path}")
        
        # Freeze embedding
        for param in self.model.embedding.parameters():
            param.requires_grad = False
        logger.info("Embedding parameters frozen")
        
        return best_identity_loss
    
    def train_shift(self, shift_idx):
        """
        Train a specific shift model.
        
        Parameters:
        -----------
        shift_idx : int
            Index of the shift to train (0-based)
            
        Returns:
        --------
        Tuple[float, float, float]
            Tuple of (baseline_ratio, forward_loss, backward_loss)
        """
        logger.info(f"Training shift {shift_idx+1}")
        
        # Set the shift start step and max step
        start_Kstep = shift_idx
        max_Kstep = shift_idx + 1
        
        # Create the appropriate trainer based on backpropagation mode
        backpropagation_mode = self.training_config['backpropagation_mode']
        
        trainer_kwargs = {
            "max_Kstep": max_Kstep,
            "start_Kstep": start_Kstep,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "learning_rate_change": self.config.learning_rate_change,
            "num_epochs": self.config.num_epochs,
            "decayEpochs": self.config.decay_epochs,
            "loss_weights": self.config.loss_weights,
            "mask_value": self.config.mask_value,
            "early_stop": self.config.early_stop,
            "patience": self.config.patience,
            "baseline": self.baseline,
            "model_name": self.model.__class__.__name__,
            "wandb_log": self.use_wandb,
            "verbose": self.config.verbose,
            "model_dict_save_dir": self.model_dict_save_dir
        }
        
        if backpropagation_mode == 'step':
            logger.info("Using step-wise backpropagation")
            trainer = Koop_Step_Trainer(model=self.model, train_dl=self.train_loader,
                                        test_dl=self.test_loader, **trainer_kwargs)
        else:
            logger.info("Using full backpropagation")
            trainer = Koop_Full_Trainer(model=self.model, train_dl=self.train_loader,
                                        test_dl=self.test_loader, **trainer_kwargs)
        
        # Train the shift
        best_baseline_ratio, best_fwd_loss, best_bwd_loss = trainer.train()
        
        # Save best model if early stopping was used
        if hasattr(trainer, 'early_stopping') and hasattr(trainer.early_stopping, 'model_path'):
            # Load the best model from early stopping
            self.model.load_state_dict(torch.load(trainer.early_stopping.model_path, map_location=self.device))
            logger.info(f"Loaded best model for shift {shift_idx+1} from {trainer.early_stopping.model_path}")
        
        # Save the trained shift model
        shift_path = self.shift_paths[shift_idx]
        torch.save(self.model.state_dict(), shift_path)
        logger.info(f"Shift {shift_idx+1} model saved to {shift_path}")
        
        # Calculate combined loss
        combined_loss = (best_fwd_loss + best_bwd_loss) / 2
        
        # Store metrics for this shift
        shift_metrics = {
            'shift_idx': shift_idx,
            'baseline_ratio': best_baseline_ratio,
            'forward_loss': best_fwd_loss,
            'backward_loss': best_bwd_loss,
            'combined_loss': combined_loss,
            'model_path': shift_path
        }
        
        self.shift_metrics.append(shift_metrics)
        
        return best_baseline_ratio, best_fwd_loss, best_bwd_loss, combined_loss
    
    def train(self):
        """
        Train the model using the sequential shift training approach.
        
        Returns:
        --------
        Dict
            Best shift metrics
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_modular_shift",
                tags=["modular_shift", self.training_config['backpropagation_mode']],
                group=self.group
            )
        
        try:
            # Step 1: Train embedding
            embedding_loss = self.train_embedding()
            logger.info(f"Embedding training completed with best identity loss: {embedding_loss}")
            
            # Save base model with trained embedding
            torch.save(self.model.state_dict(), self.base_model_path)
            logger.info(f"Base model saved to {self.base_model_path}")
            
            # Step 2: Train each shift sequentially
            best_shift_metrics = None
            best_shift_idx = -1
            best_combined_loss = float('inf')
            
            for shift_idx in range(self.num_shifts):
                # Always start from the base model with trained embedding
                self.model.load_state_dict(torch.load(self.base_model_path, map_location=self.device))
                logger.info(f"Loaded base model for training shift {shift_idx+1}")
                
                # Train this shift
                baseline_ratio, fwd_loss, bwd_loss, combined_loss = self.train_shift(shift_idx)
                
                # Log shift metrics
                logger.info(f"Shift {shift_idx+1} training results:")
                logger.info(f"  Baseline ratio: {baseline_ratio:.6f}")
                logger.info(f"  Forward loss: {fwd_loss:.6f}")
                logger.info(f"  Backward loss: {bwd_loss:.6f}")
                logger.info(f"  Combined loss: {combined_loss:.6f}")
                
                # Update best shift if this one is better
                if combined_loss < best_combined_loss:
                    best_combined_loss = combined_loss
                    best_shift_idx = shift_idx
                    best_shift_metrics = self.shift_metrics[shift_idx]
                    logger.info(f"New best shift: Shift {shift_idx+1} with combined loss {combined_loss:.6f}")
                else:
                    logger.info(f"Shift {shift_idx+1} performance is worse than best shift {best_shift_idx+1} "
                                f"({combined_loss:.6f} > {best_combined_loss:.6f})")
                    if shift_idx > 0:  # Only break after at least training shift 1 and 2
                        logger.info(f"Stopping shift training as performance degraded")
                        break
            
            # Load the best shift model
            if best_shift_idx >= 0:
                best_model_path = self.shift_paths[best_shift_idx]
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                logger.info(f"Loaded best shift model (Shift {best_shift_idx+1}) from {best_model_path}")
                
                # Final evaluation
                final_metrics = self.evaluate()
                logger.info("Final model evaluation:")
                for metric_name, metric_value in final_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.6f}")
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model)
                    
                best_baseline_ratio = best_shift_metrics['baseline_ratio']
                best_fwd_loss = best_shift_metrics['forward_loss']
                best_bwd_loss = best_shift_metrics['backward_loss']
                
                return best_baseline_ratio, best_fwd_loss, best_bwd_loss
            else:
                logger.warning("No successful shift training completed")
                return None
                
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

class EmbeddingTrainer(BaseTrainer):
    """
    Trainer for embedding module only.
    
    This trainer only trains the embedding module (autoencoder).
    """
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = False,
                 model_dict_save_dir=None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the EmbeddingTrainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Wheter to print all losses per epoch
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, project_name)
    
    def train(self) -> float:
        """
        Train the embedding module.
        
        Returns:
        --------
        float
            Best validation metric (identity loss)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_embedding",
                tags=["embedding"],
                group=self.group
            )
        
        try:
            # Create trainer
            trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                learning_rate_change=self.config.learning_rate_change,
                num_epochs=self.config.num_epochs,
                decayEpochs=self.config.decay_epochs,
                mask_value=self.mask_value,
                early_stop=self.config.early_stop,
                patience=self.config.patience,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                verbose=self.config.verbose,
                model_dict_save_dir = self.model_dict_save_dir

            )
            
            # Train embedding
            best_metric = trainer.train()
            
            # Save best model
            if hasattr(trainer, 'early_stopping') and hasattr(trainer.early_stopping, 'model_path'):
                # Load only embedding part
                self.model.embedding.load_state_dict(torch.load(trainer.early_stopping.model_path, map_location=self.device))
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model.embedding, f"{self.model.__class__.__name__}_embedding_best")
            
            return best_metric
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

def create_trainer(model: nn.Module, train_loader, test_loader, config, mask_value,
                   use_wandb: bool = False, print_losses: bool = False,
                    baseline=None, model_dict_save_dir: Optional[str] = None,
                    group: Optional[str] = None, project_name: Optional[str] = 'KOOPOMICS') -> BaseTrainer:
    """
    Create a trainer based on the training mode.
    
    Parameters:
    -----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    test_loader : DataLoader
        Testing data loader
    config : ConfigManager
        Configuration manager
    mask_value : Int
        Mask_value
    use_wandb : bool, default=False
        Whether to use Weights & Biases for logging
    baseline : nn.Module, default=None
        Baseline model for comparison
    
    Returns:
    --------
    BaseTrainer
        Trainer instance
    """
    training_mode = config.training_mode
    
    if training_mode == 'full':
        trainer = FullTrainer(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
    elif training_mode == 'modular':
        trainer = ModularTrainer(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
    elif training_mode == 'modular_shift':
        trainer = ModularShiftTrainer(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
    elif training_mode == 'embedding':
        trainer = EmbeddingTrainer(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
    
    # Set baseline if provided
    if baseline is not None:
        trainer.baseline = baseline
    
    return trainer
