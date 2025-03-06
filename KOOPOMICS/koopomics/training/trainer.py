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
from .wandb_utils import WandbManager

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
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, use_wandb: bool = False, print_losses: bool = False):
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
        self.device = config.device
        self.use_wandb = use_wandb
        self.print_losses = print_losses
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Get training configuration
        self.training_config = config.get_training_config()
        
        # Create baseline model
        self.baseline = NaiveMeanPredictor(train_loader, mask_value=config.mask_value)
        
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
                project_name="KOOPOMICS",
                entity=None  # Set to your wandb username or team name
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
                run_name=f"{self.model.__class__.__name__}_{self.training_config['mode']}",
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
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, use_wandb: bool = False, print_losses: bool = True):
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
        super().__init__(model, train_loader, test_loader, config, use_wandb, print_losses)
    
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
                tags=["full", self.training_config['backpropagation_mode']]
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
                    mask_value=self.config.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    print_losses=self.print_losses
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
                    mask_value=self.config.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    print_losses=self.print_losses
                )
            
            # Train model
            best_metric = trainer.train()
            
            # Save best model
            if hasattr(trainer, 'early_stopping') and hasattr(trainer.early_stopping, 'model_path'):
                self.load_model(trainer.early_stopping.model_path)
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model, f"{self.model.__class__.__name__}_best")
            
            return best_metric
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

class ModularTrainer(BaseTrainer):
    """
    Trainer for modular model training.
    
    This trainer first trains the embedding module, then freezes it and trains the operator module.
    """
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, use_wandb: bool = False, print_losses: bool = True):
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
        super().__init__(model, train_loader, test_loader, config, use_wandb, print_losses)
        
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
                tags=["modular", self.training_config['backpropagation_mode']]
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
                mask_value=self.config.mask_value,
                early_stop=self.config.early_stop,
                patience=self.config.patience,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                print_losses=self.print_losses
            )
            
            embedding_trainer.train()
            
            # Save embedding
            torch.save(self.model.embedding.state_dict(), self.embedding_path)
            logger.info(f"Embedding saved to {self.embedding_path}")
            
            # Log embedding to wandb if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.log_model(self.model.embedding, f"{self.model.__class__.__name__}_embedding")
            
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
                    mask_value=self.config.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    print_losses=self.print_losses
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
                    mask_value=self.config.mask_value,
                    early_stop=self.config.early_stop,
                    patience=self.config.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    print_losses=self.print_losses
                )
            
            # Train operator
            best_metric = operator_trainer.train()
            
            # Save best model
            if hasattr(operator_trainer, 'early_stopping') and hasattr(operator_trainer.early_stopping, 'model_path'):
                self.load_model(operator_trainer.early_stopping.model_path)
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model, f"{self.model.__class__.__name__}_best")
            
            return best_metric
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

class EmbeddingTrainer(BaseTrainer):
    """
    Trainer for embedding module only.
    
    This trainer only trains the embedding module (autoencoder).
    """
    
    def __init__(self, model: nn.Module, train_loader, test_loader, config, use_wandb: bool = False, print_losses: bool = False):
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
        super().__init__(model, train_loader, test_loader, config, use_wandb, print_losses)
    
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
                tags=["embedding"]
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
                mask_value=self.config.mask_value,
                early_stop=self.config.early_stop,
                patience=self.config.patience,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                print_losses=self.print_losses
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

def create_trainer(model: nn.Module, train_loader, test_loader, config, use_wandb: bool = False, print_losses: bool = False, baseline=None) -> BaseTrainer:
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
        trainer = FullTrainer(model, train_loader, test_loader, config, use_wandb, print_losses)
    elif training_mode == 'modular':
        trainer = ModularTrainer(model, train_loader, test_loader, config, use_wandb, print_losses)
    elif training_mode == 'embedding':
        trainer = EmbeddingTrainer(model, train_loader, test_loader, config, use_wandb, print_losses)
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
    
    # Set baseline if provided
    if baseline is not None:
        trainer.baseline = baseline
    
    return trainer
