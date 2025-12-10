"""
KOOPOMICS Test Utilities Module

This module provides evaluation and testing utilities for Koopman models, including:
1. NaiveMeanPredictor - A baseline model that predicts the mean of training data
2. Evaluator - A class for evaluating model performance with various metrics

Author: KOOPOMICS Team
"""

from koopomics.utils import torch, pd, np, wandb
from typing import Optional, Dict, Any, List

import torch.nn as nn

from ..training.koopman_metrics import KoopmanMetricsMixin

import logging

logger = logging.getLogger("koopomics")


class NaiveMeanPredictor(nn.Module):
    """
    A simple baseline model that predicts the mean values of the training data.
    
    This model serves as a baseline for comparison with more complex models.
    It computes the mean of each feature in the training data and returns
    these means as predictions, regardless of the input.
    
    Attributes:
        means (nn.Parameter): Tensor of mean values for each feature
        mask_value (float): Value used to mask missing data points
        device (torch.device): Device to use for computation
    """
    def __init__(self, train_data, mask_value=None):
        """
        Initialize the NaiveMeanPredictor.
        
        Args:
            train_data (DataLoader or DataFrame): Training data to compute means from
            mask_value (float, optional): Value to mask in the data
        """
        super().__init__()
        self.means = None
        self.mask_value = mask_value
        # Always keep device attribute to CPU initially, then move tensors as needed
        self.device = torch.device("cpu")
    
        if isinstance(train_data, torch.utils.data.DataLoader):
            self.get_means_dl(train_data)
        elif isinstance(train_data, pd.DataFrame):
            self.get_means_df(train_data)
        else:
            raise ValueError("train_data must be either a DataLoader or a DataFrame")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))
            self.device = torch.device("cuda")

    def get_means_dl(self, dl):
        """
        Compute mean values from a DataLoader.
        
        Args:
            dl (DataLoader): DataLoader containing the training data
        """
        # Get a sample batch to infer tensor shape
        for data in dl:
            input_data = data[0].to(self.device)
            break

        # Initialize sum and count tensors on the device
        sum_values = torch.zeros(input_data.shape[-1], dtype=torch.float32, device=self.device)
        count_values = torch.zeros(input_data.shape[-1], dtype=torch.float32, device=self.device)
        
        # Compute means efficiently in a single pass
        with torch.no_grad():
            for data in dl:
                input_data = data[0].to(self.device)
                
                # Create mask and apply in one step
                if self.mask_value is not None:
                    mask = (input_data != self.mask_value)
                    masked_input_data = torch.where(mask, input_data, torch.zeros_like(input_data))
                else:
                    masked_input_data = input_data
                    mask = torch.ones_like(input_data, dtype=torch.bool)

                # Sum across appropriate dimensions based on shape
                if masked_input_data.shape[1] == 1:
                    sum_values += masked_input_data.sum(dim=0).squeeze()
                    count_values += mask.sum(dim=0).squeeze()
                else:
                    sum_values += masked_input_data.sum(dim=(0, 1))
                    count_values += mask.sum(dim=(0, 1))

        # Compute means and store as non-trainable parameter
        self.means_values = sum_values / count_values
        self.means = nn.Parameter(self.means_values.clone().detach(), requires_grad=False)

    def get_means_df(self, df):
        """
        Compute mean values from a DataFrame.
        
        Args:
            df (DataFrame): DataFrame containing the training data
        """
        # Ensure feature_list is available
        if not hasattr(self, 'feature_list'):
            raise ValueError("feature_list attribute not set for DataFrame processing")
            
        # Create a mask to filter out rows that contain the mask_value
        if self.mask_value is not None:
            mask = (df[self.feature_list] != self.mask_value).all(axis=1)
            filtered_df = df[mask]
        else:
            filtered_df = df

        # Calculate means and convert to tensor - more efficiently with numpy
        self.means_values = filtered_df[self.feature_list].mean().values
        
        # Create tensor directly on the correct device
        self.means = nn.Parameter(
            torch.tensor(self.means_values, dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        
    def kmatrix(self):
        """Return placeholder Koopman matrices for API compatibility."""
        return torch.zeros(4, 4, device=self.device), torch.zeros(4, 4, device=self.device)
        
    def forward(self, input_vector, fwd=0, bwd=0):
        """
        Forward pass that returns the precomputed mean values.
        
        Args:
            input_vector (torch.Tensor): Input tensor (ignored except for shape)
            fwd (int, optional): Forward steps (ignored, included for API compatibility)
            bwd (int, optional): Backward steps (ignored, included for API compatibility)
            
        Returns:
            torch.Tensor: Tensor of mean values expanded to match input shape
        """
        device = input_vector.device
        input_shape = input_vector.shape
        
        # Match output shape to input shape
        if len(input_shape) == 2:
            batch_size, num_features = input_shape
            expanded_means = self.means.to(device).unsqueeze(0)
            return expanded_means.expand(batch_size, num_features)
        
        elif len(input_shape) == 3:
            batch_size, timepoints, num_features = input_shape
            expanded_means = self.means.to(device).unsqueeze(0).unsqueeze(0)
            return expanded_means.expand(batch_size, timepoints, num_features)
        
        else:
            raise ValueError("Input must be either 2D or 3D.")

"""
🧮 Evaluator
============

Unified evaluation module for KOOPOMICS models.

Computes:
- Forward & backward prediction losses
- Reconstruction, consistency & stability metrics
- Embedding-only evaluation
- Baseline comparison and ratios

Fully compatible with unified `Training_Settings`
and all Koopman trainers.
"""

class Evaluator(KoopmanMetricsMixin):
    """🧮 Unified evaluator for Koopman models and baselines."""

    def __init__(self, model: torch.nn.Module, train_loader, test_loader, settings):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Koopman model (defines `.embedding` and `.operator`).
        train_loader, test_loader : DataLoader
            Data for evaluation.
        settings : Training_Settings
            Unified runtime configuration object.
        """
        # Core
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.settings = settings

        # Runtime + hyper settings
        self.device = settings.runtime.device
        self.mask_value = settings.hyper.mask_value
        self.max_Kstep = settings.hyper.max_Kstep
        self.loss_weights = settings.hyper.loss_weights
        self.effective_loss_weights = self.loss_weights.copy()

        # Criterion & baseline
        self.criterion = settings.runtime.criterion  # ✅ unified masked criterion
        self.baseline = getattr(settings.baseline, "model", None)

        # Device sync
        self.model.to(self.device)
        if self.baseline is not None:
            self.baseline.to(self.device)

    # ==================================================================
    # 🚀 MAIN EVALUATION ENTRY
    # ==================================================================
    def __call__(self, train_metrics: bool = False):
        """Evaluate test set (and optionally train set), with optional baseline ratio."""
        train_metrics_dict = self.evaluate(self.train_loader) if train_metrics else {}
        test_metrics_dict = self.evaluate(self.test_loader)
        baseline_metrics = {}

        if self.baseline is not None:
            baseline_metrics = self.evaluate_baseline()
            base_loss = (baseline_metrics["forward_loss"] + baseline_metrics["backward_loss"]) / 2
            test_loss = (test_metrics_dict["forward_loss"] + test_metrics_dict["backward_loss"]) / 2
            ratio = (base_loss - test_loss) / base_loss if base_loss > 0 else 0.0
            test_metrics_dict["baseline_ratio"] = ratio

        logger.info(
            f"✅ Eval done | FWD={test_metrics_dict['forward_loss']:.6f}, "
            f"BWD={test_metrics_dict['backward_loss']:.6f}, "
            f"Baseline ratio={test_metrics_dict.get('baseline_ratio', 0):.4f}"
        )

        return train_metrics_dict, test_metrics_dict, baseline_metrics

    # ==================================================================
    # 🔬 CORE EVALUATION LOOP
    # ==================================================================
    def evaluate(self, loader):
        """Compute losses on a given loader using settings-driven config."""
        self.model.eval()
        total = self._init_loss_dict()

        with torch.no_grad():
            for data_seq in loader:
                input_fwd = data_seq[0].to(self.device)
                input_bwd = data_seq[-1].to(self.device)
                rev_seq = torch.flip(data_seq, dims=[0])

                # temporal consistency buffers if requested
                if self.loss_weights.get("tempcons", 0) > 0 and self.max_Kstep > 1:
                    self.temporal_cons_fwd_storage = torch.zeros(self.max_Kstep, *input_fwd.shape, device=self.device)
                    self,temporal_cons_bwd_storage = torch.zeros(self.max_Kstep, *input_bwd.shape, device=self.device)

                # multi-step prediction
                for step in range(1, self.max_Kstep + 1):
                    self.current_step = step
                    if self.loss_weights.get("fwd", 0) > 0:
                        tgt = data_seq[step].to(self.device)
                        lf, lz = self.compute_forward_loss(input_fwd, tgt, fwd=step)
                        total["fwd"] += lf
                        total["latent"] += lz
                    if self.loss_weights.get("bwd", 0) > 0:
                        tgt = rev_seq[step].to(self.device)
                        lb, lz = self.compute_backward_loss(input_bwd, tgt, bwd=step)
                        total["bwd"] += lb
                        total["latent"] += lz

                # reconstruction + regularization terms
                if self.loss_weights.get("identity", 0) > 0:
                    total["identity"] += self.compute_identity_loss(input_fwd, input_fwd)
                if self.loss_weights.get("invcons", 0) > 0:
                    total["invcons"] += self.compute_inverse_consistency(input_fwd, None)
                if self.loss_weights.get("tempcons", 0) > 0 and self.max_Kstep > 1:
                    total["tempcons"] += (
                        self.compute_temporal_consistency(temporal_cons_fwd_storage)
                        + self.compute_temporal_consistency(temporal_cons_bwd_storage, bwd=True)
                    ) * 0.5

        # unbiased normalization by sample count
        denom = max(1, sum(len(batch) for batch in loader))
        for k in total.keys():
            total[k] /= denom

        prediction_loss = (total["fwd"] + total["bwd"]) / 2

        return {
            "forward_loss": total["fwd"].detach(),
            "backward_loss": total["bwd"].detach(),
            "reconstruction_loss": total["identity"].detach(),
            "prediction_loss": prediction_loss.detach(),
            "total_loss": self.calculate_total_loss(
                total["fwd"], total["bwd"], total["latent"],
                total["identity"], total["orth"],
                total["invcons"], total["tempcons"], total["stability"]
            ).detach(),
        }

    # ==================================================================
    # 🧩 BASELINE (PREDICTION) EVALUATION
    # ==================================================================
    def evaluate_baseline(self):
        """Evaluate the baseline model’s forward/backward prediction MSE."""
        if self.baseline is None:
            logger.warning("⚠️ Baseline not initialized; skipping baseline evaluation.")
            return {"forward_loss": torch.tensor(0.0), "backward_loss": torch.tensor(0.0)}

        self.baseline.eval()
        total_fwd = torch.tensor(0.0, device=self.device)
        total_bwd = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for data_seq in self.test_loader:
                input_fwd = data_seq[0].to(self.device)
                input_bwd = data_seq[-1].to(self.device)
                rev_seq = torch.flip(data_seq, dims=[0])

                for step in range(self.max_Kstep):
                    tgt_fwd = data_seq[step + 1].to(self.device)
                    tgt_bwd = rev_seq[step + 1].to(self.device)
                    if self.loss_weights.get("fwd", 0) > 0:
                        total_fwd += self.criterion(self.baseline(input_fwd), tgt_fwd)
                    if self.loss_weights.get("bwd", 0) > 0:
                        total_bwd += self.criterion(self.baseline(input_bwd), tgt_bwd)

        denom = max(1, sum(len(batch) for batch in self.test_loader))
        return {
            "forward_loss": (total_fwd / denom).detach(),
            "backward_loss": (total_bwd / denom).detach(),
        }

    # ==================================================================
    # 🎯 EMBEDDING-ONLY EVALUATION
    # ==================================================================
    def metrics_embedding(self):
        """Evaluate embedding reconstruction (optionally vs baseline)."""
        model_metrics = self.evaluate_embedding()
        baseline_metrics = self.evaluate_baseline_embedding() if self.baseline else {}
        return model_metrics, baseline_metrics

    def evaluate_embedding(self):
        """Reconstruction loss of the embedding module."""
        self.model.eval()
        total = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for data_seq in self.test_loader:
                for step in range(data_seq.shape[0]):
                    x = data_seq[step].to(self.device)
                    total += self.compute_identity_loss(x, x)
        avg = total / max(1, len(self.test_loader))
        return {"identity_loss": avg.detach()}

    def evaluate_baseline_embedding(self):
        """Reconstruction loss using the baseline network."""
        if self.baseline is None:
            return {"identity_loss": torch.tensor(0.0)}
        self.baseline.eval()
        total = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for data_seq in self.test_loader:
                for step in range(data_seq.shape[0]):
                    x = data_seq[step].to(self.device)
                    out = self.baseline(x)
                    total += self.criterion(out, x)
        avg = total / max(1, len(self.test_loader))
        return {"identity_loss": avg.detach()}

    # ==================================================================
    # 🔎 FEATURE-WISE PREDICTION ERRORS
    # ==================================================================

    def compute_prediction_errors(self, loader=None, feature_names=None, per_feature=True):
        """
        Compute per-feature forward/backward prediction errors using
        the same criterion type as training (with reduction='none').
        """
        self.model.eval()
        loader = loader or self.test_loader

        # 🔧 Dynamically instantiate the same loss function but with reduction='none'
        crit_class = getattr(torch.nn, self.settings.hyper.criterion_name)
        crit_none = crit_class(reduction="none")
        crit_none = self.masked_criterion(crit_none, mask_value=self.mask_value)

        total_fwd = torch.tensor(0.0, device=self.device)
        total_bwd = torch.tensor(0.0, device=self.device)
        n_batches = 0
        fwd_feat_sum = bwd_feat_sum = None

        with torch.no_grad():
            for data_seq in loader:
                n_batches += 1
                input_fwd = data_seq[0].to(self.device)
                input_bwd = data_seq[-1].to(self.device)
                rev_seq = torch.flip(data_seq, dims=[0])

                if per_feature and fwd_feat_sum is None:
                    n_feat = input_fwd.shape[-1]
                    fwd_feat_sum = torch.zeros(n_feat, device=self.device)
                    bwd_feat_sum = torch.zeros(n_feat, device=self.device)

                for step in range(1, self.max_Kstep + 1):
                    tgt_fwd = data_seq[step].to(self.device)
                    tgt_bwd = rev_seq[step].to(self.device)

                    _, fwd_seq = self.model.predict(input_fwd, fwd=step)
                    fwd_pred = fwd_seq[-1]

                    bwd_seq, _ = self.model.predict(input_bwd, bwd=step)
                    bwd_pred = bwd_seq[-1]

                    # ✅ Apply dynamically created criterion
                    per_elem_fwd = crit_none(fwd_pred, tgt_fwd)
                    per_elem_bwd = crit_none(bwd_pred, tgt_bwd)

                    total_fwd += per_elem_fwd.mean()
                    total_bwd += per_elem_bwd.mean()

                    if per_feature:
                        reduce_dims = tuple(range(per_elem_fwd.dim() - 1))
                        fwd_feat_sum += per_elem_fwd.mean(dim=reduce_dims)
                        bwd_feat_sum += per_elem_bwd.mean(dim=reduce_dims)

        denom = max(1, n_batches * self.max_Kstep)
        result = {
            "total_fwd_loss": float((total_fwd / denom).item()),
            "total_bwd_loss": float((total_bwd / denom).item()),
        }

        if per_feature and fwd_feat_sum is not None:
            fwd_vec = (fwd_feat_sum / denom).tolist()
            bwd_vec = (bwd_feat_sum / denom).tolist()
            if feature_names and len(feature_names) == len(fwd_vec):
                result["fwd_feature_errors"] = dict(zip(feature_names, map(float, fwd_vec)))
                result["bwd_feature_errors"] = dict(zip(feature_names, map(float, bwd_vec)))
            else:
                result["fwd_feature_errors"] = {i: float(v) for i, v in enumerate(fwd_vec)}
                result["bwd_feature_errors"] = {i: float(v) for i, v in enumerate(bwd_vec)}
        return result

    # ==================================================================
    # ⚙️ HELPERS
    # ==================================================================
    def _init_loss_dict(self):
        """Zero-init all accumulators (same keys as trainers)."""
        return {k: torch.tensor(0.0, device=self.device) for k in [
            "fwd", "bwd", "latent", "identity", "orth", "invcons", "tempcons", "stability"
        ]}

class Evaluator_(KoopmanMetricsMixin):
    """
    Class for evaluating Koopman models and computing various performance metrics.
    
    This evaluator computes forward and backward prediction losses, reconstruction losses,
    and can compare model performance against a baseline.
    
    Attributes:
        model (nn.Module): The Koopman model to evaluate
        test_loader (DataLoader): DataLoader for the test dataset
        train_loader (DataLoader): DataLoader for the training dataset
        mask_value (float): Value used to mask missing data points
        max_Kstep (int): Maximum number of Koopman steps for prediction
        baseline (nn.Module): Optional baseline model for comparison
        device (torch.device): Device to use for computation
        criterion (function): Loss function for evaluation
        loss_weights (list): Weights for different loss components
    """
    def __init__(self, model, train_loader, test_loader, **kwargs):
        """
        Initialize the Evaluator.
        
        Args:
            model (nn.Module): The model to evaluate
            train_loader (DataLoader): DataLoader for training data
            test_loader (DataLoader): DataLoader for test data
            **kwargs: Additional keyword arguments:
                - mask_value (float): Value to mask in the data (default: -2)
                - max_Kstep (int): Maximum Koopman steps (default: 1)
                - baseline (nn.Module): Baseline model for comparison
                - model_name (str): Name of the model
                - criterion (function): Custom loss function
                - loss_weights (list): Weights for different loss components
        """
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.mask_value = kwargs.get('mask_value', -2)
        self.max_Kstep = kwargs.get('max_Kstep', 1)
        self.baseline = kwargs.get('baseline', None)
        self.model_name = kwargs.get('model_name', 'Koop')
        
        # Ensure model and data are on the same device
        self.device = next(model.parameters()).device
        
        # Set up loss function - ensure criterion is not None
        base_criterion = nn.MSELoss().to(self.device)
        provided_criterion = kwargs.get('criterion')
        
        if provided_criterion is not None:
            self.criterion = provided_criterion
        else:
            self.criterion = self.masked_criterion(base_criterion, mask_value=self.mask_value)
        
        # Set loss weights
        self.loss_weights = kwargs.get('loss_weights', [1, 1, 1, 1, 1, 1])
        self.effective_loss_weights = self.loss_weights

        # Initialize state
        self.current_step = 0
        self.metrics = {}
        
        # Move baseline to the same device as the model if it exists
        if self.baseline is not None:
            self.baseline.to(self.device)
        
    def __call__(self, train_metrics=False):
        """
        Evaluate the model on test data and optionally on training data.
        
        Args:
            train_metrics (bool): Whether to evaluate on training data
            
        Returns:
            tuple: (train_metrics, test_metrics, baseline_metrics)
        """
        # Evaluate on training data if requested
        train_model_metrics = {}
        if train_metrics:
            train_model_metrics = self.evaluate(self.train_loader)
            
        # Always evaluate on test data
        test_model_metrics = self.evaluate(self.test_loader)

        # Compute baseline metrics if a baseline model is provided
        baseline_metrics = {}
        if self.baseline:
            baseline_metrics = self.compute_baseline_performance()
            
            # Calculate baseline ratio (improvement over baseline)
            combined_test_loss = test_model_metrics['prediction_loss']
            combined_baseline_loss = (baseline_metrics['forward_loss'] + baseline_metrics['backward_loss']) / 2
            baseline_ratio = (combined_baseline_loss - combined_test_loss) / combined_baseline_loss
            
            # Add baseline ratio to metrics
            test_model_metrics['baseline_ratio'] = baseline_ratio

        return train_model_metrics, test_model_metrics, baseline_metrics

    def metrics_embedding(self):
        """
        Evaluate embedding performance of the model.
        
        Returns:
            tuple: (model_metrics, baseline_metrics)
        """
        model_metrics = self.evaluate_embedding()

        baseline_metrics = {}
        if self.baseline:
            baseline_metrics = self.compute_baseline_performance_embedding()

        return model_metrics, baseline_metrics

    def evaluate(self, dl):
        """
        Evaluate model performance on a given DataLoader.
        
        Args:
            dl (DataLoader): DataLoader containing evaluation data
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Initialize tensors for accumulating losses
        test_fwd_loss = torch.tensor(0.0, device=self.device)
        test_bwd_loss = torch.tensor(0.0, device=self.device)
        test_identity_loss = torch.tensor(0.0, device=self.device)
        total_test_loss = torch.tensor(0.0, device=self.device)
   
        with torch.no_grad():
            for data_list in dl:
                # Initialize batch losses
                loss_fwd_batch = torch.tensor(0.0, device=self.device)
                loss_bwd_batch = torch.tensor(0.0, device=self.device)
                loss_latent_identity_batch = torch.tensor(0.0, device=self.device)
                loss_identity_batch = torch.tensor(0.0, device=self.device)
                loss_inv_cons_batch = torch.tensor(0.0, device=self.device)
                loss_temp_cons_batch = torch.tensor(0.0, device=self.device)
    
                # Prepare forward and backward inputs
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
    
                # Get reconstruction loss
                if self.loss_weights["identity"] >0:

                    embedded_output, identity_output = self.model.embed(input_fwd)
                    reconstruction_loss = self.criterion(identity_output, input_fwd)
                    loss_identity_batch += reconstruction_loss

                # Evaluate each prediction step
                for step in range(1, self.max_Kstep+1):
                    self.current_step = step
                    target_fwd = data_list[step].to(self.device)
                    target_bwd = reverse_data_list[step].to(self.device)
    
                    # Initialize temporal consistency storage if needed
                    # Ensure proper device for temporal consistency storage
                    if self.max_Kstep > 1 and self.loss_weights["tempcons"] > 0:
                        # Create storage tensors directly on the same device as input
                        self.temporal_cons_fwd_storage = torch.zeros(
                            self.max_Kstep, *input_fwd.shape,
                            dtype=input_fwd.dtype,
                            device=input_fwd.device
                        )
                        self.temporal_cons_bwd_storage = torch.zeros(
                            self.max_Kstep, *input_bwd.shape,
                            dtype=input_bwd.dtype,
                            device=input_bwd.device
                        )
    
                    # Compute forward prediction loss
                    if self.loss_weights["fwd"] > 0:
                        loss_fwd_step, loss_latent_fwd_identity_step = self.compute_forward_loss(input_fwd, target_fwd, fwd=step)
                        loss_fwd_batch += loss_fwd_step
                        loss_latent_identity_batch += loss_latent_fwd_identity_step
    
                    # Compute backward prediction loss
                    if self.loss_weights["bwd"] > 0:
                        loss_bwd_step, loss_latent_bwd_identity_step = self.compute_backward_loss(input_bwd, target_bwd, bwd=step)
                        loss_bwd_batch += loss_bwd_step
                        loss_latent_identity_batch += loss_latent_bwd_identity_step
    
                    # Compute identity loss
                    if self.loss_weights["identity"] > 0:
                        loss_identity_step = (self.compute_identity_loss(input_fwd, target_fwd) +
                                             self.compute_identity_loss(input_bwd, target_bwd)) / 2
                        loss_identity_batch += loss_identity_step
    
                    # Compute inverse consistency loss
                    if self.loss_weights["invcons"] > 0:
                        loss_inv_cons_step = (self.compute_inverse_consistency(input_fwd, target_fwd) +
                                            self.compute_inverse_consistency(input_bwd, target_bwd)) / 2
                        loss_inv_cons_batch += loss_inv_cons_step
    
                    # Compute temporal consistency loss
                    if self.loss_weights["tempcons"] > 0 and self.current_step > 1:
                        loss_temp_cons_step = (self.compute_temporal_consistency(self.temporal_cons_fwd_storage) +
                                              self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)) / 2
                        loss_temp_cons_batch += loss_temp_cons_step
    
                # Calculate total batch loss
                loss_total_batch = self.calculate_total_loss(
                    loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch,
                    loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch
                )

                # Accumulate batch losses
                test_fwd_loss += loss_fwd_batch
                test_bwd_loss += loss_bwd_batch
                test_identity_loss += loss_identity_batch
                total_test_loss += loss_total_batch
                
        # Compute average losses
        avg_test_fwd_loss = test_fwd_loss / (len(dl) * self.max_Kstep)
        avg_test_bwd_loss = test_bwd_loss / (len(dl) * self.max_Kstep)
        avg_test_identity_loss = test_identity_loss / len(dl)
        avg_total_test_loss = total_test_loss / (len(dl) * self.max_Kstep)
        
        # Calculate prediction loss as average of forward and backward loss
        prediction_loss = (avg_test_fwd_loss + avg_test_bwd_loss) / 2
        
        # Return detached tensors to avoid memory leaks
        return {
            'forward_loss': avg_test_fwd_loss.detach(),
            'backward_loss': avg_test_bwd_loss.detach(),
            'reconstruction_loss': avg_test_identity_loss.detach(),
            'prediction_loss': prediction_loss.detach(),
            'total_loss': avg_total_test_loss.detach()
        }
    
    def compute_baseline_performance(self):
        """
        Evaluate baseline model performance on test data.
        
        Returns:
            dict: Dictionary of baseline evaluation metrics
        """
        self.baseline.eval()
        
        # Initialize accumulator tensors
        test_fwd_loss = torch.tensor(0.0, device=self.device)
        test_bwd_loss = torch.tensor(0.0, device=self.device)
    
        with torch.no_grad():
            for data_list in self.test_loader:
                # Initialize batch losses
                loss_fwd_batch = torch.tensor(0.0, device=self.device)
                loss_bwd_batch = torch.tensor(0.0, device=self.device)
    
                # Prepare inputs and targets
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
    
                # Evaluate each step
                for step in range(self.max_Kstep):
                    target_fwd = data_list[step + 1].to(self.device)
                    target_bwd = reverse_data_list[step + 1].to(self.device)
    
                    # Forward prediction
                    if self.loss_weights["fwd"] > 0:
                        baseline_output = self.baseline(input_fwd)
                        loss_fwd = self.criterion(baseline_output, target_fwd)
                        loss_fwd_batch += loss_fwd
                        
                    # Backward prediction
                    if self.loss_weights["bwd"] > 0:
                        baseline_output = self.baseline(input_bwd)
                        loss_bwd = self.criterion(baseline_output, target_bwd)
                        loss_bwd_batch += loss_bwd

                # Accumulate batch losses
                test_fwd_loss += loss_fwd_batch
                test_bwd_loss += loss_bwd_batch
    
        # Compute average losses
        avg_test_fwd_loss = test_fwd_loss / (len(self.test_loader) * self.max_Kstep)
        avg_test_bwd_loss = test_bwd_loss / (len(self.test_loader) * self.max_Kstep)
    
        # Return detached tensors
        return {
            'forward_loss': avg_test_fwd_loss.detach(),
            'backward_loss': avg_test_bwd_loss.detach(),
        }

    def evaluate_embedding(self):
        """
        Evaluate embedding performance on test data.
        
        Returns:
            dict: Dictionary of embedding evaluation metrics
        """
        self.model.eval()
        
        test_identity_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for data_list in self.test_loader:
                loss_identity_batch = torch.tensor(0.0, device=self.device)
                
                for step in range(data_list.shape[0]):
                    # Compute reconstruction loss for each time step
                    input_identity = data_list[step].to(self.device)
                    target_identity = data_list[step].to(self.device)
                    loss_identity_step = self.compute_identity_loss(input_identity, target_identity)
                    loss_identity_batch += loss_identity_step
                
                # Accumulate batch loss
                test_identity_loss += loss_identity_batch
    
        # Compute average loss
        avg_test_identity_loss = test_identity_loss / len(self.test_loader)

        return {
            'identity_loss': avg_test_identity_loss.detach(),
        }

    def compute_baseline_performance_embedding(self):
        """
        Evaluate baseline embedding performance on test data.
        
        Returns:
            dict: Dictionary of baseline embedding evaluation metrics
        """
        self.baseline.eval()
        
        test_identity_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for data_list in self.test_loader:
                loss_identity_batch = torch.tensor(0.0, device=self.device)

                for step in range(data_list.shape[0]):
                    # Compute reconstruction loss for baseline model
                    input_identity = data_list[step].to(self.device)
                    target_identity = data_list[step].to(self.device)
                    baseline_output = self.baseline(input_identity)
                    loss_identity_step = self.criterion(baseline_output, target_identity)
                    loss_identity_batch += loss_identity_step
                
                # Accumulate batch loss
                test_identity_loss += loss_identity_batch
    
        # Compute average loss
        avg_test_identity_loss = test_identity_loss / len(self.test_loader)

        return {
            'identity_loss': avg_test_identity_loss.detach(),
        }
        
    def compute_prediction_errors(self, dataloader, featurewise=True):
        """
        Compute prediction errors for both forward and backward predictions.
        
        This function calculates per-feature prediction errors, which is useful
        for identifying which features are predicted well and which are not.
        
        Args:
            dataloader (DataLoader): DataLoader containing evaluation data
            featurewise (bool): Whether to compute per-feature errors
            
        Returns:
            dict: Dictionary of prediction errors
        """
        self.model.eval()
        
        # Use reduction='none' to get per-feature errors
        base_criterion = nn.MSELoss(reduction='none')
        criterion = self.masked_criterion(base_criterion, mask_value=self.mask_value)
    
        # Initialize accumulators as tensors on the correct device
        total_fwd_loss = torch.tensor(0.0, device=self.device)
        total_bwd_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        # These will be initialized after we see the first batch
        fwd_feature_errors = None
        bwd_feature_errors = None
    
        with torch.no_grad():
            for data_list in dataloader:
                num_batches += 1
                # Prepare inputs and ensure they're on the right device
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
                
                # Initialize feature error accumulators after first batch
                if featurewise and fwd_feature_errors is None:
                    num_features = input_fwd.shape[-1]
                    fwd_feature_errors = torch.zeros(num_features, device=self.device)
                    bwd_feature_errors = torch.zeros(num_features, device=self.device)
    
                # Evaluate each step
                for step in range(1, self.max_Kstep+1):
                    self.current_step = step
                    target_fwd = data_list[step].to(self.device)
                    target_bwd = reverse_data_list[step].to(self.device)
    
                    # Forward prediction
                    bwd_output, fwd_output = self.model.predict(input_fwd, fwd=step)
                    per_feature_loss_fwd = criterion(fwd_output[-1], target_fwd)
                    
                    # Compute per-feature errors more efficiently
                    if featurewise:
                        # Mean across batch dimensions for each feature
                        feature_means = per_feature_loss_fwd.mean(dim=(0, 1))
                        fwd_feature_errors += feature_means
                            
                    total_fwd_loss += per_feature_loss_fwd.mean()
                    
                    # Backward prediction
                    bwd_output, fwd_output = self.model.predict(input_bwd, bwd=step)
                    per_feature_loss_bwd = criterion(bwd_output[-1], target_bwd)
    
                    if featurewise:
                        # Mean across batch dimensions for each feature
                        feature_means = per_feature_loss_bwd.mean(dim=(0, 1))
                        bwd_feature_errors += feature_means
                    
                    total_bwd_loss += per_feature_loss_bwd.mean()
                    
        # Normalize by the number of batches
        normalization_factor = num_batches * self.max_Kstep
        
        # Convert tensor results to dictionary format for compatibility
        fwd_feature_loss_dict = {}
        bwd_feature_loss_dict = {}
        
        if featurewise and fwd_feature_errors is not None:
            # Normalize and convert to dictionary
            normalized_fwd_errors = fwd_feature_errors / normalization_factor
            normalized_bwd_errors = bwd_feature_errors / normalization_factor
            
            fwd_feature_loss_dict = {i: normalized_fwd_errors[i].item() for i in range(len(normalized_fwd_errors))}
            bwd_feature_loss_dict = {i: normalized_bwd_errors[i].item() for i in range(len(normalized_bwd_errors))}
            
        # Normalize total losses
        total_fwd_loss = total_fwd_loss / normalization_factor
        total_bwd_loss = total_bwd_loss / normalization_factor
    
        return {
            'fwd_feature_errors': fwd_feature_loss_dict,
            'bwd_feature_errors': bwd_feature_loss_dict,
            'total_fwd_loss': total_fwd_loss.item(),
            'total_bwd_loss': total_bwd_loss.item()
        }

