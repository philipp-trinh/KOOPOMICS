
from koopomics.utils import torch, pd, np, wandb




def masked_criterion(criterion, mask_value=-1):
# Inner function that applies the mask and then computes the loss
    def compute_loss(predictions, targets):
        mask = targets != mask_value  
        
        masked_targets = torch.where(mask, targets, torch.tensor(0.0, device=targets.device))
        
        masked_predictions = torch.where(mask, predictions, torch.tensor(0.0, device=targets.device))
        # If all values are masked, return 0 loss
        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Calculate the loss with the given criterion
        return criterion(masked_predictions, masked_targets)

    return compute_loss  
 



class KoopmanMetricsMixin:

    def spectral_loss(pred, target):
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        return torch.mean(torch.abs(pred_fft - target_fft))
        
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def masked_criterion(self, criterion, mask_value=-1):
        """Creates a masked version of the given criterion with robust type handling"""
        def compute_loss(predictions, targets):
            # Ensure we're working with tensors
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, device=predictions.device)

            # Create mask (always returns tensor)
            mask = targets != mask_value

            # Handle case where mask is a single boolean (convert to tensor)
            if isinstance(mask, bool):
                mask = torch.tensor([mask], device=predictions.device)
            
            # Check if any elements are unmasked
            if not mask.any():
                return torch.tensor(0.0, device=predictions.device)
            
            # Apply mask safely
            masked_targets = targets.masked_fill(~mask, 0)
            masked_predictions = predictions.masked_fill(~mask, 0)
            
            return criterion(masked_predictions, masked_targets)
        
        return compute_loss

    def compute_forward_loss_(self, input_fwd, target_fwd, fwd=1):
        # Print the step and forward parameter to track which iteration is being processed
        print(f"Step {self.current_step}: Computing forward loss for fwd={fwd}")

        # Print input tensors and their shapes
        print(f"input_fwd shape: {input_fwd.shape}")
        print(f"input_fwd: {input_fwd}")
        print(f"target_fwd shape: {target_fwd.shape}")
        print(f"target_fwd: {target_fwd}")

        # Initialize loss tensors
        loss_fwd_step = torch.tensor(0.0, device=self.device)
        loss_latent_fwd_identity_step = torch.tensor(0.0, device=self.device)
        print(f"Initial loss_fwd_step: {loss_fwd_step}")
        print(f"Initial loss_latent_fwd_identity_step: {loss_latent_fwd_identity_step}")

        # Encode input to latent representation
        latent_fwd = self.model.embedding.encode(input_fwd)
        print(f"latent_fwd after encoding shape: {latent_fwd.shape}")
        print(f"latent_fwd after encoding: {latent_fwd}")

        # Loop over forward steps
        for f in range(fwd):
            print(f"Before fwd_step {f+1}/{fwd}: latent_fwd = {latent_fwd}")
            latent_fwd = self.model.operator.fwd_step(latent_fwd)  # Shift f times forward
            print(f"After fwd_step {f+1}/{fwd}: latent_fwd = {latent_fwd}")

            # Check condition for temporal consistency storage
            if self.current_step > 1 and self.loss_weights[5] > 0:
                shifted_fwd_storage = self.model.embedding.decode(latent_fwd)
                self.temporal_cons_fwd_storage[f] = shifted_fwd_storage
                print(f"Storing shifted_fwd_storage for f={f}, shape: {shifted_fwd_storage.shape}")
                print(f"shifted_fwd_storage: {shifted_fwd_storage}")

        # Decode final latent forward to get predicted output
        shifted_fwd = self.model.embedding.decode(latent_fwd)
        print(f"Final shifted_fwd shape: {shifted_fwd.shape}")
        print(f"Final shifted_fwd: {shifted_fwd}")

        # Compute forward loss
        loss_fwd_step += self.criterion(shifted_fwd, target_fwd)
        print(f"loss_fwd_step after computation: {loss_fwd_step}")

        # Compute latent identity loss if condition is met
        if self.loss_weights[2] > 0:
            latent_target_fwd = self.model.embedding.encode(target_fwd)
            print(f"latent_target_fwd shape: {latent_target_fwd.shape}")
            print(f"latent_target_fwd: {latent_target_fwd}")

            mask = target_fwd != self.mask_value
            print(f"mask shape: {mask.shape}")
            print(f"mask: {mask}")

            mask_values = mask[:, :, 0]
            print(f"mask_values shape: {mask_values.shape}")
            print(f"mask_values: {mask_values}")

            latent_mask = mask_values.unsqueeze(-1).repeat(1, 1, latent_target_fwd.shape[-1])
            print(f"latent_mask shape: {latent_mask.shape}")
            print(f"latent_mask: {latent_mask}")

            corrected_latent_target = torch.where(latent_mask, latent_target_fwd, torch.tensor(self.mask_value, device=latent_target_fwd.device))
            print(f"corrected_latent_target shape: {corrected_latent_target.shape}")
            print(f"corrected_latent_target: {corrected_latent_target}")

            loss_latent_fwd_identity_step += self.criterion(latent_fwd, corrected_latent_target)
            print(f"loss_latent_fwd_identity_step after computation: {loss_latent_fwd_identity_step}")

        # Print final losses before returning
        print(f"Returning losses: loss_fwd_step={loss_fwd_step}, loss_latent_fwd_identity_step={loss_latent_fwd_identity_step}")

        return loss_fwd_step, loss_latent_fwd_identity_step    
    
    def compute_forward_loss(self, input_fwd, target_fwd, fwd=1):


        loss_fwd_step = torch.tensor(0.0, device=self.device)
        loss_latent_fwd_identity_step = torch.tensor(0.0, device=self.device)
        
        latent_fwd = self.model.embedding.encode(input_fwd)
        
        for f in range(fwd):
            latent_fwd = self.model.operator.fwd_step(latent_fwd) # shift f times forward

            if self.current_step > 1 and self.effective_loss_weights['tempcons'] > 0:
                shifted_fwd_storage = self.model.embedding.decode(latent_fwd.real)
                self.temporal_cons_fwd_storage[f] = shifted_fwd_storage
        
        shifted_fwd = self.model.embedding.decode(latent_fwd.real)


        loss_fwd_step += self.criterion(shifted_fwd, target_fwd)

        # Compute latent identity loss with shifted latent prediction
        if self.effective_loss_weights["latent_identity"] > 0:
            latent_target_fwd = self.model.embedding.encode(target_fwd)
            mask = target_fwd != self.mask_value  
            
            mask_values = mask[:, :, 0]
            latent_mask = mask_values.unsqueeze(-1).repeat(1, 1, latent_target_fwd.shape[-1])


            corrected_latent_target = torch.where(latent_mask, latent_target_fwd, torch.tensor(self.mask_value, device=latent_target_fwd.device))
            

            loss_latent_fwd_identity_step += self.criterion(latent_fwd.real, corrected_latent_target)
            
        return loss_fwd_step, loss_latent_fwd_identity_step

    def compute_backward_loss(self, input_bwd, target_bwd, bwd=1):
        loss_bwd_step = torch.tensor(0.0, device=self.device)
        loss_latent_bwd_identity_step = torch.tensor(0.0, device=self.device)

        latent_bwd = self.model.embedding.encode(input_bwd)

        for b in range(bwd):
            latent_bwd = self.model.operator.bwd_step(latent_bwd) # shift b times backward

            if self.current_step > 1 and self.effective_loss_weights['tempcons'] > 0:
                shifted_bwd_storage = self.model.embedding.decode(latent_bwd.real)
                self.temporal_cons_bwd_storage[b-1] = shifted_bwd_storage
        
        
        shifted_bwd = self.model.embedding.decode(latent_bwd.real)

        loss_bwd_step += self.criterion(shifted_bwd, target_bwd)

        # Compute latent identity loss with shifted latent prediction
        if self.effective_loss_weights['latent_identity'] > 0:
            latent_target_bwd = self.model.embedding.encode(target_bwd)
            mask = target_bwd != self.mask_value 
            
            mask_values = mask[:, :, 0]
            latent_mask = mask_values.unsqueeze(-1).repeat(1, 1, latent_target_bwd.shape[-1])


            corrected_latent_target = torch.where(latent_mask, latent_target_bwd, torch.tensor(self.mask_value, device=latent_target_bwd.device))
                        
            loss_latent_bwd_identity_step += self.criterion(latent_bwd.real, corrected_latent_target)
            
        return loss_bwd_step, loss_latent_bwd_identity_step


    def compute_identity_loss(self, input_tensor, shift_target_tensor=None):
        """
        Computes the reconstruction (identity) loss of the autoencoder.
        """
        # Determine which tensor to reconstruct
        target_tensor = shift_target_tensor if shift_target_tensor is not None else input_tensor

        # Encode and decode
        latent = self.model.embedding.encode(target_tensor)
        reconstructed = self.model.embedding.decode(latent)

        # Compute reconstruction loss
        loss_identity = self.criterion(reconstructed, target_tensor)

        return loss_identity


    def compute_orthogonality_loss(self, latent_tensor):
        """
        Computes the orthogonality loss on the latent space.
        Expects a tensor of shape [batch*time, latent_dim] or [batch, time, latent_dim].
        """
        if latent_tensor.dim() == 3:
            batch_size, time_steps, latent_dim = latent_tensor.shape
            Z = latent_tensor.reshape(batch_size * time_steps, latent_dim)
        else:
            Z = latent_tensor  # already [batch*time, latent_dim]

        cov = Z.T @ Z / Z.shape[0]
        I = torch.eye(Z.shape[1], device=Z.device)
        loss_orth = torch.norm(cov - I, p='fro') ** 2

        return loss_orth


    def compute_identity_loss_(self, input_tensor, shift_target_tensor):

        loss_identity_step = torch.tensor(0.0, device=self.device)
        
        if shift_target_tensor is not None:
            latent_target = self.model.embedding.encode(shift_target_tensor)
            autoencoded_target = self.model.embedding.decode(latent_target)
        
            loss_identity_step += self.criterion(autoencoded_target, shift_target_tensor) 
        else:
            latent_input = self.model.embedding.encode(input_tensor)
            autoencoded_input = self.model.embedding.decode(latent_input)
            loss_identity_step += self.criterion(autoencoded_input, input_tensor)



        return loss_identity_step

    def compute_inverse_consistency(
        self,
        input_tensor,
        target_tensor=None,
        latent_invcons_weight: float | None = 0.02,
    ):
        """
        Compute inverse-consistency loss between forward and backward Koopman operations.

        Enforces that forward→backward and backward→forward dynamics approximately 
        return to the same decoded observation (real domain).
        Optionally adds a latent-space inverse consistency term for stability.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input observation at time t (real).
        target_tensor : torch.Tensor, optional
            Next observation (real). If provided, consistency is also enforced on it.
        latent_weight : float or None, optional
            Weight for latent-space inverse consistency regularization.
            If None or 0, the term is skipped. Default = 0.02.

        Returns
        -------
        loss_inv_cons_step : torch.Tensor
            Total inverse consistency loss (scalar).
        """
        loss_inv_cons_step = torch.tensor(0.0, device=self.device)

        # --------------------------------------------------------
        # 🔹 Encode input (real → complex)
        # --------------------------------------------------------
        latent_input = self.model.embedding.encode(input_tensor).to(torch.complex64)

        # Forward→Backward
        fwd_bwd_input = self.model.operator.bwd_step(
            self.model.operator.fwd_step(latent_input)
        )
        decoded_fwd_bwd_input = self.model.embedding.decode(fwd_bwd_input.real)
        loss_inv_cons_step += self.criterion(decoded_fwd_bwd_input, input_tensor)

        # Backward→Forward
        bwd_fwd_input = self.model.operator.fwd_step(
            self.model.operator.bwd_step(latent_input)
        )
        decoded_bwd_fwd_input = self.model.embedding.decode(bwd_fwd_input.real)
        loss_inv_cons_step += self.criterion(decoded_bwd_fwd_input, input_tensor)

        # --------------------------------------------------------
        # 🔹 Optionally apply same consistency to target
        # --------------------------------------------------------
        if target_tensor is not None:
            latent_target = self.model.embedding.encode(target_tensor).to(torch.complex64)

            fwd_bwd_target = self.model.operator.bwd_step(
                self.model.operator.fwd_step(latent_target)
            )
            decoded_fwd_bwd_target = self.model.embedding.decode(fwd_bwd_target.real)
            loss_inv_cons_step += self.criterion(decoded_fwd_bwd_target, target_tensor)

            bwd_fwd_target = self.model.operator.fwd_step(
                self.model.operator.bwd_step(latent_target)
            )
            decoded_bwd_fwd_target = self.model.embedding.decode(bwd_fwd_target.real)
            loss_inv_cons_step += self.criterion(decoded_bwd_fwd_target, target_tensor)

        # --------------------------------------------------------
        # ⚙️ Optional latent-space consistency (complex domain)
        # --------------------------------------------------------
        if latent_invcons_weight and latent_invcons_weight > 0:
            latent_fwd_bwd = self.model.operator.bwd_step(
                self.model.operator.fwd_step(latent_input)
            )
            latent_bwd_fwd = self.model.operator.fwd_step(
                self.model.operator.bwd_step(latent_input)
            )
            loss_latent_cons = (
                self.criterion(latent_fwd_bwd.real, latent_input.real)
                + self.criterion(latent_fwd_bwd.imag, latent_input.imag)
                + self.criterion(latent_bwd_fwd.real, latent_input.real)
                + self.criterion(latent_bwd_fwd.imag, latent_input.imag)
            )
            loss_inv_cons_step += latent_invcons_weight * loss_latent_cons

        return loss_inv_cons_step


    def compute_temporal_consistency(self, temporal_cons_storage, bwd = False):

        loss_temp_cons_step = torch.tensor(0.0, device=self.device)

        if bwd:
            temporal_cons_storage = torch.flip(temporal_cons_storage, dims=[0])

        diagonals = self.get_top_right_to_bottom_left_diagonals(temporal_cons_storage)

        for sample_tensor in diagonals:
            for temp_diag in sample_tensor:
                diag_loss = torch.tensor(0.0, device=self.device)
                for i in range(len(temp_diag) - 1):
                    diag_loss += self.criterion(temp_diag[i], temp_diag[i+1])
                
                loss_temp_cons_step += diag_loss / (len(diagonals) - 1) if len(diagonals) > 1 else torch.tensor(0.0, device=self.device)    
        
        return loss_temp_cons_step
        
    def get_top_right_to_bottom_left_diagonals(self, tensor):
        """
        Extract all diagonals from the top-right to bottom-left of a 3D tensor for evaluating temporal consistency.
    
        This function retrieves diagonals from a tensor that represents temporally shifted predictions. 
        Each diagonal corresponds to time points that are aligned for comparison, allowing for the analysis 
        of temporal consistency in Koopman predictions.
    
        Example:
            Given the input tensor `predictions`:
            
            predictions = torch.tensor([
                [[1, 1], [2, 2], [3, 3], [4, 4]],
                [[2, 2], [3, 3], [4, 4], [5, 5]],
                [[3, 3], [4, 4], [5, 5], [6, 6]],
                [[4, 4], [5, 5], [6, 6], [7, 7]],
            ])
            
            The extracted diagonals represent time-shifted time points used for consistency evaluation.
            For instance, the diagonals extracted from the tensor would be (mask_value = -2):
            
                Diagonal 0: tensor([[1, 1],
                        [-2, -2],
                        [-2, -2],
                        [-2, -2]])
                Diagonal 1: tensor([[2, 2],
                        [2, 2],
                        [-2, -2],
                        [-2, -2]])
                Diagonal 2: tensor([[3, 3],
                        [3, 3],
                        [3, 3],
                        [-2, -2]])
            and so on....
    
        Args:
            tensor (torch.Tensor): A 3D tensor of shape [num_predictions, num_timepoints, num_features].
        
        Returns:
            List[torch.Tensor]: A list of 2D tensors, where each tensor corresponds to a diagonal of 
                               the input tensor, representing aligned time points across predictions 
                               for temporal consistency analysis.
        """
        num_predictions, num_samples, num_timepoints, num_features = tensor.shape
        reshaped_tensor = tensor.permute(1, 0, 2, 3)

        max_diag_len = min(num_predictions, num_timepoints)

        diagonals = torch.full((num_samples, num_predictions + num_timepoints - 1, max_diag_len, num_features),
                               fill_value=self.mask_value, device=tensor.device, dtype=tensor.dtype)

        for sample_idx in range(num_samples):
            # Get diagonals starting from the top row
            diag_idx = 0
            for start_col in range(num_timepoints):
                diag_len = min(start_col + 1, num_predictions)
                diag_values = torch.stack([reshaped_tensor[sample_idx, i, start_col - i] for i in range(diag_len)])
                diagonals[sample_idx, diag_idx, :diag_len] = diag_values
                diag_idx += 1
    
        
            # Get diagonals starting from the rightmost column (excluding the first row)
            for start_row in range(1, num_predictions):
                diag_len = min(num_timepoints, num_predictions - start_row)
                diag_values = torch.stack([reshaped_tensor[sample_idx, start_row + i, num_timepoints - 1 - i] for i in range(diag_len)])
                diagonals[sample_idx, diag_idx, :diag_len] = diag_values
                diag_idx += 1

        return diagonals

                
    def calculate_total_loss(self, *losses):
        """
        Calculate the weighted total loss from a list of loss tensors.

        The order of the loss tensors in `*losses` must match the order of keys
        in `self.loss_weights`. The expected order is:

            1. "fwd"             : forward prediction loss
            2. "bwd"             : backward prediction loss
            3. "latent_identity" : latent identity loss
            4. "identity"        : autoencoder reconstruction loss
            5. "orthogonality"   : latent space orthogonality loss
            6. "invcons"         : inverse consistency loss
            7. "tempcons"        : temporal consistency loss

        Parameters
        ----------
        *losses : torch.Tensor
            Individual loss components in the order listed above.

        Returns
        -------
        torch.Tensor
            Weighted total loss.
        """
        loss_total = 0.0
        for loss, key in zip(losses, self.effective_loss_weights.keys()):
            weight = self.effective_loss_weights[key]
            loss_total += loss * weight
        return loss_total

    def koopman_stability_loss(self, max_radius: float = 1.05, koopman_stab_weight: float = 1e-3) -> torch.Tensor:
        """
        Penalize Koopman eigenvalues with |λ| > max_radius to softly enforce spectral stability.
        Works for both forward and bidirectional (InvKoop) operators.

        Parameters
        ----------
        model : torch.nn.Module
            The full Koopman autoencoder model (must have .operator defined).
        max_radius : float, optional
            Maximum allowed spectral radius before penalty starts.
        strength : float, optional
            Regularization strength for penalizing unstable eigenvalues.

        Returns
        -------
        torch.Tensor
            Scalar stability loss (ready to add to total loss).
        """

        op = getattr(self.model, "operator", None)
        if op is None:
            raise AttributeError("Model has no 'operator' attribute.")

        # ----------------------------------------------------------------------
        # 🔍 Select appropriate Koopman matrix (forward direction)
        # ----------------------------------------------------------------------
        reg = str(getattr(op, "op_reg", "none")).lower()

        try:
            if reg in ["none", "no"]:
                K = op.fwdkoop
            elif reg == "banded":
                K = op.bandedkoop_fwd.kmatrix()
            elif reg == "skewsym":
                # Skew-symmetric → purely imaginary spectrum, stable by construction
                return torch.tensor(0.0, device=op.device)
            elif reg == "nondelay":
                K = op.nondelay_fwd.kmatrix()
            else:
                return torch.tensor(0.0, device=op.device)
        except Exception as e:
            print(f"⚠️ Could not extract Koopman matrix for stability loss: {e}")
            return torch.tensor(0.0, device=op.device)

        # ----------------------------------------------------------------------
        # 🧮 Compute eigenvalue penalty
        # ----------------------------------------------------------------------
        try:
            eigvals = torch.linalg.eigvals(K)
            mags = torch.abs(eigvals)
            penalty = torch.relu(mags - max_radius)
            loss_val = koopman_stab_weight * torch.mean(penalty ** 2)
            return loss_val.real
        except RuntimeError as e:
            print(f"⚠️ Stability loss skipped (eigendecomposition failed): {e}")
            return torch.tensor(0.0, device=op.device)


