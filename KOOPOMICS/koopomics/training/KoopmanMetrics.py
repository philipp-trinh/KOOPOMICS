
import torch




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
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def masked_criterion(self, criterion, mask_value=-1):
    # Inner function that applies the mask and then computes the loss
        def compute_loss(predictions, targets):
            mask = targets != mask_value  

            if not mask.any():  # Check if all values are masked
                return torch.tensor(0.0, device=predictions.device)

            masked_targets = torch.where(mask, targets, torch.tensor(0.0, device=targets.device))
            masked_predictions = torch.where(mask, predictions, torch.tensor(0.0, device=targets.device))

            # Calculate the loss with the given criterion
            return criterion(masked_predictions, masked_targets)
    
        return compute_loss  
    
    def compute_forward_loss(self, input_fwd, target_fwd, fwd=1):
        loss_fwd_step = torch.tensor(0.0, device=self.device)
        loss_latent_fwd_identity_step = torch.tensor(0.0, device=self.device)
        
        latent_fwd = self.model.embedding.encode(input_fwd)

        for f in range(fwd):
            latent_fwd = self.model.operator.fwd_step(latent_fwd) # shift f times forward

            if self.current_step > 1 and self.loss_weights[5] > 0:
                shifted_fwd_storage = self.model.embedding.decode(latent_fwd)
                self.temporal_cons_fwd_storage[f] = shifted_fwd_storage
        
        shifted_fwd = self.model.embedding.decode(latent_fwd)
        
        loss_fwd_step += self.criterion(shifted_fwd, target_fwd)

        # Compute latent identity loss with shifted latent prediction
        if self.loss_weights[2] > 0:
            latent_target_fwd = self.model.embedding.encode(target_fwd)
            
            loss_latent_fwd_identity_step += self.criterion(latent_fwd, latent_target_fwd)

        return loss_fwd_step, loss_latent_fwd_identity_step

    def compute_backward_loss(self, input_bwd, target_bwd, bwd=1):
        loss_bwd_step = torch.tensor(0.0, device=self.device)
        loss_latent_bwd_identity_step = torch.tensor(0.0, device=self.device)

        latent_bwd = self.model.embedding.encode(input_bwd)

        for b in range(bwd):
            latent_bwd = self.model.operator.bwd_step(latent_bwd) # shift b times backward

            if self.current_step > 1 and self.loss_weights[5] > 0:
                shifted_bwd_storage = self.model.embedding.decode(latent_bwd)
                self.temporal_cons_bwd_storage[b-1] = shifted_bwd_storage
        
        
        shifted_bwd = self.model.embedding.decode(latent_bwd)

        loss_bwd_step += self.criterion(shifted_bwd, target_bwd)

        # Compute latent identity loss with shifted latent prediction
        if self.loss_weights[2] > 0:
            latent_target_bwd = self.model.embedding.encode(target_bwd)
            
            loss_latent_bwd_identity_step += self.criterion(latent_bwd, latent_target_bwd)
            
        return loss_bwd_step, loss_latent_bwd_identity_step


    def compute_identity_loss(self, input_tensor, shift_target_tensor):

        loss_identity_step = torch.tensor(0.0, device=self.device)
        
        latent_input = self.model.embedding.encode(input_tensor)
        autoencoded_input = self.model.embedding.decode(latent_input)
        loss_identity_step += self.criterion(input_tensor, autoencoded_input)
 
        if shift_target_tensor is not None:
            latent_target = self.model.embedding.encode(shift_target_tensor)
            autoencoded_target = self.model.embedding.decode(latent_target)
        
            loss_identity_step += self.criterion(shift_target_tensor, autoencoded_target) 

        return loss_identity_step

    def compute_inverse_consistency(self, input_tensor, target_tensor):
        
        loss_inv_cons_step = torch.tensor(0.0, device=self.device)
        
        latent_input = self.model.embedding.encode(input_tensor)
        
        inverted_bwd_fwd_input = self.model.embedding.decode(self.model.operator.bwd_step(self.model.operator.fwd_step(latent_input)))
        loss_inv_cons_step += self.criterion(inverted_bwd_fwd_input, input_tensor)

        inverted_fwd_bwd_input = self.model.embedding.decode(self.model.operator.fwd_step(self.model.operator.bwd_step(latent_input)))
        loss_inv_cons_step += self.criterion(inverted_fwd_bwd_input, input_tensor)

        if target_tensor is not None:
            latent_target = self.model.embedding.encode(target_tensor)
        
            inverted_bwd_fwd_target = self.model.embedding.decode(self.model.operator.bwd_step(self.model.operator.fwd_step(latent_target)))
            loss_inv_cons_step += self.criterion(inverted_bwd_fwd_target, target_tensor)
    
            inverted_fwd_bwd_target = self.model.embedding.decode(self.model.operator.fwd_step(self.model.operator.bwd_step(latent_target)))
            loss_inv_cons_step += self.criterion(inverted_fwd_bwd_target, target_tensor)

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
        return sum(loss * weight for loss, weight in zip(losses, self.loss_weights))

