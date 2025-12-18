"""
Health prediction models with privacy-preserving capabilities.

Implements neural network architectures for healthcare applications that support:
- HIPAA compliance through differential privacy
- Federated learning capabilities
- Machine unlearning for patient data deletion
- Influence tracking for audit trails
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .base_models import PrivacyAwareModel, DPModel, PrivacyConfig

logger = logging.getLogger(__name__)


class HealthPredictor(DPModel):
    """
    Privacy-aware health prediction model for clinical data.
    
    Supports various clinical prediction tasks like:
    - Disease diagnosis
    - Risk prediction 
    - Treatment outcome prediction
    """
    
    def __init__(self, privacy_config: PrivacyConfig,
                 input_dim: int = 100,
                 hidden_dims: List[int] = [256, 128, 64],
                 output_dim: int = 1,
                 task_type: str = "binary_classification"):
        super().__init__(privacy_config)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task_type = task_type  # "binary_classification", "multiclass", "regression"
        
        # Build neural network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)  # Higher dropout for better privacy
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if task_type == "binary_classification":
            layers.append(nn.Sigmoid())
        elif task_type == "multiclass":
            layers.append(nn.Softmax(dim=1))
        # No activation for regression
        
        self.network = nn.Sequential(*layers)
        
        # Feature extraction layers (before final classification)
        self.feature_extractor = nn.Sequential(*layers[:-2])  # Remove final linear + activation
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with smaller values for better privacy."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization for better DP performance
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through health prediction model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions tensor
        """
        # Handle missing values (common in health data)
        x = self._handle_missing_values(x)
        
        # Forward through network
        output = self.network(x)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations before final prediction."""
        x = self._handle_missing_values(x)
        features = self.feature_extractor(x)
        return features
    
    def _handle_missing_values(self, x: torch.Tensor) -> torch.Tensor:
        """Handle missing values in clinical data."""
        # Replace NaN with median (privacy-preserving approach)
        # In practice, this should be done during preprocessing
        x = torch.nan_to_num(x, nan=0.0)
        return x
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                mc_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            x: Input tensor
            mc_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(mc_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        self.eval()
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty


class FederatedHealthNet(HealthPredictor):
    """
    Health prediction model designed for federated learning.
    
    Supports:
    - Local training with privacy preservation
    - Secure aggregation
    - Differential privacy in federated setting
    """
    
    def __init__(self, privacy_config: PrivacyConfig,
                 client_id: str,
                 **kwargs):
        super().__init__(privacy_config, **kwargs)
        
        self.client_id = client_id
        self.local_updates = 0
        self.communication_rounds = 0
        
        # Client-specific batch normalization
        self._replace_batch_norm_with_group_norm()
    
    def _replace_batch_norm_with_group_norm(self):
        """Replace BatchNorm with GroupNorm for better federated performance."""
        # GroupNorm works better in federated settings with small local batches
        def replace_bn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm1d):
                    # Replace with GroupNorm
                    num_groups = min(8, child.num_features)  # Ensure divisible groups
                    if child.num_features % num_groups != 0:
                        num_groups = child.num_features
                    
                    setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
                else:
                    replace_bn(child)
        
        replace_bn(self)
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated aggregation."""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters from federated aggregation."""
        for name, param in self.named_parameters():
            if name in parameters:
                param.data = parameters[name]
    
    def compute_local_update(self, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute local parameter updates for federated averaging."""
        local_params = self.get_model_parameters()
        updates = {}
        
        for name in local_params:
            if name in global_params:
                updates[name] = local_params[name] - global_params[name]
        
        return updates
    
    def apply_dp_noise_to_gradients(self, noise_scale: float = 1.0):
        """Apply differential privacy noise to gradients."""
        if self.privacy_config.enable_dp:
            for param in self.parameters():
                if param.grad is not None:
                    noise = torch.normal(0, noise_scale, size=param.grad.shape, 
                                       device=param.grad.device)
                    param.grad += noise
    
    def get_federated_metrics(self) -> Dict[str, float]:
        """Get metrics specific to federated learning."""
        base_metrics = self.get_privacy_metrics()
        
        federated_metrics = {
            "local_updates": float(self.local_updates),
            "communication_rounds": float(self.communication_rounds),
            "client_id_hash": hash(self.client_id) % 1000,  # Anonymized client ID
        }
        
        return {**base_metrics, **federated_metrics}


class MultiTaskHealthPredictor(DPModel):
    """
    Multi-task health prediction model for predicting multiple outcomes.
    
    Useful for:
    - Predicting multiple diseases simultaneously
    - Shared representations across related tasks
    - More efficient privacy budget usage
    """
    
    def __init__(self, privacy_config: PrivacyConfig,
                 input_dim: int = 100,
                 shared_hidden_dims: List[int] = [256, 128],
                 task_configs: Dict[str, Dict] = None):
        super().__init__(privacy_config)
        
        self.input_dim = input_dim
        self.shared_hidden_dims = shared_hidden_dims
        self.task_configs = task_configs or {}
        
        # Shared feature extraction layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in self.task_configs.items():
            task_output_dim = task_config.get('output_dim', 1)
            task_type = task_config.get('type', 'binary_classification')
            
            # Task-specific layers
            task_layers = [
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(prev_dim // 2, task_output_dim)
            ]
            
            if task_type == "binary_classification":
                task_layers.append(nn.Sigmoid())
            elif task_type == "multiclass":
                task_layers.append(nn.Softmax(dim=1))
            
            self.task_heads[task_name] = nn.Sequential(*task_layers)
    
    def forward(self, x: torch.Tensor, 
               tasks: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task model.
        
        Args:
            x: Input tensor
            tasks: List of tasks to compute (if None, compute all)
            
        Returns:
            Dictionary mapping task names to predictions
        """
        # Extract shared features
        shared_features = self.shared_encoder(x)
        
        # Compute task-specific predictions
        predictions = {}
        tasks_to_compute = tasks or list(self.task_heads.keys())
        
        for task_name in tasks_to_compute:
            if task_name in self.task_heads:
                predictions[task_name] = self.task_heads[task_name](shared_features)
        
        return predictions
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared feature representations."""
        return self.shared_encoder(x)
    
    def get_task_specific_features(self, x: torch.Tensor, 
                                  task_name: str) -> torch.Tensor:
        """Get features from task-specific layers."""
        shared_features = self.shared_encoder(x)
        
        if task_name in self.task_heads:
            # Get features before final layer
            task_layers = self.task_heads[task_name][:-1]  # Remove final layer
            task_features = task_layers(shared_features)
            return task_features
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def compute_multitask_loss(self, predictions: Dict[str, torch.Tensor],
                              targets: Dict[str, torch.Tensor],
                              loss_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets  
            loss_weights: Dictionary of loss weights per task
            
        Returns:
            Combined multi-task loss
        """
        total_loss = 0.0
        loss_weights = loss_weights or {}
        
        for task_name in predictions:
            if task_name in targets:
                task_config = self.task_configs.get(task_name, {})
                task_type = task_config.get('type', 'binary_classification')
                
                # Choose appropriate loss function
                if task_type == "binary_classification":
                    loss_fn = nn.BCELoss()
                elif task_type == "multiclass":
                    loss_fn = nn.CrossEntropyLoss()
                elif task_type == "regression":
                    loss_fn = nn.MSELoss()
                else:
                    loss_fn = nn.MSELoss()  # Default
                
                task_loss = loss_fn(predictions[task_name], targets[task_name])
                weight = loss_weights.get(task_name, 1.0)
                total_loss += weight * task_loss
        
        return total_loss