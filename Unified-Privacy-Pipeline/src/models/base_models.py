"""
Base model architectures for privacy-preserving ML.

Provides abstract base classes and common functionality for models
that support differential privacy, machine unlearning, and influence tracking.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for privacy-preserving training."""
    
    # Differential Privacy
    enable_dp: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    
    # Machine Unlearning
    enable_unlearning: bool = True
    unlearning_method: str = "gradient_ascent"
    unlearning_lr: float = 0.01
    
    # Influence Functions
    enable_influence: bool = True
    influence_method: str = "lissa"
    influence_recursion_depth: int = 1000
    
    # General
    seed: int = 42
    

class PrivacyAwareModel(nn.Module, ABC):
    """
    Abstract base class for models that support privacy-preserving operations.
    
    Provides common interface for:
    - Differential privacy training
    - Machine unlearning
    - Influence function computation
    """
    
    def __init__(self, privacy_config: PrivacyConfig):
        super().__init__()
        self.privacy_config = privacy_config
        self.is_dp_enabled = False
        self.unlearning_history = []
        self.influence_cache = {}
        
        # Set random seed for reproducibility
        torch.manual_seed(privacy_config.seed)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations from input."""
        pass
    
    def enable_differential_privacy(self, privacy_engine=None):
        """Enable differential privacy for this model."""
        if privacy_engine is not None:
            self.privacy_engine = privacy_engine
            self.is_dp_enabled = True
            logger.info(f"Enabled DP with ε={self.privacy_config.dp_epsilon}, δ={self.privacy_config.dp_delta}")
        else:
            logger.warning("No privacy engine provided for DP")
    
    def compute_influence_scores(self, train_loader, test_sample, method="lissa"):
        """
        Compute influence scores for training samples on test sample.
        
        Args:
            train_loader: DataLoader for training data
            test_sample: Single test sample
            method: Influence computation method ("lissa", "exact")
            
        Returns:
            influence_scores: Tensor of influence scores for each training sample
        """
        if method == "lissa":
            return self._compute_lissa_influence(train_loader, test_sample)
        elif method == "exact":
            return self._compute_exact_influence(train_loader, test_sample)
        else:
            raise ValueError(f"Unknown influence method: {method}")
    
    def _compute_lissa_influence(self, train_loader, test_sample):
        """Compute influence using LiSSA approximation."""
        # Implementation will use pyDVL or custom implementation
        logger.info("Computing LiSSA influence scores...")
        # Placeholder - will implement with pyDVL integration
        batch_size = len(train_loader.dataset)
        return torch.zeros(batch_size)
    
    def _compute_exact_influence(self, train_loader, test_sample):
        """Compute exact influence using Hessian inverse."""
        logger.info("Computing exact influence scores...")
        # Placeholder - expensive computation
        batch_size = len(train_loader.dataset)
        return torch.zeros(batch_size)
    
    def unlearn_samples(self, forget_indices: List[int], retain_loader, method="gradient_ascent"):
        """
        Remove influence of specific samples from the model.
        
        Args:
            forget_indices: Indices of samples to forget
            retain_loader: DataLoader for samples to retain
            method: Unlearning method to use
        """
        logger.info(f"Unlearning {len(forget_indices)} samples using {method}")
        
        if method == "gradient_ascent":
            self._gradient_ascent_unlearning(forget_indices, retain_loader)
        elif method == "influence_based":
            self._influence_based_unlearning(forget_indices, retain_loader)
        else:
            raise ValueError(f"Unknown unlearning method: {method}")
            
        # Record unlearning operation
        self.unlearning_history.append({
            "method": method,
            "forget_indices": forget_indices,
            "num_samples": len(forget_indices)
        })
    
    def _gradient_ascent_unlearning(self, forget_indices, retain_loader):
        """Implement gradient ascent unlearning."""
        # Placeholder implementation
        logger.info("Performing gradient ascent unlearning...")
        
    def _influence_based_unlearning(self, forget_indices, retain_loader):
        """Implement influence-based unlearning."""
        # Use influence scores to guide unlearning
        logger.info("Performing influence-based unlearning...")
    
    def get_privacy_metrics(self) -> Dict[str, float]:
        """Get privacy-related metrics for the model."""
        metrics = {
            "dp_enabled": float(self.is_dp_enabled),
            "unlearning_operations": len(self.unlearning_history),
            "epsilon": self.privacy_config.dp_epsilon if self.is_dp_enabled else 0.0,
            "delta": self.privacy_config.dp_delta if self.is_dp_enabled else 0.0,
        }
        return metrics
    
    def save_privacy_state(self, filepath: str):
        """Save privacy-related state to file."""
        state = {
            "privacy_config": self.privacy_config,
            "unlearning_history": self.unlearning_history,
            "influence_cache": self.influence_cache,
            "model_state": self.state_dict()
        }
        torch.save(state, filepath)
        logger.info(f"Saved privacy state to {filepath}")
    
    def load_privacy_state(self, filepath: str):
        """Load privacy-related state from file."""
        state = torch.load(filepath, weights_only=False)
        self.privacy_config = state["privacy_config"]
        self.unlearning_history = state["unlearning_history"]
        self.influence_cache = state["influence_cache"]
        self.load_state_dict(state["model_state"])
        logger.info(f"Loaded privacy state from {filepath}")


class DPModel(PrivacyAwareModel):
    """
    Base class specifically for differential privacy-enabled models.
    
    Integrates with Opacus for DP training.
    """
    
    def __init__(self, privacy_config: PrivacyConfig):
        super().__init__(privacy_config)
        self.dp_accountant = None
        
    def setup_dp_training(self, optimizer, data_loader):
        """Setup differential privacy training with Opacus."""
        try:
            from opacus import PrivacyEngine
            
            privacy_engine = PrivacyEngine()
            
            # Make model, optimizer, and dataloader privacy-aware
            self, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=self,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=1,  # Will be set properly during training
                target_epsilon=self.privacy_config.dp_epsilon,
                target_delta=self.privacy_config.dp_delta,
                max_grad_norm=self.privacy_config.dp_max_grad_norm,
            )
            
            self.enable_differential_privacy(privacy_engine)
            self.dp_accountant = privacy_engine.accountant
            
            logger.info("Successfully setup DP training with Opacus")
            return self, optimizer, data_loader
            
        except ImportError:
            logger.error("Opacus not installed. Cannot enable differential privacy.")
            raise
        except Exception as e:
            logger.error(f"Failed to setup DP training: {e}")
            raise
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get privacy budget spent so far."""
        if self.dp_accountant is not None:
            return self.dp_accountant.get_privacy_spent(self.privacy_config.dp_delta)
        return 0.0, 0.0
    
    def get_dp_metrics(self) -> Dict[str, float]:
        """Get DP-specific metrics."""
        epsilon, delta = self.get_privacy_spent()
        metrics = self.get_privacy_metrics()
        metrics.update({
            "privacy_spent_epsilon": epsilon,
            "privacy_spent_delta": delta,
            "privacy_remaining_epsilon": max(0, self.privacy_config.dp_epsilon - epsilon),
        })
        return metrics