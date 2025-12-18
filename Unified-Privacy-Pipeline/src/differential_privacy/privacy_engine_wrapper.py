"""
Privacy Engine wrapper for enhanced functionality.

Provides additional features on top of Opacus PrivacyEngine including:
- Custom noise mechanisms
- Advanced privacy accounting
- Model-specific optimizations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging

try:
    from opacus import PrivacyEngine
    from opacus.accountants import RDPAccountant, GaussianAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrivacyEngineWrapper:
    """
    Enhanced wrapper around Opacus PrivacyEngine.
    
    Provides additional functionality for privacy-preserving training
    including custom accountants and noise mechanisms.
    """
    
    def __init__(self, 
                 accountant: str = "rdp",
                 secure_mode: bool = False,
                 custom_noise_scale: Optional[float] = None):
        
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required. Install with: pip install opacus")
        
        self.accountant_type = accountant
        self.secure_mode = secure_mode
        self.custom_noise_scale = custom_noise_scale
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine(
            accountant=accountant,
            secure_mode=secure_mode
        )
        
        self.is_attached = False
        self.privacy_history = []
        
    def make_private(self,
                    module: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    data_loader: torch.utils.data.DataLoader,
                    noise_multiplier: float,
                    max_grad_norm: float,
                    **kwargs) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        """
        Make model, optimizer, and data loader privacy-aware.
        
        Args:
            module: PyTorch model
            optimizer: PyTorch optimizer
            data_loader: PyTorch data loader
            noise_multiplier: Noise multiplier for DP-SGD
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional arguments for privacy engine
            
        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        logger.info(f"Making model private with noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}")
        
        # Apply custom noise scale if specified
        if self.custom_noise_scale is not None:
            noise_multiplier = self.custom_noise_scale
            logger.info(f"Using custom noise scale: {noise_multiplier}")
        
        private_module, private_optimizer, private_data_loader = self.privacy_engine.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            **kwargs
        )
        
        self.is_attached = True
        logger.info("Successfully made model private")
        
        return private_module, private_optimizer, private_data_loader
    
    def make_private_with_epsilon(self,
                                 module: nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 data_loader: torch.utils.data.DataLoader,
                                 epochs: int,
                                 target_epsilon: float,
                                 target_delta: float,
                                 max_grad_norm: float,
                                 **kwargs) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        """
        Make private with automatic noise calibration for target epsilon.
        
        Args:
            module: PyTorch model
            optimizer: PyTorch optimizer  
            data_loader: PyTorch data loader
            epochs: Number of training epochs
            target_epsilon: Target privacy parameter epsilon
            target_delta: Target privacy parameter delta
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        logger.info(f"Making private with target (ε,δ): ({target_epsilon}, {target_delta}) for {epochs} epochs")
        
        private_module, private_optimizer, private_data_loader = self.privacy_engine.make_private_with_epsilon(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            **kwargs
        )
        
        self.is_attached = True
        
        # Log initial privacy parameters
        noise_multiplier = private_optimizer.noise_multiplier
        logger.info(f"Auto-calibrated noise multiplier: {noise_multiplier}")
        
        return private_module, private_optimizer, private_data_loader
    
    def get_epsilon(self, delta: float) -> float:
        """Get current epsilon value."""
        if not self.is_attached:
            logger.warning("Privacy engine not attached to model")
            return 0.0
            
        return self.privacy_engine.accountant.get_epsilon(delta=delta)
    
    def get_privacy_spent(self, delta: float) -> Tuple[float, float]:
        """Get current privacy expenditure."""
        epsilon = self.get_epsilon(delta)
        return epsilon, delta
    
    def step(self):
        """Manual step for privacy accounting (if needed)."""
        if hasattr(self.privacy_engine.accountant, 'step'):
            self.privacy_engine.accountant.step()
    
    def get_noise_multiplier(self) -> float:
        """Get current noise multiplier."""
        if hasattr(self.privacy_engine, 'noise_multiplier'):
            return self.privacy_engine.noise_multiplier
        return 0.0
    
    def update_privacy_history(self, step: int, loss: float, epsilon: float, delta: float):
        """Update privacy history tracking."""
        self.privacy_history.append({
            "step": step,
            "loss": loss,
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": self.get_noise_multiplier()
        })
    
    def get_privacy_analysis(self) -> Dict[str, Any]:
        """Get comprehensive privacy analysis."""
        if not self.is_attached:
            return {"error": "Privacy engine not attached"}
        
        analysis = {
            "accountant_type": self.accountant_type,
            "secure_mode": self.secure_mode,
            "is_attached": self.is_attached,
            "noise_multiplier": self.get_noise_multiplier(),
            "privacy_history_length": len(self.privacy_history)
        }
        
        # Add accountant-specific information
        accountant = self.privacy_engine.accountant
        if hasattr(accountant, 'history'):
            analysis["accountant_steps"] = len(accountant.history)
        
        return analysis
    
    def save_privacy_state(self, filepath: str):
        """Save privacy engine state."""
        state = {
            "accountant_type": self.accountant_type,
            "secure_mode": self.secure_mode,
            "privacy_history": self.privacy_history,
            "accountant_state": None
        }
        
        # Save accountant state if possible
        if hasattr(self.privacy_engine.accountant, 'state_dict'):
            state["accountant_state"] = self.privacy_engine.accountant.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Privacy state saved to {filepath}")
    
    def load_privacy_state(self, filepath: str):
        """Load privacy engine state."""
        state = torch.load(filepath, weights_only=False)
        
        self.accountant_type = state["accountant_type"]
        self.secure_mode = state["secure_mode"]  
        self.privacy_history = state["privacy_history"]
        
        # Load accountant state if available
        if state["accountant_state"] is not None and hasattr(self.privacy_engine.accountant, 'load_state_dict'):
            self.privacy_engine.accountant.load_state_dict(state["accountant_state"])
        
        logger.info(f"Privacy state loaded from {filepath}")
    
    def validate_privacy_parameters(self, epsilon: float, delta: float) -> bool:
        """Validate privacy parameters."""
        if epsilon <= 0:
            logger.error(f"Invalid epsilon: {epsilon}. Must be positive.")
            return False
            
        if delta <= 0 or delta >= 1:
            logger.error(f"Invalid delta: {delta}. Must be in (0, 1).")
            return False
            
        # Check if parameters are reasonable for practical use
        if epsilon > 10:
            logger.warning(f"High epsilon value: {epsilon}. Consider lower values for better privacy.")
            
        if delta > 1e-3:
            logger.warning(f"High delta value: {delta}. Consider lower values for better privacy.")
        
        return True
    
    def estimate_noise_for_epsilon(self,
                                  target_epsilon: float,
                                  target_delta: float,
                                  epochs: int,
                                  batch_size: int,
                                  dataset_size: int) -> float:
        """
        Estimate required noise multiplier for target epsilon.
        
        Args:
            target_epsilon: Target epsilon value
            target_delta: Target delta value  
            epochs: Number of training epochs
            batch_size: Batch size
            dataset_size: Size of training dataset
            
        Returns:
            Estimated noise multiplier
        """
        steps = epochs * (dataset_size // batch_size)
        
        if self.accountant_type == "rdp":
            # Rough approximation for RDP accountant
            # This is a simplified calculation
            noise_multiplier = max(1.0, target_epsilon / (2 * epochs * torch.log(torch.tensor(1.0 / target_delta))))
        else:
            # Default conservative estimate
            noise_multiplier = 1.0
        
        logger.info(f"Estimated noise multiplier: {noise_multiplier} for (ε,δ): ({target_epsilon}, {target_delta})")
        return float(noise_multiplier)
    
    def get_privacy_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed privacy cost breakdown."""
        if not self.is_attached:
            return {"error": "Privacy engine not attached"}
        
        accountant = self.privacy_engine.accountant
        breakdown = {
            "total_steps": len(accountant.history) if hasattr(accountant, 'history') else 0,
            "accountant_type": type(accountant).__name__
        }
        
        # Add RDP-specific information
        if hasattr(accountant, 'orders'):
            breakdown["rdp_orders"] = accountant.orders
            
        if hasattr(accountant, 'rdp'):
            breakdown["rdp_values"] = accountant.rdp
        
        return breakdown