"""
Advanced Influence-Guided Unlearning Algorithms.

Implements state-of-the-art unlearning methods that leverage influence functions
to guide the unlearning process, providing more precise and efficient data removal
while preserving model utility.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from influence_functions.influence_computation import InfluenceComputation
from machine_unlearning.unlearning_methods import BaseUnlearningMethod

logger = logging.getLogger(__name__)


@dataclass
class InfluenceGuidedConfig:
    """Configuration for influence-guided unlearning methods."""
    # Influence computation parameters
    influence_method: str = "lissa"  # "lissa", "exact", "conjugate_gradient"
    lissa_iterations: int = 1000
    lissa_damping: float = 0.01
    lissa_scale: float = 10.0
    
    # Unlearning parameters
    unlearning_lr: float = 0.01
    max_unlearning_iterations: int = 100
    convergence_threshold: float = 1e-5
    influence_threshold: float = 0.1  # Minimum influence to consider
    
    # Adaptive parameters
    adaptive_lr: bool = True
    lr_decay_factor: float = 0.9
    patience: int = 10
    
    # Regularization
    l2_regularization: float = 1e-4
    gradient_clipping: float = 1.0
    
    # Verification parameters
    verify_unlearning: bool = True
    verification_samples: int = 100


class InfluenceGuidedUnlearner(BaseUnlearningMethod, ABC):
    """
    Base class for influence-guided unlearning algorithms.
    """
    
    def __init__(self, config: InfluenceGuidedConfig):
        self.config = config
        self.influence_computer = InfluenceComputation()
        
        # State tracking
        self.unlearning_history = []
        self.influence_scores = {}
        self.converged = False
        
    @abstractmethod
    def compute_unlearning_direction(self, 
                                   model: nn.Module,
                                   forget_sample: torch.Tensor,
                                   forget_target: torch.Tensor,
                                   influence_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the direction for parameter updates based on influence scores."""
        pass
    
    def unlearn(self, 
                model: nn.Module,
                forget_loader: torch.utils.data.DataLoader,
                retain_loader: torch.utils.data.DataLoader,
                device: torch.device = torch.device('cpu'),
                **kwargs) -> Dict[str, Any]:
        """
        Perform influence-guided unlearning.
        
        Args:
            model: Model to unlearn from
            forget_loader: Data to forget
            retain_loader: Data to retain
            device: Computation device
            
        Returns:
            Dictionary with unlearning results and metrics
        """
        logger.info("Starting influence-guided unlearning...")
        
        model = model.to(device)
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Compute influence scores for forget samples
        logger.info("Computing influence scores...")
        influence_results = self._compute_influences(model, forget_loader, retain_loader, device)
        
        # Perform guided unlearning
        unlearning_results = self._guided_unlearning_loop(
            model, forget_loader, retain_loader, influence_results, device
        )
        
        # Verify unlearning if requested
        if self.config.verify_unlearning:
            verification_results = self._verify_unlearning(
                original_params, model, forget_loader, device
            )
            unlearning_results['verification'] = verification_results
        
        logger.info("Influence-guided unlearning completed")
        return unlearning_results
    
    def _compute_influences(self,
                          model: nn.Module,
                          forget_loader: torch.utils.data.DataLoader,
                          retain_loader: torch.utils.data.DataLoader,
                          device: torch.device) -> Dict[str, Any]:
        """Compute influence scores for all forget samples."""
        
        model.eval()
        all_influences = {}
        forget_samples = []
        
        # Collect forget samples
        for batch_idx, batch_data in enumerate(forget_loader):
            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)
            
            forget_samples.extend([(data[i], target[i]) for i in range(len(data))])
        
        # Compute influences for each forget sample
        sample_influences = []
        
        for sample_idx, (forget_data, forget_target) in enumerate(forget_samples[:self.config.verification_samples]):
            logger.debug(f"Computing influence for forget sample {sample_idx}")
            
            try:
                # Compute influence using selected method
                if self.config.influence_method == "lissa":
                    influences = self.influence_computer.compute_influence_lissa(
                        model, 
                        forget_data.unsqueeze(0), 
                        forget_target.unsqueeze(0),
                        retain_loader,
                        device,
                        iterations=self.config.lissa_iterations,
                        damping=self.config.lissa_damping,
                        scale=self.config.lissa_scale
                    )
                elif self.config.influence_method == "exact":
                    influences = self.influence_computer.compute_influence_exact(
                        model,
                        forget_data.unsqueeze(0),
                        forget_target.unsqueeze(0),
                        retain_loader,
                        device
                    )
                else:
                    # Fallback to simple gradient-based influence
                    influences = self._compute_simple_influence(
                        model, forget_data.unsqueeze(0), forget_target.unsqueeze(0), device
                    )
                
                sample_influences.append({
                    'sample_idx': sample_idx,
                    'influences': influences,
                    'magnitude': sum(torch.norm(inf).item() for inf in influences.values())
                })
                
            except Exception as e:
                logger.warning(f"Failed to compute influence for sample {sample_idx}: {e}")
                continue
        
        # Aggregate influence results
        influence_results = {
            'sample_influences': sample_influences,
            'total_samples': len(forget_samples),
            'computed_samples': len(sample_influences),
            'average_magnitude': np.mean([si['magnitude'] for si in sample_influences]) if sample_influences else 0
        }
        
        logger.info(f"Computed influences for {len(sample_influences)} samples")
        return influence_results
    
    def _compute_simple_influence(self,
                                model: nn.Module,
                                data: torch.Tensor,
                                target: torch.Tensor,
                                device: torch.device) -> Dict[str, torch.Tensor]:
        """Compute simplified influence scores (gradient-based approximation)."""
        
        model.train()
        
        # Forward pass
        output = model(data)
        
        # Compute loss based on model type
        if output.dim() == 1 or output.shape[1] == 1:
            # Binary classification or regression
            if target.dtype == torch.long:
                loss = nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
            else:
                loss = nn.MSELoss()(output.squeeze(), target)
        else:
            # Multi-class classification
            loss = nn.CrossEntropyLoss()(output, target.long())
        
        # Compute gradients
        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        
        # Package gradients by parameter name
        influence_dict = {}
        for (name, param), grad in zip(model.named_parameters(), gradients):
            influence_dict[name] = grad.detach()
        
        return influence_dict
    
    def _guided_unlearning_loop(self,
                              model: nn.Module,
                              forget_loader: torch.utils.data.DataLoader,
                              retain_loader: torch.utils.data.DataLoader,
                              influence_results: Dict[str, Any],
                              device: torch.device) -> Dict[str, Any]:
        """Main guided unlearning optimization loop."""
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.unlearning_lr)
        if self.config.adaptive_lr:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.config.lr_decay_factor, patience=self.config.patience
            )
        
        # Track metrics
        iteration_metrics = []
        best_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(self.config.max_unlearning_iterations):
            model.train()
            
            # Compute unlearning updates
            forget_loss, retain_loss, total_loss = self._compute_guided_updates(
                model, forget_loader, retain_loader, influence_results, optimizer, device
            )
            
            # Track metrics
            metrics = {
                'iteration': iteration,
                'forget_loss': forget_loss,
                'retain_loss': retain_loss,
                'total_loss': total_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            iteration_metrics.append(metrics)
            
            # Learning rate scheduling
            if self.config.adaptive_lr:
                scheduler.step(total_loss)
            
            # Convergence check
            if total_loss < best_loss - self.config.convergence_threshold:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.patience:
                logger.info(f"Converged after {iteration} iterations")
                self.converged = True
                break
            
            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}: Forget Loss {forget_loss:.4f}, "
                           f"Retain Loss {retain_loss:.4f}, Total Loss {total_loss:.4f}")
        
        return {
            'converged': self.converged,
            'iterations': len(iteration_metrics),
            'final_metrics': iteration_metrics[-1] if iteration_metrics else {},
            'iteration_history': iteration_metrics,
            'influence_guided': True
        }
    
    def _compute_guided_updates(self,
                              model: nn.Module,
                              forget_loader: torch.utils.data.DataLoader,
                              retain_loader: torch.utils.data.DataLoader,
                              influence_results: Dict[str, Any],
                              optimizer: torch.optim.Optimizer,
                              device: torch.device) -> Tuple[float, float, float]:
        """Compute parameter updates guided by influence scores."""
        
        optimizer.zero_grad()
        
        total_forget_loss = 0.0
        total_retain_loss = 0.0
        num_forget_batches = 0
        num_retain_batches = 0
        
        # Process forget samples with influence guidance
        for batch_data in forget_loader:
            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)
            
            # Compute forget loss (to minimize/reverse)
            output = model(data)
            
            if output.dim() == 1 or output.shape[1] == 1:
                if target.dtype == torch.long:
                    forget_loss = nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
                else:
                    forget_loss = nn.MSELoss()(output.squeeze(), target)
            else:
                forget_loss = nn.CrossEntropyLoss()(output, target.long())
            
            # Apply influence-guided weighting
            influence_weight = self._get_influence_weight(batch_data, influence_results)
            weighted_forget_loss = -influence_weight * forget_loss  # Negative to encourage forgetting
            
            weighted_forget_loss.backward(retain_graph=True)
            total_forget_loss += forget_loss.item()
            num_forget_batches += 1
        
        # Process retain samples to maintain utility
        for batch_data in retain_loader:
            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)
            
            # Compute retain loss (to maintain)
            output = model(data)
            
            if output.dim() == 1 or output.shape[1] == 1:
                if target.dtype == torch.long:
                    retain_loss = nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
                else:
                    retain_loss = nn.MSELoss()(output.squeeze(), target)
            else:
                retain_loss = nn.CrossEntropyLoss()(output, target.long())
            
            retain_loss.backward(retain_graph=True)
            total_retain_loss += retain_loss.item()
            num_retain_batches += 1
        
        # Apply gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
        
        # L2 regularization
        if self.config.l2_regularization > 0:
            l2_penalty = sum(torch.norm(param)**2 for param in model.parameters())
            l2_loss = self.config.l2_regularization * l2_penalty
            l2_loss.backward()
        
        optimizer.step()
        
        # Compute average losses
        avg_forget_loss = total_forget_loss / max(num_forget_batches, 1)
        avg_retain_loss = total_retain_loss / max(num_retain_batches, 1)
        total_loss = avg_forget_loss + avg_retain_loss
        
        return avg_forget_loss, avg_retain_loss, total_loss
    
    def _get_influence_weight(self, 
                            batch_data: Union[Dict, Tuple], 
                            influence_results: Dict[str, Any]) -> float:
        """Get influence weight for a batch of data."""
        
        # For now, use average influence magnitude
        # In practice, you'd match samples to their computed influences
        sample_influences = influence_results.get('sample_influences', [])
        
        if not sample_influences:
            return 1.0
        
        avg_magnitude = influence_results.get('average_magnitude', 1.0)
        
        # Scale weight based on influence magnitude
        if avg_magnitude > self.config.influence_threshold:
            return min(2.0, avg_magnitude)  # Cap at 2x normal weight
        else:
            return 0.5  # Reduce weight for low-influence samples
    
    def _verify_unlearning(self,
                         original_params: Dict[str, torch.Tensor],
                         unlearned_model: nn.Module,
                         forget_loader: torch.utils.data.DataLoader,
                         device: torch.device) -> Dict[str, Any]:
        """Verify that unlearning was successful."""
        
        logger.info("Verifying unlearning effectiveness...")
        
        # Compute parameter changes
        param_changes = {}
        total_change = 0.0
        
        for name, param in unlearned_model.named_parameters():
            if name in original_params:
                change = torch.norm(param - original_params[name]).item()
                param_changes[name] = change
                total_change += change
        
        # Test model's memory of forget samples
        unlearned_model.eval()
        forget_losses = []
        
        with torch.no_grad():
            for batch_data in forget_loader:
                if isinstance(batch_data, dict):
                    data = batch_data.get('image', batch_data.get('features')).to(device)
                    target = batch_data.get('identity', batch_data.get('target')).to(device)
                else:
                    data, target = batch_data[0].to(device), batch_data[1].to(device)
                
                output = unlearned_model(data)
                
                # Compute loss on forget samples (should be high if forgotten)
                if output.dim() == 1 or output.shape[1] == 1:
                    if target.dtype == torch.long:
                        loss = nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
                    else:
                        loss = nn.MSELoss()(output.squeeze(), target)
                else:
                    loss = nn.CrossEntropyLoss()(output, target.long())
                
                forget_losses.append(loss.item())
        
        verification_results = {
            'total_parameter_change': total_change,
            'parameter_changes': param_changes,
            'avg_forget_loss': np.mean(forget_losses) if forget_losses else 0.0,
            'forget_loss_std': np.std(forget_losses) if forget_losses else 0.0,
            'unlearning_detected': total_change > self.config.convergence_threshold,
            'forget_quality': min(1.0, np.mean(forget_losses) / 2.0) if forget_losses else 0.0  # Heuristic
        }
        
        logger.info(f"Unlearning verification complete. "
                   f"Parameter change: {total_change:.6f}, "
                   f"Avg forget loss: {verification_results['avg_forget_loss']:.4f}")
        
        return verification_results


class LISSAGuidedUnlearner(InfluenceGuidedUnlearner):
    """
    Influence-guided unlearning using LiSSA (Linear time Stochastic Second-order Algorithm).
    """
    
    def compute_unlearning_direction(self,
                                   model: nn.Module,
                                   forget_sample: torch.Tensor,
                                   forget_target: torch.Tensor,
                                   influence_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute unlearning direction using LiSSA-estimated influences."""
        
        # The direction is directly given by the influence scores
        # Scale by learning rate and influence magnitude
        unlearning_directions = {}
        
        for param_name, influence in influence_scores.items():
            # Direction opposite to influence (to reverse the effect)
            direction = -influence * self.config.unlearning_lr
            unlearning_directions[param_name] = direction
        
        return unlearning_directions


class ExactInfluenceUnlearner(InfluenceGuidedUnlearner):
    """
    Influence-guided unlearning using exact influence function computation.
    """
    
    def compute_unlearning_direction(self,
                                   model: nn.Module,
                                   forget_sample: torch.Tensor,
                                   forget_target: torch.Tensor,
                                   influence_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute unlearning direction using exact influences."""
        
        unlearning_directions = {}
        
        for param_name, influence in influence_scores.items():
            # Use exact influence with careful scaling
            direction = -influence * self.config.unlearning_lr * 0.5  # More conservative
            unlearning_directions[param_name] = direction
        
        return unlearning_directions


class AdaptiveInfluenceUnlearner(InfluenceGuidedUnlearner):
    """
    Advanced influence-guided unlearner with adaptive mechanisms.
    """
    
    def __init__(self, config: InfluenceGuidedConfig):
        super().__init__(config)
        self.adaptation_history = []
        self.influence_threshold_adaptive = config.influence_threshold
    
    def compute_unlearning_direction(self,
                                   model: nn.Module,
                                   forget_sample: torch.Tensor,
                                   forget_target: torch.Tensor,
                                   influence_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute adaptive unlearning direction."""
        
        unlearning_directions = {}
        
        # Adaptive threshold based on influence distribution
        influence_magnitudes = [torch.norm(inf).item() for inf in influence_scores.values()]
        median_influence = np.median(influence_magnitudes) if influence_magnitudes else 0.1
        
        # Update adaptive threshold
        self.influence_threshold_adaptive = 0.7 * self.influence_threshold_adaptive + 0.3 * median_influence
        
        for param_name, influence in influence_scores.items():
            influence_magnitude = torch.norm(influence).item()
            
            # Adaptive scaling based on influence magnitude
            if influence_magnitude > self.influence_threshold_adaptive:
                scale_factor = min(2.0, influence_magnitude / self.influence_threshold_adaptive)
            else:
                scale_factor = 0.5
            
            direction = -influence * self.config.unlearning_lr * scale_factor
            unlearning_directions[param_name] = direction
        
        # Track adaptation
        self.adaptation_history.append({
            'adaptive_threshold': self.influence_threshold_adaptive,
            'median_influence': median_influence,
            'num_high_influence': sum(1 for mag in influence_magnitudes 
                                    if mag > self.influence_threshold_adaptive)
        })
        
        return unlearning_directions


class HybridInfluenceUnlearner(InfluenceGuidedUnlearner):
    """
    Hybrid unlearner combining multiple influence computation methods.
    """
    
    def __init__(self, config: InfluenceGuidedConfig):
        super().__init__(config)
        self.ensemble_weights = {'lissa': 0.6, 'exact': 0.3, 'simple': 0.1}
    
    def _compute_influences(self,
                          model: nn.Module,
                          forget_loader: torch.utils.data.DataLoader,
                          retain_loader: torch.utils.data.DataLoader,
                          device: torch.device) -> Dict[str, Any]:
        """Compute influences using multiple methods and ensemble them."""
        
        logger.info("Computing ensemble influences using multiple methods...")
        
        all_influences = {'lissa': [], 'exact': [], 'simple': []}
        forget_samples = []
        
        # Collect forget samples
        for batch_data in forget_loader:
            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)
            
            forget_samples.extend([(data[i], target[i]) for i in range(len(data))])
        
        # Compute influences using different methods
        sample_influences = []
        
        for sample_idx, (forget_data, forget_target) in enumerate(forget_samples[:self.config.verification_samples]):
            sample_results = {}
            
            # LiSSA method
            try:
                lissa_influences = self.influence_computer.compute_influence_lissa(
                    model, forget_data.unsqueeze(0), forget_target.unsqueeze(0),
                    retain_loader, device,
                    iterations=self.config.lissa_iterations // 2,  # Faster computation
                    damping=self.config.lissa_damping,
                    scale=self.config.lissa_scale
                )
                sample_results['lissa'] = lissa_influences
            except Exception as e:
                logger.warning(f"LiSSA influence computation failed: {e}")
                sample_results['lissa'] = None
            
            # Simple gradient-based method
            try:
                simple_influences = self._compute_simple_influence(
                    model, forget_data.unsqueeze(0), forget_target.unsqueeze(0), device
                )
                sample_results['simple'] = simple_influences
            except Exception as e:
                logger.warning(f"Simple influence computation failed: {e}")
                sample_results['simple'] = None
            
            # Ensemble influences
            if any(sample_results.values()):
                ensembled_influences = self._ensemble_influences(sample_results)
                
                sample_influences.append({
                    'sample_idx': sample_idx,
                    'influences': ensembled_influences,
                    'magnitude': sum(torch.norm(inf).item() for inf in ensembled_influences.values()),
                    'methods_used': [k for k, v in sample_results.items() if v is not None]
                })
        
        influence_results = {
            'sample_influences': sample_influences,
            'total_samples': len(forget_samples),
            'computed_samples': len(sample_influences),
            'average_magnitude': np.mean([si['magnitude'] for si in sample_influences]) if sample_influences else 0,
            'ensemble_method': 'hybrid'
        }
        
        logger.info(f"Computed hybrid influences for {len(sample_influences)} samples")
        return influence_results
    
    def _ensemble_influences(self, method_results: Dict[str, Optional[Dict]]) -> Dict[str, torch.Tensor]:
        """Ensemble influence results from multiple methods."""
        
        valid_methods = {k: v for k, v in method_results.items() if v is not None}
        
        if not valid_methods:
            raise ValueError("No valid influence computation methods")
        
        # Get parameter names from first valid method
        first_method = next(iter(valid_methods.values()))
        param_names = first_method.keys()
        
        ensembled_influences = {}
        
        for param_name in param_names:
            weighted_influence = torch.zeros_like(first_method[param_name])
            total_weight = 0.0
            
            for method_name, influences in valid_methods.items():
                if param_name in influences:
                    weight = self.ensemble_weights.get(method_name, 0.1)
                    weighted_influence += weight * influences[param_name]
                    total_weight += weight
            
            if total_weight > 0:
                ensembled_influences[param_name] = weighted_influence / total_weight
            else:
                ensembled_influences[param_name] = weighted_influence
        
        return ensembled_influences
    
    def compute_unlearning_direction(self,
                                   model: nn.Module,
                                   forget_sample: torch.Tensor,
                                   forget_target: torch.Tensor,
                                   influence_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute unlearning direction using ensembled influences."""
        
        unlearning_directions = {}
        
        for param_name, influence in influence_scores.items():
            # Use hybrid approach with moderate scaling
            direction = -influence * self.config.unlearning_lr * 0.8
            unlearning_directions[param_name] = direction
        
        return unlearning_directions


def create_influence_guided_config(**kwargs) -> InfluenceGuidedConfig:
    """Create influence-guided unlearning configuration with sensible defaults."""
    
    return InfluenceGuidedConfig(**kwargs)


def create_influence_guided_unlearner(method: str = "hybrid", 
                                     **config_kwargs) -> InfluenceGuidedUnlearner:
    """
    Factory function to create influence-guided unlearners.
    
    Args:
        method: Type of influence-guided unlearner
        **config_kwargs: Configuration parameters
        
    Returns:
        Configured influence-guided unlearner
    """
    
    config = create_influence_guided_config(**config_kwargs)
    
    if method == "lissa":
        return LISSAGuidedUnlearner(config)
    elif method == "exact":
        return ExactInfluenceUnlearner(config)
    elif method == "adaptive":
        return AdaptiveInfluenceUnlearner(config)
    elif method == "hybrid":
        return HybridInfluenceUnlearner(config)
    else:
        raise ValueError(f"Unknown influence-guided unlearning method: {method}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = create_influence_guided_config(
        influence_method="lissa",
        unlearning_lr=0.01,
        max_unlearning_iterations=50,
        verify_unlearning=True
    )
    
    # Create unlearner
    unlearner = create_influence_guided_unlearner("hybrid", **config.__dict__)
    
    print("Advanced influence-guided unlearner created successfully!")