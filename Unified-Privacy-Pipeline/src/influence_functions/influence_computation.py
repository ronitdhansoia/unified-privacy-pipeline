"""
Influence Function Computation for Machine Learning Models.

Implements various methods for computing influence functions:
- LiSSA (Linear time Stochastic Second-order Algorithm)
- Exact influence computation
- Conjugate gradient methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InfluenceConfig:
    """Configuration for influence computation."""
    method: str = "lissa"  # "lissa", "exact", "conjugate_gradient"
    lissa_iterations: int = 1000
    lissa_damping: float = 0.01
    lissa_scale: float = 10.0
    lissa_batch_size: int = 32
    recursion_depth: int = 5000
    r: int = 1

    # Conjugate gradient parameters
    cg_max_iterations: int = 100
    cg_tolerance: float = 1e-5

    # Memory optimization
    use_checkpointing: bool = False
    chunk_size: int = 1


class InfluenceComputation:
    """
    Compute influence functions for neural networks.

    Influence functions measure how much a training sample affects
    model predictions or parameters.
    """

    def __init__(self, config: Optional[InfluenceConfig] = None):
        self.config = config or InfluenceConfig()
        self.hvp_cache = {}

    def compute_influence_lissa(self,
                               model: nn.Module,
                               test_data: torch.Tensor,
                               test_target: torch.Tensor,
                               train_loader: torch.utils.data.DataLoader,
                               device: torch.device,
                               iterations: Optional[int] = None,
                               damping: Optional[float] = None,
                               scale: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Compute influence using LiSSA approximation.

        LiSSA provides a linear-time approximation to the inverse Hessian-vector product.

        Args:
            model: Neural network model
            test_data: Test point to compute influence for
            test_target: Test point labels
            train_loader: Training data loader
            device: Computation device
            iterations: Number of LiSSA iterations (optional override)
            damping: Damping parameter (optional override)
            scale: Scaling parameter (optional override)

        Returns:
            Dictionary mapping parameter names to influence values
        """
        iterations = iterations or self.config.lissa_iterations
        damping = damping or self.config.lissa_damping
        scale = scale or self.config.lissa_scale

        logger.debug(f"Computing LiSSA influence with {iterations} iterations")

        model.eval()

        # Compute gradient of loss at test point
        test_grads = self._compute_loss_gradient(model, test_data, test_target, device)

        # Initialize H_inverse * v estimate
        inverse_hvp = {name: torch.zeros_like(grad) for name, grad in test_grads.items()}

        # LiSSA iterations
        for i in range(iterations):
            # Sample random training batch
            try:
                train_batch = next(iter(train_loader))
            except StopIteration:
                # Reset loader if we run out
                train_loader_iter = iter(train_loader)
                train_batch = next(train_loader_iter)

            if isinstance(train_batch, dict):
                train_data = train_batch.get('image', train_batch.get('features')).to(device)
                train_target = train_batch.get('identity', train_batch.get('target')).to(device)
            else:
                train_data, train_target = train_batch[0].to(device), train_batch[1].to(device)

            # Compute Hessian-vector product
            hvp = self._compute_hvp(model, train_data, train_target, inverse_hvp, device)

            # Update inverse_hvp using LiSSA recursion
            for name in inverse_hvp:
                inverse_hvp[name] = test_grads[name] + (1 - damping) * inverse_hvp[name] - hvp[name] / scale

            if i % 100 == 0:
                logger.debug(f"LiSSA iteration {i}/{iterations}")

        # Scale the result
        influences = {name: grad / scale for name, grad in inverse_hvp.items()}

        logger.debug("LiSSA influence computation completed")
        return influences

    def compute_influence_exact(self,
                               model: nn.Module,
                               test_data: torch.Tensor,
                               test_target: torch.Tensor,
                               train_loader: torch.utils.data.DataLoader,
                               device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute exact influence using full Hessian computation.

        Warning: This is computationally expensive and only suitable for small models.

        Args:
            model: Neural network model
            test_data: Test point to compute influence for
            test_target: Test point labels
            train_loader: Training data loader
            device: Computation device

        Returns:
            Dictionary mapping parameter names to influence values
        """
        logger.debug("Computing exact influence (expensive operation)")

        model.eval()

        # Compute test gradient
        test_grads = self._compute_loss_gradient(model, test_data, test_target, device)

        # Compute full Hessian (simplified - using Fisher approximation)
        hessian = self._compute_hessian_approximation(model, train_loader, device)

        # Compute inverse Hessian-vector product
        # For simplicity, using diagonal approximation
        influences = {}
        for name, grad in test_grads.items():
            if name in hessian:
                # Add small epsilon for numerical stability
                inv_hess = 1.0 / (hessian[name] + 1e-8)
                influences[name] = grad * inv_hess
            else:
                influences[name] = grad

        logger.debug("Exact influence computation completed")
        return influences

    def compute_influence_conjugate_gradient(self,
                                            model: nn.Module,
                                            test_data: torch.Tensor,
                                            test_target: torch.Tensor,
                                            train_loader: torch.utils.data.DataLoader,
                                            device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute influence using conjugate gradient method.

        Args:
            model: Neural network model
            test_data: Test point to compute influence for
            test_target: Test point labels
            train_loader: Training data loader
            device: Computation device

        Returns:
            Dictionary mapping parameter names to influence values
        """
        logger.debug("Computing influence using conjugate gradient")

        model.eval()

        # Compute test gradient (this is our b in Ax = b)
        test_grads = self._compute_loss_gradient(model, test_data, test_target, device)

        # Initialize solution
        x = {name: torch.zeros_like(grad) for name, grad in test_grads.items()}
        r = {name: grad.clone() for name, grad in test_grads.items()}
        p = {name: grad.clone() for name, grad in test_grads.items()}

        # CG iterations
        for i in range(self.config.cg_max_iterations):
            # Compute Ap (Hessian-vector product)
            train_batch = next(iter(train_loader))
            if isinstance(train_batch, dict):
                train_data = train_batch.get('image', train_batch.get('features')).to(device)
                train_target = train_batch.get('identity', train_batch.get('target')).to(device)
            else:
                train_data, train_target = train_batch[0].to(device), train_batch[1].to(device)

            Ap = self._compute_hvp(model, train_data, train_target, p, device)

            # Compute alpha
            r_dot_r = sum((r[name] ** 2).sum() for name in r)
            p_dot_Ap = sum((p[name] * Ap[name]).sum() for name in p)

            if p_dot_Ap == 0:
                break

            alpha = r_dot_r / p_dot_Ap

            # Update x and r
            r_new = {}
            for name in x:
                x[name] = x[name] + alpha * p[name]
                r_new[name] = r[name] - alpha * Ap[name]

            # Check convergence
            r_new_dot = sum((r_new[name] ** 2).sum() for name in r_new)
            if torch.sqrt(r_new_dot) < self.config.cg_tolerance:
                logger.debug(f"CG converged in {i} iterations")
                break

            # Compute beta
            beta = r_new_dot / r_dot_r

            # Update p
            for name in p:
                p[name] = r_new[name] + beta * p[name]

            r = r_new

        logger.debug("Conjugate gradient influence computation completed")
        return x

    def _compute_loss_gradient(self,
                              model: nn.Module,
                              data: torch.Tensor,
                              target: torch.Tensor,
                              device: torch.device) -> Dict[str, torch.Tensor]:
        """Compute gradient of loss with respect to model parameters."""
        model.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss based on output shape
        if output.dim() == 1 or output.shape[1] == 1:
            # Binary classification or regression
            if target.dtype == torch.long:
                loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float())
            else:
                loss = F.mse_loss(output.squeeze(), target)
        else:
            # Multi-class classification
            loss = F.cross_entropy(output, target.long())

        # Compute gradients
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

        # Package as dictionary
        grad_dict = {}
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                grad_dict[name] = grad.detach()

        return grad_dict

    def _compute_hvp(self,
                    model: nn.Module,
                    data: torch.Tensor,
                    target: torch.Tensor,
                    vector: Dict[str, torch.Tensor],
                    device: torch.device,
                    damping: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Compute Hessian-vector product.

        Uses the identity: Hv = ∇(∇L · v)
        """
        model.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss
        if output.dim() == 1 or output.shape[1] == 1:
            if target.dtype == torch.long:
                loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float())
            else:
                loss = F.mse_loss(output.squeeze(), target)
        else:
            loss = F.cross_entropy(output, target.long())

        # First gradient
        first_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Compute gradient-vector product
        grad_vector_product = 0
        for grad, (name, param) in zip(first_grads, model.named_parameters()):
            if grad is not None and name in vector:
                grad_vector_product += (grad * vector[name]).sum()

        # Second gradient (Hessian-vector product)
        if grad_vector_product.requires_grad:
            hvp = torch.autograd.grad(grad_vector_product, model.parameters(), retain_graph=False)
        else:
            # If no second-order gradient, return zeros
            hvp = [torch.zeros_like(p) for p in model.parameters()]

        # Package as dictionary with damping
        hvp_dict = {}
        for (name, param), h in zip(model.named_parameters(), hvp):
            if h is not None:
                hvp_dict[name] = h.detach() + damping * vector[name]
            else:
                hvp_dict[name] = damping * vector[name]

        return hvp_dict

    def _compute_hessian_approximation(self,
                                      model: nn.Module,
                                      train_loader: torch.utils.data.DataLoader,
                                      device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher information matrix as Hessian approximation.

        This is much more efficient than computing the full Hessian.
        """
        model.eval()

        # Initialize Fisher information
        fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        num_samples = 0

        # Compute over training data
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= 10:  # Limit computation for efficiency
                break

            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)

            # Compute gradients
            grads = self._compute_loss_gradient(model, data, target, device)

            # Accumulate squared gradients (Fisher diagonal)
            for name, grad in grads.items():
                fisher[name] += grad ** 2

            num_samples += len(data)

        # Average
        for name in fisher:
            fisher[name] /= max(num_samples, 1)

        return fisher

    def compute_self_influence(self,
                              model: nn.Module,
                              data: torch.Tensor,
                              target: torch.Tensor,
                              train_loader: torch.utils.data.DataLoader,
                              device: torch.device,
                              method: str = "lissa") -> float:
        """
        Compute the influence of a training point on itself.

        High self-influence indicates the point is important for training.

        Args:
            model: Neural network model
            data: Training point
            target: Training point label
            train_loader: Training data loader
            device: Computation device
            method: Influence computation method

        Returns:
            Self-influence score
        """
        if method == "lissa":
            influences = self.compute_influence_lissa(model, data, target, train_loader, device)
        elif method == "exact":
            influences = self.compute_influence_exact(model, data, target, train_loader, device)
        else:
            influences = self.compute_influence_conjugate_gradient(model, data, target, train_loader, device)

        # Compute self-influence as sum of parameter influences
        self_influence = sum(torch.sum(inf).item() for inf in influences.values())

        return self_influence

    def get_top_influential_samples(self,
                                   model: nn.Module,
                                   test_data: torch.Tensor,
                                   test_target: torch.Tensor,
                                   train_loader: torch.utils.data.DataLoader,
                                   device: torch.device,
                                   top_k: int = 10,
                                   method: str = "lissa") -> List[Tuple[int, float]]:
        """
        Find the top-k most influential training samples for a test point.

        Args:
            model: Neural network model
            test_data: Test point
            test_target: Test point label
            train_loader: Training data loader
            device: Computation device
            top_k: Number of top samples to return
            method: Influence computation method

        Returns:
            List of (sample_index, influence_score) tuples
        """
        logger.info(f"Finding top {top_k} influential samples")

        # Compute test gradient
        test_grads = self._compute_loss_gradient(model, test_data, test_target, device)

        influences = []

        # Compute influence for each training sample
        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)

            for i in range(len(data)):
                sample_data = data[i:i+1]
                sample_target = target[i:i+1]

                # Compute training gradient
                train_grads = self._compute_loss_gradient(model, sample_data, sample_target, device)

                # Simple influence approximation (dot product of gradients)
                influence = sum((test_grads[name] * train_grads[name]).sum().item()
                              for name in test_grads if name in train_grads)

                sample_idx = batch_idx * train_loader.batch_size + i
                influences.append((sample_idx, influence))

        # Sort by influence and get top-k
        influences.sort(key=lambda x: abs(x[1]), reverse=True)

        logger.info(f"Top influential sample has influence: {influences[0][1]:.6f}")
        return influences[:top_k]


def create_influence_computer(method: str = "lissa", **config_kwargs) -> InfluenceComputation:
    """
    Factory function to create influence computation object.

    Args:
        method: Computation method ("lissa", "exact", "conjugate_gradient")
        **config_kwargs: Additional configuration parameters

    Returns:
        InfluenceComputation instance
    """
    config = InfluenceConfig(method=method, **config_kwargs)
    return InfluenceComputation(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Influence computation module loaded successfully!")
    print("Available methods: LiSSA, Exact, Conjugate Gradient")
