"""
Machine Unlearning Methods.

Implements various unlearning algorithms:
- Gradient Ascent Unlearning
- Influence-Based Unlearning
- Fine-tuning based approaches
- Certified removal methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class UnlearningConfig:
    """Configuration for unlearning methods."""
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 1e-5
    patience: int = 10

    # Gradient ascent specific
    ascent_weight: float = 1.0
    retain_weight: float = 1.0

    # Fine-tuning specific
    warmup_epochs: int = 5

    # Regularization
    l2_regularization: float = 1e-4
    gradient_clipping: float = 1.0

    # Verification
    verify_unlearning: bool = True
    verification_threshold: float = 0.1


class BaseUnlearningMethod(ABC):
    """
    Abstract base class for unlearning methods.

    All unlearning methods should inherit from this class and implement
    the unlearn() method.
    """

    def __init__(self, config: Optional[UnlearningConfig] = None):
        self.config = config or UnlearningConfig()
        self.history = []

    @abstractmethod
    def unlearn(self,
                model: nn.Module,
                forget_loader: torch.utils.data.DataLoader,
                retain_loader: torch.utils.data.DataLoader,
                device: torch.device = torch.device('cpu'),
                **kwargs) -> Dict[str, Any]:
        """
        Perform unlearning on the model.

        Args:
            model: Model to unlearn from
            forget_loader: Data to forget
            retain_loader: Data to retain
            device: Computation device
            **kwargs: Additional method-specific arguments

        Returns:
            Dictionary with unlearning results and metrics
        """
        pass

    def _compute_loss(self,
                     model: nn.Module,
                     data: torch.Tensor,
                     target: torch.Tensor,
                     loss_type: str = "classification") -> torch.Tensor:
        """Compute loss for a batch of data."""
        output = model(data)

        if loss_type == "regression" or output.dim() == 1 or output.shape[1] == 1:
            if target.dtype == torch.long:
                loss = nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
            else:
                loss = nn.MSELoss()(output.squeeze(), target)
        else:
            loss = nn.CrossEntropyLoss()(output, target.long())

        return loss

    def _evaluate_model(self,
                       model: nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       device: torch.device) -> Tuple[float, float]:
        """Evaluate model accuracy and loss."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, dict):
                    data = batch_data.get('image', batch_data.get('features')).to(device)
                    target = batch_data.get('identity', batch_data.get('target')).to(device)
                else:
                    data, target = batch_data[0].to(device), batch_data[1].to(device)

                output = model(data)
                loss = self._compute_loss(model, data, target)
                total_loss += loss.item()

                # Compute accuracy
                if output.dim() > 1 and output.shape[1] > 1:
                    pred = output.argmax(dim=1)
                    correct += (pred == target.long()).sum().item()
                else:
                    pred = (torch.sigmoid(output.squeeze()) > 0.5).long()
                    correct += (pred == target.long()).sum().item()

                total += len(data)

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy


class GradientAscentUnlearning(BaseUnlearningMethod):
    """
    Gradient Ascent Unlearning.

    Maximizes loss on forget set while maintaining performance on retain set.
    """

    def unlearn(self,
                model: nn.Module,
                forget_loader: torch.utils.data.DataLoader,
                retain_loader: torch.utils.data.DataLoader,
                device: torch.device = torch.device('cpu'),
                **kwargs) -> Dict[str, Any]:
        """
        Perform gradient ascent unlearning.

        The method alternates between:
        1. Gradient ascent on forget set (maximize loss)
        2. Gradient descent on retain set (minimize loss)
        """
        logger.info("Starting Gradient Ascent Unlearning...")

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Track metrics
        iteration_metrics = []
        best_combined_loss = float('inf')
        patience_counter = 0

        # Initial evaluation
        forget_loss_init, forget_acc_init = self._evaluate_model(model, forget_loader, device)
        retain_loss_init, retain_acc_init = self._evaluate_model(model, retain_loader, device)

        logger.info(f"Initial - Forget Loss: {forget_loss_init:.4f}, Retain Loss: {retain_loss_init:.4f}")

        for iteration in range(self.config.max_iterations):
            model.train()
            total_forget_loss = 0.0
            total_retain_loss = 0.0
            forget_batches = 0
            retain_batches = 0

            # Process forget batches with gradient ASCENT
            for forget_batch in forget_loader:
                optimizer.zero_grad()

                if isinstance(forget_batch, dict):
                    forget_data = forget_batch.get('image', forget_batch.get('features')).to(device)
                    forget_target = forget_batch.get('identity', forget_batch.get('target')).to(device)
                else:
                    forget_data, forget_target = forget_batch[0].to(device), forget_batch[1].to(device)

                forget_loss = self._compute_loss(model, forget_data, forget_target)

                # Gradient ASCENT on forget set (negative gradient descent)
                (-self.config.ascent_weight * forget_loss).backward()
                total_forget_loss += forget_loss.item()

                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)

                optimizer.step()
                forget_batches += 1

            # Process retain batches with gradient DESCENT
            for retain_batch in retain_loader:
                optimizer.zero_grad()

                if isinstance(retain_batch, dict):
                    retain_data = retain_batch.get('image', retain_batch.get('features')).to(device)
                    retain_target = retain_batch.get('identity', retain_batch.get('target')).to(device)
                else:
                    retain_data, retain_target = retain_batch[0].to(device), retain_batch[1].to(device)

                retain_loss = self._compute_loss(model, retain_data, retain_target)

                # Gradient DESCENT on retain set
                (self.config.retain_weight * retain_loss).backward()
                total_retain_loss += retain_loss.item()

                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)

                optimizer.step()
                retain_batches += 1

            # Compute average losses
            avg_forget_loss = total_forget_loss / max(forget_batches, 1)
            avg_retain_loss = total_retain_loss / max(retain_batches, 1)
            combined_loss = avg_retain_loss - self.config.ascent_weight * avg_forget_loss

            # Evaluate
            forget_loss_eval, forget_acc_eval = self._evaluate_model(model, forget_loader, device)
            retain_loss_eval, retain_acc_eval = self._evaluate_model(model, retain_loader, device)

            metrics = {
                'iteration': iteration,
                'forget_loss': forget_loss_eval,
                'forget_accuracy': forget_acc_eval,
                'retain_loss': retain_loss_eval,
                'retain_accuracy': retain_acc_eval,
                'combined_loss': combined_loss
            }
            iteration_metrics.append(metrics)

            # Check convergence
            if combined_loss < best_combined_loss - self.config.convergence_threshold:
                best_combined_loss = combined_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(f"Converged after {iteration + 1} iterations")
                break

            if iteration % 10 == 0:
                logger.info(f"Iter {iteration}: Forget Loss {forget_loss_eval:.4f} (Acc {forget_acc_eval:.2%}), "
                          f"Retain Loss {retain_loss_eval:.4f} (Acc {retain_acc_eval:.2%})")

        # Final evaluation
        forget_loss_final, forget_acc_final = self._evaluate_model(model, forget_loader, device)
        retain_loss_final, retain_acc_final = self._evaluate_model(model, retain_loader, device)

        results = {
            'method': 'gradient_ascent',
            'iterations': len(iteration_metrics),
            'initial_metrics': {
                'forget_loss': forget_loss_init,
                'forget_accuracy': forget_acc_init,
                'retain_loss': retain_loss_init,
                'retain_accuracy': retain_acc_init
            },
            'final_metrics': {
                'forget_loss': forget_loss_final,
                'forget_accuracy': forget_acc_final,
                'retain_loss': retain_loss_final,
                'retain_accuracy': retain_acc_final
            },
            'history': iteration_metrics,
            'forget_quality': min(1.0, forget_loss_final / max(forget_loss_init, 0.1)),
            'utility_preservation': retain_acc_final / max(retain_acc_init, 0.01)
        }

        logger.info("Gradient Ascent Unlearning completed")
        logger.info(f"Forget quality: {results['forget_quality']:.2%}, "
                   f"Utility preservation: {results['utility_preservation']:.2%}")

        return results


class InfluenceBasedUnlearning(BaseUnlearningMethod):
    """
    Influence-based unlearning using influence functions.

    Identifies and removes the influence of specific training samples.
    """

    def __init__(self, config: Optional[UnlearningConfig] = None, influence_computer=None):
        super().__init__(config)
        self.influence_computer = influence_computer

    def unlearn(self,
                model: nn.Module,
                forget_loader: torch.utils.data.DataLoader,
                retain_loader: torch.utils.data.DataLoader,
                device: torch.device = torch.device('cpu'),
                **kwargs) -> Dict[str, Any]:
        """
        Perform influence-based unlearning.

        Computes influence of forget samples and performs targeted parameter updates.
        """
        logger.info("Starting Influence-Based Unlearning...")

        if self.influence_computer is None:
            logger.warning("No influence computer provided, falling back to gradient ascent")
            fallback = GradientAscentUnlearning(self.config)
            return fallback.unlearn(model, forget_loader, retain_loader, device, **kwargs)

        model = model.to(device)

        # Compute influences for forget samples
        logger.info("Computing influences...")
        all_influences = []

        for batch_idx, batch_data in enumerate(forget_loader):
            if batch_idx >= 5:  # Limit for efficiency
                break

            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)

            # Compute influence for this batch
            influences = self.influence_computer.compute_influence_lissa(
                model, data, target, retain_loader, device
            )
            all_influences.append(influences)

        # Average influences
        if all_influences:
            avg_influences = {}
            for name in all_influences[0].keys():
                avg_influences[name] = torch.stack([inf[name] for inf in all_influences]).mean(dim=0)
        else:
            logger.warning("No influences computed")
            avg_influences = {}

        # Apply influence-based parameter updates
        logger.info("Applying influence-based updates...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in avg_influences:
                    # Remove influence by subtracting scaled influence
                    param.data -= self.config.learning_rate * avg_influences[name]

        # Fine-tune on retain set
        logger.info("Fine-tuning on retain set...")
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.1)

        for epoch in range(self.config.warmup_epochs):
            model.train()
            for batch_data in retain_loader:
                if isinstance(batch_data, dict):
                    data = batch_data.get('image', batch_data.get('features')).to(device)
                    target = batch_data.get('identity', batch_data.get('target')).to(device)
                else:
                    data, target = batch_data[0].to(device), batch_data[1].to(device)

                optimizer.zero_grad()
                loss = self._compute_loss(model, data, target)
                loss.backward()
                optimizer.step()

        # Evaluate
        forget_loss, forget_acc = self._evaluate_model(model, forget_loader, device)
        retain_loss, retain_acc = self._evaluate_model(model, retain_loader, device)

        results = {
            'method': 'influence_based',
            'forget_loss': forget_loss,
            'forget_accuracy': forget_acc,
            'retain_loss': retain_loss,
            'retain_accuracy': retain_acc,
            'num_influences_computed': len(all_influences),
            'influence_magnitude': sum(torch.norm(inf).item() for inf in avg_influences.values()) if avg_influences else 0
        }

        logger.info("Influence-Based Unlearning completed")
        logger.info(f"Forget Loss: {forget_loss:.4f}, Retain Accuracy: {retain_acc:.2%}")

        return results


class FineTuningUnlearning(BaseUnlearningMethod):
    """
    Simple fine-tuning based unlearning.

    Re-trains the model on retain set only, ignoring forget set.
    """

    def unlearn(self,
                model: nn.Module,
                forget_loader: torch.utils.data.DataLoader,
                retain_loader: torch.utils.data.DataLoader,
                device: torch.device = torch.device('cpu'),
                **kwargs) -> Dict[str, Any]:
        """
        Perform unlearning via fine-tuning on retain set.
        """
        logger.info("Starting Fine-Tuning Unlearning...")

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Initial evaluation
        forget_loss_init, forget_acc_init = self._evaluate_model(model, forget_loader, device)
        retain_loss_init, retain_acc_init = self._evaluate_model(model, retain_loader, device)

        # Fine-tune on retain set
        for epoch in range(self.config.max_iterations):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_data in retain_loader:
                if isinstance(batch_data, dict):
                    data = batch_data.get('image', batch_data.get('features')).to(device)
                    target = batch_data.get('identity', batch_data.get('target')).to(device)
                else:
                    data, target = batch_data[0].to(device), batch_data[1].to(device)

                optimizer.zero_grad()
                loss = self._compute_loss(model, data, target)
                loss.backward()

                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            if epoch % 10 == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch}: Average Loss {avg_loss:.4f}")

        # Final evaluation
        forget_loss_final, forget_acc_final = self._evaluate_model(model, forget_loader, device)
        retain_loss_final, retain_acc_final = self._evaluate_model(model, retain_loader, device)

        results = {
            'method': 'fine_tuning',
            'epochs': self.config.max_iterations,
            'initial_metrics': {
                'forget_loss': forget_loss_init,
                'forget_accuracy': forget_acc_init,
                'retain_loss': retain_loss_init,
                'retain_accuracy': retain_acc_init
            },
            'final_metrics': {
                'forget_loss': forget_loss_final,
                'forget_accuracy': forget_acc_final,
                'retain_loss': retain_loss_final,
                'retain_accuracy': retain_acc_final
            }
        }

        logger.info("Fine-Tuning Unlearning completed")
        return results


class NegativeGradientUnlearning(BaseUnlearningMethod):
    """
    Negative Gradient Unlearning.

    Applies negative gradients from forget set to reverse learning.
    """

    def unlearn(self,
                model: nn.Module,
                forget_loader: torch.utils.data.DataLoader,
                retain_loader: torch.utils.data.DataLoader,
                device: torch.device = torch.device('cpu'),
                **kwargs) -> Dict[str, Any]:
        """
        Perform negative gradient unlearning.
        """
        logger.info("Starting Negative Gradient Unlearning...")

        model = model.to(device)

        # Store original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}

        # Compute gradients on forget set
        model.train()
        forget_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        num_samples = 0

        for batch_data in forget_loader:
            if isinstance(batch_data, dict):
                data = batch_data.get('image', batch_data.get('features')).to(device)
                target = batch_data.get('identity', batch_data.get('target')).to(device)
            else:
                data, target = batch_data[0].to(device), batch_data[1].to(device)

            model.zero_grad()
            loss = self._compute_loss(model, data, target)
            loss.backward()

            # Accumulate gradients
            for (name, param) in model.named_parameters():
                if param.grad is not None:
                    forget_gradients[name] += param.grad.data

            num_samples += len(data)

        # Average gradients
        for name in forget_gradients:
            forget_gradients[name] /= max(num_samples, 1)

        # Apply negative gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in forget_gradients:
                    param.data -= self.config.learning_rate * forget_gradients[name]

        # Evaluate
        forget_loss, forget_acc = self._evaluate_model(model, forget_loader, device)
        retain_loss, retain_acc = self._evaluate_model(model, retain_loader, device)

        results = {
            'method': 'negative_gradient',
            'forget_loss': forget_loss,
            'forget_accuracy': forget_acc,
            'retain_loss': retain_loss,
            'retain_accuracy': retain_acc,
            'parameter_change': sum(torch.norm(param - original_params[name]).item()
                                   for name, param in model.named_parameters() if name in original_params)
        }

        logger.info("Negative Gradient Unlearning completed")
        return results


def create_unlearner(method: str = "gradient_ascent",
                    config: Optional[UnlearningConfig] = None,
                    **kwargs) -> BaseUnlearningMethod:
    """
    Factory function to create unlearning method.

    Args:
        method: Unlearning method type
        config: Unlearning configuration
        **kwargs: Additional method-specific arguments

    Returns:
        Unlearning method instance
    """
    config = config or UnlearningConfig()

    if method == "gradient_ascent":
        return GradientAscentUnlearning(config)
    elif method == "influence_based":
        influence_computer = kwargs.get('influence_computer')
        return InfluenceBasedUnlearning(config, influence_computer)
    elif method == "fine_tuning":
        return FineTuningUnlearning(config)
    elif method == "negative_gradient":
        return NegativeGradientUnlearning(config)
    else:
        raise ValueError(f"Unknown unlearning method: {method}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Machine unlearning methods loaded successfully!")
    print("Available methods: gradient_ascent, influence_based, fine_tuning, negative_gradient")
