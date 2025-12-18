"""
Differential Privacy trainer using Opacus.

Implements privacy-preserving training with automatic privacy accounting,
gradient clipping, and noise injection for various model architectures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import wandb

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None

from ..evaluation.privacy_metrics import PrivacyAccountant

logger = logging.getLogger(__name__)


@dataclass
class DPTrainingConfig:
    """Configuration for differential privacy training."""
    
    # Privacy parameters
    target_epsilon: float = 1.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Training parameters
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    
    # Memory management
    max_physical_batch_size: Optional[int] = None
    
    # Logging and monitoring
    log_interval: int = 10
    evaluate_interval: int = 1
    save_interval: int = 5
    
    # Privacy accounting
    strict_privacy_accounting: bool = True
    composition_method: str = "rdp"  # "rdp" or "gaussian"
    
    # Model validation
    validate_model: bool = True
    fix_model: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    experiment_name: str = "dp_training"
    
    # Additional configuration
    secure_mode: bool = False
    grad_sample_mode: str = "hooks"  # "hooks" or "functorch"


class DPTrainer:
    """
    Trainer for differential privacy-enabled models using Opacus.
    
    Provides a high-level interface for DP training with automatic
    privacy accounting, memory management, and evaluation.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: DPTrainingConfig,
                 device: torch.device = torch.device('cpu')):
        
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for DP training. Install with: pip install opacus")
        
        self.config = config
        self.device = device
        self.model = model.to(device)
        
        # Privacy components
        self.privacy_engine = None
        self.privacy_accountant = PrivacyAccountant(config.target_epsilon, config.target_delta)
        
        # Training components
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        
        # Tracking
        self.training_history = []
        self.privacy_history = []
        self.current_epoch = 0
        
        # Initialize model validation
        if config.validate_model:
            self._validate_and_fix_model()
        
        # Initialize experiment tracking
        if config.use_wandb:
            self._init_wandb()
    
    def _validate_and_fix_model(self):
        """Validate and fix model for differential privacy."""
        logger.info("Validating model for DP compatibility...")
        
        # Check if model is compatible with Opacus
        errors = ModuleValidator.validate(self.model, strict=False)
        
        if errors:
            logger.warning(f"Model validation found {len(errors)} issues:")
            for error in errors:
                logger.warning(f"  - {error}")
            
            if self.config.fix_model:
                logger.info("Attempting to fix model automatically...")
                self.model = ModuleValidator.fix(self.model)
                
                # Re-validate after fixing
                remaining_errors = ModuleValidator.validate(self.model, strict=False)
                if remaining_errors:
                    logger.error(f"Could not fix {len(remaining_errors)} model issues:")
                    for error in remaining_errors:
                        logger.error(f"  - {error}")
                    raise ValueError("Model cannot be made DP-compatible automatically")
                else:
                    logger.info("Model successfully fixed for DP training")
            else:
                raise ValueError("Model is not compatible with DP training. Set fix_model=True to attempt automatic fixes.")
        else:
            logger.info("Model is compatible with DP training")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project="unified-privacy-pipeline",
            name=self.config.experiment_name,
            config={
                "target_epsilon": self.config.target_epsilon,
                "target_delta": self.config.target_delta,
                "max_grad_norm": self.config.max_grad_norm,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            }
        )
    
    def setup_training(self,
                      optimizer: torch.optim.Optimizer,
                      criterion: nn.Module,
                      train_loader: DataLoader,
                      test_loader: Optional[DataLoader] = None):
        """
        Setup training components with differential privacy.
        
        Args:
            optimizer: PyTorch optimizer
            criterion: Loss function
            train_loader: Training data loader
            test_loader: Optional test data loader
        """
        self.optimizer = optimizer
        self.criterion = criterion
        self.test_loader = test_loader
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine(secure_mode=self.config.secure_mode)
        
        # Make model, optimizer, and dataloader private
        logger.info("Setting up differential privacy...")
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            epochs=self.config.epochs,
            target_epsilon=self.config.target_epsilon,
            target_delta=self.config.target_delta,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        # Set up memory management if needed
        if self.config.max_physical_batch_size is not None:
            logger.info(f"Using batch memory manager with physical batch size: {self.config.max_physical_batch_size}")
            self.train_loader = BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=self.config.max_physical_batch_size,
                optimizer=self.optimizer
            )
        
        logger.info(f"DP training setup complete. Target (ε,δ): ({self.config.target_epsilon}, {self.config.target_delta})")
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with differential privacy.
        
        Returns:
            Dictionary with training metrics for the epoch
        """
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.config.target_delta)
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.6f}, ε: {epsilon:.2f}"
                )
                
                if self.config.use_wandb:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epsilon": epsilon,
                        "batch": batch_idx + self.current_epoch * len(self.train_loader)
                    })
        
        # Epoch metrics
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time
        current_epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.config.target_delta)
        
        epoch_metrics = {
            "epoch": self.current_epoch,
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "epsilon": current_epsilon,
            "delta": self.config.target_delta
        }
        
        # Update privacy accounting
        self.privacy_accountant.add_privacy_cost(
            epsilon=current_epsilon,
            delta=self.config.target_delta,
            operation=f"epoch_{self.current_epoch}"
        )
        
        self.training_history.append(epoch_metrics)
        
        logger.info(
            f"Epoch {self.current_epoch} complete - "
            f"Avg Loss: {avg_loss:.6f}, ε: {current_epsilon:.2f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        return epoch_metrics
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            data_loader: Optional data loader (uses self.test_loader if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if data_loader is None:
            data_loader = self.test_loader
            
        if data_loader is None:
            logger.warning("No test data loader available for evaluation")
            return {}
        
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Compute loss
                test_loss += self.criterion(output, target).item()
                
                # Compute accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        accuracy = correct / total
        avg_test_loss = test_loss / len(data_loader)
        
        evaluation_metrics = {
            "test_loss": avg_test_loss,
            "test_accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        logger.info(f"Evaluation - Loss: {avg_test_loss:.6f}, Accuracy: {accuracy:.4f}")
        return evaluation_metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Complete training loop with differential privacy.
        
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting DP training for {self.config.epochs} epochs...")
        
        training_results = {
            "config": self.config,
            "training_history": [],
            "evaluation_history": [],
            "privacy_spent": [],
            "final_epsilon": 0.0,
            "budget_exceeded": False
        }
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Evaluate if needed
            if epoch % self.config.evaluate_interval == 0:
                eval_metrics = self.evaluate()
                epoch_metrics.update(eval_metrics)
                training_results["evaluation_history"].append(eval_metrics)
            
            training_results["training_history"].append(epoch_metrics)
            
            # Log to wandb if enabled
            if self.config.use_wandb:
                wandb.log(epoch_metrics)
            
            # Check privacy budget
            current_epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.config.target_delta)
            if current_epsilon > self.config.target_epsilon:
                logger.warning(f"Privacy budget exceeded! Current ε: {current_epsilon:.2f}, Target: {self.config.target_epsilon}")
                training_results["budget_exceeded"] = True
                if self.config.strict_privacy_accounting:
                    logger.error("Stopping training due to privacy budget violation")
                    break
            
            # Save checkpoint if needed
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        # Final privacy accounting
        final_epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.config.target_delta)
        training_results["final_epsilon"] = final_epsilon
        
        privacy_report = self.privacy_accountant.get_privacy_report()
        training_results["privacy_report"] = privacy_report
        
        logger.info(f"DP training complete. Final ε: {final_epsilon:.2f}")
        
        if self.config.use_wandb:
            wandb.finish()
        
        return training_results
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy expenditure."""
        if self.privacy_engine is not None:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.config.target_delta)
            return epsilon, self.config.target_delta
        return 0.0, 0.0
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_history": self.training_history,
            "privacy_spent": self.get_privacy_spent(),
            "privacy_accountant": self.privacy_accountant.get_privacy_report()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.training_history = checkpoint["training_history"]
        
        logger.info(f"Checkpoint loaded from {filepath}, resuming from epoch {self.current_epoch}")
    
    def get_model_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report for the model."""
        epsilon, delta = self.get_privacy_spent()
        
        report = {
            "privacy_parameters": {
                "target_epsilon": self.config.target_epsilon,
                "target_delta": self.config.target_delta,
                "spent_epsilon": epsilon,
                "spent_delta": delta,
                "max_grad_norm": self.config.max_grad_norm
            },
            "training_info": {
                "epochs_completed": self.current_epoch,
                "total_epochs": self.config.epochs,
                "budget_exceeded": epsilon > self.config.target_epsilon
            },
            "privacy_accountant_report": self.privacy_accountant.get_privacy_report(),
            "model_info": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        return report