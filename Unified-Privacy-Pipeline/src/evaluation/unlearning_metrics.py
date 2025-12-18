"""
Machine unlearning evaluation metrics.

Implements metrics to evaluate the effectiveness of unlearning operations:
- Forget quality (how well forgotten samples are removed)
- Retention quality (how well retained samples are preserved)
- Unlearning verification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UnlearningResult:
    """Results from unlearning evaluation."""
    forget_quality: float
    retention_quality: float
    model_utility: float
    verification_passed: bool
    metadata: Dict[str, Any]


class UnlearningEvaluator:
    """
    Comprehensive evaluator for machine unlearning effectiveness.
    """
    
    def __init__(self):
        self.baseline_performance = None
        self.evaluation_history = []
    
    def set_baseline(self, model: nn.Module, 
                    retain_loader: torch.utils.data.DataLoader,
                    forget_loader: torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader,
                    device: torch.device = torch.device('cpu')):
        """Set baseline performance before unlearning."""
        self.baseline_performance = {
            'retain_accuracy': self._compute_accuracy(model, retain_loader, device),
            'forget_accuracy': self._compute_accuracy(model, forget_loader, device),
            'test_accuracy': self._compute_accuracy(model, test_loader, device)
        }
        logger.info(f"Baseline set - Retain: {self.baseline_performance['retain_accuracy']:.4f}, "
                   f"Forget: {self.baseline_performance['forget_accuracy']:.4f}, "
                   f"Test: {self.baseline_performance['test_accuracy']:.4f}")
    
    def evaluate_unlearning(self, model: nn.Module,
                           retain_loader: torch.utils.data.DataLoader,
                           forget_loader: torch.utils.data.DataLoader, 
                           test_loader: torch.utils.data.DataLoader,
                           device: torch.device = torch.device('cpu')) -> UnlearningResult:
        """
        Evaluate unlearning effectiveness.
        
        Args:
            model: Model after unlearning
            retain_loader: Data that should be retained
            forget_loader: Data that should be forgotten
            test_loader: General test data
            device: Device for computation
            
        Returns:
            UnlearningResult with evaluation metrics
        """
        if self.baseline_performance is None:
            raise ValueError("Baseline not set. Call set_baseline first.")
        
        # Compute current performance
        current_performance = {
            'retain_accuracy': self._compute_accuracy(model, retain_loader, device),
            'forget_accuracy': self._compute_accuracy(model, forget_loader, device),
            'test_accuracy': self._compute_accuracy(model, test_loader, device)
        }
        
        # Compute forget quality (lower is better for forgotten data)
        forget_quality = 1.0 - (current_performance['forget_accuracy'] / 
                               self.baseline_performance['forget_accuracy'])
        
        # Compute retention quality (should remain high)
        retention_quality = (current_performance['retain_accuracy'] / 
                           self.baseline_performance['retain_accuracy'])
        
        # Overall model utility
        model_utility = (current_performance['test_accuracy'] / 
                        self.baseline_performance['test_accuracy'])
        
        # Verification - check if unlearning criteria are met
        verification_passed = (
            forget_quality > 0.1 and  # At least 10% reduction in forget accuracy
            retention_quality > 0.9 and  # At most 10% reduction in retain accuracy
            model_utility > 0.8  # At most 20% reduction in overall utility
        )
        
        result = UnlearningResult(
            forget_quality=forget_quality,
            retention_quality=retention_quality,
            model_utility=model_utility,
            verification_passed=verification_passed,
            metadata={
                'baseline_performance': self.baseline_performance,
                'current_performance': current_performance,
                'num_retain_samples': len(retain_loader.dataset),
                'num_forget_samples': len(forget_loader.dataset),
                'num_test_samples': len(test_loader.dataset)
            }
        )
        
        self.evaluation_history.append(result)
        
        logger.info(f"Unlearning evaluation - Forget: {forget_quality:.4f}, "
                   f"Retention: {retention_quality:.4f}, Utility: {model_utility:.4f}, "
                   f"Verified: {verification_passed}")
        
        return result
    
    def _compute_accuracy(self, model: nn.Module, 
                         data_loader: torch.utils.data.DataLoader,
                         device: torch.device) -> float:
        """Compute accuracy on given data loader."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        return correct / total if total > 0 else 0.0


class ForgetQuality:
    """
    Metrics for evaluating forget quality (how well data is forgotten).
    """
    
    @staticmethod
    def membership_inference_score(model: nn.Module,
                                 forget_data: torch.utils.data.DataLoader,
                                 retain_data: torch.utils.data.DataLoader,
                                 device: torch.device = torch.device('cpu')) -> float:
        """
        Compute membership inference score for forgotten data.
        Lower scores indicate better forgetting.
        """
        model.eval()
        
        forget_confidences = []
        retain_confidences = []
        
        # Get confidence scores for forget data
        with torch.no_grad():
            for data, _ in forget_data:
                data = data.to(device)
                output = model(data)
                confidence = torch.softmax(output, dim=1).max(dim=1)[0]
                forget_confidences.extend(confidence.cpu().numpy())
        
        # Get confidence scores for retain data
        with torch.no_grad():
            for data, _ in retain_data:
                data = data.to(device)
                output = model(data)
                confidence = torch.softmax(output, dim=1).max(dim=1)[0]
                retain_confidences.extend(confidence.cpu().numpy())
        
        # Lower average confidence on forget data indicates better forgetting
        avg_forget_confidence = np.mean(forget_confidences)
        avg_retain_confidence = np.mean(retain_confidences)
        
        # Forget quality: larger gap indicates better forgetting
        forget_quality = max(0, avg_retain_confidence - avg_forget_confidence)
        
        return forget_quality
    
    @staticmethod
    def activation_distance(model: nn.Module,
                          forget_data: torch.utils.data.DataLoader,
                          original_activations: Dict[str, torch.Tensor],
                          device: torch.device = torch.device('cpu')) -> float:
        """
        Compute distance between current and original activations for forgotten data.
        """
        model.eval()
        
        # Hook to capture activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Compute current activations
        current_activations = {}
        with torch.no_grad():
            for data, _ in forget_data:
                data = data.to(device)
                model(data)
                
                # Collect activations
                for name, activation in activations.items():
                    if name not in current_activations:
                        current_activations[name] = []
                    current_activations[name].append(activation.cpu())
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute distances
        total_distance = 0.0
        num_layers = 0
        
        for name in current_activations:
            if name in original_activations:
                current = torch.cat(current_activations[name], dim=0)
                original = original_activations[name]
                
                # Compute normalized distance
                distance = torch.norm(current - original) / torch.norm(original)
                total_distance += distance.item()
                num_layers += 1
        
        return total_distance / num_layers if num_layers > 0 else 0.0


class RetentionQuality:
    """
    Metrics for evaluating retention quality (how well important data is preserved).
    """
    
    @staticmethod
    def accuracy_preservation(model: nn.Module,
                            retain_data: torch.utils.data.DataLoader,
                            baseline_accuracy: float,
                            device: torch.device = torch.device('cpu')) -> float:
        """
        Compute how well accuracy is preserved on retained data.
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in retain_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        current_accuracy = correct / total if total > 0 else 0.0
        preservation = current_accuracy / baseline_accuracy if baseline_accuracy > 0 else 1.0
        
        return preservation
    
    @staticmethod
    def feature_similarity(model: nn.Module,
                         retain_data: torch.utils.data.DataLoader,
                         original_features: torch.Tensor,
                         device: torch.device = torch.device('cpu')) -> float:
        """
        Compute similarity between current and original feature representations.
        """
        model.eval()
        current_features = []
        
        with torch.no_grad():
            for data, _ in retain_data:
                data = data.to(device)
                if hasattr(model, 'get_features'):
                    features = model.get_features(data)
                else:
                    features = model(data)  # Use output as features
                current_features.append(features.cpu())
        
        if not current_features:
            return 0.0
        
        current_features = torch.cat(current_features, dim=0)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            current_features.flatten(), 
            original_features.flatten(), 
            dim=0
        )
        
        return cos_sim.item()