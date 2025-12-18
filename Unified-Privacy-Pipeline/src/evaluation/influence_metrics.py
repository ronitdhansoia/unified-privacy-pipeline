"""
Influence function evaluation metrics.

Implements metrics to evaluate the accuracy and consistency of influence
function computations for training data attribution.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InfluenceResult:
    """Results from influence function evaluation."""
    correlation_score: float
    ranking_accuracy: float
    consistency_score: float
    metadata: Dict[str, Any]


class InfluenceAccuracy:
    """
    Metrics for evaluating influence function accuracy.
    """
    
    @staticmethod
    def ground_truth_correlation(estimated_influences: np.ndarray,
                               ground_truth_influences: np.ndarray) -> float:
        """
        Compute correlation between estimated and ground truth influences.
        """
        correlation = np.corrcoef(estimated_influences, ground_truth_influences)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def leave_one_out_validation(model: nn.Module,
                               train_loader: torch.utils.data.DataLoader,
                               test_sample: torch.Tensor,
                               test_target: torch.Tensor,
                               influence_scores: np.ndarray,
                               device: torch.device = torch.device('cpu')) -> float:
        """
        Validate influence scores using leave-one-out retraining.
        """
        # This is computationally expensive and mainly for validation
        logger.info("Running leave-one-out validation (this may take time)...")
        
        # Get original loss
        model.eval()
        with torch.no_grad():
            original_output = model(test_sample.to(device))
            original_loss = nn.functional.cross_entropy(
                original_output, test_target.to(device)
            ).item()
        
        # Select top influential samples to validate
        top_k = min(5, len(influence_scores))  # Validate only top 5 for efficiency
        top_indices = np.argsort(np.abs(influence_scores))[-top_k:]
        
        actual_influences = []
        
        for idx in top_indices:
            # Create dataset without this sample
            remaining_data = []
            remaining_targets = []
            
            for i, (data, target) in enumerate(train_loader):
                if i != idx // train_loader.batch_size:  # Simplified logic
                    remaining_data.append(data)
                    remaining_targets.append(target)
            
            if not remaining_data:
                continue
                
            # Retrain without this sample (simplified - just one epoch)
            temp_model = type(model)(model.privacy_config, **model.__dict__.get('kwargs', {}))
            temp_model.load_state_dict(model.state_dict())
            
            # Simple retraining step
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.001)
            temp_model.train()
            
            for data, target in zip(remaining_data[:1], remaining_targets[:1]):  # Just one batch
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = temp_model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            
            # Compute new loss
            temp_model.eval()
            with torch.no_grad():
                new_output = temp_model(test_sample.to(device))
                new_loss = nn.functional.cross_entropy(
                    new_output, test_target.to(device)
                ).item()
            
            actual_influence = original_loss - new_loss
            actual_influences.append(actual_influence)
        
        if len(actual_influences) == 0:
            return 0.0
        
        # Compare with estimated influences
        estimated_subset = influence_scores[top_indices[-len(actual_influences):]]
        correlation = np.corrcoef(estimated_subset, actual_influences)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def ranking_precision_at_k(estimated_influences: np.ndarray,
                             ground_truth_influences: np.ndarray,
                             k: int = 10) -> float:
        """
        Compute precision@k for influence ranking.
        """
        # Get top-k indices for both estimated and ground truth
        est_top_k = set(np.argsort(np.abs(estimated_influences))[-k:])
        gt_top_k = set(np.argsort(np.abs(ground_truth_influences))[-k:])
        
        # Compute precision
        intersection = len(est_top_k.intersection(gt_top_k))
        precision = intersection / k
        
        return precision


class InfluenceConsistency:
    """
    Metrics for evaluating influence function consistency across different scenarios.
    """
    
    @staticmethod
    def multi_run_consistency(influence_scores_list: List[np.ndarray]) -> float:
        """
        Compute consistency across multiple runs of influence computation.
        """
        if len(influence_scores_list) < 2:
            return 1.0
        
        correlations = []
        for i in range(len(influence_scores_list)):
            for j in range(i + 1, len(influence_scores_list)):
                corr = np.corrcoef(influence_scores_list[i], influence_scores_list[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    @staticmethod
    def cross_sample_consistency(model: nn.Module,
                               train_loader: torch.utils.data.DataLoader,
                               test_samples: List[torch.Tensor],
                               device: torch.device = torch.device('cpu')) -> float:
        """
        Compute consistency of influence rankings across different test samples.
        """
        if len(test_samples) < 2:
            return 1.0
        
        # Compute influence scores for each test sample
        all_influences = []
        
        for test_sample in test_samples:
            # Simplified influence computation (placeholder)
            influences = InfluenceConsistency._compute_simple_influences(
                model, train_loader, test_sample, device
            )
            all_influences.append(influences)
        
        # Compute pairwise ranking correlations
        ranking_correlations = []
        for i in range(len(all_influences)):
            for j in range(i + 1, len(all_influences)):
                # Compute Spearman rank correlation
                rank_i = np.argsort(np.argsort(all_influences[i]))
                rank_j = np.argsort(np.argsort(all_influences[j]))
                corr = np.corrcoef(rank_i, rank_j)[0, 1]
                if not np.isnan(corr):
                    ranking_correlations.append(corr)
        
        return np.mean(ranking_correlations) if ranking_correlations else 0.0
    
    @staticmethod
    def _compute_simple_influences(model: nn.Module,
                                 train_loader: torch.utils.data.DataLoader,
                                 test_sample: torch.Tensor,
                                 device: torch.device) -> np.ndarray:
        """
        Compute simplified influence scores (for testing purposes).
        """
        model.eval()
        influences = []
        
        # Get test sample gradient
        test_sample = test_sample.to(device)
        test_output = model(test_sample.unsqueeze(0))
        test_loss = test_output.sum()  # Simplified loss
        
        test_grads = torch.autograd.grad(test_loss, model.parameters(), create_graph=False)
        test_grad_vec = torch.cat([g.flatten() for g in test_grads])
        
        # Compute dot product with each training sample gradient
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                for i in range(len(data)):
                    sample_output = model(data[i:i+1])
                    sample_loss = sample_output.sum()
                    
                    sample_grads = torch.autograd.grad(sample_loss, model.parameters(), create_graph=False)
                    sample_grad_vec = torch.cat([g.flatten() for g in sample_grads])
                    
                    # Influence approximation: dot product of gradients
                    influence = torch.dot(test_grad_vec, sample_grad_vec).item()
                    influences.append(influence)
        
        return np.array(influences)
    
    @staticmethod
    def parameter_stability(model: nn.Module,
                          influence_scores_before: np.ndarray,
                          influence_scores_after: np.ndarray) -> float:
        """
        Compute stability of influence scores before and after model updates.
        """
        correlation = np.corrcoef(influence_scores_before, influence_scores_after)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0