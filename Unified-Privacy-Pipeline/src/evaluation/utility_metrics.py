"""
Utility metrics for evaluating model performance and fairness.

Implements various metrics to assess:
- Model accuracy and performance
- Fairness across different groups
- Utility preservation after privacy operations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UtilityResult:
    """Results from utility evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float]
    confusion_matrix: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class FairnessResult:
    """Results from fairness evaluation."""
    demographic_parity: Dict[str, float]
    equalized_odds: Dict[str, float]
    calibration_scores: Dict[str, float]
    overall_fairness: float
    metadata: Dict[str, Any]


class AccuracyMetrics:
    """
    Comprehensive accuracy and performance metrics.
    """
    
    def __init__(self, task_type: str = "binary_classification"):
        self.task_type = task_type
        
    def evaluate_model(self, model: nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device = torch.device('cpu')) -> UtilityResult:
        """
        Evaluate model performance on given dataset.
        
        Args:
            model: Model to evaluate
            data_loader: Data to evaluate on
            device: Device for computation
            
        Returns:
            UtilityResult with performance metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                
                if self.task_type == "binary_classification":
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                    if len(probabilities.shape) > 1:
                        probabilities = probabilities.squeeze()
                    predictions = (probabilities > 0.5).astype(int)
                elif self.task_type == "multiclass":
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    predictions = np.argmax(probabilities, axis=1)
                else:  # regression
                    probabilities = outputs.cpu().numpy()
                    predictions = probabilities
                
                # Ensure consistent shapes
                preds_flat = predictions.flatten() if predictions.ndim > 1 else predictions
                targets_flat = targets.cpu().numpy().flatten()
                probs_flat = probabilities.flatten() if probabilities.ndim > 1 else probabilities
                
                # Ensure same length (truncate if needed)
                min_len = min(len(preds_flat), len(targets_flat), len(probs_flat))
                all_predictions.extend(preds_flat[:min_len])
                all_targets.extend(targets_flat[:min_len])
                all_probabilities.extend(probs_flat[:min_len])
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        if self.task_type in ["binary_classification", "multiclass"]:
            accuracy = accuracy_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
            
            try:
                if self.task_type == "binary_classification" or len(np.unique(all_targets)) == 2:
                    auc = roc_auc_score(all_targets, all_probabilities)
                else:
                    auc = roc_auc_score(all_targets, all_probabilities, average='weighted', multi_class='ovr')
            except ValueError as e:
                logger.warning(f"Could not compute AUC: {e}")
                auc = None
                
            conf_matrix = confusion_matrix(all_targets, all_predictions)
        else:  # regression
            # For regression, compute R² and MAE
            from sklearn.metrics import r2_score, mean_absolute_error
            accuracy = r2_score(all_targets, all_predictions)
            precision = mean_absolute_error(all_targets, all_predictions)
            recall = 0.0  # Not applicable for regression
            f1 = 0.0      # Not applicable for regression
            auc = None
            conf_matrix = np.array([[0]])
        
        result = UtilityResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            confusion_matrix=conf_matrix,
            metadata={
                "task_type": self.task_type,
                "num_samples": len(all_targets),
                "num_classes": len(np.unique(all_targets)) if self.task_type != "regression" else 1
            }
        )
        
        logger.info(f"Accuracy evaluation complete - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return result
    
    def compare_models(self, baseline_result: UtilityResult,
                      privacy_result: UtilityResult) -> Dict[str, float]:
        """
        Compare utility between baseline and privacy-preserving models.
        
        Args:
            baseline_result: Results from baseline model
            privacy_result: Results from privacy-preserving model
            
        Returns:
            Dictionary with utility preservation metrics
        """
        utility_preservation = {
            "accuracy_preservation": privacy_result.accuracy / baseline_result.accuracy,
            "precision_preservation": privacy_result.precision / baseline_result.precision if baseline_result.precision > 0 else 1.0,
            "recall_preservation": privacy_result.recall / baseline_result.recall if baseline_result.recall > 0 else 1.0,
            "f1_preservation": privacy_result.f1_score / baseline_result.f1_score if baseline_result.f1_score > 0 else 1.0,
            "accuracy_drop": baseline_result.accuracy - privacy_result.accuracy,
            "precision_drop": baseline_result.precision - privacy_result.precision,
            "recall_drop": baseline_result.recall - privacy_result.recall,
            "f1_drop": baseline_result.f1_score - privacy_result.f1_score,
        }
        
        if baseline_result.auc_score is not None and privacy_result.auc_score is not None:
            utility_preservation.update({
                "auc_preservation": privacy_result.auc_score / baseline_result.auc_score,
                "auc_drop": baseline_result.auc_score - privacy_result.auc_score,
            })
        
        logger.info(f"Utility comparison - Accuracy preservation: {utility_preservation['accuracy_preservation']:.4f}")
        return utility_preservation


class FairnessMetrics:
    """
    Fairness evaluation metrics for different demographic groups.
    """
    
    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attributes = sensitive_attributes
        
    def evaluate_fairness(self, model: nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         sensitive_data: Dict[str, np.ndarray],
                         device: torch.device = torch.device('cpu')) -> FairnessResult:
        """
        Evaluate fairness metrics across sensitive attributes.
        
        Args:
            model: Model to evaluate
            data_loader: Data to evaluate on
            sensitive_data: Dictionary mapping attribute names to attribute values
            device: Device for computation
            
        Returns:
            FairnessResult with fairness metrics
        """
        # Get model predictions
        accuracy_evaluator = AccuracyMetrics()
        utility_result = accuracy_evaluator.evaluate_model(model, data_loader, device)
        
        # Get predictions and probabilities
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(device)
                outputs = model(data)
                
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_targets.extend(targets.numpy())
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Compute fairness metrics for each sensitive attribute
        demographic_parity = {}
        equalized_odds = {}
        calibration_scores = {}
        
        for attr_name in self.sensitive_attributes:
            if attr_name in sensitive_data:
                attr_values = sensitive_data[attr_name]
                
                # Demographic Parity (Statistical Parity)
                dp_scores = self._compute_demographic_parity(all_predictions, attr_values)
                demographic_parity[attr_name] = dp_scores
                
                # Equalized Odds
                eo_scores = self._compute_equalized_odds(all_predictions, all_targets, attr_values)
                equalized_odds[attr_name] = eo_scores
                
                # Calibration
                calib_scores = self._compute_calibration(all_probabilities, all_targets, attr_values)
                calibration_scores[attr_name] = calib_scores
        
        # Compute overall fairness score
        overall_fairness = self._compute_overall_fairness(
            demographic_parity, equalized_odds, calibration_scores
        )
        
        result = FairnessResult(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            calibration_scores=calibration_scores,
            overall_fairness=overall_fairness,
            metadata={
                "sensitive_attributes": self.sensitive_attributes,
                "num_samples": len(all_targets),
                "base_accuracy": utility_result.accuracy
            }
        )
        
        logger.info(f"Fairness evaluation complete - Overall fairness: {overall_fairness:.4f}")
        return result
    
    def _compute_demographic_parity(self, predictions: np.ndarray,
                                   sensitive_attr: np.ndarray) -> Dict[str, float]:
        """Compute demographic parity (statistical parity) scores."""
        unique_groups = np.unique(sensitive_attr)
        positive_rates = {}
        
        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            group_predictions = predictions[group_mask]
            positive_rate = np.mean(group_predictions)
            positive_rates[str(group)] = positive_rate
        
        # Compute parity difference (max - min positive rate)
        rates = list(positive_rates.values())
        parity_difference = max(rates) - min(rates)
        
        return {
            "positive_rates": positive_rates,
            "parity_difference": parity_difference,
            "demographic_parity_score": 1.0 - parity_difference  # Higher is better
        }
    
    def _compute_equalized_odds(self, predictions: np.ndarray,
                               targets: np.ndarray,
                               sensitive_attr: np.ndarray) -> Dict[str, float]:
        """Compute equalized odds scores."""
        unique_groups = np.unique(sensitive_attr)
        tpr_scores = {}  # True Positive Rate
        fpr_scores = {}  # False Positive Rate
        
        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            group_predictions = predictions[group_mask]
            group_targets = targets[group_mask]
            
            # True Positive Rate
            true_positives = np.sum((group_predictions == 1) & (group_targets == 1))
            positives = np.sum(group_targets == 1)
            tpr = true_positives / positives if positives > 0 else 0
            
            # False Positive Rate  
            false_positives = np.sum((group_predictions == 1) & (group_targets == 0))
            negatives = np.sum(group_targets == 0)
            fpr = false_positives / negatives if negatives > 0 else 0
            
            tpr_scores[str(group)] = tpr
            fpr_scores[str(group)] = fpr
        
        # Compute equalized odds difference
        tpr_values = list(tpr_scores.values())
        fpr_values = list(fpr_scores.values())
        
        tpr_difference = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_difference = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        # Overall equalized odds score
        eq_odds_score = 1.0 - max(tpr_difference, fpr_difference)
        
        return {
            "true_positive_rates": tpr_scores,
            "false_positive_rates": fpr_scores,
            "tpr_difference": tpr_difference,
            "fpr_difference": fpr_difference,
            "equalized_odds_score": eq_odds_score
        }
    
    def _compute_calibration(self, probabilities: np.ndarray,
                           targets: np.ndarray,
                           sensitive_attr: np.ndarray) -> Dict[str, float]:
        """Compute calibration scores across groups."""
        unique_groups = np.unique(sensitive_attr)
        calibration_errors = {}
        
        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            group_probs = probabilities[group_mask]
            group_targets = targets[group_mask]
            
            # Compute calibration error (Expected Calibration Error)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Get predictions in this bin
                in_bin = (group_probs > bin_lower) & (group_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = group_targets[in_bin].mean()
                    avg_confidence_in_bin = group_probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_errors[str(group)] = ece
        
        # Compute calibration difference across groups
        error_values = list(calibration_errors.values())
        calibration_difference = max(error_values) - min(error_values) if error_values else 0
        
        return {
            "calibration_errors": calibration_errors,
            "calibration_difference": calibration_difference,
            "calibration_score": 1.0 - calibration_difference
        }
    
    def _compute_overall_fairness(self, demographic_parity: Dict[str, Dict],
                                equalized_odds: Dict[str, Dict],
                                calibration_scores: Dict[str, Dict]) -> float:
        """Compute overall fairness score."""
        fairness_scores = []
        
        for attr in self.sensitive_attributes:
            if attr in demographic_parity:
                dp_score = demographic_parity[attr].get("demographic_parity_score", 0)
                eo_score = equalized_odds[attr].get("equalized_odds_score", 0)
                calib_score = calibration_scores[attr].get("calibration_score", 0)
                
                # Weighted average of fairness metrics
                attr_fairness = (dp_score + eo_score + calib_score) / 3
                fairness_scores.append(attr_fairness)
        
        return np.mean(fairness_scores) if fairness_scores else 0.0


class UtilityPreservation:
    """
    Utility preservation evaluation after privacy-preserving operations.
    """
    
    def __init__(self):
        self.baseline_metrics = {}
        self.privacy_metrics = {}
        
    def set_baseline(self, model: nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    device: torch.device = torch.device('cpu')):
        """Set baseline performance metrics."""
        evaluator = AccuracyMetrics()
        self.baseline_metrics = evaluator.evaluate_model(model, data_loader, device)
        logger.info("Baseline metrics set")
        
    def evaluate_preservation(self, model: nn.Module,
                            data_loader: torch.utils.data.DataLoader,
                            device: torch.device = torch.device('cpu')) -> Dict[str, float]:
        """Evaluate utility preservation compared to baseline."""
        if not self.baseline_metrics:
            raise ValueError("Baseline metrics not set. Call set_baseline first.")
        
        evaluator = AccuracyMetrics()
        current_metrics = evaluator.evaluate_model(model, data_loader, device)
        
        preservation_scores = evaluator.compare_models(self.baseline_metrics, current_metrics)
        
        return preservation_scores
    
    def continuous_monitoring(self, model: nn.Module,
                            data_loader: torch.utils.data.DataLoader,
                            operation_name: str,
                            device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
        """Continuously monitor utility during privacy operations."""
        preservation_scores = self.evaluate_preservation(model, data_loader, device)
        
        monitoring_result = {
            "operation": operation_name,
            "timestamp": torch.utils.data.DataLoader,  # Would use actual timestamp in practice
            "preservation_scores": preservation_scores,
            "utility_alert": any(score < 0.8 for score in preservation_scores.values() if "preservation" in str(score))
        }
        
        return monitoring_result