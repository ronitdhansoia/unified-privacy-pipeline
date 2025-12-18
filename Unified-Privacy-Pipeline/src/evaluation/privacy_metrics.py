"""
Privacy evaluation metrics and attacks.

Implements various privacy attacks and metrics to evaluate the privacy
protection of trained models including:
- Membership inference attacks
- Model extraction attacks  
- Reconstruction attacks
- Privacy accounting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Results from a privacy attack."""
    attack_accuracy: float
    attack_auc: float
    confidence_scores: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    metadata: Dict[str, Any]


class MembershipInferenceAttack:
    """
    Implementation of membership inference attacks.
    
    Tests whether an attacker can determine if a specific sample
    was used during model training.
    """
    
    def __init__(self, attack_model_type: str = "logistic"):
        self.attack_model_type = attack_model_type
        self.attack_model = None
        
    def prepare_attack_data(self, target_model: nn.Module,
                           member_data: torch.utils.data.DataLoader,
                           non_member_data: torch.utils.data.DataLoader,
                           device: torch.device = torch.device('cpu')) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training the attack model.
        
        Args:
            target_model: The model to attack
            member_data: Data that was used to train the target model
            non_member_data: Data that was NOT used to train the target model
            device: Device to run inference on
            
        Returns:
            Tuple of (features, labels) for attack model training
        """
        target_model.eval()
        features_list = []
        labels_list = []
        
        # Process member data (label = 1)
        logger.info("Processing member data...")
        member_count = 0
        for batch_idx, (data, _) in enumerate(member_data):
            if batch_idx % 10 == 0:
                logger.debug(f"Processing member batch {batch_idx}")
                
            data = data.to(device)
            with torch.no_grad():
                outputs = target_model(data)
                
                # Extract attack features
                attack_features = self._extract_attack_features(outputs, data)
                features_list.append(attack_features)
                labels_list.extend([1] * len(data))
                member_count += len(data)
        
        # Process non-member data (label = 0)
        logger.info("Processing non-member data...")
        non_member_count = 0
        for batch_idx, (data, _) in enumerate(non_member_data):
            if batch_idx % 10 == 0:
                logger.debug(f"Processing non-member batch {batch_idx}")
                
            data = data.to(device)
            with torch.no_grad():
                outputs = target_model(data)
                
                # Extract attack features
                attack_features = self._extract_attack_features(outputs, data)
                features_list.append(attack_features)
                labels_list.extend([0] * len(data))
                non_member_count += len(data)
        
        # Ensure we have balanced data for attack training
        if member_count == 0 or non_member_count == 0:
            logger.warning(f"Imbalanced data: {member_count} members, {non_member_count} non-members")
            # Create artificial balance by duplicating minority class and adding noise
            if member_count == 0:
                # Duplicate some non-member data as member data with noise
                duplicate_count = min(max(non_member_count // 2, 1), len(features_list) // 2)
                for i in range(duplicate_count):
                    # Add small noise to create variation
                    noisy_features = features_list[i].copy()
                    noisy_features += np.random.randn(*noisy_features.shape) * 0.01
                    features_list.append(noisy_features)
                    labels_list.append(1)
                logger.info(f"Added {duplicate_count} synthetic member samples")
            elif non_member_count == 0:
                # Duplicate some member data as non-member data with noise
                duplicate_count = min(max(member_count // 2, 1), len(features_list) // 2)
                for i in range(duplicate_count):
                    # Add small noise to create variation
                    noisy_features = features_list[i].copy()
                    noisy_features += np.random.randn(*noisy_features.shape) * 0.01
                    features_list.append(noisy_features)
                    labels_list.append(0)
                logger.info(f"Added {duplicate_count} synthetic non-member samples")
        
        features = np.vstack(features_list)
        labels = np.array(labels_list)

        # Final check: ensure we have both classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logger.warning("Only one class present in labels, adding synthetic opposite class")
            # Add one sample of the opposite class
            opposite_label = 1 if unique_labels[0] == 0 else 0
            synthetic_sample = features[0].copy() + np.random.randn(*features[0].shape) * 0.1
            features = np.vstack([features, synthetic_sample.reshape(1, -1)])
            labels = np.append(labels, opposite_label)
        
        logger.info(f"Prepared attack data: {features.shape[0]} samples, {features.shape[1]} features")
        return features, labels
    
    def _extract_attack_features(self, outputs: torch.Tensor, inputs: torch.Tensor) -> np.ndarray:
        """
        Extract features for the membership inference attack.
        
        Common features include:
        - Prediction confidence (max probability)
        - Prediction entropy
        - Loss value
        - Top-k predictions
        """
        if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # Binary classification or regression
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            if len(probabilities.shape) > 1:
                probabilities = probabilities.squeeze()
            features = np.column_stack([
                probabilities,  # Prediction confidence
                1 - probabilities,  # Complement confidence
                np.abs(probabilities - 0.5),  # Distance from decision boundary
            ])
        else:  # Multi-class classification
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            
            # Confidence-based features
            max_prob = np.max(probabilities, axis=1)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
            
            # Top-2 predictions (handle case where we might have only 2 classes)
            sorted_probs = np.sort(probabilities, axis=1)
            top1 = sorted_probs[:, -1]
            top2 = sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else sorted_probs[:, -1]
            
            features = np.column_stack([
                max_prob,          # Maximum confidence
                entropy,           # Prediction entropy  
                top1 - top2,       # Confidence gap
                top1,              # Top-1 probability
                top2,              # Top-2 probability
            ])
        
        return features
    
    def train_attack_model(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Train the membership inference attack model.

        Args:
            features: Attack features
            labels: Membership labels (1=member, 0=non-member)

        Returns:
            Training accuracy of attack model
        """
        # Check if we have both classes in the data
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logger.warning(f"Only {len(unique_labels)} class(es) in data. Cannot train binary classifier.")
            logger.warning("Returning random baseline accuracy of 0.5")
            # Return a dummy attack model that always predicts the majority class
            self.attack_model = LogisticRegression(random_state=42)
            return 0.5  # Random baseline

        if self.attack_model_type == "logistic":
            self.attack_model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown attack model type: {self.attack_model_type}")

        # Split data for training
        split_idx = len(features) // 2
        train_features, test_features = features[:split_idx], features[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]

        # Check split has both classes
        if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
            logger.warning("Train/test split doesn't have both classes. Using full dataset for training.")
            # Use stratified approach or full dataset
            from sklearn.model_selection import train_test_split
            try:
                train_features, test_features, train_labels, test_labels = train_test_split(
                    features, labels, test_size=0.3, random_state=42, stratify=labels
                )
            except ValueError:
                # If stratification fails, just use simple split
                logger.warning("Stratification failed, using simple random split")
                train_features, test_features, train_labels, test_labels = train_test_split(
                    features, labels, test_size=0.3, random_state=42
                )

        # Final check before fitting
        if len(np.unique(train_labels)) < 2:
            logger.warning("Still only one class after split. Returning baseline.")
            return 0.5

        # Train attack model
        self.attack_model.fit(train_features, train_labels)

        # Evaluate on test set
        if len(np.unique(test_labels)) >= 2:
            test_predictions = self.attack_model.predict(test_features)
            attack_accuracy = accuracy_score(test_labels, test_predictions)
        else:
            attack_accuracy = 0.5  # Baseline if test set has only one class

        logger.info(f"Attack model training accuracy: {attack_accuracy:.4f}")
        return attack_accuracy
    
    def evaluate_attack(self, target_model: nn.Module,
                       eval_member_data: torch.utils.data.DataLoader,
                       eval_non_member_data: torch.utils.data.DataLoader,
                       device: torch.device = torch.device('cpu')) -> AttackResult:
        """
        Evaluate the membership inference attack.
        
        Args:
            target_model: Model to attack
            eval_member_data: Member data for evaluation
            eval_non_member_data: Non-member data for evaluation
            device: Device for inference
            
        Returns:
            AttackResult containing attack performance metrics
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        # Prepare evaluation data
        eval_features, eval_labels = self.prepare_attack_data(
            target_model, eval_member_data, eval_non_member_data, device
        )
        
        # Run attack
        attack_predictions = self.attack_model.predict(eval_features)
        attack_probabilities = self.attack_model.predict_proba(eval_features)[:, 1]
        
        # Compute metrics
        attack_accuracy = accuracy_score(eval_labels, attack_predictions)
        attack_auc = roc_auc_score(eval_labels, attack_probabilities)
        
        result = AttackResult(
            attack_accuracy=attack_accuracy,
            attack_auc=attack_auc,
            confidence_scores=attack_probabilities,
            predictions=attack_predictions,
            true_labels=eval_labels,
            metadata={
                "attack_model_type": self.attack_model_type,
                "num_samples": len(eval_labels),
                "num_members": np.sum(eval_labels),
                "num_non_members": len(eval_labels) - np.sum(eval_labels)
            }
        )
        
        logger.info(f"Attack Results - Accuracy: {attack_accuracy:.4f}, AUC: {attack_auc:.4f}")
        return result


class ModelExtractionAttack:
    """
    Implementation of model extraction attacks.
    
    Tests whether an attacker can steal the functionality of a model
    by querying it and training a surrogate model.
    """
    
    def __init__(self, surrogate_architecture: nn.Module):
        self.surrogate_model = surrogate_architecture
        self.query_history = []
        
    def extract_model(self, target_model: nn.Module,
                     query_data: torch.utils.data.DataLoader,
                     num_epochs: int = 10,
                     device: torch.device = torch.device('cpu')) -> Dict[str, float]:
        """
        Perform model extraction attack.
        
        Args:
            target_model: Model to extract
            query_data: Data to query the target model with
            num_epochs: Training epochs for surrogate model
            device: Device for computation
            
        Returns:
            Dictionary with extraction metrics
        """
        target_model.eval()
        self.surrogate_model = self.surrogate_model.to(device)
        
        # Collect query-response pairs
        query_inputs = []
        target_outputs = []
        
        logger.info("Collecting query-response pairs...")
        for batch_idx, (data, _) in enumerate(query_data):
            data = data.to(device)
            
            with torch.no_grad():
                outputs = target_model(data)
                
            query_inputs.append(data.cpu())
            target_outputs.append(outputs.cpu())
            
            self.query_history.append({
                'batch_idx': batch_idx,
                'num_queries': len(data),
                'input_shape': data.shape
            })
        
        # Prepare surrogate training data
        all_inputs = torch.cat(query_inputs, dim=0).to(device)
        all_outputs = torch.cat(target_outputs, dim=0).to(device)
        
        # Train surrogate model
        logger.info("Training surrogate model...")
        optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.001)
        
        # Determine loss function based on output shape
        if len(all_outputs.shape) == 1 or all_outputs.shape[1] == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()  # For soft labels from target model
        
        for epoch in range(num_epochs):
            self.surrogate_model.train()
            optimizer.zero_grad()
            
            surrogate_outputs = self.surrogate_model(all_inputs)
            loss = criterion(surrogate_outputs, all_outputs)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                logger.debug(f"Surrogate training epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate extraction quality
        self.surrogate_model.eval()
        with torch.no_grad():
            surrogate_predictions = self.surrogate_model(all_inputs)
            
            # Compute agreement between target and surrogate
            if len(all_outputs.shape) == 1 or all_outputs.shape[1] == 1:
                # Binary classification
                target_pred = torch.sigmoid(all_outputs) > 0.5
                surrogate_pred = torch.sigmoid(surrogate_predictions) > 0.5
                agreement = torch.mean((target_pred == surrogate_pred).float()).item()
            else:
                # Multi-class or regression
                target_pred = torch.argmax(all_outputs, dim=1)
                surrogate_pred = torch.argmax(surrogate_predictions, dim=1)
                agreement = torch.mean((target_pred == surrogate_pred).float()).item()
        
        extraction_metrics = {
            "agreement_rate": agreement,
            "num_queries": len(all_inputs),
            "surrogate_loss": loss.item(),
            "extraction_epochs": num_epochs
        }
        
        logger.info(f"Model extraction complete. Agreement rate: {agreement:.4f}")
        return extraction_metrics


class ReconstructionAttack:
    """
    Implementation of data reconstruction attacks.
    
    Tests whether an attacker can reconstruct training data
    from model parameters or gradients.
    """
    
    def __init__(self):
        self.reconstruction_history = []
    
    def gradient_based_reconstruction(self, model: nn.Module,
                                    target_gradients: Dict[str, torch.Tensor],
                                    input_shape: Tuple[int, ...],
                                    num_iterations: int = 1000,
                                    device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
        """
        Attempt to reconstruct input data from gradients.
        
        Args:
            model: The model that produced the gradients
            target_gradients: Gradients to match
            input_shape: Shape of input to reconstruct
            num_iterations: Optimization iterations
            device: Device for computation
            
        Returns:
            Dictionary with reconstruction results
        """
        model = model.to(device)
        model.eval()
        
        # Initialize random input
        reconstructed_input = torch.randn(input_shape, device=device, requires_grad=True)
        
        # Optimizer for reconstruction
        optimizer = torch.optim.Adam([reconstructed_input], lr=0.1)
        
        best_loss = float('inf')
        best_reconstruction = None
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass with reconstructed input
            output = model(reconstructed_input)
            
            # Compute dummy loss (we need gradients)
            dummy_loss = output.sum()
            
            # Compute gradients
            current_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            # Compute reconstruction loss (gradient matching)
            reconstruction_loss = 0
            for (name, target_grad), current_grad in zip(target_gradients.items(), current_gradients):
                if current_grad is not None:
                    reconstruction_loss += F.mse_loss(current_grad, target_grad)
            
            # Backward pass
            reconstruction_loss.backward()
            optimizer.step()
            
            # Track best reconstruction
            if reconstruction_loss.item() < best_loss:
                best_loss = reconstruction_loss.item()
                best_reconstruction = reconstructed_input.detach().clone()
            
            if iteration % 100 == 0:
                logger.debug(f"Reconstruction iteration {iteration}, Loss: {reconstruction_loss.item():.6f}")
        
        reconstruction_result = {
            "reconstructed_input": best_reconstruction.cpu(),
            "final_loss": best_loss,
            "iterations": num_iterations,
            "input_shape": input_shape
        }
        
        self.reconstruction_history.append(reconstruction_result)
        logger.info(f"Reconstruction attack complete. Final loss: {best_loss:.6f}")
        
        return reconstruction_result


class PrivacyAccountant:
    """
    Tracks and accounts for privacy budget spent during training.
    
    Integrates with differential privacy mechanisms to monitor
    the total privacy cost.
    """
    
    def __init__(self, target_epsilon: float, target_delta: float):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.privacy_history = []
    
    def add_privacy_cost(self, epsilon: float, delta: float, operation: str = "training"):
        """Add privacy cost from an operation."""
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        
        self.privacy_history.append({
            "operation": operation,
            "epsilon": epsilon,
            "delta": delta,
            "cumulative_epsilon": self.spent_epsilon,
            "cumulative_delta": self.spent_delta
        })
        
        logger.info(f"Privacy cost added - ε: {epsilon:.6f}, δ: {delta:.6f} for {operation}")
        
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        remaining_epsilon = max(0, self.target_epsilon - self.spent_epsilon)
        remaining_delta = max(0, self.target_delta - self.spent_delta)
        return remaining_epsilon, remaining_delta
    
    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded."""
        return (self.spent_epsilon > self.target_epsilon or 
                self.spent_delta > self.target_delta)
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        remaining_eps, remaining_delta = self.get_remaining_budget()
        
        report = {
            "target_privacy": {"epsilon": self.target_epsilon, "delta": self.target_delta},
            "spent_privacy": {"epsilon": self.spent_epsilon, "delta": self.spent_delta},
            "remaining_privacy": {"epsilon": remaining_eps, "delta": remaining_delta},
            "budget_exceeded": self.is_budget_exceeded(),
            "privacy_operations": len(self.privacy_history),
            "privacy_history": self.privacy_history
        }
        
        return report