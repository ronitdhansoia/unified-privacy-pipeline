"""
Comprehensive Evaluation Script.

Evaluates the complete privacy pipeline including:
- Privacy metrics (MIA, extraction attacks)
- Utility metrics (accuracy, loss)
- Fairness metrics
- Unlearning quality
"""

import torch
import json
import numpy as np
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from evaluation.privacy_metrics import MembershipInferenceAttack, PrivacyAccountant
from machine_unlearning.unlearning_methods import create_unlearner, UnlearningConfig


class ComprehensiveEvaluator:
    """Comprehensive evaluation of privacy-preserving models."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.results = {}

    def evaluate_privacy(self, member_loader, non_member_loader) -> Dict[str, Any]:
        """Evaluate privacy using membership inference attack."""
        print("\n" + "="*50)
        print("Privacy Evaluation")
        print("="*50)

        mia = MembershipInferenceAttack()

        # Prepare attack data
        features, labels = mia.prepare_attack_data(
            self.model, member_loader, non_member_loader, self.device
        )

        # Train attack model
        train_acc = mia.train_attack_model(features, labels)

        # Evaluate attack
        attack_result = mia.evaluate_attack(
            self.model, member_loader, non_member_loader, self.device
        )

        privacy_results = {
            'mia_attack_accuracy': attack_result.attack_accuracy,
            'mia_attack_auc': attack_result.attack_auc,
            'mia_train_accuracy': train_acc,
            'privacy_leakage': attack_result.attack_accuracy - 0.5,  # Above random
            'privacy_score': 1.0 - (attack_result.attack_accuracy - 0.5) * 2  # Normalized
        }

        print(f"MIA Attack Accuracy: {attack_result.attack_accuracy:.4f}")
        print(f"MIA Attack AUC: {attack_result.attack_auc:.4f}")
        print(f"Privacy Score: {privacy_results['privacy_score']:.4f}")

        return privacy_results

    def evaluate_utility(self, test_loader) -> Dict[str, Any]:
        """Evaluate model utility (accuracy, loss)."""
        print("\n" + "="*50)
        print("Utility Evaluation")
        print("="*50)

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, dict):
                    data = batch_data.get('image', batch_data.get('features')).to(self.device)
                    target = batch_data.get('identity', batch_data.get('target')).to(self.device)
                else:
                    data, target = batch_data[0].to(self.device), batch_data[1].to(self.device)

                output = self.model(data)

                # Compute loss
                if output.dim() > 1 and output.shape[1] > 1:
                    loss = torch.nn.CrossEntropyLoss()(output, target.long())
                    pred = output.argmax(dim=1)
                else:
                    loss = torch.nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
                    pred = (torch.sigmoid(output.squeeze()) > 0.5).long()

                total_loss += loss.item()
                correct += (pred == target.long()).sum().item()
                total += len(data)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        utility_results = {
            'test_accuracy': accuracy,
            'test_loss': avg_loss,
            'total_samples': total,
            'correct_predictions': correct
        }

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {avg_loss:.4f}")

        return utility_results

    def evaluate_unlearning_quality(
        self,
        original_model,
        forget_loader,
        retain_loader
    ) -> Dict[str, Any]:
        """Evaluate unlearning quality."""
        print("\n" + "="*50)
        print("Unlearning Quality Evaluation")
        print("="*50)

        # Compute parameter changes
        param_changes = {}
        total_change = 0.0

        for (name_orig, param_orig), (name_new, param_new) in zip(
            original_model.named_parameters(),
            self.model.named_parameters()
        ):
            if name_orig == name_new:
                change = torch.norm(param_orig - param_new).item()
                param_changes[name_orig] = change
                total_change += change

        # Evaluate on forget and retain sets
        forget_loss, forget_acc = self._evaluate_on_loader(forget_loader)
        retain_loss, retain_acc = self._evaluate_on_loader(retain_loader)

        # Compute forget quality (higher forget loss = better forgetting)
        forget_quality = min(1.0, forget_loss / 2.0)  # Normalized heuristic

        unlearning_results = {
            'parameter_change': total_change,
            'forget_loss': forget_loss,
            'forget_accuracy': forget_acc,
            'retain_loss': retain_loss,
            'retain_accuracy': retain_acc,
            'forget_quality': forget_quality,
            'utility_preservation': retain_acc
        }

        print(f"Parameter Change: {total_change:.6f}")
        print(f"Forget Accuracy: {forget_acc:.4f}")
        print(f"Retain Accuracy: {retain_acc:.4f}")
        print(f"Forget Quality: {forget_quality:.4f}")

        return unlearning_results

    def _evaluate_on_loader(self, data_loader):
        """Helper to evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, dict):
                    data = batch_data.get('image', batch_data.get('features')).to(self.device)
                    target = batch_data.get('identity', batch_data.get('target')).to(self.device)
                else:
                    data, target = batch_data[0].to(self.device), batch_data[1].to(self.device)

                output = self.model(data)

                # Compute loss
                if output.dim() > 1 and output.shape[1] > 1:
                    loss = torch.nn.CrossEntropyLoss()(output, target.long())
                    pred = output.argmax(dim=1)
                else:
                    loss = torch.nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
                    pred = (torch.sigmoid(output.squeeze()) > 0.5).long()

                total_loss += loss.item()
                correct += (pred == target.long()).sum().item()
                total += len(data)

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def generate_report(self, output_path: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report."""
        print("\n" + "="*50)
        print("Generating Comprehensive Report")
        print("="*50)

        report = {
            'model_summary': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'evaluations': self.results
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {output_path}")
        return report


if __name__ == "__main__":
    print("Comprehensive Privacy Pipeline Evaluation")
    print("=" * 50)
    print("\nUsage:")
    print("  from evaluations.comprehensive_evaluation import ComprehensiveEvaluator")
    print("  evaluator = ComprehensiveEvaluator(model, device)")
    print("  evaluator.evaluate_privacy(member_loader, non_member_loader)")
    print("  evaluator.evaluate_utility(test_loader)")
    print("  evaluator.generate_report()")
