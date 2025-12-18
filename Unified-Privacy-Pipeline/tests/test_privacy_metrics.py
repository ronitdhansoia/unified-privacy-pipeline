"""
Unit tests for privacy evaluation metrics.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.privacy_metrics import (
    MembershipInferenceAttack,
    ModelExtractionAttack,
    PrivacyAccountant
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def trained_model():
    """Create and train a simple model."""
    model = SimpleModel()
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(20):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

    return model


@pytest.fixture
def member_non_member_data():
    """Create member and non-member datasets."""
    # Member data (used in training)
    X_member = torch.randn(50, 10)
    y_member = torch.randint(0, 2, (50,))
    member_dataset = TensorDataset(X_member, y_member)
    member_loader = DataLoader(member_dataset, batch_size=10)

    # Non-member data (not used in training)
    X_non_member = torch.randn(50, 10)
    y_non_member = torch.randint(0, 2, (50,))
    non_member_dataset = TensorDataset(X_non_member, y_non_member)
    non_member_loader = DataLoader(non_member_dataset, batch_size=10)

    return member_loader, non_member_loader


def test_mia_initialization():
    """Test MIA initialization."""
    mia = MembershipInferenceAttack(attack_model_type="logistic")
    assert mia.attack_model_type == "logistic"
    assert mia.attack_model is None


def test_mia_prepare_attack_data(trained_model, member_non_member_data):
    """Test MIA attack data preparation."""
    member_loader, non_member_loader = member_non_member_data

    mia = MembershipInferenceAttack()
    device = torch.device('cpu')

    features, labels = mia.prepare_attack_data(
        trained_model, member_loader, non_member_loader, device
    )

    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(features) == len(labels)
    assert features.shape[0] > 0


def test_mia_train_attack_model(trained_model, member_non_member_data):
    """Test training the MIA attack model."""
    member_loader, non_member_loader = member_non_member_data

    mia = MembershipInferenceAttack()
    device = torch.device('cpu')

    features, labels = mia.prepare_attack_data(
        trained_model, member_loader, non_member_loader, device
    )

    accuracy = mia.train_attack_model(features, labels)

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
    assert mia.attack_model is not None


def test_mia_full_pipeline(trained_model, member_non_member_data):
    """Test full MIA pipeline."""
    member_loader, non_member_loader = member_non_member_data

    mia = MembershipInferenceAttack()
    device = torch.device('cpu')

    # Prepare and train
    features, labels = mia.prepare_attack_data(
        trained_model, member_loader, non_member_loader, device
    )
    mia.train_attack_model(features, labels)

    # Evaluate attack
    result = mia.evaluate_attack(
        trained_model, member_loader, non_member_loader, device
    )

    assert result.attack_accuracy >= 0
    assert result.attack_auc >= 0
    assert 'attack_model_type' in result.metadata


def test_model_extraction_attack(trained_model, member_non_member_data):
    """Test model extraction attack."""
    query_loader, _ = member_non_member_data

    # Create surrogate model
    surrogate = SimpleModel()

    extractor = ModelExtractionAttack(surrogate)
    device = torch.device('cpu')

    metrics = extractor.extract_model(
        trained_model, query_loader, num_epochs=5, device=device
    )

    assert 'agreement_rate' in metrics
    assert 'num_queries' in metrics
    assert 0 <= metrics['agreement_rate'] <= 1


def test_privacy_accountant():
    """Test privacy accountant."""
    accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)

    assert accountant.target_epsilon == 1.0
    assert accountant.target_delta == 1e-5

    # Add some privacy cost
    accountant.add_privacy_cost(0.5, 1e-6, "training_epoch_1")

    remaining_eps, remaining_delta = accountant.get_remaining_budget()
    assert remaining_eps == 0.5
    assert remaining_delta == (1e-5 - 1e-6)

    # Check if budget exceeded
    assert not accountant.is_budget_exceeded()

    # Add more cost to exceed budget
    accountant.add_privacy_cost(0.7, 1e-5, "training_epoch_2")
    assert accountant.is_budget_exceeded()


def test_privacy_accountant_report():
    """Test privacy report generation."""
    accountant = PrivacyAccountant(target_epsilon=2.0, target_delta=1e-5)
    accountant.add_privacy_cost(0.8, 5e-6, "operation_1")
    accountant.add_privacy_cost(0.5, 3e-6, "operation_2")

    report = accountant.get_privacy_report()

    assert 'target_privacy' in report
    assert 'spent_privacy' in report
    assert 'remaining_privacy' in report
    assert 'budget_exceeded' in report
    assert report['privacy_operations'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
