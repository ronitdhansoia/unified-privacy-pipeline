"""
Unit tests for machine unlearning methods.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from machine_unlearning.unlearning_methods import (
    GradientAscentUnlearning,
    InfluenceBasedUnlearning,
    FineTuningUnlearning,
    NegativeGradientUnlearning,
    UnlearningConfig,
    create_unlearner
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
def simple_model():
    """Create a simple model for testing."""
    model = SimpleModel()
    # Pre-train the model a bit
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

    return model


@pytest.fixture
def data_loaders():
    """Create forget and retain data loaders."""
    # Forget set
    X_forget = torch.randn(50, 10)
    y_forget = torch.randint(0, 2, (50,))
    forget_dataset = TensorDataset(X_forget, y_forget)
    forget_loader = DataLoader(forget_dataset, batch_size=10, shuffle=True)

    # Retain set
    X_retain = torch.randn(100, 10)
    y_retain = torch.randint(0, 2, (100,))
    retain_dataset = TensorDataset(X_retain, y_retain)
    retain_loader = DataLoader(retain_dataset, batch_size=10, shuffle=True)

    return forget_loader, retain_loader


def test_unlearning_config():
    """Test unlearning configuration."""
    config = UnlearningConfig(
        learning_rate=0.01,
        max_iterations=50
    )
    assert config.learning_rate == 0.01
    assert config.max_iterations == 50


def test_gradient_ascent_unlearning(simple_model, data_loaders):
    """Test gradient ascent unlearning."""
    forget_loader, retain_loader = data_loaders

    config = UnlearningConfig(max_iterations=5, patience=2)
    unlearner = GradientAscentUnlearning(config)

    device = torch.device('cpu')
    results = unlearner.unlearn(simple_model, forget_loader, retain_loader, device)

    assert 'method' in results
    assert results['method'] == 'gradient_ascent'
    assert 'final_metrics' in results
    assert 'forget_loss' in results['final_metrics']
    assert 'retain_loss' in results['final_metrics']


def test_fine_tuning_unlearning(simple_model, data_loaders):
    """Test fine-tuning unlearning."""
    forget_loader, retain_loader = data_loaders

    config = UnlearningConfig(max_iterations=5)
    unlearner = FineTuningUnlearning(config)

    device = torch.device('cpu')
    results = unlearner.unlearn(simple_model, forget_loader, retain_loader, device)

    assert 'method' in results
    assert results['method'] == 'fine_tuning'
    assert 'final_metrics' in results


def test_negative_gradient_unlearning(simple_model, data_loaders):
    """Test negative gradient unlearning."""
    forget_loader, retain_loader = data_loaders

    config = UnlearningConfig()
    unlearner = NegativeGradientUnlearning(config)

    device = torch.device('cpu')
    results = unlearner.unlearn(simple_model, forget_loader, retain_loader, device)

    assert 'method' in results
    assert results['method'] == 'negative_gradient'
    assert 'parameter_change' in results
    assert results['parameter_change'] > 0  # Parameters should have changed


def test_influence_based_unlearning_fallback(simple_model, data_loaders):
    """Test influence-based unlearning (should fallback to gradient ascent)."""
    forget_loader, retain_loader = data_loaders

    config = UnlearningConfig(max_iterations=5, patience=2)
    # No influence computer provided, should fallback
    unlearner = InfluenceBasedUnlearning(config, influence_computer=None)

    device = torch.device('cpu')
    results = unlearner.unlearn(simple_model, forget_loader, retain_loader, device)

    assert results is not None


def test_create_unlearner_factory():
    """Test factory function for creating unlearners."""
    config = UnlearningConfig()

    # Test different methods
    ga_unlearner = create_unlearner("gradient_ascent", config)
    assert isinstance(ga_unlearner, GradientAscentUnlearning)

    ft_unlearner = create_unlearner("fine_tuning", config)
    assert isinstance(ft_unlearner, FineTuningUnlearning)

    ng_unlearner = create_unlearner("negative_gradient", config)
    assert isinstance(ng_unlearner, NegativeGradientUnlearning)


def test_unlearner_evaluation(simple_model, data_loaders):
    """Test model evaluation method."""
    forget_loader, retain_loader = data_loaders

    unlearner = GradientAscentUnlearning()
    device = torch.device('cpu')

    loss, accuracy = unlearner._evaluate_model(simple_model, forget_loader, device)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
