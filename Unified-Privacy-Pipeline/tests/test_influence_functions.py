"""
Unit tests for influence function computation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from influence_functions.influence_computation import (
    InfluenceComputation,
    InfluenceConfig,
    create_influence_computer
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
    return SimpleModel()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    return loader


@pytest.fixture
def test_sample():
    """Create a test sample."""
    return torch.randn(1, 10), torch.tensor([1])


def test_influence_computation_init():
    """Test influence computation initialization."""
    config = InfluenceConfig()
    influence_comp = InfluenceComputation(config)
    assert influence_comp.config is not None
    assert influence_comp.config.method == "lissa"


def test_lissa_influence(simple_model, sample_data, test_sample):
    """Test LiSSA influence computation."""
    influence_comp = create_influence_computer(method="lissa", lissa_iterations=10)

    test_data, test_target = test_sample
    device = torch.device('cpu')

    influences = influence_comp.compute_influence_lissa(
        simple_model, test_data, test_target, sample_data, device, iterations=10
    )

    assert isinstance(influences, dict)
    assert len(influences) > 0
    for name, influence in influences.items():
        assert isinstance(influence, torch.Tensor)


def test_exact_influence(simple_model, sample_data, test_sample):
    """Test exact influence computation."""
    influence_comp = create_influence_computer(method="exact")

    test_data, test_target = test_sample
    device = torch.device('cpu')

    influences = influence_comp.compute_influence_exact(
        simple_model, test_data, test_target, sample_data, device
    )

    assert isinstance(influences, dict)
    assert len(influences) > 0


def test_loss_gradient_computation(simple_model, test_sample):
    """Test gradient computation."""
    influence_comp = InfluenceComputation()

    test_data, test_target = test_sample
    device = torch.device('cpu')

    grads = influence_comp._compute_loss_gradient(
        simple_model, test_data, test_target, device
    )

    assert isinstance(grads, dict)
    assert len(grads) > 0
    for name, grad in grads.items():
        assert isinstance(grad, torch.Tensor)


def test_hvp_computation(simple_model, sample_data):
    """Test Hessian-vector product computation."""
    influence_comp = InfluenceComputation()

    # Get a batch of data
    data, target = next(iter(sample_data))
    device = torch.device('cpu')

    # Create a random vector
    vector = {name: torch.randn_like(param) for name, param in simple_model.named_parameters()}

    hvp = influence_comp._compute_hvp(simple_model, data, target, vector, device)

    assert isinstance(hvp, dict)
    assert len(hvp) > 0


def test_self_influence(simple_model, sample_data, test_sample):
    """Test self-influence computation."""
    influence_comp = create_influence_computer(method="lissa", lissa_iterations=10)

    test_data, test_target = test_sample
    device = torch.device('cpu')

    self_inf = influence_comp.compute_self_influence(
        simple_model, test_data, test_target, sample_data, device, method="lissa"
    )

    assert isinstance(self_inf, float)


def test_factory_function():
    """Test factory function for creating influence computers."""
    inf_comp = create_influence_computer(method="lissa", lissa_iterations=100)
    assert isinstance(inf_comp, InfluenceComputation)
    assert inf_comp.config.lissa_iterations == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
