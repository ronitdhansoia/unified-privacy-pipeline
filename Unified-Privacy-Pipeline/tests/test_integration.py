"""
Integration tests for the complete privacy pipeline.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from influence_functions.influence_computation import create_influence_computer
from machine_unlearning.unlearning_methods import create_unlearner, UnlearningConfig
from evaluation.privacy_metrics import MembershipInferenceAttack, PrivacyAccountant


class SimpleClassifier(nn.Module):
    """Simple classifier for integration testing."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def complete_pipeline_setup():
    """Set up a complete pipeline for testing."""
    # Create model
    model = SimpleClassifier()

    # Create datasets
    X_train = torch.randn(200, 10)
    y_train = torch.randint(0, 2, (200,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    X_forget = torch.randn(50, 10)
    y_forget = torch.randint(0, 2, (50,))
    forget_dataset = TensorDataset(X_forget, y_forget)
    forget_loader = DataLoader(forget_dataset, batch_size=10, shuffle=False)

    X_retain = torch.randn(150, 10)
    y_retain = torch.randint(0, 2, (150,))
    retain_dataset = TensorDataset(X_retain, y_retain)
    retain_loader = DataLoader(retain_dataset, batch_size=20, shuffle=True)

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = nn.CrossEntropyLoss()(output, y_batch)
            loss.backward()
            optimizer.step()

    return {
        'model': model,
        'train_loader': train_loader,
        'forget_loader': forget_loader,
        'retain_loader': retain_loader
    }


def test_end_to_end_unlearning_pipeline(complete_pipeline_setup):
    """Test complete unlearning pipeline."""
    setup = complete_pipeline_setup
    model = setup['model']
    forget_loader = setup['forget_loader']
    retain_loader = setup['retain_loader']

    device = torch.device('cpu')

    # Step 1: Create unlearner
    config = UnlearningConfig(max_iterations=5, patience=2)
    unlearner = create_unlearner("gradient_ascent", config)

    # Step 2: Perform unlearning
    results = unlearner.unlearn(model, forget_loader, retain_loader, device)

    # Verify results
    assert 'method' in results
    assert 'final_metrics' in results
    assert results['final_metrics']['forget_loss'] >= 0
    assert results['final_metrics']['retain_accuracy'] >= 0


def test_privacy_evaluation_pipeline(complete_pipeline_setup):
    """Test privacy evaluation with MIA."""
    setup = complete_pipeline_setup
    model = setup['model']
    train_loader = setup['train_loader']
    retain_loader = setup['retain_loader']

    device = torch.device('cpu')

    # Create MIA
    mia = MembershipInferenceAttack()

    # Prepare attack data
    features, labels = mia.prepare_attack_data(
        model, train_loader, retain_loader, device
    )

    # Train attack
    mia.train_attack_model(features, labels)

    # Evaluate
    result = mia.evaluate_attack(model, train_loader, retain_loader, device)

    assert result.attack_accuracy >= 0
    assert result.attack_auc >= 0


def test_influence_computation_pipeline(complete_pipeline_setup):
    """Test influence computation in pipeline."""
    setup = complete_pipeline_setup
    model = setup['model']
    forget_loader = setup['forget_loader']
    retain_loader = setup['retain_loader']

    device = torch.device('cpu')

    # Create influence computer
    inf_computer = create_influence_computer(method="lissa", lissa_iterations=10)

    # Get a test sample
    test_data, test_target = next(iter(forget_loader))
    test_data = test_data[0:1]
    test_target = test_target[0:1]

    # Compute influences
    influences = inf_computer.compute_influence_lissa(
        model, test_data, test_target, retain_loader, device, iterations=10
    )

    assert isinstance(influences, dict)
    assert len(influences) > 0


def test_combined_influence_and_unlearning(complete_pipeline_setup):
    """Test influence computation combined with unlearning."""
    setup = complete_pipeline_setup
    model = setup['model']
    forget_loader = setup['forget_loader']
    retain_loader = setup['retain_loader']

    device = torch.device('cpu')

    # Create influence computer
    inf_computer = create_influence_computer(method="lissa", lissa_iterations=10)

    # Create influence-based unlearner
    config = UnlearningConfig(max_iterations=5)
    unlearner = create_unlearner("influence_based", config, influence_computer=inf_computer)

    # Perform unlearning
    results = unlearner.unlearn(model, forget_loader, retain_loader, device)

    assert results is not None
    assert 'method' in results


def test_privacy_accounting_workflow():
    """Test privacy accounting workflow."""
    accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)

    # Simulate multiple training epochs with privacy cost
    for epoch in range(5):
        epsilon_cost = 0.3
        delta_cost = 1e-6
        accountant.add_privacy_cost(epsilon_cost, delta_cost, f"epoch_{epoch}")

    # Check budget (5 epochs * 0.3 = 1.5, which exceeds 1.0)
    is_exceeded = accountant.is_budget_exceeded()
    assert is_exceeded  # Should exceed after 5 epochs

    # Generate report
    report = accountant.get_privacy_report()
    assert report['privacy_operations'] == 5
    assert report['spent_privacy']['epsilon'] == 1.5


def test_full_privacy_pipeline_workflow(complete_pipeline_setup):
    """Test complete privacy-preserving ML workflow."""
    setup = complete_pipeline_setup
    model = setup['model']
    forget_loader = setup['forget_loader']
    retain_loader = setup['retain_loader']
    train_loader = setup['train_loader']

    device = torch.device('cpu')

    # Initialize privacy accountant
    accountant = PrivacyAccountant(target_epsilon=3.0, target_delta=1e-5)

    # Step 1: Initial privacy evaluation
    mia_initial = MembershipInferenceAttack()
    features_init, labels_init = mia_initial.prepare_attack_data(
        model, train_loader, retain_loader, device
    )
    mia_initial.train_attack_model(features_init, labels_init)
    result_init = mia_initial.evaluate_attack(model, train_loader, retain_loader, device)

    initial_privacy_leakage = result_init.attack_accuracy

    # Step 2: Perform unlearning
    config = UnlearningConfig(max_iterations=5, patience=2)
    unlearner = create_unlearner("gradient_ascent", config)
    unlearning_results = unlearner.unlearn(model, forget_loader, retain_loader, device)

    accountant.add_privacy_cost(0.5, 1e-6, "unlearning_operation")

    # Step 3: Post-unlearning privacy evaluation
    mia_post = MembershipInferenceAttack()
    features_post, labels_post = mia_post.prepare_attack_data(
        model, forget_loader, retain_loader, device
    )
    mia_post.train_attack_model(features_post, labels_post)
    result_post = mia_post.evaluate_attack(model, forget_loader, retain_loader, device)

    # Verify pipeline completed
    assert unlearning_results is not None
    assert result_post.attack_accuracy >= 0
    assert not accountant.is_budget_exceeded()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
