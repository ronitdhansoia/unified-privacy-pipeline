#!/usr/bin/env python3
"""
Quick validation script to verify the build is complete and functional.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all critical modules can be imported."""
    print("Testing imports...")
    try:
        from influence_functions.influence_computation import InfluenceComputation, create_influence_computer
        from machine_unlearning.unlearning_methods import (
            GradientAscentUnlearning,
            create_unlearner,
            UnlearningConfig
        )
        from evaluation.privacy_metrics import MembershipInferenceAttack, PrivacyAccountant
        print("[PASS] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from influence_functions.influence_computation import create_influence_computer
        from machine_unlearning.unlearning_methods import create_unlearner, UnlearningConfig

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()

        # Create simple data
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)

        # Test influence computation
        print("  Testing influence computation...")
        inf_computer = create_influence_computer(method="lissa", lissa_iterations=5)
        test_data = X[0:1]
        test_target = y[0:1]
        influences = inf_computer.compute_influence_lissa(
            model, test_data, test_target, loader, torch.device('cpu'), iterations=5
        )
        assert isinstance(influences, dict)
        print("  [PASS] Influence computation works")

        # Test unlearning
        print("  Testing unlearning...")
        config = UnlearningConfig(max_iterations=2, patience=1)
        unlearner = create_unlearner("gradient_ascent", config)

        forget_loader = DataLoader(TensorDataset(X[0:20], y[0:20]), batch_size=5)
        retain_loader = DataLoader(TensorDataset(X[20:50], y[20:50]), batch_size=5)

        results = unlearner.unlearn(model, forget_loader, retain_loader, torch.device('cpu'))
        assert 'method' in results
        print("  [PASS] Unlearning works")

        print("[PASS] All functionality tests passed")
        return True

    except Exception as e:
        print(f"[FAIL] Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all required files and directories exist."""
    print("\nTesting file structure...")

    required_files = [
        'src/influence_functions/influence_computation.py',
        'src/machine_unlearning/unlearning_methods.py',
        'tests/test_influence_functions.py',
        'tests/test_unlearning_methods.py',
        'tests/test_privacy_metrics.py',
        'tests/test_integration.py',
        'scripts/run_tests.sh',
        'scripts/train_model.py',
        'LICENSE',
        'CONTRIBUTING.md',
        'notebooks/01_getting_started.ipynb',
        'experiments/benchmarks/privacy_utility_tradeoff.py',
        'experiments/evaluations/comprehensive_evaluation.py'
    ]

    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"  [PASS] {file_path}")
        else:
            print(f"  [FAIL] {file_path} - MISSING")
            all_exist = False

    if all_exist:
        print("[PASS] All required files present")
    else:
        print("[FAIL] Some files are missing")

    return all_exist


def main():
    print("="*70)
    print("UNIFIED PRIVACY PIPELINE - BUILD VALIDATION")
    print("="*70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Functionality", test_basic_functionality()))

    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name}: {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\nBuild validation successful! Project is ready for presentation.")
        return 0
    else:
        print("\nSome validation tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
