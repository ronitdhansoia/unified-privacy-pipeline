"""
Privacy-Utility Tradeoff Benchmark.

Evaluates the relationship between privacy guarantees and model utility
across different privacy budgets and unlearning methods.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from machine_unlearning.unlearning_methods import create_unlearner, UnlearningConfig
from evaluation.privacy_metrics import MembershipInferenceAttack


def run_privacy_utility_benchmark(
    model: nn.Module,
    forget_loader,
    retain_loader,
    test_loader,
    methods: List[str] = None,
    iterations_range: List[int] = None
) -> Dict:
    """
    Run privacy-utility tradeoff benchmark.

    Args:
        model: Model to benchmark
        forget_loader: Forget data loader
        retain_loader: Retain data loader
        test_loader: Test data loader
        methods: Unlearning methods to test
        iterations_range: Range of iterations to test

    Returns:
        Benchmark results dictionary
    """
    if methods is None:
        methods = ['gradient_ascent', 'fine_tuning', 'negative_gradient']

    if iterations_range is None:
        iterations_range = [10, 25, 50, 100]

    results = {
        'methods': {},
        'metadata': {
            'iterations_tested': iterations_range,
            'methods_tested': methods
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for method in methods:
        print(f"\nTesting method: {method}")
        method_results = []

        for max_iter in iterations_range:
            print(f"  Iterations: {max_iter}")

            # Create fresh model copy
            model_copy = type(model)()
            model_copy.load_state_dict(model.state_dict())

            # Configure and run unlearning
            config = UnlearningConfig(
                max_iterations=max_iter,
                patience=max_iter // 5
            )
            unlearner = create_unlearner(method, config)

            unlearn_results = unlearner.unlearn(
                model_copy, forget_loader, retain_loader, device
            )

            method_results.append({
                'iterations': max_iter,
                'forget_loss': unlearn_results.get('final_metrics', {}).get('forget_loss', 0),
                'forget_accuracy': unlearn_results.get('final_metrics', {}).get('forget_accuracy', 0),
                'retain_loss': unlearn_results.get('final_metrics', {}).get('retain_loss', 0),
                'retain_accuracy': unlearn_results.get('final_metrics', {}).get('retain_accuracy', 0),
                'forget_quality': unlearn_results.get('forget_quality', 0),
                'utility_preservation': unlearn_results.get('utility_preservation', 0)
            })

        results['methods'][method] = method_results

    return results


def plot_results(results: Dict, output_path: str = 'privacy_utility_plot.png'):
    """Plot privacy-utility tradeoff results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Privacy-Utility Tradeoff Analysis', fontsize=16)

    methods = list(results['methods'].keys())

    for method in methods:
        method_data = results['methods'][method]
        iterations = [r['iterations'] for r in method_data]
        forget_acc = [r['forget_accuracy'] for r in method_data]
        retain_acc = [r['retain_accuracy'] for r in method_data]
        forget_quality = [r['forget_quality'] for r in method_data]
        utility_pres = [r['utility_preservation'] for r in method_data]

        # Forget accuracy over iterations
        axes[0, 0].plot(iterations, forget_acc, marker='o', label=method)
        axes[0, 0].set_xlabel('Iterations')
        axes[0, 0].set_ylabel('Forget Accuracy')
        axes[0, 0].set_title('Forgetting Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Retain accuracy over iterations
        axes[0, 1].plot(iterations, retain_acc, marker='s', label=method)
        axes[0, 1].set_xlabel('Iterations')
        axes[0, 1].set_ylabel('Retain Accuracy')
        axes[0, 1].set_title('Utility Preservation')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Forget quality
        axes[1, 0].plot(iterations, forget_quality, marker='^', label=method)
        axes[1, 0].set_xlabel('Iterations')
        axes[1, 0].set_ylabel('Forget Quality')
        axes[1, 0].set_title('Unlearning Quality')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Privacy-Utility tradeoff
        axes[1, 1].plot(forget_quality, utility_pres, marker='D', label=method)
        axes[1, 1].set_xlabel('Forget Quality (Privacy)')
        axes[1, 1].set_ylabel('Utility Preservation')
        axes[1, 1].set_title('Privacy-Utility Tradeoff')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def save_results(results: Dict, output_path: str = 'benchmark_results.json'):
    """Save benchmark results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    print("Privacy-Utility Tradeoff Benchmark")
    print("=" * 50)
    print("\nNote: This is a template benchmark script.")
    print("Requires: model, data loaders to run complete benchmark.")
    print("\nUsage:")
    print("  from benchmarks.privacy_utility_tradeoff import run_privacy_utility_benchmark")
    print("  results = run_privacy_utility_benchmark(model, forget_loader, retain_loader, test_loader)")
