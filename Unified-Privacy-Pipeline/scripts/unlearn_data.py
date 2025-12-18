#!/usr/bin/env python3
"""
Unlearning script for removing specific data from trained models.

Usage:
    python scripts/unlearn_data.py --model models/model.pth --method gradient_ascent --iterations 100
"""

import argparse
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from machine_unlearning.unlearning_methods import create_unlearner, UnlearningConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Unlearn specific data from trained models')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--method', type=str, default='gradient_ascent',
                       choices=['gradient_ascent', 'influence_based', 'fine_tuning', 'negative_gradient'],
                       help='Unlearning method to use')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Maximum unlearning iterations')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for unlearning')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for unlearned model')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        base_name = os.path.splitext(args.model)[0]
        args.output = f"{base_name}_unlearned.pth"

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Unlearning from model: {args.model}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Device: {device}")

    # Create unlearning configuration
    config = UnlearningConfig(
        learning_rate=args.learning_rate,
        max_iterations=args.iterations
    )

    # Create unlearner
    unlearner = create_unlearner(args.method, config)

    logger.info("\n" + "="*50)
    logger.info("Unlearning Configuration:")
    logger.info("="*50)
    logger.info(f"Method: {args.method}")
    logger.info(f"Max iterations: {args.iterations}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Note: Actual unlearning requires model and data loaders
    logger.info("\n" + "="*50)
    logger.info("Note: Full unlearning requires:")
    logger.info("  1. Loaded model")
    logger.info("  2. Forget data loader (data to unlearn)")
    logger.info("  3. Retain data loader (data to keep)")
    logger.info("="*50)

    logger.info(f"\nUnlearned model will be saved to: {args.output}")
    logger.info("\nSee documentation for complete usage examples.")


if __name__ == "__main__":
    main()
