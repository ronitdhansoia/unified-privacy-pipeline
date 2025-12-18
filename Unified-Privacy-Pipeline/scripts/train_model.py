#!/usr/bin/env python3
"""
Training script for privacy-preserving models.

Usage:
    python scripts/train_model.py --task face_recognition --epochs 50 --use-dp
    python scripts/train_model.py --task health_prediction --batch-size 32
"""

import argparse
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from pipelines.face_recognition_pipeline import FaceRecognitionPipeline
from pipelines.health_prediction_pipeline import HealthPredictionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train privacy-preserving models')

    parser.add_argument('--task', type=str, required=True,
                       choices=['face_recognition', 'health_prediction'],
                       help='Task to train on')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--use-dp', action='store_true',
                       help='Use differential privacy')
    parser.add_argument('--epsilon', type=float, default=2.0,
                       help='Privacy budget (epsilon)')
    parser.add_argument('--delta', type=float, default=1e-5,
                       help='Privacy parameter (delta)')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for saved models')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    logger.info(f"Training task: {args.task}")
    logger.info(f"Differential Privacy: {'Enabled' if args.use_dp else 'Disabled'}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize pipeline based on task
    if args.task == 'face_recognition':
        logger.info("Initializing Face Recognition Pipeline...")
        pipeline = FaceRecognitionPipeline(
            use_differential_privacy=args.use_dp,
            epsilon=args.epsilon,
            delta=args.delta,
            device=device
        )

        # Train the pipeline
        logger.info("Starting training...")
        results = pipeline.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        # Save model
        model_path = os.path.join(args.output_dir, 'face_recognition_model.pth')
        pipeline.save_model(model_path)

    elif args.task == 'health_prediction':
        logger.info("Initializing Health Prediction Pipeline...")
        pipeline = HealthPredictionPipeline(
            use_differential_privacy=args.use_dp,
            epsilon=args.epsilon,
            delta=args.delta,
            device=device
        )

        # Train the pipeline
        logger.info("Starting training...")
        results = pipeline.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        # Save model
        model_path = os.path.join(args.output_dir, 'health_prediction_model.pth')
        pipeline.save_model(model_path)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Training Results:")
    logger.info("="*50)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    logger.info(f"\nModel saved to: {model_path}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
