#!/usr/bin/env python3
"""
Unified Privacy Pipeline - FastAPI Backend
Real-time privacy-preserving ML server with all pipeline integrations.
"""

import sys
import os
sys.path.insert(0, 'src')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import time
import logging
from threading import Thread, Lock
from typing import Optional, Dict, List, Any

# Differential Privacy imports
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available - Differential Privacy will be disabled")

# Import privacy pipeline modules
from datasets.real_data_loaders import create_unlearning_loaders
from machine_unlearning.unlearning_methods import create_unlearner, UnlearningConfig
from evaluation.privacy_metrics import MembershipInferenceAttack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Privacy Pipeline API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_state = {
    'status': 'idle',
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'metrics': {},
    'logs': [],
    'model': None,
    'task': None
}
state_lock = Lock()


# Models
class SimpleFaceNet(nn.Module):
    """Optimized CNN for face recognition - faster for demos."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 - Streamlined
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2 - Streamlined
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3 - Streamlined
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Classifier - Smaller for speed
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.features(x)


class SimpleHealthNet(nn.Module):
    """Enhanced deep network for health prediction with LayerNorm (DP-compatible) and Dropout."""
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Input -> 128
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 128 -> 256
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 256 -> 128
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 4: 128 -> 64
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# Request models
class TrainRequest(BaseModel):
    task: str = 'face_recognition'
    epochs: int = 5
    use_dp: bool = False


class UnlearnRequest(BaseModel):
    method: str = 'gradient_ascent'
    iterations: int = 10


# Helper functions
def add_log(message: str, level: str = 'info'):
    """Add a log entry."""
    timestamp = time.strftime('%H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    with state_lock:
        training_state['logs'].append(log_entry)
        if len(training_state['logs']) > 100:
            training_state['logs'] = training_state['logs'][-100:]
    logger.info(f"[{level.upper()}] {message}")


def update_progress(status=None, progress=None, epoch=None, loss=None, metrics=None):
    """Update training progress."""
    with state_lock:
        if status:
            training_state['status'] = status
        if progress is not None:
            training_state['progress'] = progress
        if epoch is not None:
            training_state['current_epoch'] = epoch
        if loss is not None:
            training_state['current_loss'] = loss
        if metrics:
            training_state['metrics'].update(metrics)


def train_face_recognition(epochs=5, use_dp=False):
    """Train face recognition model - optimized for live demos."""
    try:
        add_log("Starting face recognition training (demo mode)...", "info")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        add_log(f"Using device: {device}", "info")

        # Load data - smaller dataset for faster demo
        add_log("Loading LFW dataset (optimized for demo)...", "info")
        forget_loader, retain_loader, test_loader = create_unlearning_loaders(
            dataset_type='lfw', forget_ratio=0.1, batch_size=64  # Larger batch for speed
        )

        base_dataset = forget_loader.dataset.dataset
        num_classes = len(base_dataset.person_to_id)
        add_log(f"Dataset loaded: {num_classes} unique people", "success")

        # Create model
        add_log("Creating optimized model...", "info")
        model = SimpleFaceNet(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Higher LR for faster convergence
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Simpler scheduler
        criterion = nn.CrossEntropyLoss()

        # Use fewer epochs for fast demo - still maintains good accuracy
        epochs = min(epochs, 8)  # Cap at 8 epochs for demo speed

        update_progress(status='training', epoch=0, progress=0)
        training_state['total_epochs'] = epochs
        training_state['model'] = model
        training_state['task'] = 'face_recognition'

        # Training loop - train on all batches
        add_log(f"Training for {epochs} epochs with improved architecture...", "info")
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            correct = 0
            total = 0

            for images, labels in retain_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Track accuracy during training
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            train_acc = 100. * correct / total if total > 0 else 0
            progress = int((epoch + 1) / epochs * 100)

            # Adjust learning rate
            scheduler.step()

            update_progress(
                epoch=epoch + 1,
                loss=avg_loss,
                progress=progress
            )

            # Log every epoch for better demo feedback
            add_log(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1f}%", "info")

        # Evaluation
        add_log("Evaluating model...", "info")
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total if total > 0 else 0

        update_progress(
            status='completed',
            progress=100,
            metrics={'test_accuracy': accuracy / 100}
        )

        add_log(f"Training completed! Test accuracy: {accuracy:.2f}%", "success")

    except Exception as e:
        add_log(f"Training failed: {str(e)}", "error")
        update_progress(status='error')
        logger.exception("Training error:")


def train_health_prediction(epochs=50, use_dp=False):
    """Train health prediction model - optimized for high accuracy."""
    try:
        add_log("Starting health prediction training (high accuracy mode)...", "info")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        add_log(f"Using device: {device}", "info")

        # Load data with smaller batches for better generalization
        add_log("Loading heart disease dataset...", "info")
        forget_loader, retain_loader, test_loader = create_unlearning_loaders(
            dataset_type='heart', forget_ratio=0.1, batch_size=16
        )
        add_log(f"Dataset loaded: {len(retain_loader.dataset)} training samples", "success")

        # Create enhanced model
        add_log("Creating enhanced health prediction model...", "info")
        model = SimpleHealthNet(input_dim=13, output_dim=2, hidden_dim=128).to(device)

        # Better optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Apply Differential Privacy if requested
        if use_dp and OPACUS_AVAILABLE:
            add_log("Applying Differential Privacy with ε=1.0, δ=1e-5...", "info")
            privacy_engine = PrivacyEngine()

            model, optimizer, retain_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=retain_loader,
                epochs=epochs,
                target_epsilon=1.0,
                target_delta=1e-5,
                max_grad_norm=1.0,
            )
            add_log("Differential Privacy enabled successfully", "success")
        elif use_dp and not OPACUS_AVAILABLE:
            add_log("DP requested but Opacus not installed - training without DP", "warning")

        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        criterion = nn.CrossEntropyLoss()

        # Use more epochs for better accuracy (no time limit)
        epochs = min(epochs, 100)  # Cap at 100 for safety

        update_progress(status='training', epoch=0, progress=0)
        training_state['total_epochs'] = epochs
        training_state['model'] = model
        training_state['task'] = 'health_prediction'

        # Training loop with validation tracking
        add_log(f"Training for {epochs} epochs (high accuracy mode)...", "info")
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15

        for epoch in range(epochs):
            # Training phase
            model.train()
            total_loss = 0
            num_batches = 0
            correct_train = 0
            total_train = 0

            for features, labels in retain_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Track training accuracy
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            train_acc = 100. * correct_train / total_train if total_train > 0 else 0

            # Adjust learning rate based on loss
            scheduler.step(avg_loss)

            # Early stopping check
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                add_log(f"Early stopping at epoch {epoch + 1} - no improvement", "info")
                break

            progress = int((epoch + 1) / epochs * 100)

            update_progress(
                epoch=epoch + 1,
                loss=avg_loss,
                progress=progress
            )

            # Log every 5 epochs or last epoch
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                add_log(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1f}%", "info")

        # Evaluation
        add_log("Evaluating model...", "info")
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total if total > 0 else 0

        update_progress(
            status='completed',
            progress=100,
            metrics={'test_accuracy': accuracy / 100}
        )

        add_log(f"Training completed! Test accuracy: {accuracy:.2f}%", "success")

    except Exception as e:
        add_log(f"Training failed: {str(e)}", "error")
        update_progress(status='error')
        logger.exception("Training error:")


def perform_unlearning(method='gradient_ascent', iterations=10):
    """Perform machine unlearning."""
    try:
        add_log(f"Starting unlearning with {method}...", "info")
        update_progress(status='unlearning', progress=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = training_state.get('model')
        task = training_state.get('task')

        if model is None:
            raise ValueError("No trained model found. Train a model first.")

        # Unwrap model from PrivacyEngine if it was trained with DP
        # This is necessary because Opacus wraps the model and is incompatible with unlearning
        if hasattr(model, '_module'):
            add_log("Unwrapping model from Differential Privacy engine...", "info")
            model = model._module

        # Load dataset
        add_log("Loading dataset for unlearning...", "info")
        if task == 'face_recognition':
            forget_loader, retain_loader, _ = create_unlearning_loaders(
                dataset_type='lfw', forget_ratio=0.1, batch_size=16
            )
        else:
            forget_loader, retain_loader, _ = create_unlearning_loaders(
                dataset_type='heart', forget_ratio=0.1, batch_size=16
            )

        # Perform unlearning with optimized config
        add_log("Unlearning in progress...", "info")
        config = UnlearningConfig(
            max_iterations=iterations,
            patience=3,
            learning_rate=0.001,  # Higher learning rate for faster convergence
            gradient_clipping=1.0
        )
        unlearner = create_unlearner(method, config)
        results = unlearner.unlearn(model, forget_loader, retain_loader, device)

        # Update the model in training_state with unwrapped version
        with state_lock:
            training_state['model'] = model

        forget_acc = results['final_metrics']['forget_accuracy']
        retain_acc = results['final_metrics']['retain_accuracy']
        forget_quality = results['forget_quality']

        update_progress(
            status='completed',
            progress=100,
            metrics={
                'forget_accuracy': forget_acc,
                'retain_accuracy': retain_acc,
                'forget_quality': forget_quality
            }
        )

        add_log(f"Unlearning completed!", "success")
        add_log(f"Forget quality: {forget_quality:.2%}", "success")

    except Exception as e:
        add_log(f"Unlearning failed: {str(e)}", "error")
        update_progress(status='error')
        logger.exception("Unlearning error:")


def evaluate_privacy():
    """Evaluate privacy with MIA."""
    try:
        add_log("Starting privacy evaluation...", "info")
        update_progress(status='evaluating', progress=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = training_state.get('model')
        task = training_state.get('task')

        if model is None:
            raise ValueError("No trained model found. Train a model first.")

        # Load dataset
        add_log("Preparing attack data...", "info")
        if task == 'face_recognition':
            forget_loader, retain_loader, _ = create_unlearning_loaders(
                dataset_type='lfw', forget_ratio=0.1, batch_size=32
            )
        else:
            forget_loader, retain_loader, _ = create_unlearning_loaders(
                dataset_type='heart', forget_ratio=0.1, batch_size=32
            )

        # Perform MIA
        add_log("Training attack model...", "info")
        mia = MembershipInferenceAttack()
        features, labels = mia.prepare_attack_data(model, forget_loader, retain_loader, device)
        mia.train_attack_model(features, labels)

        add_log("Evaluating attack...", "info")
        attack_result = mia.evaluate_attack(model, forget_loader, retain_loader, device)

        attack_acc = attack_result.attack_accuracy
        privacy_protection = (1 - (attack_acc - 0.5) * 2)

        update_progress(
            status='completed',
            progress=100,
            metrics={
                'attack_accuracy': attack_acc,
                'privacy_protection': privacy_protection
            }
        )

        add_log(f"Privacy evaluation completed!", "success")
        add_log(f"Privacy protection: {privacy_protection*100:.1f}%", "success")

    except Exception as e:
        add_log(f"Privacy evaluation failed: {str(e)}", "error")
        update_progress(status='error')
        logger.exception("Privacy evaluation error:")


# API Routes
@app.get("/")
def root():
    return {"message": "Unified Privacy Pipeline API", "status": "running"}


@app.get("/status")
def get_status():
    """Get current status."""
    with state_lock:
        # Return a copy to avoid race conditions during JSON serialization
        return {
            'status': training_state['status'],
            'progress': training_state['progress'],
            'current_epoch': training_state['current_epoch'],
            'total_epochs': training_state['total_epochs'],
            'current_loss': training_state['current_loss'],
            'metrics': dict(training_state['metrics']),  # Create a copy
            'logs': list(training_state['logs']),  # Create a copy
        }


@app.get("/logs")
def get_logs():
    """Get logs."""
    with state_lock:
        return {"logs": training_state['logs']}


@app.post("/train")
def start_training(request: TrainRequest):
    """Start training."""
    with state_lock:
        training_state['status'] = 'idle'
        training_state['progress'] = 0
        training_state['current_epoch'] = 0
        training_state['total_epochs'] = 0
        training_state['current_loss'] = 0.0
        training_state['metrics'] = {}
        training_state['logs'] = []

    def run_training():
        if request.task == 'face_recognition':
            train_face_recognition(epochs=request.epochs, use_dp=request.use_dp)
        elif request.task == 'health_prediction':
            train_health_prediction(epochs=request.epochs, use_dp=request.use_dp)

    thread = Thread(target=run_training)
    thread.daemon = True
    thread.start()

    return {"success": True, "message": "Training started"}


@app.post("/unlearn")
def start_unlearning(request: UnlearnRequest):
    """Start unlearning."""
    def run_unlearning():
        perform_unlearning(method=request.method, iterations=request.iterations)

    thread = Thread(target=run_unlearning)
    thread.daemon = True
    thread.start()

    return {"success": True, "message": "Unlearning started"}


@app.post("/evaluate")
def start_evaluation():
    """Start privacy evaluation."""
    def run_evaluation():
        evaluate_privacy()

    thread = Thread(target=run_evaluation)
    thread.daemon = True
    thread.start()

    return {"success": True, "message": "Privacy evaluation started"}


@app.post("/reset")
def reset_state():
    """Reset state."""
    with state_lock:
        training_state['status'] = 'idle'
        training_state['progress'] = 0
        training_state['current_epoch'] = 0
        training_state['total_epochs'] = 0
        training_state['current_loss'] = 0.0
        training_state['metrics'] = {}
        training_state['logs'] = []
        training_state['model'] = None
        training_state['task'] = None

    return {"success": True, "message": "State reset"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("UNIFIED PRIVACY PIPELINE - FASTAPI BACKEND")
    print("=" * 70)
    print("\nStarting server...")
    print("API documentation: http://localhost:8000/docs")
    print("API base URL: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
