"""
Differential Privacy implementation using Opacus.

Provides privacy-preserving training capabilities including:
- DP-SGD training
- Privacy accounting
- Gradient clipping and noise addition
- Privacy budget management
"""

from .dp_trainer import DPTrainer, DPTrainingConfig
from .privacy_engine_wrapper import PrivacyEngineWrapper
from .noise_mechanisms import GaussianMechanism, LaplaceMechanism
from .accountant import PrivacyAccountantWrapper

__all__ = [
    "DPTrainer",
    "DPTrainingConfig", 
    "PrivacyEngineWrapper",
    "GaussianMechanism",
    "LaplaceMechanism",
    "PrivacyAccountantWrapper"
]