"""
Model architectures for the Unified Privacy Pipeline.

Contains baseline implementations for:
- Face recognition models
- Health prediction models  
- Privacy-preserving architectures
"""

from .face_models import FaceEncoder, FaceClassifier, PrivacyAwareFaceNet
from .health_models import HealthPredictor, FederatedHealthNet
from .base_models import PrivacyAwareModel, DPModel

__all__ = [
    "FaceEncoder",
    "FaceClassifier", 
    "PrivacyAwareFaceNet",
    "HealthPredictor",
    "FederatedHealthNet",
    "PrivacyAwareModel",
    "DPModel"
]