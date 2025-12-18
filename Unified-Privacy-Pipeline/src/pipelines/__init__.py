"""
Unified Privacy Pipeline Core Module

This module contains the main pipeline implementations that integrate:
- Machine unlearning algorithms
- Influence function estimation
- Differential privacy mechanisms
"""

from .unified_pipeline import UnifiedPrivacyPipeline
from .face_recognition_pipeline import FaceRecognitionPipeline
from .health_prediction_pipeline import HealthPredictionPipeline

__all__ = [
    "UnifiedPrivacyPipeline",
    "FaceRecognitionPipeline", 
    "HealthPredictionPipeline"
]