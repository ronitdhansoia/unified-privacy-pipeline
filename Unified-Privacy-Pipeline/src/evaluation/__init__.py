"""
Evaluation framework for privacy-preserving machine learning.

Contains metrics and tools for evaluating:
- Privacy protection (membership inference, model extraction)
- Utility preservation (accuracy, fairness)
- Unlearning effectiveness
- Influence function accuracy
"""

from .privacy_metrics import (
    MembershipInferenceAttack,
    ModelExtractionAttack,
    ReconstructionAttack,
    PrivacyAccountant
)

from .utility_metrics import (
    AccuracyMetrics,
    FairnessMetrics, 
    UtilityPreservation
)

from .unlearning_metrics import (
    UnlearningEvaluator,
    ForgetQuality,
    RetentionQuality
)

from .influence_metrics import (
    InfluenceAccuracy,
    InfluenceConsistency
)

__all__ = [
    "MembershipInferenceAttack",
    "ModelExtractionAttack", 
    "ReconstructionAttack",
    "PrivacyAccountant",
    "AccuracyMetrics",
    "FairnessMetrics",
    "UtilityPreservation",
    "UnlearningEvaluator",
    "ForgetQuality",
    "RetentionQuality",
    "InfluenceAccuracy",
    "InfluenceConsistency"
]