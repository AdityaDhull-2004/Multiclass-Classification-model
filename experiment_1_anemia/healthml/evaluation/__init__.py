"""Metrics, cross-validation, statistical testing and plotting."""

from healthml.evaluation.metrics import (
    classification_metrics,
    per_class_report,
    roc_auc_ovr,
)
from healthml.evaluation.crossval import cross_validate_models, fold_scores
from healthml.evaluation.stats import paired_ttest, describe_significance
from healthml.evaluation import plots

__all__ = [
    "classification_metrics",
    "per_class_report",
    "roc_auc_ovr",
    "cross_validate_models",
    "fold_scores",
    "paired_ttest",
    "describe_significance",
    "plots",
]
