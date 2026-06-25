"""Classification metrics for the evaluation protocol.

We compute:
  * Accuracy                (Eq. 7)
  * Macro F1-score          (Eq. 8-9)  -- the headline metric for imbalance
  * ROC-AUC, one-vs-rest

We add **balanced accuracy** and **weighted F1** because on a 74%-negative
dataset plain accuracy is misleading: predicting "negative" for everyone already
scores ~0.74. Macro-averaged metrics weight every class equally, so a model that
ignores rare diseases is correctly penalised.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def roc_auc_ovr(y_true: np.ndarray, y_proba: np.ndarray, *, average: str = "macro") -> float:
    """One-vs-rest multi-class ROC-AUC.

    Each class is scored as "this class vs all others" and the per-class AUCs are
    macro-averaged. Returns ``nan`` if a class is absent from ``y_true`` (AUC
    undefined). ``y_proba`` must have one column per class, in label order.
    """
    n_classes = y_proba.shape[1]
    classes = np.arange(n_classes)
    y_bin = label_binarize(y_true, classes=classes)
    # Guard: classes missing in this subset make AUC undefined.
    present = y_bin.sum(axis=0) > 0
    if not present.all():
        y_bin = y_bin[:, present]
        y_proba = y_proba[:, present]
    try:
        return float(roc_auc_score(y_bin, y_proba, average=average, multi_class="ovr"))
    except ValueError:
        return float("nan")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the headline scalar metrics in one call."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        out["roc_auc_ovr"] = roc_auc_ovr(y_true, y_proba)
    return out


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    """Human-readable precision/recall/F1 table per class."""
    labels = list(range(len(class_names)))
    return classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0, digits=4
    )
