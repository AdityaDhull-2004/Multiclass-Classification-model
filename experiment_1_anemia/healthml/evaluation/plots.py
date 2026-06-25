"""Reusable plotting helpers. All functions save a PNG and return its path.

Uses the non-interactive 'Agg' backend so scripts run headless (no display).
The figures include a confusion matrix and feature importance, plus ROC curves
and calibration.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve, auc  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

from healthml.config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, filename: str) -> Path:
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(class_dist: pd.Series, title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=class_dist.values, y=class_dist.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("count")
    ax.set_title(title)
    for i, v in enumerate(class_dist.values):
        ax.text(v, i, f" {v}", va="center")
    return _save(fig, filename)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    filename: str,
    normalize: bool = False,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 2, 1.0 * len(class_names) + 2))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="magma", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return _save(fig, filename)


def plot_roc_ovr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    title: str,
    filename: str,
) -> Path:
    """One-vs-rest ROC curve per class."""
    classes = np.arange(len(class_names))
    y_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, filename)


def plot_feature_importance(
    importances: pd.Series, title: str, filename: str, top_n: int = 15
) -> Path:
    top = importances.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1.5))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("importance")
    ax.set_title(title)
    return _save(fig, filename)


def plot_model_comparison(
    summary: pd.DataFrame, metric_means: pd.Series, title: str, filename: str
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metric_means.sort_values().plot.barh(ax=ax, color="#55A868")
    ax.set_xlabel("mean CV macro-F1")
    ax.set_title(title)
    for i, v in enumerate(metric_means.sort_values().values):
        ax.text(v, i, f" {v:.3f}", va="center")
    return _save(fig, filename)


def plot_reliability(
    y_true_binary: np.ndarray, y_prob: np.ndarray, title: str, filename: str
) -> Path:
    """Calibration (reliability) curve for one class's probability estimates."""
    frac_pos, mean_pred = calibration_curve(y_true_binary, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(fig, filename)
