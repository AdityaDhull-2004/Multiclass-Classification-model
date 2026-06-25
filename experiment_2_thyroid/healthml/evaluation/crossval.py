"""Stratified k-fold cross-validation that returns *fold-wise* scores.

We must keep the individual fold scores (not just their mean) for two reasons:
  1. We report ``mean +/- std`` across folds (Eq. 6).
  2. The paired t-test (Eq. 10) compares two models *fold by fold*, so every
     model is evaluated on the **same** fold splits -- that pairing is what makes
     the test "paired" and far more sensitive than comparing independent means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from healthml.config import CV_FOLDS, RANDOM_SEED
from healthml.evaluation.metrics import classification_metrics


def fold_scores(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cv: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "roc_auc_ovr"),
) -> pd.DataFrame:
    """Evaluate one model with stratified k-fold CV; return a (folds x metrics) frame.

    A *fresh clone* of the model is fit on each training fold -- including its
    preprocessor -- so nothing learned on a validation fold leaks back.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rows = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        est = clone(model)
        est.fit(X_tr, y_tr)

        y_pred = est.predict(X_val)
        y_proba = est.predict_proba(X_val) if hasattr(est, "predict_proba") else None
        scored = classification_metrics(y_val, y_pred, y_proba)
        rows.append({m: scored.get(m, float("nan")) for m in metrics})
    return pd.DataFrame(rows, index=[f"fold{i + 1}" for i in range(cv)])


def cross_validate_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cv: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "roc_auc_ovr"),
) -> dict[str, pd.DataFrame]:
    """Run :func:`fold_scores` for several models on identical fold splits.

    Returns a dict ``name -> (folds x metrics) DataFrame``. Because the same
    ``seed`` drives ``StratifiedKFold`` for every model, fold *i* contains the
    same patients across models -> valid paired comparison.
    """
    return {
        name: fold_scores(model, X, y, cv=cv, seed=seed, metrics=metrics)
        for name, model in models.items()
    }


def summarize_cv(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collapse fold scores into a ``mean +/- std`` summary table per model."""
    rows = {}
    for name, df in results.items():
        rows[name] = {
            f"{col}": f"{df[col].mean():.4f} +/- {df[col].std(ddof=1):.4f}"
            for col in df.columns
        }
    return pd.DataFrame(rows).T
