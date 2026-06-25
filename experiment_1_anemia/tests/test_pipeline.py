"""Fast, network-free tests for the core pipeline.

Run from the repo root with:  PYTHONPATH=src python -m pytest tests/ -q
(or simply:  PYTHONPATH=src python tests/test_pipeline.py)

These build a tiny in-memory synthetic Dataset so the whole stack -- preprocessing,
the hybrid model, metrics, cross-validation, the paired t-test, and SHAP local
explanations -- is exercised without downloading anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from healthml.data.preprocess import Dataset, stratified_split, build_preprocessor
from healthml.models import build_hybrid, build_model_zoo
from healthml.evaluation.metrics import classification_metrics
from healthml.evaluation.crossval import cross_validate_models
from healthml.evaluation.stats import paired_ttest
from healthml.explain import ShapExplainer


def _toy_dataset(n_per_class: int = 80, seed: int = 0) -> Dataset:
    """Three well-separated classes with a numeric and a missing-prone feature."""
    rng = np.random.default_rng(seed)
    centers = {0: (0, 0), 1: (5, 5), 2: (0, 5)}
    rows, y = [], []
    for cls, (a, b) in centers.items():
        for _ in range(n_per_class):
            v = rng.normal(b, 1.0)
            rows.append({"feat_a": rng.normal(a, 1.0), "feat_b": v,
                         "cat": "x" if rng.random() > 0.5 else "y"})
            y.append(cls)
    X = pd.DataFrame(rows)
    X.loc[X.sample(frac=0.1, random_state=seed).index, "feat_b"] = np.nan  # missingness
    return Dataset(X=X, y=np.array(y), class_names=["A", "B", "C"],
                   numeric_features=["feat_a", "feat_b"], categorical_features=["cat"],
                   name="toy", description="synthetic 3-class")


def test_preprocessor_handles_missing_and_categorical():
    ds = _toy_dataset()
    prep = build_preprocessor(ds)
    Xt = prep.fit_transform(ds.X)
    assert not np.isnan(Xt).any(), "imputation should remove all NaNs"
    # numeric(2) + missing-indicator(1 for feat_b) + one-hot(cat -> 2) = 5 columns
    assert Xt.shape[1] >= 4


def test_hybrid_fits_and_scores_well():
    ds = _toy_dataset()
    X_tr, X_te, y_tr, y_te = stratified_split(ds, test_size=0.25)
    model = build_hybrid(ds)
    model.fit(X_tr, y_tr)
    m = classification_metrics(y_te, model.predict(X_te), model.predict_proba(X_te))
    assert m["macro_f1"] > 0.8, m
    assert 0.0 <= m["roc_auc_ovr"] <= 1.0


def test_cv_and_paired_ttest_align_on_folds():
    ds = _toy_dataset()
    zoo = build_model_zoo(ds, include_extensions=False)
    res = cross_validate_models(zoo, ds.X, ds.y, cv=3)
    assert set(res) == set(zoo)
    for df in res.values():
        assert df.shape[0] == 3  # one row per fold
    sig = paired_ttest(res["Hybrid (RF+XGB)"]["macro_f1"].to_numpy(),
                       res["Random Forest"]["macro_f1"].to_numpy())
    assert sig.n == 3
    assert 0.0 <= sig.p_value <= 1.0


def test_shap_local_explanation_is_additive():
    ds = _toy_dataset()
    X_tr, X_te, y_tr, y_te = stratified_split(ds, test_size=0.25)
    model = build_hybrid(ds)
    model.fit(X_tr, y_tr)
    expl = ShapExplainer(model, X_te.reset_index(drop=True), ds.class_names)
    le = expl.explain_instance(0, y_true=np.asarray(y_te))
    # SHAP additivity: base value + sum(contributions) == final score (Eq. 11).
    assert abs((le.base_value + le.contributions["shap"].sum()) - le.final_score) < 1e-4
    assert le.predicted_label in ds.class_names
    assert "predicted:" in le.narrative


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("ALL TESTS PASSED")

