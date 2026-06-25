"""Model factory: base learners, the hybrid, and stronger extensions.

Every model is returned as a scikit-learn :class:`~sklearn.pipeline.Pipeline`
whose first step is the dataset's preprocessor. Two reasons:

* **No leakage.** Imputation/encoding are *fit inside* the pipeline, so during
  cross-validation they only ever see the training fold.
* **One object to fit/predict/serialize.** Experiment code stays trivial.

The "hybrid" is a soft-voting ensemble of Random Forest and XGBoost
(Eq. 5: ``y_hat = argmax_c (1/M) * sum_m P_m(c|x)``). We add LightGBM and a
stacking meta-learner as the "more complex" extension.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from healthml.config import RANDOM_SEED
from healthml.data.preprocess import Dataset, build_preprocessor


# --------------------------------------------------------------------------- #
# Base learners (classifier objects, *without* the preprocessor)
# --------------------------------------------------------------------------- #
def _rf(seed: int, balanced: bool) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced_subsample" if balanced else None,
    )


def _xgb(seed: int) -> XGBClassifier:
    # XGBoost has no class_weight; imbalance is handled via per-sample weights
    # passed at fit time (see crossval / experiment runners).
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def _lgbm(seed: int, balanced: bool) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced" if balanced else None,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #
def make_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    """Glue a preprocessor and a classifier into one estimator."""
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def _prep(dataset: Dataset, scale: bool = False) -> ColumnTransformer:
    return build_preprocessor(dataset, scale=scale)


def build_random_forest(dataset: Dataset, seed: int = RANDOM_SEED, balanced: bool = False) -> Pipeline:
    """Random Forest baseline (Eq. 2: f_RF(x) = (1/T) sum_t h_t(x))."""
    return make_pipeline(_prep(dataset), _rf(seed, balanced))


def build_xgboost(dataset: Dataset, seed: int = RANDOM_SEED) -> Pipeline:
    """XGBoost gradient-boosted trees (Eq. 3 regularised objective)."""
    return make_pipeline(_prep(dataset), _xgb(seed))


def build_lightgbm(dataset: Dataset, seed: int = RANDOM_SEED, balanced: bool = False) -> Pipeline:
    """LightGBM -- a faster leaf-wise booster (Experiment 2 extension)."""
    return make_pipeline(_prep(dataset), _lgbm(seed, balanced))


def build_hybrid(
    dataset: Dataset,
    seed: int = RANDOM_SEED,
    balanced: bool = False,
    weights: tuple[float, float] | None = None,
) -> Pipeline:
    """The hybrid: soft-voting RF + XGBoost.

    Soft voting averages the two classifiers' posterior probabilities and takes
    the arg-max (Eq. 5). The preprocessor sits *once* in front of the voter, so
    both base learners see identical transformed features, with the feature
    input fed simultaneously to two models.
    """
    voter = VotingClassifier(
        estimators=[("rf", _rf(seed, balanced)), ("xgb", _xgb(seed))],
        voting="soft",
        weights=list(weights) if weights else None,
        n_jobs=None,  # base learners already parallelise internally
    )
    return make_pipeline(_prep(dataset), voter)


def build_stacking(dataset: Dataset, seed: int = RANDOM_SEED, balanced: bool = False) -> Pipeline:
    """Stacking ensemble: RF + XGB + LightGBM with a logistic meta-learner.

    Unlike soft-voting (which *averages* probabilities with fixed weights),
    stacking *learns* how to combine the base learners' out-of-fold predictions,
    and can discover that e.g. XGBoost is more trustworthy for one class while
    RF is better for another. This is the "more complex" ensemble.
    """
    base = [
        ("rf", _rf(seed, balanced)),
        ("xgb", _xgb(seed)),
        ("lgbm", _lgbm(seed, balanced)),
    ]
    meta = LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None)
    stack = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )
    return make_pipeline(_prep(dataset), stack)


def build_model_zoo(
    dataset: Dataset,
    seed: int = RANDOM_SEED,
    balanced: bool = False,
    include_extensions: bool = True,
) -> dict[str, Pipeline]:
    """Return the standard set of models keyed by display name.

    We compare *Random Forest* (baseline) vs *Hybrid*. The extensions add
    LightGBM and stacking for the more-complex study.
    """
    zoo: dict[str, Pipeline] = {
        "Random Forest": build_random_forest(dataset, seed, balanced),
        "XGBoost": build_xgboost(dataset, seed),
        "Hybrid (RF+XGB)": build_hybrid(dataset, seed, balanced),
    }
    if include_extensions:
        zoo["LightGBM"] = build_lightgbm(dataset, seed, balanced)
        zoo["Stacking (RF+XGB+LGBM)"] = build_stacking(dataset, seed, balanced)
    return zoo


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced per-sample weights = n_samples / (n_classes * class_count).

    Used to give XGBoost (which lacks ``class_weight``) imbalance awareness.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n, k = len(y), len(classes)
    return np.array([n / (k * freq[c]) for c in y], dtype=float)
