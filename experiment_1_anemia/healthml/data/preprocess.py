"""Shared preprocessing utilities used by every dataset loader.

Design goals
------------
* **One `Dataset` object** carries everything downstream code needs: the feature
  matrix, the integer-encoded target, the human-readable class names, the feature
  names, and the column types. This keeps the experiment scripts tiny.
* **Leakage-safe by construction.** We never fit an imputer/scaler on the full
  data here; fitting happens *inside* a scikit-learn ``Pipeline`` so it is re-fit
  on each cross-validation training fold only (see ``healthml.models``).
* **Tree-friendly but model-agnostic.** Tree ensembles are invariant to monotone
  feature scaling, so scaling is *optional*. Missing
  values, however, must be handled explicitly; for clinical lab data the very
  fact a test was not ordered can be informative, so we add missingness
  indicators rather than silently imputing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from healthml.config import RANDOM_SEED, TEST_SIZE


@dataclass
class Dataset:
    """A fully-described, model-ready dataset.

    Attributes
    ----------
    X:
        Feature matrix as a :class:`pandas.DataFrame` (column names preserved so
        SHAP plots are human-readable).
    y:
        Integer-encoded target (``0 .. n_classes-1``).
    class_names:
        ``class_names[i]`` is the original label for integer class ``i``.
    numeric_features / categorical_features:
        Column-name lists used to build the preprocessing ``ColumnTransformer``.
    name:
        Short identifier ("anemia" / "thyroid") used for output filenames.
    description:
        One-line human description.
    """

    X: pd.DataFrame
    y: np.ndarray
    class_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    name: str = "dataset"
    description: str = ""

    # -- convenience views ------------------------------------------------- #
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def class_distribution(self) -> pd.Series:
        """Counts per class label, sorted from most to least frequent."""
        counts = pd.Series(self.y).value_counts().sort_index()
        counts.index = [self.class_names[i] for i in counts.index]
        return counts.sort_values(ascending=False)

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.name} -- {self.description}",
            f"  samples : {self.n_samples}",
            f"  features: {self.n_features} "
            f"({len(self.numeric_features)} numeric, "
            f"{len(self.categorical_features)} categorical)",
            f"  classes : {self.n_classes}",
            "  class distribution:",
        ]
        dist = self.class_distribution()
        for label, count in dist.items():
            pct = 100 * count / self.n_samples
            lines.append(f"      {label:<32} {count:>6}  ({pct:4.1f}%)")
        return "\n".join(lines)


def stratified_split(
    dataset: Dataset,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/test split that preserves class proportions.

    Stratification matters enormously for imbalanced multi-class problems: a plain
    random split can leave a rare class with *zero* test samples, making its
    recall undefined. ``stratify=y`` guarantees every class keeps the same
    proportion in both parts (a stratified 80/20 split).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    dataset: Dataset,
    *,
    scale: bool = False,
    add_missing_indicators: bool = True,
) -> ColumnTransformer:
    """Construct a leakage-safe ``ColumnTransformer`` for a dataset.

    Numeric columns  -> median imputation (+ optional missingness flag) (+ optional scaling)
    Categorical cols -> most-frequent imputation -> one-hot encoding

    Because this is returned *unfitted* and placed at the head of a
    :class:`~sklearn.pipeline.Pipeline`, scikit-learn re-fits it on each CV
    training fold, so test-fold statistics never leak into training.
    """
    numeric_steps: list = [
        ("impute", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if dataset.numeric_features:
        transformers.append(("num", numeric_pipe, dataset.numeric_features))
    if dataset.categorical_features:
        transformers.append(("cat", categorical_pipe, dataset.categorical_features))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
