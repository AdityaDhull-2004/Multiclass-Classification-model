"""Loader for the Garavan Institute thyroid-disease archive (``thyroid0387``).

Why this dataset (Experiment 2)
-------------------------------
The anemia study has three concrete limitations we deliberately attack here:

1. *Small, single-source data* (1,281 rows). thyroid0387 has **9,172 records**.
2. *Severe class imbalance not handled.* This dataset is ~74% "negative"; we
   treat imbalance explicitly (stratification, class weights, macro metrics).
3. *Missing values ignored.* Clinical panels are ordered selectively, so lab
   values are often missing **informatively** -- we keep that signal.

Source
------
https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/
Each line is ``29 attribute values, diagnoses[record-id]``. The diagnosis is a
string of letters (see ``DIAGNOSIS_LETTERS``); ``-`` means "no condition". A
form ``X|Y`` means "consistent with X but more likely Y" -- we take ``Y``.

The 8 raw letter-groups are folded into **7 classes** (antithyroid-treatment
codes O/P/Q, which are overwhelmingly hyperthyroid management, are folded into
``hyperthyroid``; this keeps every class large enough for reliable per-class
metrics).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from healthml.config import RAW_DATA_DIR
from healthml.data.preprocess import Dataset

_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "thyroid-disease/thyroid0387.data"
)

# Column order exactly as documented in thyroid0387.names (29 attributes).
_COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source", "diagnosis_raw",
]

# Binary clinical-history flags encoded as f/t -> 0/1, treated as numeric.
_BINARY_FLAGS = [
    "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick",
    "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
]

# Continuous lab measurements (the clinically decisive features).
_CONTINUOUS = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

# True nominal columns -> one-hot encoded.
_CATEGORICAL = ["sex", "referral_source"]

# The "*_measured" flags are perfectly collinear with the value's missingness,
# so we drop them and let the imputer's missing-indicator recreate that signal.
# TBG is missing in ~96% of rows -> drop the column entirely.
_DROP = [
    "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured",
    "FTI_measured", "TBG_measured", "TBG",
]

#: First-letter of a diagnosis code -> clinical group (see module docstring).
DIAGNOSIS_LETTERS: dict[str, str] = {
    "-": "negative",
    "A": "hyperthyroid", "B": "hyperthyroid", "C": "hyperthyroid", "D": "hyperthyroid",
    "E": "hypothyroid", "F": "hypothyroid", "G": "hypothyroid", "H": "hypothyroid",
    "I": "binding_protein", "J": "binding_protein",
    "K": "nonthyroidal_illness",
    "L": "replacement_therapy", "M": "replacement_therapy", "N": "replacement_therapy",
    "O": "hyperthyroid", "P": "hyperthyroid", "Q": "hyperthyroid",  # antithyroid mgmt
    "R": "discordant_results", "S": "discordant_results", "T": "discordant_results",
}


def _download(cache: Path) -> Path:
    """Download the raw file once and cache it under data/raw/."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_URL, cache)  # noqa: S310 (trusted UCI URL)
    return cache


def _map_diagnosis(raw: str) -> str:
    """Map a raw diagnosis string to one of the 7 grouped class labels."""
    # Strip the trailing "[record-id]".
    code = raw.split("[")[0].strip()
    # "X|Y" -> "more likely Y".
    if "|" in code:
        code = code.split("|")[-1]
    if not code or code == "-":
        return "negative"
    return DIAGNOSIS_LETTERS.get(code[0], "negative")


def load_thyroid(cache_dir: Path = RAW_DATA_DIR) -> Dataset:
    """Download, clean and return the thyroid dataset as a :class:`Dataset`."""
    path = _download(Path(cache_dir) / "thyroid0387.data")

    df = pd.read_csv(path, header=None, names=_COLUMNS, na_values="?")

    # --- target ---------------------------------------------------------- #
    label = df["diagnosis_raw"].astype(str).map(_map_diagnosis)

    # --- features -------------------------------------------------------- #
    df = df.drop(columns=["diagnosis_raw"] + _DROP)

    # Binary f/t flags -> 0/1 (kept numeric; tree splits handle them directly).
    for col in _BINARY_FLAGS:
        df[col] = df[col].map({"f": 0, "t": 1}).astype("float64")

    # Continuous columns -> numeric; clip a known bad age (>120 is non-physiological).
    for col in _CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["age"] > 120, "age"] = np.nan

    numeric_features = _CONTINUOUS + _BINARY_FLAGS
    categorical_features = _CATEGORICAL
    df = df[numeric_features + categorical_features]

    # --- encode target to integers, ordered by frequency (0 = most common) #
    order = label.value_counts().index.tolist()
    class_to_int = {c: i for i, c in enumerate(order)}
    y = label.map(class_to_int).to_numpy()

    return Dataset(
        X=df.reset_index(drop=True),
        y=y,
        class_names=order,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        name="thyroid",
        description="Garavan Institute thyroid diagnoses (9,172 patients, 7 classes)",
    )


if __name__ == "__main__":  # quick manual check
    ds = load_thyroid()
    print(ds.summary())
