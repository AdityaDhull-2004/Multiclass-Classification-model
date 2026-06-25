"""Loader for the multi-class anemia CBC dataset (Experiment 1).

This study uses an open CBC dataset of 1,281 patients, 14 numeric features and
9 diagnostic classes. You supply the CSV; this loader applies the following
preprocessing:

* auto-detect the diagnosis/target column (named ``Diagnosis``/``class``/... or
  the last column as a fallback);
* treat every remaining column as a numeric CBC feature;
* **remove duplicate records** (to prevent performance inflation);
* label-encode the diagnosis.

Expected columns (typical CBC panel; order/case do not matter)::

    WBC, LYMp, NEUTp, LYMn, NEUTn, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, PDW, PCT, Diagnosis
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from healthml.config import RAW_DATA_DIR
from healthml.data.preprocess import Dataset

#: Default location to look for the user-supplied CSV.
DEFAULT_PATHS = [
    RAW_DATA_DIR / "anemia.csv",
    RAW_DATA_DIR / "cbc.csv",
    RAW_DATA_DIR / "diagnosed_cbc_data_v4.csv",
]

#: Column-name fragments that identify the target, in priority order.
_TARGET_HINTS = ["diagnosis", "all_class", "class", "label", "target", "type"]


def _find_csv(path: str | Path | None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Anemia CSV not found at {p}")
        return p
    for cand in DEFAULT_PATHS:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No anemia CSV found. Place the dataset at "
        f"'{RAW_DATA_DIR / 'anemia.csv'}' (or pass path=...). Expected a CBC "
        "table with a diagnosis column and 14 numeric features."
    )


def _detect_target(df: pd.DataFrame) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for hint in _TARGET_HINTS:
        for low, original in lower.items():
            if hint in low:
                return original
    # Fallback: the last column.
    return df.columns[-1]


def load_anemia(path: str | Path | None = None) -> Dataset:
    """Load and clean the anemia CBC dataset into a :class:`Dataset`."""
    csv = _find_csv(path)
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]

    target_col = _detect_target(df)

    # Drop exact duplicate records to avoid leakage/inflation.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)

    y_raw = df[target_col].astype(str).str.strip()
    X = df.drop(columns=[target_col]).copy()

    # Coerce all feature columns to numeric (CBC parameters are continuous).
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    numeric_features = list(X.columns)

    order = y_raw.value_counts().index.tolist()
    class_to_int = {c: i for i, c in enumerate(order)}
    y = y_raw.map(class_to_int).to_numpy()

    desc = f"CBC anemia panel ({len(df)} patients, {len(order)} classes"
    if dropped:
        desc += f"; dropped {dropped} duplicates"
    desc += ")"

    return Dataset(
        X=X.reset_index(drop=True),
        y=y,
        class_names=order,
        numeric_features=numeric_features,
        categorical_features=[],
        name="anemia",
        description=desc,
    )


if __name__ == "__main__":
    print(load_anemia().summary())
