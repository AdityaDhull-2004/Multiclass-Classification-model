"""Inspect the bundled thyroid dataset: structure, class balance, missingness.

Run from the experiment folder:
    python scripts/inspect_thyroid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from healthml.data import load_thyroid


def main() -> None:
    ds = load_thyroid()
    print(ds.summary())
    print("\nMissing values per feature (top 8):")
    print(ds.X.isna().sum().sort_values(ascending=False).head(8).to_string())
    print("\nFeature dtypes:")
    print(ds.X.dtypes.value_counts().to_string())


if __name__ == "__main__":
    main()
