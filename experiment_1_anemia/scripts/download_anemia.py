"""Fetch the multi-class anemia CBC dataset for Experiment 1.

Origin
------
This is the Kaggle "CBC / anemia types" dataset (`diagnosed_cbc_data_v4.csv`,
1,281 rows, 14 CBC features + a 9-class `Diagnosis` column) used in Experiment 1.
Kaggle downloads require a personal API token (`~/.kaggle/kaggle.json`),
which isn't always available, so by default this script pulls the identical file
from a public GitHub mirror. Both routes yield the same CSV.

Usage
-----
    python scripts/download_anemia.py            # GitHub mirror (no auth needed)
    python scripts/download_anemia.py --kaggle   # official Kaggle API (needs token)

The file is written to data/raw/anemia.csv, where `healthml.data.load_anemia`
finds it automatically.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from healthml.config import RAW_DATA_DIR

GITHUB_RAW = (
    "https://raw.githubusercontent.com/"
    "BhaveshBhakta/Anemia-Type-Prediction-Using-ML/main/diagnosed_cbc_data_v4.csv"
)
# A few public Kaggle slugs that host this dataset (any one works with the API).
KAGGLE_DATASET = "ehababoelnaga/anemia-types-classification"

OUT = RAW_DATA_DIR / "anemia.csv"


def from_github() -> None:
    print(f"Downloading from GitHub mirror:\n  {GITHUB_RAW}")
    urllib.request.urlretrieve(GITHUB_RAW, OUT)  # noqa: S310
    print(f"Wrote {OUT}")


def from_kaggle() -> None:
    """Requires `pip install kaggle` and ~/.kaggle/kaggle.json (or env vars)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Kaggle package/credentials unavailable: {exc}\n"
                 "Run without --kaggle to use the GitHub mirror.")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), unzip=True)
    # The unzipped file may have a different name; normalise to anemia.csv.
    for p in RAW_DATA_DIR.glob("*.csv"):
        if "cbc" in p.name.lower() or "anemia" in p.name.lower():
            p.replace(OUT)
            break
    print(f"Wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kaggle", action="store_true", help="use the official Kaggle API")
    args = ap.parse_args()
    (from_kaggle if args.kaggle else from_github)()

    # Quick sanity check.
    import pandas as pd
    df = pd.read_csv(OUT)
    print(f"shape={df.shape}; columns={list(df.columns)}")


if __name__ == "__main__":
    main()

