"""Standalone training + testing pipeline for Experiment 2: thyroid (7-class).

This folder is SELF-CONTAINED. The dataset is bundled under data/raw/, the
`healthml` package is vendored locally, and nothing is downloaded. Just run:

    pip install -r requirements.txt
    python run_experiment.py

It performs the full protocol (stratified 5-fold CV of 5 models, paired
significance test, hold-out evaluation, global + per-prediction SHAP) and writes
a self-contained report to outputs/reports/thyroid_report.md plus figures under
outputs/figures/. The thyroid dataset is large, so a full run takes several
minutes (the stacking model's nested cross-validation dominates).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the vendored package and the local pipeline module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from healthml.data import load_thyroid         # noqa: E402
from pipeline import run_experiment             # noqa: E402


def main() -> None:
    # Reads the bundled data/raw/thyroid0387.data (no download performed).
    dataset = load_thyroid()
    print(dataset.summary())
    run_experiment(
        dataset,
        tag="thyroid",
        title="Experiment 2 -- Thyroid Disease (7 classes)",
        run_extensions=True,
    )


if __name__ == "__main__":
    main()
