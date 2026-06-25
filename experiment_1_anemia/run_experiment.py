"""Standalone training + testing pipeline for Experiment 1: anemia (9-class CBC).

This folder is SELF-CONTAINED. The dataset is bundled under data/raw/, the
`healthml` package is vendored locally, and nothing is downloaded. Just run:

    pip install -r requirements.txt
    python run_experiment.py

It performs the full protocol (stratified 5-fold CV of 5 models, paired
significance test, hold-out evaluation, global + per-prediction SHAP) and writes
a self-contained report to outputs/reports/anemia_report.md plus figures under
outputs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the vendored package and the local pipeline module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from healthml.data import load_anemia          # noqa: E402
from pipeline import run_experiment             # noqa: E402


def main() -> None:
    dataset = load_anemia()  # reads the bundled data/raw/anemia.csv
    print(dataset.summary())
    run_experiment(
        dataset,
        tag="anemia",
        title="Experiment 1 -- Anemia from CBC (9 classes)",
        run_extensions=True,
    )


if __name__ == "__main__":
    main()
