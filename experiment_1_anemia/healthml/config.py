"""Central configuration: paths, random seeds and shared constants.

Keeping every "magic value" in one module is itself a reproducibility practice.
We enforce *fixed random seeds* and *no data leakage*; both are enforced here and
reused everywhere so an experiment cannot accidentally use a different seed or
split ratio than another.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# Standalone layout: config.py lives at <folder>/healthml/config.py  -> parents[1]
# is the experiment folder, which is this self-contained project's root.
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Evaluation protocol
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.20          # stratified 80/20 hold-out
CV_FOLDS: int = 5                # stratified 5-fold cross-validation
SIGNIFICANCE_ALPHA: float = 0.05  # paired t-test threshold
