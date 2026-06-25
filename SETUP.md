# Setup & Run Guide

This guide explains how to set up and run **both** experiments. They are
independent and self-contained — you can run either one on its own. Nothing is
downloaded at run time; each folder already contains its dataset.

---

## 1. Prerequisites

- **Python 3.10 or newer** (`python --version` to check).
- **pip** (ships with Python).
- ~1 GB free disk and a normal CPU. No GPU is required.
- No internet needed to run (the data is bundled). Internet is only needed once,
  to install the Python libraries.

---

## 2. (Recommended) Create a virtual environment

Keeping dependencies isolated avoids clashes with other projects.

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

The two experiments share the same dependency list. Install from whichever folder
you want to run (the files are identical):

```bash
pip install -r experiment_1_anemia/requirements.txt
```

This installs scikit-learn, XGBoost, LightGBM, SHAP, matplotlib, pandas, scipy and
the few supporting libraries.

---

## 4. Run Experiment 1 — Anemia (≈ 2–4 minutes)

```bash
cd experiment_1_anemia
python run_experiment.py
```

What happens: it loads the bundled CBC dataset (`data/raw/anemia.csv`), trains and
cross-validates five models, runs the significance test, evaluates the hybrid on a
held-out test set, and produces SHAP explanations.

Outputs appear in `experiment_1_anemia/outputs/`:
- `outputs/reports/anemia_report.md` — full results with embedded figures
- `outputs/figures/*.png` — class distribution, model comparison, confusion
  matrices, ROC, SHAP bar / interaction / waterfalls

The polished IEEE write-up is in `experiment_1_anemia/report/report_anemia.pdf`.

---

## 5. Run Experiment 2 — Thyroid (≈ 10–18 minutes)

```bash
cd experiment_2_thyroid
python run_experiment.py
```

Same pipeline on the larger Garavan thyroid archive
(`data/raw/thyroid0387.data`, 9,172 patients). It takes longer mainly because of the
stacking model's nested cross-validation. Outputs land in
`experiment_2_thyroid/outputs/`, and the write-up is
`experiment_2_thyroid/report/report_thyroid.pdf`.

---

## 6. Expected results (fixed random seed = 42)

| Metric (hybrid, hold-out) | Experiment 1 (anemia) | Experiment 2 (thyroid) |
|---|---|---|
| Accuracy | ~0.984 | ~0.945 |
| Macro-F1 | ~0.947 | ~0.863 |
| ROC-AUC (one-vs-rest) | ~0.9997 | ~0.995 |
| Hybrid vs RF (paired t-test) | p ≈ 0.82 (not significant) | p ≈ 0.014 (significant) |

Numbers are reproducible: the seed is fixed, so re-running gives the same results.

---

## 7. Troubleshooting

- **`ModuleNotFoundError`** — make sure you installed the requirements (Step 3) into
  the same Python/venv you are running with.
- **LightGBM fails to load on Linux** — install OpenMP: `sudo apt-get install
  libgomp1`. On macOS: `brew install libomp`.
- **It seems stuck on Experiment 2** — that's normal; the stacking model is slow.
  Give it up to ~20 minutes. Experiment 1 is the quick one to try first.
- **Run from inside the experiment folder** (`cd experiment_1_anemia` first). The
  scripts resolve their own paths, but running from the folder is the tested path.

---

*For an overview of both experiments and the headline findings, see
[README.md](README.md). The head-to-head comparison report is in
[comparison_report/](comparison_report/).*
