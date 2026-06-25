# Project Overview

This project teaches **explainable multi-class disease classification** end-to-end,
twice:

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Disease | Anemia (9 subtypes) | Thyroid disease (7 groups) |
| Data | CBC panel, ~1,281 patients | Garavan `thyroid0387`, 9,172 patients |
| Scope | multi-class anemia classification from CBC data | the larger, harder thyroid study |
| Why bigger matters | — | tackles *small-data*, *imbalance*, and *missing-value* challenges |
| Headline goal | hybrid RF+XGB + SHAP | **deep per-prediction SHAP**: *why is this patient predicted this class?* |

Both experiments run through **one shared, leakage-safe pipeline** so the only thing
that changes is the dataset.

---

## How the code is organised

```
src/healthml/
├── config.py            # paths, seeds, split ratio, CV folds  (single source of truth)
├── data/
│   ├── preprocess.py    # Dataset container, stratified split, ColumnTransformer
│   ├── anemia.py        # Experiment 1 loader (your CSV)
│   └── thyroid.py       # Experiment 2 loader (auto-downloads + groups 7 classes)
├── models/
│   └── factory.py       # RF, XGBoost, hybrid soft-voting, LightGBM, stacking
├── evaluation/
│   ├── metrics.py       # accuracy, macro-F1, balanced-acc, ROC-AUC (OvR)
│   ├── crossval.py      # stratified k-fold returning *fold-wise* scores
│   ├── stats.py         # paired t-test + Wilcoxon  (is the gain real?)
│   └── plots.py         # confusion matrix, ROC, importance, calibration
└── explain/
    └── shap_explain.py  # global SHAP + detailed local "why this prediction"

experiments/
├── _common.py           # the full protocol, run once per dataset
├── exp1_anemia/run.py
└── exp2_thyroid/run.py

docs/                    # <- you are here; conceptual deep-dives
outputs/                 # figures, reports, saved models (generated)
```

The design rule: **each ML concept lives in exactly one small module**, so when a doc
says "stratified splitting", you can open `data/preprocess.py::stratified_split` and
read the 10 lines that do it.

---

## How to run

```powershell
# from the repo root
$env:PYTHONPATH = "src"          # PowerShell  (cmd:  set PYTHONPATH=src)

# Experiment 2 needs no manual data — it downloads thyroid0387 itself:
python experiments/exp2_thyroid/run.py

# Experiment 1 needs the CBC CSV first:
#   put it at  data/raw/anemia.csv  (a diagnosis column + 14 CBC features)
python experiments/exp1_anemia/run.py
```

Each run writes `outputs/reports/<tag>_report.md` (a self-contained results document
with embedded figures) and PNGs under `outputs/figures/`.

---

## Suggested reading order

1. **[01_concepts_ml_pipeline.md](01_concepts_ml_pipeline.md)** — the ML pipeline:
   leakage, stratification, encoding, missing data.
2. **[02_anemia_disease.md](02_anemia_disease.md)** — the medicine behind Experiment 1.
3. **[03_thyroid_disease.md](03_thyroid_disease.md)** — the medicine behind Experiment 2.
4. **[04_models.md](04_models.md)** — Random Forest, XGBoost, soft-voting, stacking (with the math).
5. **[05_evaluation_and_stats.md](05_evaluation_and_stats.md)** — metrics, cross-validation, the paired t-test.
6. **[06_shap_explained.md](06_shap_explained.md)** — Shapley values, TreeSHAP, and reading the plots.
7. **[07_results.md](07_results.md)** — what we actually found, side-by-side.
8. **[08_extensions_and_complexity.md](08_extensions_and_complexity.md)** — what makes Experiment 2 harder, and where to go next.

Every doc cross-references the exact source file and function that implements the idea.
