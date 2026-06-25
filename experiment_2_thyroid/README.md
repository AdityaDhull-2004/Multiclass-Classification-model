# Experiment 2 — Explainable Multi-Class Thyroid-Disease Classification

**Author:** Aditya Dhull · `adityadhull@iisc.ac.in`

A **fully self-contained** project that classifies a patient into **7 thyroid
diagnostic groups** from a laboratory + clinical-history panel, using a hybrid
Random-Forest + XGBoost soft-voting ensemble, and explains every prediction with
SHAP. This is the larger, harder companion to Experiment 1 — 9,172 patients,
severe class imbalance, and informative missing lab values. Everything needed —
data, source code, pipeline, tests, docs and the report — lives in this folder, and
it runs **offline** (nothing is downloaded).

---

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py
```

A full run takes ~10–18 minutes (the stacking model's nested cross-validation
dominates). Results land in `outputs/`. Requires Python 3.10+.

---

## Folder contents

```
experiment_2_thyroid/
├── run_experiment.py            # entry point — runs the whole pipeline
├── pipeline.py                  # the full protocol (CV, stats, eval, SHAP, model save)
├── healthml/                    # the SOURCE package (data · models · evaluation · explain)
├── data/raw/thyroid0387.data    # the bundled Garavan thyroid archive (9,172 records)
├── docs/                        # conceptual deep-dives (ML pipeline, models, SHAP, results…)
├── scripts/                     # helper scripts (dataset inspection)
├── tests/                       # network-free unit tests for the pipeline
├── report/                      # the IEEE report (PDF + LaTeX + figures)
├── outputs/                     # GENERATED on run: figures, report, saved model
└── requirements.txt
```

- **Source code** is the `healthml/` package (this is the "src").
- **Run the tests:** `python tests/test_pipeline.py`
- **Inspect the raw data:** `python scripts/inspect_thyroid.py`

---

## The dataset

`data/raw/thyroid0387.data` — the Garavan Institute `thyroid0387` archive: 9,172
records, 29 attributes, a diagnosis string. The loader maps the raw codes into
**7 clinically coherent classes** (negative, hypothyroid, hyperthyroid,
binding-protein anomaly, non-thyroidal illness, replacement therapy, discordant
results), keeps 6 continuous labs (TSH, T3, TT4, T4U, FTI, age), 14 binary
clinical-history flags and 2 nominal categoricals, and handles **informative
missingness** (e.g. T3 missing in ~28% of records) with median imputation **plus
missing-indicator features**. The data is severely imbalanced (73.8% negative).

---

## What the pipeline does

1. Leakage-safe preprocessing (impute + missing-indicators + one-hot, fit in-fold).
2. Stratified 5-fold cross-validation of five models — Random Forest, XGBoost, the
   RF+XGB **soft-voting hybrid**, LightGBM, and a **stacking** ensemble.
3. A paired *t*-test of the hybrid vs the Random-Forest baseline.
4. Hold-out evaluation of the hybrid: confusion matrices, one-vs-rest ROC,
   per-class report — and the trained model is saved to `outputs/models/`.
5. SHAP: global importance, a feature-interaction summary, and per-prediction
   narratives + waterfall plots.

---

## Expected results (fixed seed = 42)

| Metric (hybrid, hold-out) | Value |
|---|---|
| Accuracy | ~0.945 |
| Balanced accuracy | ~0.850 |
| Macro-F1 | ~0.863 |
| ROC-AUC (OvR) | ~0.995 |

Here the paired *t*-test reports the hybrid's gain over Random Forest as
**statistically significant** (p ≈ 0.014) — in contrast to Experiment 1, purely
because of the larger sample. Top SHAP features are TSH, T3, FTI, TT4 (high TSH →
hypothyroid; suppressed TSH + high FTI → hyperthyroid).

---

## Outputs (after running)

- `outputs/reports/thyroid_report.md` — full results with embedded figures
- `outputs/figures/*.png` — class distribution, model comparison, confusion
  matrices, ROC, SHAP bar / interaction / waterfalls
- `outputs/models/thyroid_hybrid_model.joblib` — the trained hybrid pipeline
  (reload with `joblib.load(...)`, then `.predict(...)`)

The polished write-up is `report/report_thyroid.pdf`.

---

*Educational research project — not a medical device. The archive is
single-institution and decades old (1984–87); nothing here should be used for real
diagnosis.*
