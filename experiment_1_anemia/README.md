# Experiment 1 — Explainable Multi-Class Anemia Classification (CBC)

**Author:** Aditya Dhull · `adityadhull@iisc.ac.in`

A **fully self-contained** project that classifies a patient into **9 anemia
diagnostic classes** from **14 Complete Blood Count (CBC)** parameters, using a
hybrid Random-Forest + XGBoost soft-voting ensemble, and explains every prediction
with SHAP. Everything needed — data, source code, pipeline, tests, docs and the
report — lives in this folder, and it runs **offline** (nothing is downloaded).

---

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py
```

A full run takes ~2–4 minutes. Results land in `outputs/` (figures, a markdown
report, and the saved model). Requires Python 3.10+.

---

## Folder contents

```
experiment_1_anemia/
├── run_experiment.py        # entry point — runs the whole pipeline
├── pipeline.py              # the full protocol (CV, stats, eval, SHAP, model save)
├── healthml/                # the SOURCE package (data · models · evaluation · explain)
├── data/raw/anemia.csv      # the bundled dataset (1,281 rows, 14 features + Diagnosis)
├── docs/                    # conceptual deep-dives (ML pipeline, models, SHAP, results…)
├── scripts/                 # helper scripts (data provenance, p-value stability study)
├── tests/                   # network-free unit tests for the pipeline
├── report/                  # the IEEE report (PDF + LaTeX + figures)
├── outputs/                 # GENERATED on run: figures, report, saved model
└── requirements.txt
```

- **Source code** is the `healthml/` package (this is the "src").
- **Run the tests:** `python tests/test_pipeline.py`
- **Reproduce the SHAP p-value-stability study:** `python scripts/pvalue_stability.py`
- **Re-fetch the dataset** (already bundled): `python scripts/download_anemia.py`

---

## The dataset

`data/raw/anemia.csv` — the public CBC / anemia-types dataset
(`diagnosed_cbc_data_v4`): 1,281 patients, 14 numeric CBC features
(WBC, LYMp, NEUTp, LYMn, NEUTn, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, PDW, PCT) and a
9-class `Diagnosis`. The loader removes exact duplicate records (→ 1,232 patients).

---

## What the pipeline does

1. Leakage-safe preprocessing (median imputation inside the CV loop).
2. Stratified 5-fold cross-validation of five models — Random Forest, XGBoost, the
   RF+XGB **soft-voting hybrid**, LightGBM, and a **stacking** ensemble.
3. A paired *t*-test of the hybrid vs the Random-Forest baseline.
4. Hold-out evaluation of the hybrid: confusion matrices, one-vs-rest ROC,
   per-class report — and the trained model is saved to `outputs/models/`.
5. SHAP: global importance, a feature-interaction summary, and per-prediction
   narratives + waterfall plots (including a misclassified case).

---

## Expected results (fixed seed = 42)

| Metric (hybrid, hold-out) | Value |
|---|---|
| Accuracy | ~0.984 |
| Macro-F1 | ~0.947 |
| ROC-AUC (OvR) | ~0.9997 |

The paired *t*-test reports the hybrid's gain over Random Forest as **not
statistically significant** (p ≈ 0.82) — an honest result on ~1,200 patients. Top
SHAP features are HGB, MCV, WBC, MCHC, MCH, PLT (the MCV morphological framework).

---

## Outputs (after running)

- `outputs/reports/anemia_report.md` — full results with embedded figures
- `outputs/figures/*.png` — class distribution, model comparison, confusion
  matrices, ROC, SHAP bar / interaction / waterfalls
- `outputs/models/anemia_hybrid_model.joblib` — the trained hybrid pipeline
  (reload with `joblib.load(...)`, then `.predict(...)`)

The polished write-up is `report/report_anemia.pdf`.

---

*Educational research project — not a medical device. Single-source dataset;
nothing here should be used for real diagnosis.*
