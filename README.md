# Explainable Multi-Class Disease Classification — Two Experiments

**Author:** Aditya Dhull · `adityadhull@iisc.ac.in`

This repository contains **two self-contained, explainable machine-learning
experiments** that classify diseases from routine laboratory panels, plus a report
comparing them. Each experiment trains a hybrid Random-Forest + XGBoost soft-voting
ensemble (with LightGBM and stacking for comparison), evaluates it rigorously, and
explains every prediction with SHAP.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Disease | **Anemia** (9 subtypes) | **Thyroid disease** (7 groups) |
| Data | Complete Blood Count, 1,232 patients | Garavan archive, 9,172 patients |
| Folder | [`experiment_1_anemia/`](experiment_1_anemia/) | [`experiment_2_thyroid/`](experiment_2_thyroid/) |
| Hybrid macro-F1 (test) | ~0.947 | ~0.863 |
| Hybrid vs RF significance | not significant (p≈0.82) | significant (p≈0.014) |

Both experiments are **completely self-contained**: the dataset is committed inside
the folder, the code is bundled, and **nothing is downloaded at run time**. Download
this repository as a ZIP, and they run offline.

---

## Repository contents

```
README.md                     <- you are here (overview of both experiments)
SETUP.md                      <- step-by-step setup & run guide
comparison_report/            <- IEEE report comparing both experiments (PDF + LaTeX)
experiment_1_anemia/          <- data + pipeline + report (self-contained, runnable)
experiment_2_thyroid/         <- data + pipeline + report (self-contained, runnable)
```

Each experiment folder contains:

```
data/raw/<dataset>            the bundled training/testing data
healthml/ + pipeline.py + run_experiment.py + requirements.txt   the pipeline
report/                       the experiment's IEEE report (PDF + LaTeX + figures)
```

---

## Run it (short version)

```bash
cd experiment_1_anemia          # or experiment_2_thyroid
pip install -r requirements.txt
python run_experiment.py
```

Full instructions, prerequisites, expected outputs and runtimes are in
**[SETUP.md](SETUP.md)**.

---

## What each run produces

1. Stratified 5-fold cross-validation of five models — Random Forest, XGBoost, the
   RF+XGB soft-voting **hybrid**, LightGBM, and a **stacking** ensemble
   (accuracy, macro-F1, ROC-AUC; mean ± std).
2. A paired *t*-test of the hybrid vs the Random-Forest baseline.
3. Hold-out test evaluation: confusion matrices, one-vs-rest ROC, per-class report.
4. SHAP explanations: global importance, a feature-interaction summary, and
   **per-prediction** narratives + waterfall plots (including misclassified cases).

Results are written to each folder's `outputs/` directory; the polished write-up for
each experiment lives in its `report/` folder, and the head-to-head comparison is in
[`comparison_report/`](comparison_report/).

---

## The headline finding

The *same* hybrid-vs-baseline comparison is **not statistically significant** on the
~1,200-patient anemia set (p≈0.82) but **is significant** on the ~9,200-patient
thyroid set (p≈0.014). Same kind of model edge — what changes is **statistical power
from a larger sample**, not a better model. That contrast is why both experiments are
presented together.

---

*Educational research project — not a medical device. Both datasets are
single-institution; nothing here should be used for real diagnosis.*
