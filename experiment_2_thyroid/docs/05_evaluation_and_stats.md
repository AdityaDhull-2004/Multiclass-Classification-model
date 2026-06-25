# 05 — Evaluation & Statistics

Code:
[`evaluation/metrics.py`](../src/healthml/evaluation/metrics.py) ·
[`evaluation/crossval.py`](../src/healthml/evaluation/crossval.py) ·
[`evaluation/stats.py`](../src/healthml/evaluation/stats.py)

On an imbalanced multi-class problem, *how* you score a model matters as much as the
model. This doc explains every metric and test we use and why.

---

## 1. Why plain accuracy lies

Thyroid is **73.8% `negative`.** A model that predicts `negative` for *every* patient
scores **0.738 accuracy** while being clinically useless — it never detects a single
disease. Accuracy rewards getting the majority right and is blind to the rare classes
that actually matter in medicine.

Eq. 7:  Accuracy = (1/N) · Σ_i  𝟙(y_i = ŷ_i)

We report accuracy for context, but never rely on it alone.

---

## 2. Macro-F1 — the headline metric

**Precision** = of the patients we *called* class c, how many truly were c?
**Recall** = of the patients who truly *were* c, how many did we catch?
**F1** is their harmonic mean — high only when *both* are high.

Eq. 8:  F1_i = 2 · (Precision_i · Recall_i) / (Precision_i + Recall_i)
Eq. 9:  Macro-F1 = (1/K) · Σ_i  F1_i

The key word is **macro**: compute F1 *per class*, then average the classes with
**equal weight**, regardless of size. A model that ignores `hyperthyroid` (2.9% of
data) gets F1 ≈ 0 for that class, which drags the macro average down hard — exactly
the penalty we want. The "predict-everything-negative" model scores macro-F1 ≈ 0.12,
correctly exposing it as terrible.

> **Macro vs weighted vs micro:**
> - *macro* — average per-class F1 equally → **emphasises rare classes** (we use this).
> - *weighted* — average per-class F1 weighted by class size → tracks overall but is
>   dominated by the majority (we report it too, for contrast).
> - *micro* — pool all predictions → equals accuracy for single-label problems.

We also report **balanced accuracy** (mean per-class recall) as a second imbalance-
aware summary.

---

## 3. ROC-AUC, one-vs-rest

Code: `metrics.py::roc_auc_ovr`

The ROC curve plots true-positive rate against false-positive rate as you sweep the
decision threshold; **AUC** (area under it) is the probability the model ranks a random
positive above a random negative. AUC = 1.0 is perfect, 0.5 is coin-flipping.

For K classes we use **one-vs-rest (OvR)**: score each class as "this class vs all
others", get K AUCs, and **macro-average** them. Unlike accuracy/F1, AUC uses the
predicted *probabilities* (not the thresholded label), so it measures how well the
model *separates* classes independent of any threshold choice.

Caveat we handle in code: if a CV fold happens to contain zero samples of some rare
class, that class's AUC is undefined; we drop it from the average rather than crash.

---

## 4. Cross-validation: mean ± std, kept fold-wise

Code: `crossval.py::fold_scores`, `cross_validate_models`

We use **stratified 5-fold CV**: split the data into 5 class-balanced folds, train on
4, validate on the 5th, rotate. This yields 5 scores per metric.

Eq. 6 (the std across folds):  σ = sqrt( (1/4) · Σ_j (S_j − μ)² )

- The **mean** estimates expected performance.
- The **std** estimates *stability* — a model with high mean but huge variance across
  folds is risky. We find the hybrid had a *lower* macro-F1 std than the baseline,
  i.e. more stable across folds.

Critically, we **return every fold's score**, not just the summary, because the next
step compares two models fold by fold. And every model is evaluated on the **same**
folds (same `StratifiedKFold` seed) — without that, a "paired" test would be invalid.

---

## 5. The paired t-test — is the gain real or noise?

Code: `stats.py::paired_ttest`

Suppose Hybrid beats Random Forest on mean macro-F1. Two explanations:
- **H1:** the hybrid is genuinely better.
- **H0 (null):** they are equally good, and the hybrid only "won" because of how the
  folds randomly fell.

A **paired** t-test discriminates between them. For each fold *i* compute the
difference d_i = F1_hybrid,i − F1_RF,i. Then:

Eq. 10:  t = d̄ / (s_d / √n)

where d̄ is the mean difference, s_d its standard deviation, n = 5 folds. The test
asks: *is the mean per-fold difference far enough from zero, relative to its scatter,
to be unlikely under H0?* It outputs a **p-value**:

- **p < 0.05** → reject H0 → the improvement is statistically significant.
- **p ≥ 0.05** → fail to reject → we *cannot* claim a real improvement.

In Experiment 1 this test concludes the hybrid's lift over the baseline is **not**
significant; reporting that honestly — rather than overselling a tiny gain — is a core
lesson of the project. Our `SignificanceResult.describe()` prints the verdict in plain
English.

Why *paired* and not a normal (independent) t-test? Because the two models see the
*same* folds, their scores are correlated; pairing removes the fold-to-fold variance
and makes the test far more sensitive to the model difference we actually care about.

### Wilcoxon signed-rank — the safety net

Code: `stats.py::wilcoxon`

The t-test assumes the differences are roughly normal. With only **5 folds** that
assumption is shaky. The **Wilcoxon signed-rank test** is a non-parametric paired
alternative that ranks the differences instead of assuming normality. If the t-test
and Wilcoxon agree, you can trust the verdict; if they disagree, treat the result as
inconclusive (and get more folds/data).

---

## 6. The full evaluation, assembled

`experiments/_common.py::run_experiment` ties it together for each dataset:

1. `cross_validate_models(zoo, X, y)` → fold scores for all 5 models on identical folds.
2. `summarize_cv` → the `mean ± std` table.
3. `paired_ttest(hybrid_folds, rf_folds)` → the significance verdict (the p-value).
4. Refit the hybrid on the 80% train split, evaluate once on the held-out 20%:
   metrics + per-class report + confusion matrix + ROC curves.

The confusion matrix is the most diagnostic single picture: its
diagonal is correct predictions, and off-diagonal cells show *which* classes get
confused for which — e.g. `normocytic hypochromic` vs `normocytic normochromic` in the
anemia study, or `hypothyroid` vs `replacement_therapy` in thyroid.

---

## Metric → code map

| Metric / test | Code |
|---|---|
| Accuracy, balanced accuracy | `metrics.py::classification_metrics` |
| Macro-F1, weighted-F1 | `metrics.py::classification_metrics` |
| ROC-AUC (OvR macro) | `metrics.py::roc_auc_ovr` |
| Per-class precision/recall/F1 | `metrics.py::per_class_report` |
| Stratified k-fold (fold-wise) | `crossval.py::fold_scores` |
| mean ± std summary | `crossval.py::summarize_cv` |
| Paired t-test | `stats.py::paired_ttest` |
| Wilcoxon signed-rank | `stats.py::wilcoxon` |
| Confusion matrix / ROC plots | `plots.py` |
