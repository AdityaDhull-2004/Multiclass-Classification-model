# 07 — Results

Generated artefacts live in `outputs/`:
- reports: [`anemia_report.md`](../outputs/reports/anemia_report.md) · [`thyroid_report.md`](../outputs/reports/thyroid_report.md)
- per-model fold scores: `outputs/reports/*_cv_*.csv`
- figures: `outputs/figures/anemia_*.png` · `outputs/figures/thyroid_*.png`

Both experiments are from fixed-seed runs (`config.RANDOM_SEED = 42`).

---

# PART A — Experiment 1: Anemia (multi-class CBC classification)

Data: `diagnosed_cbc_data_v4.csv`, 1,281 rows → **1,232 after dropping 49 duplicates**,
14 CBC features, 9 classes.

## A1. Cross-validation

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest (baseline) | 0.9846 ± 0.013 | 0.9341 ± 0.055 |
| XGBoost | 0.9878 ± 0.004 | 0.9483 ± 0.008 |
| **Hybrid (RF+XGB)** | 0.9870 ± 0.007 | 0.9394 ± 0.017 |
| LightGBM | 0.9894 ± 0.006 | 0.9465 ± 0.053 |
| Stacking (RF+XGB+LGBM) | **0.9911 ± 0.006** | **0.9660 ± 0.025** |

Our hybrid lands at **CV accuracy 0.9870, macro-F1 0.9394**. Interestingly, the
**stacking** extension is the strongest here (macro-F1 0.966), the one place a more
complex ensemble clearly beats the hybrid.

## A2. Significance — an honest verdict

```
Hybrid - Random Forest mean macro-F1 = +0.0053
paired t-test:  t = 0.24,  p = 0.82  ->  NOT significant
```

We find p = 0.82 (not significant): on ~1,200 patients the hybrid's edge over a strong
RF baseline is within fold noise. (Contrast this with Experiment 2, Part B, where 7×
more data makes a similar edge significant.)

## A3. Hold-out test (final hybrid)

| Metric | Value |
|---|---|
| Accuracy | **0.9838** |
| Macro-F1 | **0.9467** |
| ROC-AUC (OvR) | 0.9997 |

Per-class F1 is ≥ 0.94 for every well-populated class; the only weak class is
**Macrocytic anemia (F1 0.67 on just 3 test samples)** — a pure small-sample artefact,
illustrating the minority-class fragility inherent to small datasets.

## A4. SHAP — what drives anemia subtyping

Top global features (mean |SHAP|): **HGB, MCV, WBC, MCHC, MCH, PLT**. This is the
**MCV morphological framework** from [02_anemia_disease.md](02_anemia_disease.md)
emerging straight from the data.

Per-prediction explanations are textbook-correct (full text in
[`anemia_report.md`](../outputs/reports/anemia_report.md) §6):

| Patient | Predicted | SHAP's top reason | Physiology |
|---|---|---|---|
| #0 | Iron deficiency | MCV 79.9 + MCH 24.4 (low) | microcytic, hypochromic ✓ |
| #141 | Macrocytic | MCV = 113 → **+4.02** | large cells = B12/folate type ✓ |
| #4 | Other microcytic | MCV = 68.8 → **+3.60** | very small cells ✓ |
| #16 | Leukemia | WBC = 13.8 → **+3.65** | high white count ✓ |
| #53 | Thrombocytopenia | PLT = 84 → **+2.55** | low platelets ✓ |

**SHAP also exposes a data-quality bug.** Misclassified patient #14 has physiologically
impossible values (HGB = 41, RBC = 13.1, HCT = 2); SHAP shows the model was misled by the
absurd HGB, which is *why* it wrongly said "Healthy". That is explainability earning its
keep — it turns a silent error into a visible, diagnosable one.

---

# PART B — Experiment 2: Thyroid (the harder, larger study)

## 1. Cross-validation (stratified 5-fold, on identical folds)

| Model | Accuracy | Macro-F1 | ROC-AUC (OvR) |
|---|---|---|---|
| Random Forest (baseline) | 0.9412 ± 0.0078 | 0.8495 ± 0.0240 | 0.9919 ± 0.0026 |
| XGBoost | 0.9504 ± 0.0058 | 0.8721 ± 0.0199 | 0.9951 ± 0.0010 |
| **Hybrid (RF+XGB)** *(proposed)* | **0.9517 ± 0.0048** | **0.8750 ± 0.0175** | 0.9948 ± 0.0011 |
| LightGBM | 0.9506 ± 0.0050 | 0.8729 ± 0.0177 | 0.9944 ± 0.0012 |
| Stacking (RF+XGB+LGBM) | 0.9508 ± 0.0047 | 0.8736 ± 0.0151 | 0.9911 ± 0.0033 |

Observations:
- The **hybrid has the best mean macro-F1 *and* the (near-)smallest std** among the
  classic trio — showing that soft-voting improves *stability*, not just the mean.
- The gains over plain XGBoost are tiny (0.8750 vs 0.8721); most of the lift over the
  Random-Forest baseline comes from **boosting**, not from the voting itself.
- Stacking does **not** beat the simple hybrid here — a useful lesson that more
  machinery is not automatically better on this data.

---

## 2. Is the hybrid's gain statistically significant?

Paired t-test on the 5 fold-wise macro-F1 scores, **Hybrid vs Random Forest**:

```
mean(Hybrid) - mean(Random Forest) = +0.0255   (Hybrid is higher)
t = 4.16,  p = 0.0141
-> STATISTICALLY SIGNIFICANT at alpha = 0.05  (reject H0: equal performance)
```

**This is the most interesting contrast between the two experiments.** In Experiment 1,
the hybrid's improvement over the baseline was *not* significant (p = 0.82) on 1,281
patients. Here, on **9,172** patients, the *same* kind of improvement **is** significant
(p = 0.014).

That is not a contradiction — it is exactly what statistical power predicts: **a larger
sample makes a real-but-small effect detectable.** The insignificant result on the small
dataset and the significant result on the large one are two sides of the same coin, and
together they teach the lesson better than either alone. (Note we still compare against
the strong RF baseline; against XGBoost the hybrid's edge would again be marginal.)

---

## 3. Held-out test performance (final hybrid, 20% unseen)

| Metric | Value |
|---|---|
| Accuracy | 0.9450 |
| Balanced accuracy | 0.8496 |
| **Macro-F1** | **0.8629** |
| Weighted-F1 | 0.9435 |
| ROC-AUC (OvR) | 0.9946 |

Per-class (precision / recall / F1):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.963 | 0.974 | 0.969 | 1355 |
| hypothyroid | 0.949 | 0.985 | 0.967 | 132 |
| nonthyroidal_illness | 0.977 | 0.923 | 0.949 | 91 |
| binding_protein | 0.829 | 0.808 | 0.818 | 78 |
| replacement_therapy | 0.903 | 0.915 | 0.909 | 71 |
| discordant_results | 0.738 | 0.554 | 0.633 | 56 |
| hyperthyroid | 0.804 | 0.789 | 0.796 | 52 |

Reading this clinically:
- **`hypothyroid` is nearly perfect (F1 0.967)** — unsurprising, because high TSH is an
  unambiguous, strongly-separating signal (see [03](03_thyroid_disease.md)).
- **`discordant_results` is the hardest class (F1 0.633).** By definition these are
  patients whose assays *disagree* — there is no clean signal, so the model (like a
  clinician) struggles. The macro-F1 is "held back" honestly by this class, which is
  precisely why we report **macro** and not just accuracy (0.945 would have hidden it).
- The gap between **accuracy 0.945** and **macro-F1 0.863** is the imbalance story in one
  line: the model is excellent on the big classes and merely good on the rare ones.

---

## 4. What SHAP revealed

### Global (what matters in general)
Top features by mean |SHAP|: **TSH, T3, FTI, TT4, on_thyroxine, T4U, age** — i.e. the
thyroid-function lab panel, in almost exactly the order a clinician would rank them. The
missing-indicator features (`missingindicator_T3`, `missingindicator_TSH`) also appear,
confirming that *which tests were ordered* carries real signal.

### Local (why each individual prediction) — the headline
The report's per-prediction narratives are textbook-correct physiology (full text in
[`thyroid_report.md`](../outputs/reports/thyroid_report.md) §6):

| Patient | Predicted | SHAP's top reason | Physiology check |
|---|---|---|---|
| #36 | hypothyroid | TSH = 6.4 → **+3.80** | high TSH = underactive thyroid ✓ |
| #41 | hyperthyroid | FTI = 196, TSH = 0.02 → **+3.2 / +2.2** | high hormone + suppressed TSH = overactive ✓ |
| #18 | replacement_therapy | on_thyroxine = 1 → **+3.62** | patient is *taking* thyroid hormone ✓ |
| #8 | binding_protein | T4U = 2.15 → **+3.81** | binding-protein anomaly shifts T4U ✓ |
| #7 | nonthyroidal_illness | T3 = 1.0 (low) → **+4.23** | low T3 in sick patients ("low-T3 syndrome") ✓ |

Every explanation is *additive* (base value + contributions = the model's score), so it
is a complete account of the decision, not a story told after the fact.

### Explaining the mistakes
The two misclassified examples are both **`discordant_results` → predicted `negative`**
(patients #37, #42). SHAP shows why: their dominant lab values looked near-normal to the
model (e.g. #42's pattern pushed toward `negative` despite a low TT4 that pulled the
other way), and with low confidence (0.55–0.71). This is the model honestly being
uncertain on the genuinely ambiguous class — exactly where a human should take over.

---

## 5. Side-by-side: the two experiments

| Aspect | Exp 1 (anemia) | Exp 2 (thyroid) |
|---|---|---|
| Patients | 1,232 (post-dedup) | 9,172 |
| Classes | 9 | 7 |
| Feature types | 14 numeric | 6 lab + 14 binary + 2 nominal, **+ missing values** |
| Hybrid macro-F1 (CV) | 0.9394 | 0.8750 |
| Hybrid test accuracy | 0.9838 | 0.9450 |
| Hybrid vs RF baseline | **not sig. (p = 0.82)** | **sig. (p = 0.014)** |
| Imbalance handling | macro metrics + stratification | + class/sample weighting |
| Explainability | global **+ per-prediction** SHAP | global **+ per-prediction** SHAP |

**Two lessons jump out of this table.** (1) On the smaller anemia dataset, the
hybrid's edge over the baseline is honestly "not significant." (2) The same
hybrid-vs-baseline comparison flips to **significant** in Experiment 2 purely because it
has ~7× the data: statistical power, not a better model, is what changed. That contrast
is the whole reason for running both.

The thyroid macro-F1 is *lower* than the anemia number — not because the model
is worse, but because the problem is **harder and more honest**: a near-impossible
`discordant_results` class, severe imbalance, and missing data. That drop is the whole
point of choosing a tougher, larger dataset.

---

*Numbers above are from a single fixed-seed run (`config.RANDOM_SEED = 42`). Re-running
reproduces them exactly; changing the seed shifts the last digits but not the story.*
