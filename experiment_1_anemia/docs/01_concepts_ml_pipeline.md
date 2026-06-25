# 01 — The Machine-Learning Pipeline (Concepts)

This document explains the *general* ML concepts that both experiments rely on,
and points at the exact code that implements each one. If you understand this
file, the rest of the project is just "apply these ideas to two datasets".

The pipeline, end to end:

```
raw data ─▶ clean & encode ─▶ split ─▶ [ preprocess ─▶ model ]  ─▶ evaluate ─▶ explain
                                         └── fit only on train fold ──┘
```

The single most important principle is in the brackets: **everything that learns
from data — imputers, encoders, scalers, the model — must be fit on training data
only.** Violating this is *data leakage*, and it is the #1 cause of results that
look great in a notebook and fall apart in the clinic.

---

## 1. The `Dataset` container

Code: [`src/healthml/data/preprocess.py`](../src/healthml/data/preprocess.py)

Instead of passing around loose `X`, `y`, and label lists, every loader returns one
`Dataset` object that carries:

- `X` — a **pandas DataFrame** (we keep column names so SHAP plots read `TSH`, not `f17`),
- `y` — an integer-encoded target (`0 … n_classes-1`),
- `class_names` — so integer `3` can be printed as `"hypothyroid"`,
- `numeric_features` / `categorical_features` — column lists that drive preprocessing,
- helper methods: `.summary()`, `.class_distribution()`.

Why integer-encode the target? scikit-learn, XGBoost and LightGBM all expect the
label to be `0..K-1`. We sort classes by frequency, so class `0` is always the
majority class — convenient when reading confusion matrices.

---

## 2. Train/test split — and why *stratified*

Code: `preprocess.py::stratified_split`

We hold out 20% of patients as a **test set** that the model never sees during
training or tuning (an 80/20 split). The test set exists to answer one
question honestly: *how well does this generalise to unseen patients?*

The split is **stratified** (`stratify=y`): it preserves each class's proportion in
both halves. This matters enormously when classes are rare. Thyroid's
`hyperthyroid` class is ~2.9% of patients. A naive random split could, by chance,
put almost all of them in training and leave the test set with two or three — making
the test estimate of `hyperthyroid` recall pure noise. Stratification guarantees the
test set mirrors reality.

> **Rule of thumb:** the more imbalanced your problem, the more you *need*
> stratification — both for the train/test split and for cross-validation folds.

---

## 3. Encoding & the feature types

Real clinical tables mix three kinds of columns; each needs different handling.

| Kind | Examples | How we encode it | Why |
|---|---|---|---|
| **Continuous** | TSH, MCV, HGB, age | leave numeric | trees split on thresholds directly |
| **Binary flag** | `on_thyroxine` (f/t) | map to 0/1, treat as numeric | a 0/1 column *is* a valid split feature |
| **Nominal** | `sex`, `referral_source` | **one-hot encode** | no meaningful order between categories |

One-hot encoding turns `sex ∈ {M, F}` into two 0/1 columns `sex_M`, `sex_F`. We never
integer-encode a nominal feature as `M=0, F=1, other=2`, because that invents a fake
ordering (`other > F > M`) the model would wrongly exploit.

Code: `preprocess.py::build_preprocessor` builds a `ColumnTransformer` that routes
numeric columns down one path and categorical columns down another.

---

## 4. Missing values — and why missingness is *informative* in medicine

Code: `preprocess.py::build_preprocessor` (the `SimpleImputer` steps)

In the thyroid data, **T3 is missing for 28% of patients, TSH for 9%.** That is not
random: a doctor orders a test when they suspect something. The very *absence* of a
T3 measurement carries signal.

We handle this in two layers:
1. **Imputation** — fill numeric gaps with the column **median** (robust to outliers),
   fill categorical gaps with the **most frequent** value. A model cannot consume
   `NaN`, so something must fill the hole.
2. **Missing-indicator** — `SimpleImputer(add_indicator=True)` adds a companion 0/1
   column `missingindicator_T3` that records "this value was originally missing".
   This lets the model *learn from the absence itself*, instead of pretending the
   imputed median was a real measurement.

This is a concrete way Experiment 2 handles real clinical data, where missing
values are pervasive and often informative.

> **Leakage trap:** the median used for imputation must come from the *training fold
> only*. We get this for free by putting the imputer **inside the pipeline** (next
> section), so scikit-learn re-fits it on each fold.

---

## 5. The `Pipeline` — leakage-safety by construction

Code: [`src/healthml/models/factory.py`](../src/healthml/models/factory.py) → `make_pipeline`

Every model is a scikit-learn `Pipeline`:

```python
Pipeline([("prep", preprocessor), ("clf", classifier)])
```

When you call `pipeline.fit(X_train, y_train)`:
1. the preprocessor is **fit** on `X_train` (learns medians, one-hot categories),
2. it **transforms** `X_train`,
3. the classifier is **fit** on the transformed training data.

When you later call `pipeline.predict(X_test)`, the preprocessor only *applies* the
already-learned transforms. Because cross-validation clones and re-fits the *whole*
pipeline on each fold, test-fold statistics can never leak into training. This single
design decision removes an entire category of subtle bugs.

Do tree models need feature scaling? **No** — Random Forest, XGBoost and LightGBM
split on thresholds, so they are invariant to any monotonic rescaling of a feature.
We expose a `scale=` flag anyway for completeness, but leave
it off for the tree ensembles.

---

## 6. Cross-validation — a better estimate than a single split

Code: [`src/healthml/evaluation/crossval.py`](../src/healthml/evaluation/crossval.py)

A single 80/20 split gives *one* performance number, which can be lucky or unlucky.
**Stratified k-fold cross-validation** (we use k=5) splits the training data into 5
equal, class-balanced folds, then trains 5 times — each time holding out a different
fold for validation. You get 5 scores, and their **mean ± std** tells you both the
expected performance *and* how stable it is.

Crucially, our `fold_scores` keeps the **individual** fold scores (not just the mean),
because the statistical test in the next section compares two models *fold by fold*.

---

## 7. The honest question: is an improvement *real*?

Code: [`src/healthml/evaluation/stats.py`](../src/healthml/evaluation/stats.py)

If the hybrid scores macro-F1 0.94 and Random Forest scores 0.93, is the hybrid
*actually* better, or did it just win the coin-flip of how the folds fell? A
**paired t-test** on the 5 fold-wise scores answers this. We apply this discipline
rather than overselling a tiny gain, and add the non-parametric **Wilcoxon
signed-rank** test as a robustness check (5 folds is too few to trust the t-test's
normality assumption blindly).

See [05_evaluation_and_stats.md](05_evaluation_and_stats.md) for the full treatment.

---

## 8. Explanation — closing the loop

Code: [`src/healthml/explain/shap_explain.py`](../src/healthml/explain/shap_explain.py)

A model that is accurate but unexplainable is hard to trust in a clinic. **SHAP**
decomposes each prediction into per-feature contributions, so we can say "*this*
patient was called hypothyroid because their TSH was high and their FTI was low" —
and a clinician can check that against physiology. This is the project's headline
deliverable; [06_shap_explained.md](06_shap_explained.md) covers it in depth.

---

## Concept → code quick map

| Concept | File · symbol |
|---|---|
| Dataset container | `data/preprocess.py` · `Dataset` |
| Stratified split | `data/preprocess.py` · `stratified_split` |
| Encoding / imputation / missing-indicator | `data/preprocess.py` · `build_preprocessor` |
| Leakage-safe pipeline | `models/factory.py` · `make_pipeline` |
| Models | `models/factory.py` · `build_*` |
| Metrics | `evaluation/metrics.py` |
| Cross-validation (fold-wise) | `evaluation/crossval.py` |
| Paired t-test / Wilcoxon | `evaluation/stats.py` |
| SHAP global + local | `explain/shap_explain.py` |
| Full protocol | `experiments/_common.py` · `run_experiment` |
