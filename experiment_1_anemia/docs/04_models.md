# 04 — The Models (with the math)

Code: [`src/healthml/models/factory.py`](../src/healthml/models/factory.py)

All five models are tree-based. Trees are a natural fit for clinical tabular data:
they handle mixed feature types, need no scaling, capture non-linear thresholds
("TSH > 6 *and* FTI < 60"), and — importantly here — are **exactly explainable with
TreeSHAP**.

We build, in increasing sophistication:

1. Decision tree (the building block — conceptual)
2. **Random Forest** — the baseline
3. **XGBoost** — the second base learner
4. **Hybrid soft-voting (RF + XGB)** — the *proposed* model
5. **LightGBM** and **Stacking** — the "more complex" extensions

---

## 0. The building block: a decision tree

A decision tree recursively splits the data on the feature/threshold that best
separates the classes (measured by **Gini impurity** or **entropy**). A leaf
predicts the class distribution of the training samples that landed in it.

- **Strength:** interpretable, non-linear, no scaling needed.
- **Weakness:** a single deep tree **overfits** — it memorises the training set and
  generalises poorly (high variance).

The two great ideas below both fix the variance problem, in opposite ways.

---

## 1. Random Forest — *bagging* (variance reduction)

Eq. 2:  f_RF(x) = (1/T) · Σ_t  h_t(x)

A Random Forest trains **T independent trees**, each on:
- a **bootstrap sample** (random rows drawn with replacement), and
- a **random subset of features** at each split.

These two randomisations make the trees *decorrelated*. Averaging many noisy-but-
unbiased trees cancels their individual errors — that is **bagging** (bootstrap
aggregating). The forest's variance is far lower than any single tree's, while bias
barely rises.

```python
RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample" | None)
```

- `n_estimators=300` — number of trees T.
- `class_weight="balanced_subsample"` — our imbalance switch: rare classes get
  proportionally up-weighted inside each tree, so `hyperthyroid` is not ignored.

For multi-class, each tree votes a class distribution; the forest averages them and
takes the arg-max. `predict_proba` returns those averaged class probabilities — which
the hybrid needs.

---

## 2. XGBoost — *boosting* (bias reduction)

Eq. 3:  L = Σ_i  l(y_i, ŷ_i)  +  Σ_k  Ω(f_k)

Boosting builds trees **sequentially**, each new tree fitting the *errors* (gradients
of the loss) left by the ensemble so far. Where bagging reduces variance, boosting
reduces **bias** — it keeps refining the decision boundary.

XGBoost is a particularly strong, regularised gradient booster:
- the first term `l(y, ŷ)` is the training loss (multi-class log-loss here),
- the second term `Ω(f_k) = γ·T + ½·λ·‖w‖²` **penalises tree complexity** (number of
  leaves T and leaf-weight magnitude), which is what stops boosting from overfitting.

```python
XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
              subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
              objective="multi:softprob", eval_metric="mlogloss")
```

- `learning_rate=0.1` — shrinks each tree's contribution (slower but more accurate).
- `subsample`/`colsample_bytree` — row/column sampling, borrowing bagging's
  decorrelation trick to regularise.
- `reg_lambda=1.0` — the λ in Ω, the L2 penalty on leaf weights.
- `objective="multi:softprob"` — output a full probability vector per class.

XGBoost has **no `class_weight`.** To make it imbalance-aware we instead pass
**per-sample weights** (`factory.py::compute_sample_weights`), which up-weight rare
classes' rows — mathematically equivalent to class weighting.

---

## 3. The Hybrid — soft voting (the proposed model)

Eq. 4–5:  ŷ = arg max_c  (1/M) · Σ_m  P_m(c | x),    M = 2

The hybrid runs Random Forest and XGBoost **in parallel** and **averages their class
probabilities**, then predicts the arg-max class. This is **soft voting** (it
averages probabilities, unlike *hard* voting which counts discrete votes).

Why average two good models?
- RF reduces variance; XGB reduces bias. Their errors are partly **uncorrelated**, so
  averaging cancels some of each — the classic ensemble bet.
- Soft voting uses the *confidence* of each model, so a very-sure model sways the
  average more than a hesitant one.

```python
VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft",
)
```

Architecture note: the preprocessor sits **once** in front of the voter, so both base
learners receive *identical* transformed features — the feature input is fed
simultaneously to the two models. Our `build_hybrid` wraps the voter in the standard
`[prep, clf]` pipeline.

> **The honest caveat:** averaging two already-strong, already-correlated tree
> ensembles often yields only a *tiny* lift over the better one — small enough that a
> paired t-test may call it statistically insignificant. We measure exactly that; see
> [05](05_evaluation_and_stats.md).

---

## 4. LightGBM — a faster, leaf-wise booster (extension)

LightGBM is another gradient booster with two engineering tricks: it grows trees
**leaf-wise** (splitting the single highest-loss leaf, not a whole level at a time)
and bins continuous features into histograms for speed. It often matches or beats
XGBoost while training faster, which is why it joins the stacking ensemble.

```python
LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
               class_weight="balanced" | None)
```

---

## 5. Stacking — a *learned* combiner (the "more complex" ensemble)

Code: `factory.py::build_stacking`

Soft voting combines base models with **fixed equal weights**. **Stacking** instead
*learns* how to combine them:

1. Train base learners (RF, XGB, LightGBM) and collect their **out-of-fold**
   predicted probabilities (using internal 5-fold CV, so the meta-features are not
   contaminated by leakage).
2. Train a **meta-learner** (here, multinomial logistic regression) that takes those
   probability vectors as input and learns the best way to blend them.

This lets the ensemble discover, e.g., "trust XGBoost for `binding_protein` but RF for
`hyperthyroid`" — a flexibility fixed-weight voting cannot express.

```python
StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
    final_estimator=LogisticRegression(class_weight="balanced"),
    stack_method="predict_proba", cv=5,
)
```

Trade-off: stacking is more powerful but **slower** (it trains the base learners many
times) and **easier to overfit** if the meta-learner is complex — which is why we keep
the meta-learner deliberately simple (logistic regression).

---

## Choosing between them

| Model | Combines by | Fixes | Cost | When it shines |
|---|---|---|---|---|
| Random Forest | averaging independent trees | variance | low | strong, robust baseline |
| XGBoost | sequential error-correction | bias | medium | complex non-linear boundaries |
| Hybrid (soft vote) | averaging RF + XGB probs | both, a little | medium | squeezing a small, stable lift |
| LightGBM | leaf-wise boosting | bias | low | speed + accuracy |
| Stacking | a learned meta-model | both, adaptively | high | when base models disagree usefully |

`build_model_zoo(dataset)` returns all of them keyed by name, so the experiment driver
cross-validates the whole set and compares them on identical folds.

---

## Imbalance switch in one place

Every builder takes `balanced: bool`. When `True`:
- RF / LightGBM / the stacking meta-learner use `class_weight="balanced"`,
- XGBoost uses `compute_sample_weights(y)` (since it lacks `class_weight`).

This is how we turn imbalance handling into a controllable knob in
Experiment 2. See [08_extensions_and_complexity.md](08_extensions_and_complexity.md).
