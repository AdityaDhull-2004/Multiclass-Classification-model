# 06 — SHAP Explained: *Why* a Prediction Is What It Is

Code: [`src/healthml/explain/shap_explain.py`](../src/healthml/explain/shap_explain.py)

This is the heart of the project. A model that outputs "hypothyroid, 0.91" is not
enough for medicine — a clinician needs to know **why**. SHAP turns the black box into
a sentence: *"this patient was called hypothyroid because their TSH was high and their
FTI was low."* This document explains the idea from first principles, then shows how
our code produces both **global** ("what matters in general") and **local** ("why this
one patient") explanations.

---

## 1. The core idea: fairly splitting the credit

Imagine the model's output for a patient is a number (a score for some class). The
model started from a **baseline** (its average score for that class across everyone)
and ended at *this patient's* score. SHAP answers: **how much did each feature
contribute to the gap between baseline and final?**

Formally, SHAP gives each feature *j* a value φ_j such that

Eq. 11:   f(x)  =  φ_0  +  Σ_j  φ_j

- **φ_0** — the base value (average model output for this class).
- **φ_j** — the contribution of feature *j* for *this* patient (positive = pushed the
  score up, negative = pushed it down).
- The contributions **add up exactly** to the model's output. This *additivity* is
  what makes SHAP trustworthy: nothing is hand-waved; the parts sum to the whole.

---

## 2. Where φ comes from: Shapley values from game theory

SHAP's φ are **Shapley values**, a 1953 idea from cooperative game theory with a
beautiful guarantee. Think of the features as *players* cooperating to produce the
prediction (the *payout*). The Shapley value of a player is its **fair share** of the
payout, defined as its *average marginal contribution* over every possible order in
which players could join the coalition:

φ_j = Σ_{S ⊆ features \ {j}}  [ |S|! (M−|S|−1)! / M! ] · ( f(S ∪ {j}) − f(S) )

In words: for every subset *S* of the *other* features, measure how much adding
feature *j* changes the prediction ( f(S ∪ {j}) − f(S) ), then average those marginal
gains with the right combinatorial weights. Averaging over **all orderings** is what
makes the attribution fair — no feature is privileged by being considered "first".

Shapley values are the **only** attribution that simultaneously satisfies:
- **Efficiency** — contributions sum to the prediction (Eq. 11).
- **Symmetry** — two features that always contribute equally get equal credit.
- **Dummy** — a feature that never changes the output gets φ = 0.
- **Additivity** — explanations of an ensemble are the sum of its members'.

That uniqueness is why SHAP is the explanation method of choice for this project.

---

## 3. TreeSHAP: exact and fast for trees

The sum above is over *all 2^M subsets* of features — exponential, hopeless to compute
directly. For **tree models** (RF, XGBoost, LightGBM), the **TreeSHAP** algorithm
computes the exact Shapley values in polynomial time by exploiting tree structure. So
for our tree ensembles there is **no sampling approximation** — the explanations are
exact. This is a major reason the whole project is tree-based.

In code:

```python
self.explainer = shap.TreeExplainer(self.tree_model)
self._exp = self.explainer(self._Xbg)        # exact SHAP for every patient
# self._exp.values      shape (n_patients, n_features, n_classes)
# self._exp.base_values shape (n_patients, n_classes)
```

For a multi-class model SHAP returns a φ value **per feature, per class**: feature *j*
can push *toward* `hypothyroid` and *away from* `negative` at the same time.

### Which model do we explain?

The hybrid is RF + XGBoost. We apply SHAP to the XGBoost component:
`_extract_tree_model` reaches into the fitted `VotingClassifier` and pulls
out the `xgb` sub-estimator. We explain a single, coherent tree model — clean and
exact — rather than trying to average two explainers.

---

## 4. Global explanations — what matters *in general*

Code: `ShapExplainer.global_importance`, `plot_global_bar`, `plot_beeswarm`

### Global importance (bar)
Average **|φ_j|** over all patients and all classes:

```python
imp = np.abs(self._values).mean(axis=(0, 2))   # mean |SHAP| per feature
```

A feature with large mean |SHAP| moves predictions a lot *somewhere* — it is globally
important. In thyroid this surfaces **TSH, T3, FTI** at the top, exactly the labs a
clinician would name. This is the principled version of "feature
importance" (and is more reliable than a tree's built-in impurity importance, which can
be biased toward high-cardinality features).

### Beeswarm (directional)
The bar chart shows *how much* a feature matters but not *which way*. The **beeswarm**
plots, for one class, every patient's φ for each feature, coloured by the feature's
value. You can read sentences off it: *"high TSH (red dots) sits at positive SHAP for
`hypothyroid` → high TSH pushes toward hypothyroid"* — which is precisely the HPT-axis
physiology from [03_thyroid_disease.md](03_thyroid_disease.md). When the model's global
behaviour matches known medicine, you gain trust that it learned signal, not artefact.

---

## 5. Local explanations — *why this patient* (the headline)

Code: `ShapExplainer.explain_instance`, `plot_waterfall`

This is the capability you asked for: **detailed, per-prediction reasoning.** For a
chosen patient *i*:

1. Get the predicted class *c* (arg-max probability).
2. Pull that class's SHAP row: `φ = self._values[i, :, c]`.
3. Attach the patient's **actual clinical values** to each φ.
4. Sort by |φ| and read off the story.

`explain_instance` returns a `LocalExplanation` with a ready-made **narrative**, e.g.:

```
Patient #12  ->  predicted: hypothyroid  (confidence 96.4%;  true label: hypothyroid)
The model's baseline score for 'hypothyroid' is -1.80. This patient's measurements adjust it as follows:
   TSH         = 14.2   increases evidence for hypothyroid by +2.41
   FTI         = 58     increases evidence for hypothyroid by +1.05
   on_thyroxine= 0      increases evidence for hypothyroid by +0.33
   T3          = 1.1    increases evidence for hypothyroid by +0.20
   age         = 61     decreases evidence for hypothyroid by -0.12
Net score for 'hypothyroid' = +2.07, the highest of all 7 classes -> final prediction: hypothyroid.
```

Every line is a feature, its real value, the direction it pushed, and the magnitude.
The base value plus all contributions equals the final score (additivity, Eq. 11), so
the explanation is *complete*, not a post-hoc rationalisation.

### Waterfall plot
`plot_waterfall` renders the same decomposition visually: starting at the base value,
each feature's bar steps the score up (red) or down (blue) until it reaches the final
output. It is the single best picture for "show me why this one prediction happened".

### Explaining mistakes, too
The experiment driver deliberately includes **misclassified** patients in its local
examples. SHAP on an error is gold for debugging: it shows *which* feature values
misled the model (e.g. a missing T3 plus a borderline TSH tipping `hypothyroid` into
`negative`), pointing straight at data-quality or class-overlap problems.

---

## 6. How to *read* a SHAP explanation responsibly

- **SHAP shows association, not causation.** "High TSH pushed toward hypothyroid" is
  what *the model* learned; it aligns with physiology here, but SHAP itself proves
  nothing causal. We keep this limitation front-and-centre.
- **Units are the model's score space**, not probabilities. For multi-class XGBoost
  the φ are in margin/log-odds-like units. We describe them as "evidence for class c",
  which is faithful and readable.
- **Sanity-check against the disease docs.** The medicine docs ([02](02_anemia_disease.md),
  [03](03_thyroid_disease.md)) tell you which features *should* drive which class. If
  SHAP agrees, trust grows; if it points at something physiologically irrelevant (say,
  `referral_source` driving `hypothyroid`), you have found a leakage/bias bug.

---

## 7. The pipeline → SHAP plumbing (a subtlety worth knowing)

SHAP must explain the model in the space the model actually sees — i.e. **after**
imputation and one-hot encoding. So `ShapExplainer`:

1. fits/extracts the pipeline's `prep` step,
2. transforms the background patients into model space, **keeping feature names** via
   `prep.get_feature_names_out()` (so plots show `missingindicator_T3`, `sex_F`, etc.),
3. runs TreeSHAP on the transformed matrix,
4. when narrating, maps each model-space feature *back* to the patient's original
   clinical value where one exists, so the story reads in real units (`TSH = 14.2`),
   not standardised numbers.

That round-trip is what makes the explanations both **technically correct** (explaining
the real model input) and **human-readable** (reported in clinical units).

---

## SHAP → code map

| Capability | Code |
|---|---|
| Build exact tree explainer | `ShapExplainer.__init__` → `shap.TreeExplainer` |
| Extract XGB from the hybrid | `_extract_tree_model` |
| Global importance (mean \|SHAP\|) | `global_importance`, `plot_global_bar` |
| Directional global view | `plot_beeswarm` |
| Per-prediction narrative | `explain_instance` → `LocalExplanation.narrative` |
| Per-prediction waterfall | `plot_waterfall` |
