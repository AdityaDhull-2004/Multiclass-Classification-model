# 08 — What Makes Experiment 2 Harder, and Where to Go Next

Experiment 1 is the anemia study. Experiment 2 is the "something a little more
complex" study. This doc states *exactly* how it is more complex and which common
limitations of small-scale clinical-ML studies it addresses.

---

## 1. Common limitations of small-scale clinical-ML studies → how this work responds

Small clinical-ML studies routinely run into the same handful of limitations. We picked
thyroid disease specifically to push on them:

| Common limitation | Experiment 2's response |
|---|---|
| **Small dataset**, single open repository | **9,172 patients**, the Garavan Institute clinical archive |
| **Class imbalance** under-addressed; rare subtypes scarce | A `balanced` switch (class weights + sample weights), macro metrics, stratified folds; imbalance is *more* extreme here (74% negative) so the issue is unavoidable, not optional |
| **Missing values** ignored | Structured missingness (T3 missing 28%) handled with median imputation **+ missing-indicator features** that let the model learn from the absence |
| **Hyperparameters not tuned** | Optional Optuna tuning hook (below); stronger default learners |
| **No demographic factors** | `age` and `sex` are first-class features and show up in SHAP |
| **Only post-hoc SHAP** | Same SHAP, but pushed into **detailed per-prediction narratives** — the headline deliverable |

We do **not** claim to have solved clinical thyroid diagnosis — the dataset is also
old (1984–87) and single-institution. The point is pedagogical: feel each limitation
bite, and apply a concrete technique to it.

---

## 2. Ways Experiment 2 is genuinely more complex

1. **Mixed feature types.** Anemia is 14 clean numeric CBC values. Thyroid mixes 6
   continuous labs, 14 binary history flags, and 2 nominal categoricals — so the
   `ColumnTransformer` and one-hot encoding actually earn their keep.
2. **Real missing data**, and missingness that is *informative* (a test is ordered
   when suspected). This is the single biggest jump in realism.
3. **Severe imbalance.** The minority classes (`hyperthyroid` 2.9%) are rarer than
   anything in the anemia set, so macro-F1 and stratification are not optional niceties.
4. **A bigger model zoo.** Beyond RF/XGB/hybrid we add LightGBM and a **stacking**
   meta-learner — a *learned* combiner, strictly more expressive than soft voting.
5. **Class definition is itself a modelling choice.** The raw diagnosis is a string of
   condition letters; `thyroid.py` documents the grouping into 7 clinically coherent
   classes. (Anemia's classes come ready-made.)

---

## 3. The extensions, and how to switch them on

### Imbalance handling
Every model builder accepts `balanced=True`:

```python
from healthml.data import load_thyroid
from healthml.models import build_model_zoo

ds = load_thyroid()
zoo = build_model_zoo(ds, balanced=True)   # class/sample weighting on
```

Compare the macro-F1 of `balanced=False` vs `True`: weighting usually *raises minority
recall* at a small cost to majority precision — visible per-class in the report.

### SMOTE (synthetic minority over-sampling) — an alternative
`imbalanced-learn` is installed. To resample *inside* the CV (leakage-safe), wrap the
classifier in an `imblearn` pipeline so SMOTE only ever sees the training fold:

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from healthml.data.preprocess import build_preprocessor

pipe = ImbPipeline([
    ("prep", build_preprocessor(ds)),
    ("smote", SMOTE(random_state=42)),
    ("clf", build_xgboost(ds).named_steps["clf"]),
])
```

SMOTE interpolates new minority-class points between real ones. It can help, but on
clinical data it can also fabricate implausible patients — always compare it against
plain class-weighting, never assume it wins.

### Hyperparameter tuning with Optuna
Untuned defaults are a common limitation. Optuna is installed; a minimal search:

```python
import optuna
from sklearn.model_selection import cross_val_score
from healthml.models import make_pipeline
from healthml.data.preprocess import build_preprocessor
from xgboost import XGBClassifier

def objective(trial):
    clf = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
    )
    pipe = make_pipeline(build_preprocessor(ds), clf)
    return cross_val_score(pipe, ds.X, ds.y, cv=5, scoring="f1_macro").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)
print(study.best_params)
```

Tune **only on the training split**, then evaluate the chosen config once on the held-
out test set — otherwise the test set leaks into model selection.

### Probability calibration & decision-curve analysis
`plots.plot_reliability` already draws a calibration curve. Tree ensembles (especially
RF) are often *over-confident*; wrapping a model in
`sklearn.calibration.CalibratedClassifierCV` can make `predict_proba` trustworthy —
which matters if a clinician reads "91% hypothyroid" literally.

---

## 4. Further directions (research-grade)

- **External validation.** Both diseases are single-source. The real test of
  generalisation is a *second* hospital's data — the single most important next step.
- **Cost-sensitive learning.** Missing a `leukemia`/`hyperthyroid` case is far costlier
  than a false alarm; encode that asymmetry in the loss instead of treating all errors
  equally.
- **SHAP interaction values.** `shap.TreeExplainer(...).shap_interaction_values` exposes
  pairwise feature interactions (e.g. TSH×FTI), a deeper view than main effects.
- **Counterfactual explanations.** "What minimal change to TSH would flip this patient
  from hypothyroid to negative?" complements SHAP's attribution with actionability.
- **Temporal / longitudinal modelling.** A single blood draw is a snapshot; trends over
  time carry more diagnostic signal than any one panel.

---

## 5. The one-paragraph takeaway

The anemia study teaches the *method*; pushing it onto a larger, messier, more
imbalanced disease teaches the *engineering and the honesty* — handling missing data,
respecting imbalance with macro metrics, refusing to oversell a statistically
insignificant gain, and — above all — making every prediction **explain itself** so a
human can check the machine against the medicine.
