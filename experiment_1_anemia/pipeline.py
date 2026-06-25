"""Shared experiment driver used by both Experiment 1 (anemia) and 2 (thyroid).

One function, :func:`run_experiment`, executes the full reproducible protocol and
writes a self-contained Markdown report plus figures:

  1. Dataset summary + class-distribution figure
  2. Stratified 5-fold CV of every model (mean +/- std)        [Eq. 6]
  3. Paired t-test: Hybrid vs Random-Forest baseline           [Eq. 10]
  4. Final hold-out evaluation of the hybrid
       - accuracy, macro-F1, ROC-AUC (OvR)
       - confusion matrix, ROC curves, per-class report
  5. SHAP global importance (bar + beeswarm)                   [Eq. 11]
  6. SHAP local "why this prediction" narratives + waterfalls  [the headline goal]
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# StackingClassifier feeds numpy arrays (no column names) to LightGBM during its
# internal CV; this emits a harmless feature-name warning we silence for clean logs.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from healthml.config import CV_FOLDS, MODELS_DIR, REPORTS_DIR
from healthml.data.preprocess import Dataset, stratified_split
from healthml.evaluation import plots
from healthml.evaluation.crossval import cross_validate_models, summarize_cv
from healthml.evaluation.metrics import classification_metrics, per_class_report
from healthml.evaluation.stats import paired_ttest
from healthml.explain import ShapExplainer
from healthml.models import build_model_zoo, build_hybrid


def _pick_examples(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> list[int]:
    """One correctly-classified index per class + up to two misclassified ones."""
    chosen: list[int] = []
    for c in range(len(class_names)):
        hits = np.where((y_true == c) & (y_pred == c))[0]
        if len(hits):
            chosen.append(int(hits[0]))
    misses = np.where(y_true != y_pred)[0]
    chosen.extend(int(i) for i in misses[:2])
    return chosen


def run_experiment(
    dataset: Dataset,
    *,
    tag: str,
    title: str,
    run_extensions: bool = True,
) -> Path:
    """Run the full protocol and return the path to the Markdown report."""
    md: list[str] = [f"# {title}\n", f"_Auto-generated report for the `{tag}` experiment._\n"]

    # Reports live in outputs/reports/ and figures in outputs/figures/, so embed
    # figures by a relative path that stays valid if the repo is moved/shared.
    def rel(p: Path) -> str:
        return f"../figures/{p.name}"

    # --- 1. dataset ----------------------------------------------------- #
    md.append("## 1. Dataset\n")
    md.append("```\n" + dataset.summary() + "\n```\n")
    dist_fig = plots.plot_class_distribution(
        dataset.class_distribution(), f"{title}: class distribution", f"{tag}_class_dist.png"
    )
    md.append(f"![class distribution]({rel(dist_fig)})\n")

    # --- 2. cross-validation of the model zoo --------------------------- #
    md.append(f"## 2. Stratified {CV_FOLDS}-fold cross-validation\n")
    zoo = build_model_zoo(dataset, include_extensions=run_extensions)
    cv_results = cross_validate_models(zoo, dataset.X, dataset.y)
    summary = summarize_cv(cv_results)
    md.append(summary.to_markdown() + "\n")

    macro_means = pd.Series({k: v["macro_f1"].mean() for k, v in cv_results.items()})
    cmp_fig = plots.plot_model_comparison(
        summary, macro_means, f"{title}: mean CV macro-F1", f"{tag}_model_cmp.png"
    )
    md.append(f"\n![model comparison]({rel(cmp_fig)})\n")

    # --- 3. statistical significance (Hybrid vs RF baseline) ------------ #
    md.append("## 3. Statistical significance (paired t-test on fold macro-F1)\n")
    sig = paired_ttest(
        cv_results["Hybrid (RF+XGB)"]["macro_f1"].to_numpy(),
        cv_results["Random Forest"]["macro_f1"].to_numpy(),
    )
    md.append("```\n" + sig.describe("Hybrid (RF+XGB)", "Random Forest") + "\n```\n")

    # --- 4. final hold-out evaluation of the hybrid --------------------- #
    md.append("## 4. Hold-out test evaluation (final hybrid)\n")
    X_tr, X_te, y_tr, y_te = stratified_split(dataset)
    hybrid = build_hybrid(dataset)
    hybrid.fit(X_tr, y_tr)
    # Persist the trained hybrid pipeline so it can be reloaded without retraining
    # (joblib.load(path) returns a ready-to-predict scikit-learn Pipeline).
    model_path = MODELS_DIR / f"{tag}_hybrid_model.joblib"
    joblib.dump(hybrid, model_path, compress=3)
    md.append(f"\nTrained hybrid model saved to `outputs/models/{model_path.name}`.\n")
    proba = hybrid.predict_proba(X_te)
    pred = hybrid.predict(X_te)
    test_metrics = classification_metrics(y_te, pred, proba)
    md.append(
        "| metric | value |\n|---|---|\n"
        + "\n".join(f"| {k} | {v:.4f} |" for k, v in test_metrics.items())
        + "\n"
    )
    md.append("\n```\n" + per_class_report(y_te, pred, dataset.class_names) + "\n```\n")

    cm_fig = plots.plot_confusion_matrix(
        y_te, pred, dataset.class_names, f"{title}: confusion matrix (test)", f"{tag}_cm.png"
    )
    cmn_fig = plots.plot_confusion_matrix(
        y_te, pred, dataset.class_names, f"{title}: row-normalised", f"{tag}_cm_norm.png",
        normalize=True,
    )
    roc_fig = plots.plot_roc_ovr(
        y_te, proba, dataset.class_names, f"{title}: ROC one-vs-rest", f"{tag}_roc.png"
    )
    md.append(f"\n![confusion matrix]({rel(cm_fig)})\n")
    md.append(f"\n![confusion matrix normalised]({rel(cmn_fig)})\n")
    md.append(f"\n![roc]({rel(roc_fig)})\n")

    # --- 5. SHAP global -------------------------------------------------- #
    md.append("## 5. SHAP global explanation (XGBoost component)\n")
    X_te = X_te.reset_index(drop=True)
    y_te_arr = np.asarray(y_te)
    explainer = ShapExplainer(hybrid, X_te, dataset.class_names)
    gbar = explainer.plot_global_bar(f"{title}: global SHAP importance", f"{tag}_shap_bar.png")
    md.append(f"![shap bar]({rel(gbar)})\n")
    # SHAP interaction-value summary (matrix of beeswarms) for a clinically
    # central class; diagonal = main effects, off-diagonal = pairwise interactions.
    interesting = 1 if dataset.n_classes > 1 else 0
    inter = explainer.plot_interaction_summary(
        interesting,
        f"{title}: SHAP interaction summary ({dataset.class_names[interesting]})",
        f"{tag}_shap_interaction.png",
    )
    md.append(f"\n![shap interaction summary]({rel(inter)})\n")
    md.append("\nTop global features (mean |SHAP|):\n")
    md.append("```\n" + explainer.global_importance().head(12).to_string() + "\n```\n")

    # --- 6. SHAP local "why this prediction" ---------------------------- #
    md.append("## 6. SHAP local explanations -- why each prediction was made\n")
    for j, i in enumerate(_pick_examples(y_te_arr, pred, dataset.class_names)):
        le = explainer.explain_instance(i, y_true=y_te_arr)
        wf = explainer.plot_waterfall(i, f"{tag}_waterfall_{j}.png")
        tag_str = "CORRECT" if le.true_label == le.predicted_label else "MISCLASSIFIED"
        md.append(f"### Example {j + 1} ({tag_str})\n")
        md.append("```\n" + le.narrative + "\n```\n")
        md.append(f"![waterfall]({rel(wf)})\n")

    report_path = REPORTS_DIR / f"{tag}_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    # Persist CV scores for the docs.
    for name, df in cv_results.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        df.to_csv(REPORTS_DIR / f"{tag}_cv_{safe}.csv")
    print(f"[{tag}] report written -> {report_path}")
    return report_path

