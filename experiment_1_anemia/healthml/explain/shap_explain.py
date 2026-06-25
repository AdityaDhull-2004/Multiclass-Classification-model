"""SHAP explanations: global feature attribution + detailed local "why".

SHAP (SHapley Additive exPlanations) assigns each feature a contribution so that

    f(x)  =  base_value  +  sum_j  phi_j(x)            (Eq. 11)

For a tree model and a chosen class, ``base_value`` is the model's average score
for that class and ``phi_j`` is how much feature *j* of *this* patient moved the
score up or down. Summing all contributions reproduces the model's raw score.
This is exactly what lets us answer **"why is this prediction what it is"**:
we read off the signed contributions, attach the patient's actual values, and
turn them into a sentence.

We explain a single tree model (SHAP is applied to the XGBoost component of the
hybrid). SHAP's :class:`~shap.TreeExplainer` is exact and fast for trees, so no
sampling approximation is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from healthml.config import FIGURES_DIR  # noqa: E402


def _extract_tree_model(pipeline: Pipeline, prefer: str = "xgb"):
    """Return (fitted_preprocessor, tree_classifier) from a fitted pipeline.

    Handles a plain ``[prep, clf]`` pipeline and the hybrid whose ``clf`` is a
    ``VotingClassifier`` (we pull out its named ``xgb`` / ``rf`` sub-estimator,
    explaining the XGBoost component).
    """
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "named_estimators_"):  # VotingClassifier / StackingClassifier
        if prefer in clf.named_estimators_:
            clf = clf.named_estimators_[prefer]
        else:
            clf = next(iter(clf.named_estimators_.values()))
    return prep, clf


@dataclass
class LocalExplanation:
    index: int
    true_label: str
    predicted_label: str
    predicted_proba: float
    base_value: float
    final_score: float
    contributions: pd.DataFrame  # columns: feature, value, shap, direction
    narrative: str


class ShapExplainer:
    """Compute and render SHAP explanations for a fitted tree pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        X_background: pd.DataFrame,
        class_names: list[str],
        *,
        prefer_estimator: str = "xgb",
    ):
        self.pipeline = pipeline
        self.class_names = class_names
        self.prep, self.tree_model = _extract_tree_model(pipeline, prefer_estimator)

        # Transform features into the model's input space and keep readable names.
        self.feature_names = list(self.prep.get_feature_names_out())
        self._Xbg_raw = X_background
        self._Xbg = pd.DataFrame(
            self.prep.transform(X_background),
            columns=self.feature_names,
            index=X_background.index,
        )

        self.explainer = shap.TreeExplainer(self.tree_model)
        # exp.values: (n, n_features, n_classes); exp.base_values: (n, n_classes)
        self._exp = self.explainer(self._Xbg)
        self._values = np.asarray(self._exp.values)
        self._base = np.asarray(self._exp.base_values)
        # Normalise to 3-D (n, n_features, n_classes) for uniform indexing.
        if self._values.ndim == 2:  # binary / single-output edge case
            self._values = self._values[:, :, None]
            self._base = self._base.reshape(self._base.shape[0], -1)

    # ------------------------------------------------------------------ #
    # Global explanations
    # ------------------------------------------------------------------ #
    def global_importance(self) -> pd.Series:
        """Mean |SHAP| per feature, averaged over samples and classes."""
        imp = np.abs(self._values).mean(axis=(0, 2))
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def plot_global_bar(self, title: str, filename: str, top_n: int = 15) -> Path:
        imp = self.global_importance().head(top_n)[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(imp) + 1.5))
        ax.barh(imp.index, imp.values, color="#C44E52")
        ax.set_xlabel("mean |SHAP value|  (avg contribution magnitude)")
        ax.set_title(title)
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_beeswarm(self, class_index: int, title: str, filename: str) -> Path:
        """Beeswarm summary for one class (global directional effects)."""
        expl = shap.Explanation(
            values=self._values[:, :, class_index],
            base_values=self._base[:, class_index],
            data=self._Xbg.values,
            feature_names=self.feature_names,
        )
        fig = plt.figure()
        shap.plots.beeswarm(expl, max_display=15, show=False)
        plt.title(title)
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_interaction_summary(
        self,
        class_index: int,
        title: str,
        filename: str,
        *,
        max_display: int = 7,
        max_samples: int = 700,
    ) -> Path:
        """SHAP *interaction*-value summary plot (the N x N matrix of beeswarms).

        This reproduces the layout used in the reference report: the top
        ``max_display`` features are shown on both axes; each cell is a horizontal
        beeswarm whose x-axis is the **SHAP interaction value**. Diagonal cells are
        a feature's main effect; off-diagonal cells are pairwise interaction
        effects (TreeSHAP splits each contribution into main + interaction parts).

        Interaction values are computed for a single class output (``class_index``)
        on a random sub-sample (for speed and legibility), since exact interaction
        values cost an extra factor of ``n_features`` over plain SHAP.
        """
        n = self._Xbg.shape[0]
        if n > max_samples:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, max_samples, replace=False))
        else:
            idx = np.arange(n)
        X_sub = self._Xbg.iloc[idx]

        inter = self.explainer.shap_interaction_values(X_sub)
        # Normalise multi-class output to a single (n, f, f) array for this class.
        if isinstance(inter, list):
            arr = np.asarray(inter[class_index])
        else:
            inter = np.asarray(inter)
            if inter.ndim == 4:
                # Either (n, f, f, classes) or (classes, n, f, f).
                if inter.shape[-1] == len(self.class_names):
                    arr = inter[..., class_index]
                else:
                    arr = inter[class_index]
            else:
                arr = inter  # already (n, f, f)

        # summary_plot builds its own (sub-plotted) figure for interaction values,
        # so we must grab the current figure afterwards rather than pre-creating one.
        shap.summary_plot(arr, X_sub, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle(title, y=1.02, fontsize=11)
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # Local (per-prediction) explanations -- the detailed "why"
    # ------------------------------------------------------------------ #
    def explain_instance(
        self, i: int, y_true: np.ndarray | None = None, top_k: int = 8
    ) -> LocalExplanation:
        """Explain why row ``i`` of the background set got its prediction.

        We explain the **predicted** class: read the SHAP contributions for that
        class, attach the patient's actual (pre-transform) values where possible,
        sort by magnitude, and compose a narrative.
        """
        x_raw = self._Xbg_raw.iloc[[i]]
        proba = self.pipeline.predict_proba(x_raw)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.class_names[pred_idx]
        true_label = (
            self.class_names[int(y_true[i])] if y_true is not None else "unknown"
        )

        shap_c = self._values[i, :, pred_idx]
        base_c = float(self._base[i, pred_idx])
        final_c = base_c + float(shap_c.sum())

        # Original clinical values for readability (fall back to transformed).
        orig_row = self._Xbg_raw.iloc[i]
        trans_row = self._Xbg.iloc[i]

        def value_for(feat: str):
            if feat in orig_row.index:
                return orig_row[feat]
            return trans_row[feat]

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": [value_for(f) for f in self.feature_names],
                "shap": shap_c,
            }
        )
        df["abs"] = df["shap"].abs()
        df = df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
        df["direction"] = np.where(df["shap"] >= 0, "towards", "away")

        narrative = self._narrate(
            i, true_label, pred_label, proba[pred_idx], base_c, final_c, df.head(top_k)
        )
        return LocalExplanation(
            index=i,
            true_label=true_label,
            predicted_label=pred_label,
            predicted_proba=float(proba[pred_idx]),
            base_value=base_c,
            final_score=final_c,
            contributions=df,
            narrative=narrative,
        )

    @staticmethod
    def _fmt_value(v) -> str:
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            if np.isnan(v):
                return "missing"
            return f"{v:.3g}"
        return str(v)

    def _narrate(
        self, i, true_label, pred_label, proba, base, final, top: pd.DataFrame
    ) -> str:
        lines = [
            f"Patient #{i}  ->  predicted: {pred_label}  "
            f"(confidence {proba:.1%};  true label: {true_label})",
            f"The model's baseline score for '{pred_label}' is {base:+.2f}. "
            f"This patient's measurements adjust it as follows:",
        ]
        for _, r in top.iterrows():
            arrow = "increases" if r["shap"] >= 0 else "decreases"
            lines.append(
                f"   {r['feature']:<28} = {self._fmt_value(r['value']):<10} "
                f"{arrow} evidence for {pred_label} by {r['shap']:+.2f}"
            )
        lines.append(
            f"Net score for '{pred_label}' = {final:+.2f}, the highest of all "
            f"{len(self.class_names)} classes -> final prediction: {pred_label}."
        )
        return "\n".join(lines)

    def plot_waterfall(
        self, i: int, filename: str, class_index: int | None = None, max_display: int = 12
    ) -> Path:
        """Waterfall plot: stacked feature contributions for one prediction."""
        if class_index is None:
            proba = self.pipeline.predict_proba(self._Xbg_raw.iloc[[i]])[0]
            class_index = int(np.argmax(proba))
        expl = shap.Explanation(
            values=self._values[i, :, class_index],
            base_values=self._base[i, class_index],
            data=self._Xbg.iloc[i].values,
            feature_names=self.feature_names,
        )
        fig = plt.figure()
        shap.plots.waterfall(expl, max_display=max_display, show=False)
        plt.title(f"Why patient #{i} -> {self.class_names[class_index]}")
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
