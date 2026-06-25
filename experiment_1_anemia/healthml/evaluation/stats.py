"""Statistical significance testing of model differences.

We ask the right question: *is the hybrid's improvement over the baseline real,
or just fold-to-fold noise?* We answer it with a **paired t-test** on fold-wise
macro-F1 (Eq. 10), interpreting the result against the alpha threshold to decide
whether the difference is significant.

We also offer the Wilcoxon signed-rank test, a non-parametric alternative that
does not assume the fold differences are normally distributed (a safer choice
with only 5 folds).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from healthml.config import SIGNIFICANCE_ALPHA


@dataclass
class SignificanceResult:
    statistic: float
    p_value: float
    mean_difference: float
    n: int
    test: str
    alpha: float = SIGNIFICANCE_ALPHA

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def describe(self, name_a: str = "model A", name_b: str = "model B") -> str:
        direction = "higher" if self.mean_difference > 0 else "lower"
        verdict = (
            "STATISTICALLY SIGNIFICANT" if self.significant
            else "NOT statistically significant"
        )
        return (
            f"{self.test} on {self.n} paired folds:\n"
            f"  mean({name_a}) - mean({name_b}) = {self.mean_difference:+.4f} "
            f"({name_a} is {direction})\n"
            f"  statistic = {self.statistic:.4f}, p = {self.p_value:.4f}\n"
            f"  -> {verdict} at alpha = {self.alpha} "
            f"({'reject' if self.significant else 'fail to reject'} H0: equal performance)"
        )


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> SignificanceResult:
    """Paired (dependent) t-test on fold-wise scores (Eq. 10).

    H0: the two models have equal mean performance across folds.
    ``scores_a`` and ``scores_b`` must be aligned (fold *i* of each is the same
    split). A small p means the per-fold differences are consistently one-signed.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    diff = a - b
    t_stat, p = stats.ttest_rel(a, b)
    return SignificanceResult(
        statistic=float(t_stat),
        p_value=float(p),
        mean_difference=float(diff.mean()),
        n=len(diff),
        test="Paired t-test",
    )


def wilcoxon(scores_a: np.ndarray, scores_b: np.ndarray) -> SignificanceResult:
    """Wilcoxon signed-rank test -- non-parametric paired alternative."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    try:
        stat, p = stats.wilcoxon(a, b)
    except ValueError:  # all differences zero
        stat, p = float("nan"), 1.0
    return SignificanceResult(
        statistic=float(stat),
        p_value=float(p),
        mean_difference=float((a - b).mean()),
        n=len(a),
        test="Wilcoxon signed-rank",
    )


def describe_significance(
    fold_results: dict,
    model_a: str,
    model_b: str,
    metric: str = "macro_f1",
) -> str:
    """Convenience: run a paired t-test between two models on a CV metric."""
    a = fold_results[model_a][metric].to_numpy()
    b = fold_results[model_b][metric].to_numpy()
    return paired_ttest(a, b).describe(model_a, model_b)
