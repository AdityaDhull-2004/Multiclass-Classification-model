"""Demonstrate that the Hybrid-vs-RF paired t-test p-value is UNSTABLE.

Same dataset, same models, same protocol -- we only change the fold-shuffling
seed. If the p-value bounces around wildly, that proves the exact p (0.82 in our
run) is noise, and only the verdict ("not significant") is robust.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from healthml.data import load_anemia
from healthml.models import build_random_forest, build_hybrid
from healthml.evaluation.crossval import fold_scores
from healthml.evaluation.stats import paired_ttest

ds = load_anemia()
print(f"data: {ds.n_samples} patients, {ds.n_classes} classes\n")
print(f"{'seed':>4} | {'mean(Hyb-RF)':>12} | {'t':>7} | {'p-value':>8} | verdict")
print("-" * 56)

ps = []
for seed in [0, 1, 7, 13, 21, 42, 99, 123, 2024, 31337]:
    rf = fold_scores(build_random_forest(ds, seed=42), ds.X, ds.y, seed=seed,
                     metrics=("macro_f1",))["macro_f1"].to_numpy()
    hy = fold_scores(build_hybrid(ds, seed=42), ds.X, ds.y, seed=seed,
                     metrics=("macro_f1",))["macro_f1"].to_numpy()
    sig = paired_ttest(hy, rf)
    ps.append(sig.p_value)
    verdict = "SIGNIFICANT" if sig.significant else "not sig."
    print(f"{seed:>4} | {sig.mean_difference:>+12.4f} | {sig.statistic:>7.2f} | "
          f"{sig.p_value:>8.3f} | {verdict}")

ps = np.array(ps)
print("-" * 56)
print(f"p-value range over seeds: [{ps.min():.3f}, {ps.max():.3f}]   "
      f"median={np.median(ps):.3f}")
print(f"fraction of seeds that are 'significant' (p<0.05): {(ps < 0.05).mean():.0%}")
print("\nThe exact p is seed-dependent noise; the 'not significant' verdict is what holds.")

