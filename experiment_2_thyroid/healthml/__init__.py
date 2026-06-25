"""healthml -- a reproducible, explainable multi-class disease classification toolkit.

This package implements a hybrid Random-Forest + XGBoost soft-voting framework
for multi-class anemia classification using CBC data (Experiment 1) and extends
it to a larger, more complex thyroid-disease problem with deep SHAP explanations
(Experiment 2).

The sub-packages are deliberately small and single-purpose so each ML concept is
easy to locate and study:

    healthml.data        -- loading, cleaning, encoding and splitting datasets
    healthml.models      -- base learners and the hybrid soft-voting ensemble
    healthml.evaluation  -- metrics, cross-validation, statistics and plots
    healthml.explain     -- global and per-prediction SHAP explanations
"""

from healthml import config

__all__ = ["config"]
__version__ = "0.1.0"
