"""
shap_analysis.py
=================
SHAP-based explainability for tree-based models (RandomForest, XGBoost,
LightGBM, CatBoost). Automatically skipped for MLP, since SHAP's fast
TreeExplainer doesn't apply to neural nets (KernelExplainer would work
but is too slow to run by default on every training pass).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import config

TREE_BASED_MODELS = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}


def run_shap_analysis(model, X_sample, feature_names, target_names, model_name):
    """Generate SHAP summary plots, one per target output."""
    if model_name not in TREE_BASED_MODELS:
        print(f"[INFO] Skipping SHAP for {model_name} (not tree-based).")
        return

    # Cap sample size for speed on large test sets
    if len(X_sample) > 500:
        X_sample = X_sample[:500]

    estimators = getattr(model, "estimators_", None)

    if estimators is not None:
        # MultiOutputRegressor case: one sub-model per target
        for est, target in zip(estimators, target_names):
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_sample)
            _save_summary(shap_values, X_sample, feature_names, target, model_name)
    else:
        # CatBoost native MultiRMSE: single model, multi-dimensional output
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            for sv, target in zip(shap_values, target_names):
                _save_summary(sv, X_sample, feature_names, target, model_name)
        else:
            for i, target in enumerate(target_names):
                _save_summary(shap_values[:, :, i], X_sample, feature_names, target, model_name)


def _save_summary(shap_values, X_sample, feature_names, target, model_name):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title(f"{model_name}: SHAP Summary - {target}")
    out_path = os.path.join(config.PLOT_DIR, f"{model_name}_shap_{target}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved SHAP plot: {out_path}")
