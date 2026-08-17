"""
evaluate.py
===========
Cross-validation, metric computation, and diagnostic plotting utilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone

import config


def compute_metrics(y_true, y_pred, target_names):
    """Return per-target R2, MAE, RMSE as a DataFrame."""
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for i, name in enumerate(target_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rows.append({"Target": name, "R2": r2, "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(rows)


def cross_validate_model(model, X, y, n_splits=config.CV_FOLDS):
    """
    Manual K-fold CV (works uniformly across all 5 model types).
    Returns per-fold metrics and a mean/std summary per target.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    X = np.asarray(X)
    y = np.asarray(y)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_val)

        m = compute_metrics(y_val, preds, config.TARGET_COLS)
        m["Fold"] = fold
        fold_metrics.append(m)

    all_folds = pd.concat(fold_metrics, ignore_index=True)
    summary = all_folds.groupby("Target")[["R2", "MAE", "RMSE"]].agg(["mean", "std"])
    return summary, all_folds


def plot_actual_vs_predicted(y_true, y_pred, target_names, model_name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, axes = plt.subplots(1, len(target_names), figsize=(6 * len(target_names), 5))
    if len(target_names) == 1:
        axes = [axes]
    for i, name in enumerate(target_names):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, edgecolor="k")
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max())]
        ax.plot(lims, lims, "r--", label="Ideal (y=x)")
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{model_name}: Actual vs Predicted - {name}")
        ax.legend()
    plt.tight_layout()
    out_path = os.path.join(config.PLOT_DIR, f"{model_name}_actual_vs_predicted.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_residuals(y_true, y_pred, target_names, model_name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, axes = plt.subplots(1, len(target_names), figsize=(6 * len(target_names), 5))
    if len(target_names) == 1:
        axes = [axes]
    for i, name in enumerate(target_names):
        residuals = y_true[:, i] - y_pred[:, i]
        ax = axes[i]
        ax.scatter(y_pred[:, i], residuals, alpha=0.5, edgecolor="k")
        ax.axhline(0, color="r", linestyle="--")
        ax.set_xlabel(f"Predicted {name}")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(f"{model_name}: Residuals - {name}")
    plt.tight_layout()
    out_path = os.path.join(config.PLOT_DIR, f"{model_name}_residuals.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_correlation_heatmap(df):
    plt.figure(figsize=(9, 7))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature / Target Correlation Heatmap")
    plt.tight_layout()
    out_path = os.path.join(config.PLOT_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_feature_importance(model, feature_names, target_names, model_name):
    """
    Plots feature importances if the underlying estimator(s) expose
    `.feature_importances_` (tree-based models). Silently skipped for MLP.
    """
    estimators = getattr(model, "estimators_", [model])  # MultiOutputRegressor sub-models
    fig, axes = plt.subplots(1, len(target_names), figsize=(6 * len(target_names), 5))
    if len(target_names) == 1:
        axes = [axes]

    plotted = False
    for i, (est, name) in enumerate(zip(estimators, target_names)):
        importances = getattr(est, "feature_importances_", None)
        if importances is None:
            continue
        plotted = True
        order = np.argsort(importances)[::-1]
        ax = axes[i]
        ax.barh(np.array(feature_names)[order], importances[order])
        ax.invert_yaxis()
        ax.set_title(f"{model_name}: Feature Importance - {name}")

    if plotted:
        plt.tight_layout()
        out_path = os.path.join(config.PLOT_DIR, f"{model_name}_feature_importance.png")
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] Saved plot: {out_path}")
    plt.close()
    return plotted
