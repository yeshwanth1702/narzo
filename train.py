"""
train.py
========
Main training script: preprocesses data, trains & cross-validates all
five candidate models, evaluates on the held-out test set, generates
diagnostic plots, runs SHAP analysis, and saves the best model + all
preprocessing artifacts (imputer, scaler) via Joblib.
"""

import joblib
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

import config
from data_preprocessing import prepare_dataset
from models import get_regression_models
from evaluate import (
    cross_validate_model, compute_metrics,
    plot_actual_vs_predicted, plot_residuals,
    plot_correlation_heatmap, plot_feature_importance,
)
from shap_analysis import run_shap_analysis


def main():
    # 1. Preprocess data --------------------------------------------------
    data = prepare_dataset()
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"].values
    y_test = data["y_test"].values

    plot_correlation_heatmap(data["clean_df"])

    # 2. Train & cross-validate every candidate model ---------------------
    # This batch pipeline predicts config.TARGET_COLS jointly, so every
    # (single-output) regressor from models.py is wrapped in sklearn's
    # MultiOutputRegressor here. The Streamlit app instead trains one
    # model per target directly — see streamlit_app.py for that path.
    models = {name: MultiOutputRegressor(m) for name, m in get_regression_models().items()}
    test_metrics_all = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")

        # 5-fold CV on the training data
        cv_summary, _ = cross_validate_model(model, X_train_scaled, y_train)
        print(f"[CV] {name} results:\n{cv_summary}\n")

        # Fit on the full training set, evaluate on the held-out test set
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        test_metrics = compute_metrics(y_test, preds, config.TARGET_COLS)
        test_metrics_all[name] = test_metrics
        print(f"[TEST] {name} results:\n{test_metrics}\n")

        trained_models[name] = model

        # Diagnostic plots
        plot_actual_vs_predicted(y_test, preds, config.TARGET_COLS, name)
        plot_residuals(y_test, preds, config.TARGET_COLS, name)
        plot_feature_importance(model, config.FEATURE_COLS, config.TARGET_COLS, name)

        # SHAP (tree-based models only; skipped automatically for MLP)
        run_shap_analysis(
            model, X_test_scaled, config.FEATURE_COLS, config.TARGET_COLS, name
        )

    # 3. Rank models by mean R2 (averaged across both targets) ------------
    ranking = {
        name: metrics_df["R2"].mean()
        for name, metrics_df in test_metrics_all.items()
    }
    best_name = max(ranking, key=ranking.get)
    print(f"\n[RESULT] Best model on test R2: {best_name} ({ranking[best_name]:.4f})")
    print("\nAll model rankings (mean test R2 across both targets):")
    for name, score in sorted(ranking.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")

    # 4. Persist best model + preprocessing artifacts ----------------------
    joblib.dump(trained_models[best_name], f"{config.MODEL_DIR}/best_model_{best_name}.pkl")
    joblib.dump(data["scaler"], f"{config.MODEL_DIR}/scaler.pkl")
    joblib.dump(data["imputer"], f"{config.MODEL_DIR}/imputer.pkl")
    joblib.dump(best_name, f"{config.MODEL_DIR}/best_model_name.pkl")

    # Save every trained model too, in case side-by-side comparison is needed
    for name, model in trained_models.items():
        joblib.dump(model, f"{config.MODEL_DIR}/model_{name}.pkl")

    print(f"\n[INFO] Best model ({best_name}) and preprocessing artifacts saved to "
          f"{config.MODEL_DIR}")

    # Save metric tables for reference
    summary_rows = []
    for name, df in test_metrics_all.items():
        df = df.copy()
        df["Model"] = name
        summary_rows.append(df)
    pd.concat(summary_rows, ignore_index=True).to_csv(
        f"{config.MODEL_DIR}/test_metrics_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
