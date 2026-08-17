"""
streamlit_app.py
=================
Interactive, live version of the reactor MIMO ML pipeline.

Unlike the original version, this app does NOT assume exactly 2 fixed
numeric outputs. You choose:
  - any number of feature columns (numeric or categorical)
  - any number of target columns (numeric or categorical)
For each target, the task type (Regression vs Classification) is
auto-detected from the data but you can override it in the sidebar.
Regression targets are modeled with RandomForest/XGBoost/LightGBM/
CatBoost/MLP regressors; classification targets use the classifier
counterparts of the same 5 algorithms. Optional hyperparameter tuning
(RandomizedSearchCV) can be switched on per run from the sidebar.

Run:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import joblib
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

from models import (
    get_regression_models, get_classification_models,
    REGRESSION_PARAM_DIST, CLASSIFICATION_PARAM_DIST, tune_model,
)

RANDOM_STATE = 42
TREE_BASED_MODELS = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}

st.set_page_config(page_title="Reactor MIMO ML", layout="wide", page_icon="⚗️")

if "trained" not in st.session_state:
    st.session_state.trained = False

# ----------------------------------------------------------------------
# Synthetic demo data — includes numeric + categorical features, and
# both regression and classification targets, so the flexible column
# mapping / mixed task-type logic can actually be exercised.
# ----------------------------------------------------------------------
def make_synthetic_df(n=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    pressure = rng.uniform(1.5, 8.0, n)
    temperature = rng.uniform(850, 1100, n)
    molar_ratio = rng.uniform(2.0, 5.0, n)
    flow_h2 = rng.uniform(50, 200, n)
    flow_sicl4 = rng.uniform(20, 100, n)
    catalyst_type = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    catalyst_bonus = np.select(
        [catalyst_type == "A", catalyst_type == "B", catalyst_type == "C"], [0, 3, 6]
    )

    conversion = (
        40 + 3.5 * (temperature - 850) / 250 + 5 * molar_ratio - 1.2 * pressure
        + catalyst_bonus + rng.normal(0, 2, n)
    ).clip(0, 100)
    productivity = (
        10 + 0.15 * flow_h2 + 0.08 * flow_sicl4 + 0.5 * pressure
        - 0.01 * (temperature - 950) ** 2 / 50 + rng.normal(0, 1.5, n)
    ).clip(0, None)

    # Binary classification target derived from conversion + noise
    batch_quality = np.where(conversion + rng.normal(0, 3, n) >= 55, "Pass", "Fail")
    # Multiclass classification target
    grade = pd.cut(
        conversion, bins=[-1, 45, 58, 200], labels=["Low", "Medium", "High"]
    ).astype(str)

    df = pd.DataFrame({
        "Reactor_Pressure_bar": pressure,
        "Reactor_Temperature_C": temperature,
        "Feed_Molar_Ratio": molar_ratio,
        "Flow_H2_Nm3h": flow_h2,
        "Flow_SiCl4_kgh": flow_sicl4,
        "Catalyst_Type": catalyst_type,
        "Conversion_pct": conversion,
        "Productivity_kg_m3h": productivity,
        "Batch_Quality": batch_quality,
        "Conversion_Grade": grade,
    })

    for col in ["Reactor_Pressure_bar", "Reactor_Temperature_C", "Feed_Molar_Ratio",
                "Flow_H2_Nm3h", "Flow_SiCl4_kgh"]:
        idx = rng.choice(n, size=int(0.01 * n), replace=False)
        df.loc[idx, col] = np.nan
    out_idx = rng.choice(n, size=10, replace=False)
    df.loc[out_idx, "Productivity_kg_m3h"] *= 5
    return df


# ----------------------------------------------------------------------
# Task-type inference
# ----------------------------------------------------------------------
def infer_task(series):
    """Guess 'regression' vs 'classification' for a column."""
    s = series.dropna()
    if pd.api.types.is_numeric_dtype(s):
        n_unique = s.nunique()
        looks_discrete = n_unique <= 10 and np.allclose(s, s.round())
        return "classification" if looks_discrete else "regression"
    return "classification"


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def build_preprocessor(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            categorical_features,
        ))
    return ColumnTransformer(transformers)


def get_feature_names(preprocessor, numeric_features, categorical_features):
    names = list(numeric_features)
    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names += list(ohe.get_feature_names_out(categorical_features))
    return names


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred, task):
    if task == "regression":
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def safe_cv_splits(y, task, desired=5):
    if task == "regression":
        return desired
    counts = pd.Series(y).value_counts()
    return max(2, min(desired, int(counts.min())))


def cross_validate_target(model, X, y, task, n_splits=5):
    y = np.asarray(y)
    n_splits = safe_cv_splits(y, task, n_splits)
    if task == "regression":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y)

    fold_scores = []
    for tr_idx, val_idx in split_iter:
        m = clone(model)
        m.fit(X[tr_idx], y[tr_idx])
        preds = m.predict(X[val_idx])
        fold_scores.append(compute_metrics(y[val_idx], preds, task))
    return pd.DataFrame(fold_scores).mean().to_dict()


# ----------------------------------------------------------------------
# Sidebar — data source, column mapping, task types, tuning toggle
# ----------------------------------------------------------------------
st.sidebar.title("⚗️ Reactor MIMO ML")
st.sidebar.caption("Any number of inputs & outputs · mixed regression + classification")

data_source = st.sidebar.radio("Data source", ["Upload CSV", "Use synthetic demo data"])

raw_df = None
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
else:
    n_points = st.sidebar.slider("Number of synthetic points", 20, 5000, 2000, step=10)
    raw_df = make_synthetic_df(n_points)
    st.sidebar.caption("Includes numeric + categorical features, plus regression AND "
                        "classification target columns, so you can try mixed setups.")

if raw_df is None:
    st.title("Reactor Performance — Flexible MIMO ML")
    st.info("⬅️ Upload a CSV, or switch to synthetic demo data in the sidebar, to get started.")
    st.stop()

all_cols = raw_df.columns.tolist()

st.sidebar.markdown("---")
st.sidebar.subheader("Column mapping")

target_cols = st.sidebar.multiselect(
    "Target column(s) — outputs to predict", all_cols,
    default=[c for c in ["Conversion_pct", "Productivity_kg_m3h"] if c in all_cols],
)
feature_candidates = [c for c in all_cols if c not in target_cols]
feature_cols = st.sidebar.multiselect(
    "Feature column(s) — inputs", feature_candidates, default=feature_candidates,
)

if not target_cols or not feature_cols:
    st.warning("Select at least one feature column and one target column in the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Target task types")
task_types = {}
for t in target_cols:
    default_task = infer_task(raw_df[t])
    choice = st.sidebar.selectbox(
        f"'{t}'", ["Regression", "Classification"],
        index=0 if default_task == "regression" else 1,
        key=f"task_{t}",
    )
    task_types[t] = "regression" if choice == "Regression" else "classification"

st.sidebar.markdown("---")
outlier_mult = st.sidebar.slider("Outlier IQR multiplier (numeric regression targets/features only)", 1.0, 3.0, 1.5, 0.1)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

st.sidebar.markdown("---")
tune_enabled = st.sidebar.checkbox("🎛️ Enable hyperparameter tuning (RandomizedSearchCV)", value=False)
if tune_enabled:
    st.sidebar.caption("Runs a small randomized search per model/target before the final fit. "
                        "Slower, but usually improves results.")

numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(raw_df[c])]
categorical_features = [c for c in feature_cols if c not in numeric_features]

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
st.title("⚗️ Reactor Performance — Flexible MIMO ML")
tab_data, tab_train, tab_diag, tab_predict = st.tabs(
    ["1 · Data", "2 · Train & Compare", "3 · Diagnostics", "4 · Live Prediction"]
)

# ======================================================================
# TAB 1 — DATA
# ======================================================================
with tab_data:
    st.subheader("Raw data preview")
    st.dataframe(raw_df.head(20), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(raw_df))
    c2.metric("Missing values", int(raw_df[feature_cols + target_cols].isna().sum().sum()))
    c3.metric("Duplicate rows", int(raw_df.duplicated().sum()))

    st.subheader("Detected task types")
    st.dataframe(
        pd.DataFrame({"Target": target_cols, "Task": [task_types[t] for t in target_cols]}),
        width="stretch", hide_index=True,
    )

    st.subheader("Cleaning")
    work_df = raw_df[feature_cols + target_cols].copy()
    before = len(work_df)
    work_df = work_df.drop_duplicates()
    st.write(f"- Dropped **{before - len(work_df)}** duplicate rows")

    # Impute: numeric -> median, categorical -> most frequent (done per dtype group)
    all_numeric = [c for c in (feature_cols + target_cols)
                   if pd.api.types.is_numeric_dtype(work_df[c])]
    all_categorical = [c for c in (feature_cols + target_cols) if c not in all_numeric]
    if all_numeric:
        work_df[all_numeric] = SimpleImputer(strategy="median").fit_transform(work_df[all_numeric])
    if all_categorical:
        work_df[all_categorical] = SimpleImputer(strategy="most_frequent").fit_transform(work_df[all_categorical])
    st.write("- Imputed missing values (median for numeric, most-frequent for categorical)")

    # Outlier removal — only on numeric columns that are features, or targets marked regression
    outlier_cols = [c for c in numeric_features] + \
                   [t for t in target_cols if task_types[t] == "regression" and t in all_numeric]
    if outlier_cols:
        mask = pd.Series(True, index=work_df.index)
        for col in outlier_cols:
            q1, q3 = work_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - outlier_mult * iqr, q3 + outlier_mult * iqr
            mask &= work_df[col].between(lower, upper)
        n_removed = int((~mask).sum())
        work_df = work_df[mask].reset_index(drop=True)
    else:
        n_removed = 0
    st.write(f"- Removed **{n_removed}** outlier rows (Tukey IQR × {outlier_mult})")
    st.write(f"- Clean dataset: **{len(work_df)}** rows")

    numeric_for_corr = [c for c in (feature_cols + target_cols)
                        if pd.api.types.is_numeric_dtype(work_df[c])]
    if len(numeric_for_corr) >= 2:
        st.subheader("Correlation heatmap (numeric columns)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(work_df[numeric_for_corr].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", square=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.session_state.clean_df = work_df

# ======================================================================
# TAB 2 — TRAIN & COMPARE
# ======================================================================
with tab_train:
    st.subheader("Train & compare all 5 algorithms — per target")
    st.caption("Each target gets its own set of 5 models (regressors or classifiers, "
               "depending on its task type). 5-fold CV, then evaluated on the held-out test set.")

    if st.button("🚀 Train all models", type="primary"):
        df = st.session_state.clean_df
        idx_train, idx_test = train_test_split(
            df.index, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
        )

        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_train_raw = df.loc[idx_train, feature_cols]
        X_test_raw = df.loc[idx_test, feature_cols]
        X_train_t = preprocessor.fit_transform(X_train_raw)
        X_test_t = preprocessor.transform(X_test_raw)
        X_train_t = np.asarray(X_train_t.todense()) if hasattr(X_train_t, "todense") else np.asarray(X_train_t)
        X_test_t = np.asarray(X_test_t.todense()) if hasattr(X_test_t, "todense") else np.asarray(X_test_t)
        feat_names_out = get_feature_names(preprocessor, numeric_features, categorical_features)

        trained_models, test_metrics_all, cv_metrics_all = {}, {}, {}
        label_encoders, best_params_all = {}, {}
        best_name_per_target = {}

        total_steps = len(target_cols) * 5
        step = 0
        progress = st.progress(0.0, text="Starting...")

        for target in target_cols:
            task = task_types[target]
            y_full = df[target]

            if task == "classification":
                le = LabelEncoder()
                y_full_enc = le.fit_transform(y_full.astype(str))
                label_encoders[target] = le
            else:
                y_full_enc = y_full.values.astype(float)

            y_train = y_full_enc[df.index.get_indexer(idx_train)]
            y_test = y_full_enc[df.index.get_indexer(idx_test)]

            model_defs = get_regression_models() if task == "regression" else get_classification_models()
            param_dist = REGRESSION_PARAM_DIST if task == "regression" else CLASSIFICATION_PARAM_DIST

            trained_models[target] = {}
            test_metrics_all[target] = {}
            cv_metrics_all[target] = {}
            best_params_all[target] = {}

            for name, model in model_defs.items():
                step += 1
                progress.progress(step / total_steps, text=f"Training {target} · {name}...")

                used_model = model
                if tune_enabled:
                    try:
                        used_model, best_params = tune_model(
                            model, param_dist[name], X_train_t, y_train, task
                        )
                        best_params_all[target][name] = best_params
                    except Exception as e:
                        st.warning(f"Tuning failed for {target}/{name} ({e}); using default hyperparameters.")
                        best_params_all[target][name] = None

                cv_summary = cross_validate_target(used_model, X_train_t, y_train, task)
                fitted = clone(used_model)
                fitted.fit(X_train_t, y_train)
                preds = fitted.predict(X_test_t)
                test_metrics = compute_metrics(y_test, preds, task)

                trained_models[target][name] = fitted
                test_metrics_all[target][name] = test_metrics
                cv_metrics_all[target][name] = cv_summary

            score_key = "R2" if task == "regression" else "Accuracy"
            best_name_per_target[target] = max(
                test_metrics_all[target], key=lambda n: test_metrics_all[target][n][score_key]
            )

        progress.progress(1.0, text="Done.")

        st.session_state.update({
            "trained": True,
            "trained_models": trained_models,
            "test_metrics_all": test_metrics_all,
            "cv_metrics_all": cv_metrics_all,
            "best_params_all": best_params_all,
            "best_name_per_target": best_name_per_target,
            "label_encoders": label_encoders,
            "preprocessor": preprocessor,
            "feat_names_out": feat_names_out,
            "X_train_t": X_train_t, "X_test_t": X_test_t,
            "idx_train": idx_train, "idx_test": idx_test,
            "task_types": task_types,
        })
        st.success("Training complete for all targets. Best model per target: " +
                   ", ".join(f"**{t}** → {n}" for t, n in best_name_per_target.items()))

    if st.session_state.trained:
        for target in target_cols:
            if target not in st.session_state.trained_models:
                continue
            task = st.session_state.task_types[target]
            score_key = "R2" if task == "regression" else "Accuracy"
            st.markdown(f"### {target} ({task})")
            rows = []
            for name, metrics in st.session_state.test_metrics_all[target].items():
                row = {"Model": name, **metrics}
                row["Best"] = "🏆" if name == st.session_state.best_name_per_target[target] else ""
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if tune_enabled:
                with st.expander(f"Best hyperparameters found — {target}"):
                    st.json(st.session_state.best_params_all[target])

        st.markdown("### Download a trained model")
        dl_target = st.selectbox("Target", target_cols, key="dl_target")
        dl_model_name = st.selectbox(
            "Model", list(st.session_state.trained_models[dl_target].keys()), key="dl_model"
        )
        buf = io.BytesIO()
        joblib.dump(st.session_state.trained_models[dl_target][dl_model_name], buf)
        st.download_button(
            f"⬇️ Download {dl_target}_{dl_model_name}.pkl",
            data=buf.getvalue(), file_name=f"{dl_target}_{dl_model_name}.pkl",
        )
    else:
        st.info("Click **Train all models** to run the comparison.")

# ======================================================================
# TAB 3 — DIAGNOSTICS
# ======================================================================
with tab_diag:
    if not st.session_state.trained:
        st.info("Train models first in the **Train & Compare** tab.")
    else:
        diag_target = st.selectbox("Target", target_cols, key="diag_target")
        task = st.session_state.task_types[diag_target]
        model_names = list(st.session_state.trained_models[diag_target].keys())
        diag_model_name = st.selectbox(
            "Model", model_names,
            index=model_names.index(st.session_state.best_name_per_target[diag_target]),
            key="diag_model",
        )
        model = st.session_state.trained_models[diag_target][diag_model_name]
        X_test_t = st.session_state.X_test_t
        idx_test = st.session_state.idx_test
        df = st.session_state.clean_df
        feat_names_out = st.session_state.feat_names_out

        if task == "classification":
            le = st.session_state.label_encoders[diag_target]
            y_test = le.transform(df.loc[idx_test, diag_target].astype(str))
        else:
            y_test = df.loc[idx_test, diag_target].values.astype(float)
        preds = model.predict(X_test_t)

        if task == "regression":
            st.markdown("### Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(y_test, preds, alpha=0.5, edgecolor="k")
            lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
            ax.plot(lims, lims, "r--", label="Ideal (y=x)")
            ax.set_xlabel(f"Actual {diag_target}"); ax.set_ylabel(f"Predicted {diag_target}")
            ax.legend()
            st.pyplot(fig); plt.close(fig)

            st.markdown("### Residuals")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            residuals = y_test - preds
            ax.scatter(preds, residuals, alpha=0.5, edgecolor="k")
            ax.axhline(0, color="r", linestyle="--")
            ax.set_xlabel(f"Predicted {diag_target}"); ax.set_ylabel("Residual")
            st.pyplot(fig); plt.close(fig)
        else:
            le = st.session_state.label_encoders[diag_target]
            st.markdown("### Confusion matrix")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(5, 4.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig); plt.close(fig)

            st.markdown("### Classification report")
            report = classification_report(
                y_test, preds, target_names=le.classes_, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

        st.markdown("### Feature importance")
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            st.caption("Not available for this model type.")
        else:
            order = np.argsort(importances)[::-1][:20]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(np.array(feat_names_out)[order], importances[order])
            ax.invert_yaxis()
            st.pyplot(fig); plt.close(fig)

        st.markdown("### SHAP summary")
        if diag_model_name not in TREE_BASED_MODELS:
            st.caption("SHAP TreeExplainer isn't available for MLP — skipped.")
        else:
            sample = X_test_t[:300]
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(sample)
            if task == "regression":
                vals = sv
            else:
                le = st.session_state.label_encoders[diag_target]
                if isinstance(sv, list):
                    class_idx = st.selectbox(
                        "Class to explain", list(range(len(sv))),
                        format_func=lambda i: le.classes_[i], key="shap_class",
                    )
                    vals = sv[class_idx]
                elif np.ndim(sv) == 3:
                    class_idx = st.selectbox(
                        "Class to explain", list(range(sv.shape[2])),
                        format_func=lambda i: le.classes_[i], key="shap_class",
                    )
                    vals = sv[:, :, class_idx]
                else:
                    vals = sv
            fig = plt.figure()
            shap.summary_plot(vals, sample, feature_names=feat_names_out, show=False)
            st.pyplot(fig); plt.close(fig)

# ======================================================================
# TAB 4 — LIVE PREDICTION
# ======================================================================
with tab_predict:
    if not st.session_state.trained:
        st.info("Train models first in the **Train & Compare** tab.")
    else:
        df = st.session_state.clean_df
        preprocessor = st.session_state.preprocessor

        st.markdown("#### Operating conditions")
        cols = st.columns(min(3, len(feature_cols)))
        inputs = {}
        for i, feat in enumerate(feature_cols):
            with cols[i % len(cols)]:
                if feat in numeric_features:
                    lo, hi = float(df[feat].min()), float(df[feat].max())
                    default = float(df[feat].median())
                    if np.isclose(lo, hi):
                        # Constant column in this dataset — a slider needs a
                        # real range, so just show the fixed value instead.
                        st.metric(feat, f"{lo:.4g}")
                        st.caption("constant in this dataset")
                        inputs[feat] = lo
                    else:
                        inputs[feat] = st.slider(feat, lo, hi, default, key=f"pred_{feat}")
                else:
                    options = sorted(df[feat].dropna().unique().tolist())
                    if len(options) == 1:
                        st.metric(feat, str(options[0]))
                        st.caption("constant in this dataset")
                        inputs[feat] = options[0]
                    else:
                        inputs[feat] = st.selectbox(feat, options, key=f"pred_{feat}")

        x_raw = pd.DataFrame([inputs])[feature_cols]
        x_t = preprocessor.transform(x_raw)
        x_t = np.asarray(x_t.todense()) if hasattr(x_t, "todense") else np.asarray(x_t)

        st.markdown("#### Predicted outputs")
        for target in target_cols:
            task = st.session_state.task_types[target]
            model_names = list(st.session_state.trained_models[target].keys())
            best = st.session_state.best_name_per_target[target]
            chosen = st.selectbox(
                f"Model for {target}", model_names, index=model_names.index(best),
                key=f"predict_model_{target}",
            )
            model = st.session_state.trained_models[target][chosen]

            if task == "regression":
                pred_value = model.predict(x_t)[0]
                lo, hi = float(df[target].min()), float(df[target].max())
                pad = (hi - lo) * 0.15
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=pred_value, title={"text": target},
                    gauge={"axis": {"range": [max(0, lo - pad), hi + pad]},
                           "bar": {"color": "#35D9A6"}},
                ))
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
                st.plotly_chart(fig, width="stretch")
            else:
                le = st.session_state.label_encoders[target]
                pred_class_idx = model.predict(x_t)[0]
                pred_label = le.classes_[int(pred_class_idx)]
                st.markdown(f"**{target}: `{pred_label}`**")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x_t)[0]
                    proba_df = pd.DataFrame({"Class": le.classes_, "Probability": proba})
                    fig, ax = plt.subplots(figsize=(4, 2.2))
                    ax.barh(proba_df["Class"], proba_df["Probability"], color="#FFAE42")
                    ax.set_xlim(0, 1)
                    st.pyplot(fig); plt.close(fig)

        st.caption(f"Predictions computed on {len(df)} cleaned rows "
                   f"({'tuned' if tune_enabled else 'default'} hyperparameters).")
