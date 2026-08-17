# Reactor Performance MIMO ML Pipeline

Predicts **Conversion (%)** and **Productivity** simultaneously from reactor
operating conditions, using a Multiple-Input Multiple-Output (MIMO)
regression pipeline that compares Random Forest, XGBoost, LightGBM,
CatBoost, and an MLP neural network.

Tested and verified working end-to-end on **Python 3.12**.

## Project layout

```
reactor_mimo_ml/
├── config.py                  # Column names & all hyperparameters — EDIT THIS FIRST
├── data_preprocessing.py       # Cleaning, imputation, outlier removal, scaling
├── models.py                   # RF / XGBoost / LightGBM / CatBoost / MLP definitions
├── evaluate.py                 # Metrics, cross-validation, plots
├── shap_analysis.py             # SHAP explainability (tree models only)
├── train.py                     # Orchestrates training + evaluation + saving
├── predict.py                   # CLI / importable prediction tool
├── generate_synthetic_data.py  # OPTIONAL: makes fake data to smoke-test the pipeline
├── main.py                      # Single entry point
├── requirements.txt
├── data/                        # Put your reactor_data.csv here
├── saved_models/                # Created automatically — trained models + scaler
└── plots/                       # Created automatically — all diagnostic charts
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Point the pipeline at your real dataset

Two things to do once your `reactor_data.csv` is ready:

1. Copy it to `data/reactor_data.csv`.
2. Open `config.py` and set the column names to match your actual headers:
   - `PRESSURE_COL`, `TEMPERATURE_COL`, `MOLAR_RATIO_COL`
   - `FLOW_RATE_COLS` — a list, so you can add as many individual feed
     flow streams as your process has (H₂, SiCl₄, HCl, etc.)
   - `TARGET_COLS` — defaults to `["Conversion_pct", "Productivity_kg_m3h"]`

Everything else (preprocessing, model training, plotting, SHAP, and the
prediction script) automatically adapts to however many feature/target
columns you configure — no other code changes needed.

## 3. (Optional) Smoke-test the pipeline before your real data arrives

```bash
python main.py --demo
```

This generates a synthetic `reactor_data.csv` (with realistic-ish
relationships, some missing values, and a few outliers baked in) and
runs the full pipeline against it, so you can confirm everything works
on your machine first. **Delete `data/reactor_data.csv` afterwards and
replace it with your real data.**

## 4. Train on your real data

```bash
python main.py
```

This will:
1. Load and validate `data/reactor_data.csv`
2. Clean data: drop duplicates, median-impute missing values, remove
   outliers via the Tukey IQR rule
3. Split 80:20 train/test, standardize features
4. Train all 5 models with 5-fold cross-validation
5. Evaluate each on the held-out test set (R², MAE, RMSE — per target)
6. Generate for every model: actual-vs-predicted plots, residual plots,
   feature importance, and SHAP summary plots (tree models only), plus
   one overall correlation heatmap
7. Pick the best model (highest mean test R² across both targets) and
   save it — along with the scaler, imputer, and every trained model —
   to `saved_models/` via Joblib

All plots land in `plots/`. A `test_metrics_summary.csv` with every
model's test-set metrics is saved to `saved_models/`.

## 5. Predict on new operating conditions

Interactively:

```bash
python predict.py
```

You'll be prompted for pressure, temperature, molar ratio, and each
configured flow rate, then shown predicted Conversion and Productivity.

Programmatically:

```python
from predict import predict_performance

result = predict_performance(
    pressure=5.2,
    temperature=950,
    molar_ratio=3.1,
    flows=[120.0, 45.0],   # order must match config.FLOW_RATE_COLS
)
print(result)
# {'Conversion_pct': 74.3, 'Productivity_kg_m3h': 41.8, '_model_used': 'CatBoost'}
```

## Interactive live app (Streamlit)

For a point-and-click, browser-based version of the whole pipeline, run:

```bash
streamlit run streamlit_app.py
```

This version is fully flexible — it does **not** assume 2 fixed numeric
outputs, and it handles classification targets, not just regression:

- **Any number of inputs and outputs.** Pick as many feature columns
  and target columns as you like from the sidebar — 1 target, 5
  targets, whatever your data has.
- **Mixed regression + classification in one run.** Each target's task
  type (Regression vs Classification) is auto-detected from its data
  (numeric/continuous → regression, text or a small set of discrete
  values → classification) and shown in the sidebar, where you can
  override it per target. A run can predict a continuous `Conversion_pct`
  and a categorical `Batch_Quality` (Pass/Fail) side by side — each
  gets its own set of 5 models (regressors or classifiers) under the hood.
- **Categorical *inputs* too.** Non-numeric feature columns (e.g. a
  `Catalyst_Type` column) are automatically one-hot encoded; numeric
  features are median-imputed and standardized.
- **Optional hyperparameter tuning**, toggled per run from the sidebar
  ("🎛️ Enable hyperparameter tuning"). Off by default for speed; when
  on, each of the 5 algorithms goes through a small `RandomizedSearchCV`
  (8 iterations, 3-fold) before the final fit, and the best-found
  parameters are shown in an expander next to that target's results.

Four tabs:

1. **Data** — upload your CSV (or use the synthetic demo data, which
   includes numeric + categorical features and both regression and
   classification target columns so you can try mixed setups
   immediately), map feature/target columns and confirm each target's
   task type, and see cleaning stats + a correlation heatmap.
2. **Train & Compare** — one click trains all 5 models **per target**,
   shows the right metrics for each task type (R²/MAE/RMSE for
   regression, Accuracy/F1 for classification), and lets you download
   any trained model as a `.pkl`.
3. **Diagnostics** — pick any target + model combination. Regression
   targets show actual-vs-predicted and residual plots; classification
   targets show a confusion matrix and full classification report.
   Feature importance and SHAP summaries are available for both (for
   multi-class SHAP, pick which class to explain).
4. **Live Prediction** — sliders for numeric inputs, dropdowns for
   categorical inputs. Regression targets show a live gauge; classification
   targets show the predicted class plus a probability bar chart.

Everything runs in-session — no files are written to disk unless you
click a download button.

### How this differs from the batch CLI pipeline (`train.py` / `main.py`)

The command-line pipeline (`config.py`, `data_preprocessing.py`,
`train.py`, `predict.py`) is still **regression-only** with a fixed
column mapping in `config.py`, and doesn't do hyperparameter tuning.
It's meant for scripted/scheduled batch runs. The Streamlit app is the
flexible, interactive tool — use it for exploration, mixed-task
problems, or when you want tuning. `models.py` is shared between both,
but the Streamlit app calls its `get_regression_models()` /
`get_classification_models()` / `tune_model()` directly rather than
going through the `MultiOutputRegressor` wrapping used by the batch script.

## Notes on model choices

- **RandomForest / XGBoost / LightGBM** are wrapped in scikit-learn's
  `MultiOutputRegressor`, which trains one independent model per
  target — simple, robust, and gives clean per-target feature
  importances and SHAP values.
- **CatBoost** uses its native `MultiRMSE` loss, learning both targets
  jointly in a single model.
- **MLP** (`MLPRegressor`) is inherently multi-output and often
  captures cross-target interactions the tree ensembles miss, but
  SHAP is skipped for it here since TreeExplainer doesn't apply and
  KernelExplainer is too slow to run by default.
- Everything is re-evaluated every run — the "best" model can change
  as your dataset grows, so re-run `python main.py` whenever you add
  new operating points.

## Extending the pipeline

- Add more flow streams: just append to `FLOW_RATE_COLS` in `config.py`.
- Add more targets: append to `TARGET_COLS` — plots, metrics, and SHAP
  all loop over this list automatically.
- Swap the outlier method: edit `remove_outliers_iqr` in
  `data_preprocessing.py` (e.g. to a z-score or isolation-forest method)
  if IQR proves too aggressive/lenient for your process data.
