"""
models.py
=========
Defines per-target regression and classification models (Random Forest,
XGBoost, LightGBM, CatBoost, MLP), their hyperparameter search spaces,
and a helper to run optional tuning.

Design note: every model here is a plain SINGLE-output estimator. The
app trains one instance per target column rather than wrapping
everything in one joint multi-output estimator (e.g. sklearn's
MultiOutputRegressor). That's what makes it possible for some targets
to be regression and others classification within the same run, and
for the number of targets to be anything the user selects — not fixed
to 2.
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

RANDOM_STATE = 42


def get_regression_models():
    """Unfitted regressors, one per candidate algorithm."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, objective="reg:squarederror",
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            random_state=RANDOM_STATE, verbose=False,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=2000, random_state=RANDOM_STATE,
        ),
    }


def get_classification_models():
    """Unfitted classifiers, one per candidate algorithm."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            random_state=RANDOM_STATE, verbose=False,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=2000, random_state=RANDOM_STATE,
        ),
    }


# ------------------------------------------------------------------
# Hyperparameter search spaces — only used when tuning is switched on
# ------------------------------------------------------------------
_SHARED_PARAM_DIST = {
    "RandomForest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300, 400],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "CatBoost": {
        "iterations": [200, 300, 400],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
    },
    "MLP": {
        "hidden_layer_sizes": [(32,), (64,), (64, 32), (64, 32, 16)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.005, 0.01],
    },
}

REGRESSION_PARAM_DIST = _SHARED_PARAM_DIST
CLASSIFICATION_PARAM_DIST = _SHARED_PARAM_DIST


def tune_model(estimator, param_dist, X, y, task, n_iter=8, cv=3, random_state=RANDOM_STATE):
    """
    Run a small RandomizedSearchCV over `param_dist`.

    Returns (unfitted_best_clone, best_params). The caller decides when
    to actually .fit() it (once for CV reporting, once for the final
    train/test evaluation) — this function never returns a model
    that's already been fit on the full X/y, to avoid leaking test data.
    """
    scoring = "r2" if task == "regression" else "accuracy"
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=n_iter, cv=cv, scoring=scoring,
        random_state=random_state, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    return clone(search.best_estimator_), search.best_params_
