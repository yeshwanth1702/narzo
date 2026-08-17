"""
data_preprocessing.py
======================
Loads reactor_data.csv and performs cleaning, missing value treatment,
outlier detection/removal, feature scaling, and train/test splitting.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config


def load_data(path=config.DATA_PATH):
    """Load the raw CSV file."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_columns(df):
    """Ensure all expected feature/target columns exist in the dataset."""
    required = config.FEATURE_COLS + config.TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Update config.py to match your actual reactor_data.csv headers."
        )


def remove_duplicates(df):
    """Drop exact duplicate rows (repeated steady-state logs)."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows")
    return df


def treat_missing_values(df, columns, strategy=config.MISSING_VALUE_STRATEGY):
    """Impute missing numeric values (median by default — robust to skew)."""
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df, imputer


def remove_outliers_iqr(df, columns, multiplier=config.OUTLIER_IQR_MULTIPLIER):
    """Remove rows where any of `columns` fall outside the Tukey IQR fence."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask &= df[col].between(lower, upper)
    removed = (~mask).sum()
    print(f"[INFO] Outlier removal: dropped {removed} rows "
          f"({removed / len(df):.1%} of data)")
    return df[mask].reset_index(drop=True)


def scale_features(X_train, X_test):
    """Standardize features (zero mean, unit variance)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_dataset(path=config.DATA_PATH):
    """
    Full preprocessing pipeline. Returns train/test splits (scaled + raw),
    plus fitted imputer/scaler objects for reuse at inference time.
    """
    df = load_data(path)
    validate_columns(df)

    all_cols = config.FEATURE_COLS + config.TARGET_COLS
    df = df[all_cols].copy()

    df = remove_duplicates(df)
    df, imputer = treat_missing_values(df, all_cols)
    df = remove_outliers_iqr(df, all_cols)

    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "imputer": imputer, "scaler": scaler,
        "clean_df": df,
    }


if __name__ == "__main__":
    data = prepare_dataset()
    print("[INFO] Preprocessing complete.")
    print(f"Train shape: {data['X_train'].shape}, Test shape: {data['X_test'].shape}")
