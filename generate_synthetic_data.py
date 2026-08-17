"""
generate_synthetic_data.py
============================
OPTIONAL utility — generates a synthetic reactor_data.csv so the full
pipeline (preprocessing -> training -> evaluation -> prediction) can be
smoke-tested before your real plant data is available.

Replace the generated file with your real data/reactor_data.csv once
it's ready — no other code needs to change as long as the column names
in config.py match your real headers.

Run:
    python generate_synthetic_data.py
"""

import os
import numpy as np
import pandas as pd

import config

np.random.seed(config.RANDOM_STATE)
N = 2000

os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)

pressure = np.random.uniform(1.5, 8.0, N)
temperature = np.random.uniform(850, 1100, N)
molar_ratio = np.random.uniform(2.0, 5.0, N)
flow_h2 = np.random.uniform(50, 200, N)
flow_sicl4 = np.random.uniform(20, 100, N)

# Synthetic, physically-plausible relationships + noise
conversion = (
    40
    + 3.5 * (temperature - 850) / 250
    + 5 * molar_ratio
    - 1.2 * pressure
    + np.random.normal(0, 2, N)
).clip(0, 100)

productivity = (
    10
    + 0.15 * flow_h2
    + 0.08 * flow_sicl4
    + 0.5 * pressure
    - 0.01 * (temperature - 950) ** 2 / 50
    + np.random.normal(0, 1.5, N)
).clip(0, None)

df = pd.DataFrame({
    config.PRESSURE_COL: pressure,
    config.TEMPERATURE_COL: temperature,
    config.MOLAR_RATIO_COL: molar_ratio,
    config.FLOW_RATE_COLS[0]: flow_h2,
    config.FLOW_RATE_COLS[1]: flow_sicl4,
    config.TARGET_COLS[0]: conversion,
    config.TARGET_COLS[1]: productivity,
})

# Inject a few missing values and outliers so the cleaning pipeline has
# something real to do during the smoke test.
for col in config.FEATURE_COLS:
    idx = np.random.choice(N, size=int(0.01 * N), replace=False)
    df.loc[idx, col] = np.nan

outlier_idx = np.random.choice(N, size=10, replace=False)
df.loc[outlier_idx, config.TARGET_COLS[1]] *= 5

df.to_csv(config.DATA_PATH, index=False)
print(f"[INFO] Synthetic dataset written to {config.DATA_PATH} ({N} rows)")
