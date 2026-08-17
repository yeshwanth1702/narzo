"""
config.py
=========
Central configuration for the Reactor Performance MIMO ML pipeline.
Edit this file to match your actual reactor_data.csv column names.
"""

import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "reactor_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Column names — EDIT THESE to match your real dataset headers
# ------------------------------------------------------------------
PRESSURE_COL = "Reactor_Pressure_bar"
TEMPERATURE_COL = "Reactor_Temperature_C"
MOLAR_RATIO_COL = "Feed_Molar_Ratio"

# Any number of individual feed flow columns. Add/remove as needed —
# everything downstream (config.FEATURE_COLS, predict.py, etc.) adapts
# automatically to however many you list here.
FLOW_RATE_COLS = [
    "Flow_H2_Nm3h",
    "Flow_SiCl4_kgh",
    # Add more flow streams here, e.g. "Flow_HCl_kgh"
]

FEATURE_COLS = [PRESSURE_COL, TEMPERATURE_COL, MOLAR_RATIO_COL] + FLOW_RATE_COLS

TARGET_COLS = ["Conversion_pct", "Productivity_kg_m3h"]

# ------------------------------------------------------------------
# Preprocessing / training settings
# ------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
OUTLIER_IQR_MULTIPLIER = 1.5        # Tukey's rule
MISSING_VALUE_STRATEGY = "median"   # for SimpleImputer
