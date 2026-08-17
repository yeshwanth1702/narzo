"""
predict.py
==========
Interactive / scriptable prediction tool. Loads the saved best model
plus its preprocessing artifacts, then predicts Conversion and
Productivity from user-supplied operating conditions.

Usage (interactive, prompts for each value):
    python predict.py

Usage (non-interactive, for integration into other tools/UIs):
    from predict import predict_performance
    result = predict_performance(
        pressure=5.2, temperature=950, molar_ratio=3.1,
        flows=[120.0, 45.0],   # order must match config.FLOW_RATE_COLS
    )
"""

import numpy as np
import joblib

import config


def load_artifacts():
    """Load the best model (chosen automatically during training) plus scaler."""
    best_name = joblib.load(f"{config.MODEL_DIR}/best_model_name.pkl")
    model = joblib.load(f"{config.MODEL_DIR}/best_model_{best_name}.pkl")
    scaler = joblib.load(f"{config.MODEL_DIR}/scaler.pkl")
    return model, scaler, best_name


def predict_performance(pressure, temperature, molar_ratio, flows):
    """
    Predict Conversion (%) and Productivity from raw operating conditions.

    Parameters
    ----------
    pressure : float — Reactor Pressure (bar)
    temperature : float — Reactor Temperature (°C)
    molar_ratio : float — Feed Molar Ratio
    flows : list[float] — one value per entry in config.FLOW_RATE_COLS,
                           supplied in that same order.

    Returns
    -------
    dict mapping each target name to its predicted value, plus which
    model produced the prediction under the "_model_used" key.
    """
    if len(flows) != len(config.FLOW_RATE_COLS):
        raise ValueError(
            f"Expected {len(config.FLOW_RATE_COLS)} flow rate value(s) "
            f"{config.FLOW_RATE_COLS}, got {len(flows)}."
        )

    model, scaler, best_name = load_artifacts()

    x = np.array([[pressure, temperature, molar_ratio, *flows]])
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)[0]

    result = dict(zip(config.TARGET_COLS, preds))
    result["_model_used"] = best_name
    return result


def _interactive():
    print("=== Reactor Performance Predictor ===\n")

    pressure = float(input("Enter Reactor Pressure (bar): "))
    temperature = float(input("Enter Reactor Temperature (°C): "))
    molar_ratio = float(input("Enter Feed Molar Ratio: "))

    flows = []
    for flow_col in config.FLOW_RATE_COLS:
        val = float(input(f"Enter {flow_col}: "))
        flows.append(val)

    result = predict_performance(pressure, temperature, molar_ratio, flows)

    print("\n--- Prediction Results ---")
    for target in config.TARGET_COLS:
        print(f"{target}: {result[target]:.3f}")
    print(f"(Model used: {result['_model_used']})")


if __name__ == "__main__":
    _interactive()
