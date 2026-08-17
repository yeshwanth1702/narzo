"""
main.py
=======
Single entry point to run the entire pipeline end-to-end:
    1. (optional) generate synthetic data for a smoke test
    2. preprocess data (cleaning, imputation, outlier removal, scaling)
    3. train & cross-validate all 5 models
    4. evaluate, plot, run SHAP, and save the best model + artifacts

Run:
    python main.py            # expects data/reactor_data.csv to already exist
    python main.py --demo     # generates synthetic data first (testing only)
"""

import argparse
import os

import config


def main():
    parser = argparse.ArgumentParser(description="Reactor MIMO ML Pipeline")
    parser.add_argument("--demo", action="store_true",
                         help="Generate synthetic data before training (for testing only)")
    args = parser.parse_args()

    if args.demo:
        import generate_synthetic_data  # noqa: F401  (executes on import)

    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"{config.DATA_PATH} not found. Place your reactor_data.csv there, "
            f"or run with --demo to generate synthetic test data first."
        )

    import train
    train.main()


if __name__ == "__main__":
    main()
