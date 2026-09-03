"""Run one prediction from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.model import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--vibration", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--rpm", type=float, required=True)
    parser.add_argument("--operating-hours", type=float, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args()

    model = joblib.load(args.model)
    values = [[args.temperature, args.vibration, args.pressure, args.rpm, args.operating_hours]]
    sample = pd.DataFrame(values, columns=FEATURES)
    probability = float(model.predict_proba(sample)[0, 1])
    prediction = int(probability >= 0.5)
    print(f"Prediction: {'failure risk' if prediction else 'normal operation'}")
    print(f"Failure probability: {probability:.1%}")


if __name__ == "__main__":
    main()
