"""Train and save the predictive-maintenance model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.model import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sensor_data.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model, metrics = train_and_evaluate(data)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
