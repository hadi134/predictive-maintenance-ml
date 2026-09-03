"""Generate a reproducible synthetic equipment-sensor dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sensor_data(rows: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """Return synthetic sensor readings and a binary failure label."""
    if rows < 20:
        raise ValueError("rows must be at least 20")

    rng = np.random.default_rng(seed)
    temperature = np.clip(rng.normal(82, 13, rows), 45, 130)
    vibration = np.clip(rng.gamma(shape=2.2, scale=1.25, size=rows), 0.2, 10)
    pressure = np.clip(rng.normal(30, 4.2, rows), 16, 42)
    rpm = np.clip(rng.normal(1_500, 260, rows), 600, 2_400)
    operating_hours = rng.integers(50, 8_000, rows)

    risk_score = (
        -3.1
        + 0.065 * (temperature - 80)
        + 0.48 * (vibration - 3)
        - 0.16 * (pressure - 30)
        + 0.00035 * (operating_hours - 2_500)
        + 0.00045 * (rpm - 1_500)
    )
    failure_probability = 1 / (1 + np.exp(-risk_score))
    failed = rng.binomial(1, failure_probability)

    return pd.DataFrame(
        {
            "machine_id": np.arange(1, rows + 1),
            "temperature_c": temperature.round(2),
            "vibration_mm_s": vibration.round(2),
            "pressure_bar": pressure.round(2),
            "rpm": rpm.round(0).astype(int),
            "operating_hours": operating_hours,
            "failed": failed,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/sensor_data.csv"))
    args = parser.parse_args()

    data = generate_sensor_data(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data):,} rows to {args.output}")
    print(f"Failure rate: {data['failed'].mean():.1%}")


if __name__ == "__main__":
    main()
