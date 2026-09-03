"""Model-building and evaluation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "operating_hours",
]
TARGET = "failed"


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42),
            ),
        ]
    )


def train_and_evaluate(
    data: pd.DataFrame, test_size: float = 0.25
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the model and return it with JSON-serializable metrics."""
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=test_size,
        random_state=42,
        stratify=data[TARGET],
    )
    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "test_rows": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return model, metrics
