from fastapi.testclient import TestClient

from src.api import app
from src.data import generate_sensor_data
from src.model import train_and_evaluate


def test_data_is_reproducible() -> None:
    first = generate_sensor_data(rows=100, seed=7)
    second = generate_sensor_data(rows=100, seed=7)
    assert first.equals(second)
    assert set(first["failed"].unique()) == {0, 1}


def test_model_beats_random_baseline() -> None:
    data = generate_sensor_data(rows=1_500, seed=42)
    _, metrics = train_and_evaluate(data)
    assert metrics["roc_auc"] > 0.75


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
