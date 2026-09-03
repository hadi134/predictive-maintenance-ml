# Predictive Maintenance ML

An end-to-end machine-learning project that predicts whether industrial equipment is at risk of failure from sensor readings.

This project reflects the kind of practical AI work I enjoy: inspect a problem, identify likely causes, build a baseline model, evaluate it, and make the result usable through an API.

> The data is generated synthetically and is intended for learning and portfolio demonstration. It is not production sensor data.

## What it includes

- Reproducible synthetic sensor-data generation
- Data preprocessing and logistic-regression classification
- Accuracy, precision, recall, F1, ROC AUC, and confusion-matrix reporting
- A command-line prediction tool
- A FastAPI prediction endpoint
- Automated tests and GitHub Actions CI

## Baseline result

Using 2,000 generated rows with seed `42`, the logistic-regression baseline reached **0.833 ROC AUC** and **0.719 recall** on the held-out test set. The full result is saved in [`reports/metrics.json`](reports/metrics.json).

Because failures are intentionally less common than normal operation, recall and precision are more informative here than accuracy alone.

## Project structure

```text
.
├── data/
├── reports/
├── src/
│   ├── api.py
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
└── tests/
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data --rows 2000
python -m src.train
```

Try one prediction:

```bash
python -m src.predict \
  --temperature 98 \
  --vibration 5.2 \
  --pressure 25 \
  --rpm 1700 \
  --operating-hours 4200
```

Start the API:

```bash
uvicorn src.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Example API request

```json
{
  "temperature_c": 98,
  "vibration_mm_s": 5.2,
  "pressure_bar": 25,
  "rpm": 1700,
  "operating_hours": 4200
}
```

## Next steps

- Compare additional classification models
- Track experiments and model versions
- Add drift monitoring for changing sensor distributions
- Validate with a real, permissioned maintenance dataset
