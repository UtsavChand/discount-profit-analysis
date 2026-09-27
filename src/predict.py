"""Model inference utilities."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.preprocessing import prepare_inference_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gradient_boosting.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def predict(transaction: dict) -> float:
    model = load_model()
    X = prepare_inference_data(transaction)
    return float(model.predict(X)[0])


if __name__ == "__main__":
    example = {
        "price": 1000.0,
        "quantity": 2,
        "discount": 0.20,
        "total_amount": 1600.0,
        "shipping_cost": 100.0,
        "category": "Electronics",
    }
    print(f"Predicted profit margin: {predict(example):.4f}")
