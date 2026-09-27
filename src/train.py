"""Train and save the discount profitability models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing import MODEL_FEATURES, prepare_model_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ecommerce_sales_34500.csv"
MODEL_DIR = ROOT / "models"


def build_pipeline(model, *, drop_first: bool = False) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop="first" if drop_first else None,
                    handle_unknown="ignore",
                ),
                ["category"],
            ),
            ("numeric", "passthrough", ["discount_value", "unit_revenue"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(model, X_test, y_test) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }


def train() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place ecommerce_sales_34500.csv in data/."
        )

    df = pd.read_csv(DATA_PATH)
    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear_pipeline = build_pipeline(LinearRegression(), drop_first=True)
    gbr_pipeline = build_pipeline(
        GradientBoostingRegressor(random_state=42), drop_first=False
    )

    linear_pipeline.fit(X_train, y_train)
    gbr_pipeline.fit(X_train, y_train)

    metrics = {
        "linear_regression": evaluate(linear_pipeline, X_test, y_test),
        "gradient_boosting": evaluate(gbr_pipeline, X_test, y_test),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(linear_pipeline, MODEL_DIR / "linear_regression.pkl")
    joblib.dump(gbr_pipeline, MODEL_DIR / "gradient_boosting.pkl")

    metadata = {
        "target": "profit_margin",
        "features": MODEL_FEATURES,
        "discount_filter": "> 0",
        "test_size": 0.2,
        "random_state": 42,
        "metrics": metrics,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    return metadata


if __name__ == "__main__":
    train()
