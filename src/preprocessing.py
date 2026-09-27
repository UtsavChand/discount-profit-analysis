"""Feature engineering and model preprocessing for the discount profitability project."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "price",
    "quantity",
    "discount",
    "total_amount",
    "shipping_cost",
    "category",
    "profit_margin",
}

MODEL_FEATURES = ["discount_value", "unit_revenue", "category"]
TARGET = "profit_margin"


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features used by the original notebook/paper."""
    validate_input_columns(df)
    out = df.copy()

    if (out["quantity"] <= 0).any():
        raise ValueError("quantity must be greater than zero")

    out["total_revenue"] = out["total_amount"] - out["shipping_cost"]
    out["unit_revenue"] = out["total_revenue"] / out["quantity"]
    out["original_price_total"] = out["price"] * out["quantity"]
    out["discount_value"] = out["original_price_total"] * out["discount"]

    return out


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and target using only discounted transactions."""
    out = create_features(df)
    out = out[out["discount"] > 0].copy()

    X = out[MODEL_FEATURES].copy()
    y = out[TARGET].copy()
    return X, y


def prepare_inference_data(transaction: dict) -> pd.DataFrame:
    """Convert one raw transaction into the model's three input features."""
    raw = pd.DataFrame([transaction])
    # API inputs do not contain the target, so validate the fields explicitly here.
    required = {"price", "quantity", "discount", "total_amount", "shipping_cost", "category"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing inference fields: {sorted(missing)}")

    if (raw["quantity"] <= 0).any():
        raise ValueError("quantity must be greater than zero")

    raw["total_revenue"] = raw["total_amount"] - raw["shipping_cost"]
    raw["unit_revenue"] = raw["total_revenue"] / raw["quantity"]
    raw["original_price_total"] = raw["price"] * raw["quantity"]
    raw["discount_value"] = raw["original_price_total"] * raw["discount"]

    return raw[MODEL_FEATURES]
