import pandas as pd
import pytest

from src.preprocessing import create_features, prepare_inference_data, prepare_model_data


def sample_row():
    return {
        "price": 1000.0,
        "quantity": 2,
        "discount": 0.2,
        "total_amount": 1600.0,
        "shipping_cost": 100.0,
        "category": "Electronics",
        "profit_margin": 20.0,
    }


def test_engineered_features_match_project_formulas():
    df = pd.DataFrame([sample_row()])
    result = create_features(df).iloc[0]

    assert result["total_revenue"] == 1500.0
    assert result["unit_revenue"] == 750.0
    assert result["discount_value"] == 400.0


def test_only_discounted_transactions_are_used():
    rows = [sample_row(), {**sample_row(), "discount": 0.0}]
    X, y = prepare_model_data(pd.DataFrame(rows))
    assert len(X) == 1
    assert len(y) == 1


def test_inference_features_have_expected_columns():
    transaction = {k: v for k, v in sample_row().items() if k != "profit_margin"}
    X = prepare_inference_data(transaction)
    assert list(X.columns) == ["discount_value", "unit_revenue", "category"]


def test_zero_quantity_is_rejected():
    df = pd.DataFrame([{**sample_row(), "quantity": 0}])
    with pytest.raises(ValueError, match="quantity"):
        create_features(df)
