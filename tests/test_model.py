import pandas as pd

from src.train import build_pipeline
from sklearn.ensemble import GradientBoostingRegressor


def test_gradient_boosting_pipeline_can_fit_and_predict():
    X = pd.DataFrame(
        {
            "discount_value": [10, 20, 30, 40, 50, 60],
            "unit_revenue": [100, 120, 140, 160, 180, 200],
            "category": ["A", "A", "B", "B", "A", "B"],
        }
    )
    y = [5, 8, 12, 14, 17, 20]

    model = build_pipeline(GradientBoostingRegressor(random_state=42))
    model.fit(X, y)
    predictions = model.predict(X.iloc[:2])

    assert len(predictions) == 2
    assert all(isinstance(float(x), float) for x in predictions)
