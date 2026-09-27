from fastapi.testclient import TestClient

import api.main as main


client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint(monkeypatch):
    monkeypatch.setattr(main, "predict", lambda _: 42.0)

    response = client.post(
        "/predict",
        json={
            "price": 1000,
            "quantity": 2,
            "discount": 0.2,
            "total_amount": 1600,
            "shipping_cost": 100,
            "category": "Electronics",
        },
    )

    assert response.status_code == 200
    assert response.json()["predicted_profit_margin"] == 42.0


def test_invalid_quantity_is_rejected():
    response = client.post(
        "/predict",
        json={
            "price": 1000,
            "quantity": 0,
            "discount": 0.2,
            "total_amount": 1600,
            "shipping_cost": 100,
            "category": "Electronics",
        },
    )
    assert response.status_code == 422
