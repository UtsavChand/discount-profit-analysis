# Discount Profitability ML

Machine learning project for modelling transaction-level profit margins under historical e-commerce discounting.

The project compares an interpretable **Linear Regression** baseline with a **Gradient Boosting Regressor (GBR)** and turns the research notebook into a reproducible ML pipeline with saved models, inference code, a FastAPI service and automated tests.

> **Important:** This is a predictive modelling project, not a causal discount optimizer. The model learns associations present in historical discounting decisions and should not be interpreted as evidence that increasing a discount will cause higher profit.

## Project motivation

The original study asks:

1. How do historically observed discounting practices relate to profit margins across product categories?
2. Do non-linear models outperform linear models in capturing discount-profit relationships?

The modelling focuses on discounted transactions and introduces transaction-scale features because discount percentage and product category alone were not sufficiently explanatory.

## Dataset

The dataset contains approximately 34,500 historical e-commerce transactions.

Relevant fields include:

- `category`
- `discount`
- `price`
- `quantity`
- `total_amount`
- `shipping_cost`
- `profit_margin` (target)


## Feature engineering

Two engineered features from the research workflow are used by the production pipeline:

### Discount value

```text
(price × quantity) × discount
```

This represents the absolute value of the discount rather than only its percentage.

### Unit revenue

```text
(total_amount − shipping_cost) / quantity
```

This is used as a transaction-scale / markup proxy because explicit COGS is not available in the dataset.

Only transactions with `discount > 0` are used for the main modelling task.

## Models

### Linear Regression

Used as the interpretable baseline.

### Gradient Boosting Regressor

Used to capture non-linear and interaction-driven relationships between discount value, unit revenue and product category.

Both models use the same train/test split (`test_size=0.2`, `random_state=42`). Category encoding is contained inside each saved scikit-learn pipeline so training and inference use the same transformation logic.

## Reported research results

The submitted research report reports the following test-set results for the original implementation:

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.6626 | 25.94 | 13.60 |
| Gradient Boosting Regressor | 0.7765 | 21.11 | 9.32 |

The original analysis therefore found that GBR improved explained variance and reduced prediction error relative to the linear baseline.

## Model interpretation

The notebook retains the research analysis around:

- feature importance
- partial dependence
- SHAP explanations
- discount sensitivity simulations
- residual diagnostics

The original analysis found `discount_value` and `unit_revenue` to be substantially more influential than most category indicators.

## Repository structure

```text
discount-profitability-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── discount_profitability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── api/
│   └── main.py
│
├── models/
│   ├── gradient_boosting.pkl
│   ├── linear_regression.pkl
│   └── model_metadata.json
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
└── data/
    └── README.md
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

## Train the models

From the repository root:

```bash
python -m src.train
```

This creates:

```text
models/gradient_boosting.pkl
models/linear_regression.pkl
models/model_metadata.json
```

The saved models include their preprocessing pipeline, including category encoding. This avoids manually reproducing the training-time one-hot encoded columns during inference.

## Run inference from Python

```python
from src.predict import predict

transaction = {
    "price": 1000,
    "quantity": 2,
    "discount": 0.20,
    "total_amount": 1600,
    "shipping_cost": 100,
    "category": "Electronics",
}

result = predict(transaction)
print(result)
```

## Run the FastAPI service

After training:

```bash
uvicorn api.main:app --reload
```

API documentation is available through FastAPI's interactive documentation at `/docs`.

### `POST /predict`

Example request:

```json
{
  "price": 1000,
  "quantity": 2,
  "discount": 0.20,
  "total_amount": 1600,
  "shipping_cost": 100,
  "category": "Electronics"
}
```

Example response shape:

```json
{
  "predicted_profit_margin": 42.0
}
```

The number above is illustrative; the actual response depends on the trained model artifact.

## Testing

Run the test suite with:

```bash
pytest
```

Tests cover:

- engineered feature formulas
- filtering of discounted transactions
- inference feature construction
- invalid quantities
- model pipeline fitting/prediction
- FastAPI health endpoint
- FastAPI prediction validation

## Limitations

The project deliberately keeps the limitations identified in the research work:

- discounts are historically observed rather than randomly assigned, so the model does not establish causal discount effects;
- explicit COGS is unavailable, so unit revenue is used as a proxy;
- restricting the main model to discounted transactions introduces selection bias;
- the simulation does not model demand elasticity or customer behaviour;
- SHAP and partial dependence explain model behaviour, not the underlying economic process.

## Future improvements

Potential next steps include:

- adding explicit COGS and modelling actual profit;
- including non-discounted transactions;
- modelling demand response / quantity elasticity;
- adding causal inference methods such as propensity-score approaches or causal forests;
- adding CI to run the test suite automatically on every push.

## Research reference

The repository implementation is based on the submitted project report, **"Modelling Profit Sensitivity to Discounting in E-Commerce Transactions Using Linear and Non-Linear Machine Learning Approaches"**.
