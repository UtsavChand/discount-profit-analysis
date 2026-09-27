"""FastAPI inference service."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(
    title="Discount Profitability API",
    description="Predict transaction-level profit margin for discounted e-commerce transactions.",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    discount: float = Field(gt=0, le=1)
    total_amount: float = Field(ge=0)
    shipping_cost: float = Field(ge=0)
    category: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    predicted_profit_margin: float


@app.get("/")
def root():
    return {"message": "Discount Profitability ML API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def prediction(transaction: TransactionRequest):
    value = predict(transaction.model_dump())
    return PredictionResponse(predicted_profit_margin=value)
