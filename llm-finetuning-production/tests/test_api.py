"""
API Integration and Endpoint Tests using FastAPI TestClient
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import ExtractionResponse, HealthResponse

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "device" in data

@patch("api.main.get_inference_engine")
def test_predict_endpoint_success(mock_get_engine):
    mock_engine = MagicMock()
    mock_engine.model_path_or_id = "Qwen/Qwen2.5-1.5B-Instruct"
    mock_engine.device = "cpu"
    mock_engine.extract.return_value = {
        "data": {
            "customer": "John",
            "quantity": 3,
            "product": "laptops",
            "amount": 2400.0,
            "delivery_day": "Friday"
        },
        "raw_response": '{"customer": "John", "quantity": 3, "product": "laptops", "amount": 2400.0, "delivery_day": "Friday"}',
        "is_valid_json": True,
        "error": None,
        "latency_ms": 42.5,
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct"
    }
    mock_get_engine.return_value = mock_engine

    payload = {"text": "John ordered 3 laptops for $2400 and wants delivery on Friday."}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_json"] is True
    assert data["data"]["customer"] == "John"
    assert data["data"]["quantity"] == 3
    assert data["data"]["product"] == "laptops"
    assert data["data"]["amount"] == 2400.0
    assert data["data"]["delivery_day"] == "Friday"
    assert data["latency_ms"] == 42.5

def test_predict_validation_error_empty_text():
    # text length < 3 should trigger Pydantic validation error (422)
    response = client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422
