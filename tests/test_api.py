import os
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_or_root():
    response = client.get("/")
    assert response.status_code in [200, 404]

def test_predict_unauthorized():
    payload = {
        "model_name": "XGBoost",
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [401, 403]