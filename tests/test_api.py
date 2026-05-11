import pytest

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_analyze_single_text(client):
    payload = {"text": "Apple stock prices are surging today."}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["label"] == "positive"
    assert data["results"][0]["text"] == "Apple stock prices are surging today."

def test_analyze_list_text(client):
    payload = {"text": ["Bullish trend", "Bearish market"]}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2

def test_empty_input(client):
    payload = {"text": []}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 400

def test_invalid_payload(client):
    payload = {"not_text": "invalid"}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
