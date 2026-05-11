import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.sentiment_service import analyzer

@pytest.fixture(scope="module")
def client():
    # Mock the analyzer readiness for tests
    analyzer.ready = True
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def mock_analyzer_predict(monkeypatch):
    """Mock the predict method to avoid downloading/running the actual model in tests."""
    def mock_predict(self, texts):
        return [
            {"text": text, "label": "positive", "score": 0.95}
            for text in texts
        ]
    monkeypatch.setattr("app.services.sentiment_service.SentimentAnalyzer.predict", mock_predict)
