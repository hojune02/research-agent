"""Smoke tests: app boots and core endpoints respond in mock mode."""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

settings.MOCK_LLM = True


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "enable_reranker" in body


def test_ask_empty_project_refuses_without_citations():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "user_id": "smoke-user",
                "project_id": "definitely-empty-project",
                "question": "What is the main contribution of the paper?",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "do not know" in body["answer"].lower()
        assert body["citations"] == []


def test_tools_listed():
    with TestClient(app) as client:
        response = client.get("/tools")
        assert response.status_code == 200
        assert len(response.json()["tools"]) >= 4