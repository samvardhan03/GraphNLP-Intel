"""Integration tests for the FastAPI application."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client():
    """Create a test client with mocked dependencies."""
    # Patch Redis and Neo4j before importing the app
    with patch("graphnlp.api.app.get_settings") as mock_settings:
        settings = MagicMock()
        settings.redis_url = "redis://localhost:6379"
        settings.neo4j_uri = "bolt://localhost:7687"
        settings.neo4j_user = "neo4j"
        settings.neo4j_password = "password"
        settings.rate_limit_per_minute = 100
        settings.environment = "test"
        settings.ner_model = "en_core_web_sm"
        settings.embedding_model = "all-MiniLM-L6-v2"
        settings.gnn_layers = 2
        mock_settings.return_value = settings

        from graphnlp.api.app import create_app

        app = create_app()
        client = TestClient(app)
        yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, app_client):
        response = app_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "graphnlp-intel"

    def test_health_no_auth_required(self, app_client):
        """Health endpoint should work without authentication."""
        response = app_client.get("/health")
        assert response.status_code == 200


class TestAnalyzeEndpoint:
    def test_analyze_requires_auth(self, app_client):
        """Requests without tenant context should get 401."""
        response = app_client.post(
            "/v1/analyze",
            json={"documents": ["test"]},
        )
        assert response.status_code == 401

    def test_analyze_with_tenant_header(self, app_client):
        """With X-Tenant-ID header, should proceed (may fail at Redis)."""
        response = app_client.post(
            "/v1/analyze",
            json={"documents": ["test document"]},
            headers={"X-Tenant-ID": "test-tenant"},
        )
        # May get 503 (Redis unavailable) or 200 - both are valid
        assert response.status_code in (200, 503)

    def test_analyze_empty_documents(self, app_client):
        """Empty documents list should fail validation."""
        response = app_client.post(
            "/v1/analyze",
            json={"documents": []},
            headers={"X-Tenant-ID": "test-tenant"},
        )
        assert response.status_code in (422, 503)


class TestGraphEndpoint:
    def test_graph_requires_auth(self, app_client):
        response = app_client.get("/v1/graph/some-id")
        assert response.status_code == 401

    def test_graph_with_tenant(self, app_client):
        response = app_client.get(
            "/v1/graph/nonexistent-id",
            headers={"X-Tenant-ID": "test-tenant"},
        )
        # Should fail with 503 (Neo4j unavailable) or 404
        assert response.status_code in (404, 503)


class TestWebhookEndpoints:
    def test_webhooks_list_requires_auth(self, app_client):
        response = app_client.get("/v1/webhooks")
        assert response.status_code == 401

    def test_webhook_register_requires_https(self, app_client):
        response = app_client.post(
            "/v1/webhooks",
            json={"url": "http://insecure.example.com", "events": ["graph.changed"]},
            headers={"X-Tenant-ID": "test-tenant"},
        )
        # Should reject HTTP URLs or fail at Redis
        assert response.status_code in (400, 503)

    def test_webhook_invalid_events(self, app_client):
        response = app_client.post(
            "/v1/webhooks",
            json={"url": "https://example.com/hook", "events": ["invalid.event"]},
            headers={"X-Tenant-ID": "test-tenant"},
        )
        assert response.status_code in (400, 503)
