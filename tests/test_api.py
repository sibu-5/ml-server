"""
FastAPI endpoint tests — V3.

Tests:
- Authentication (require_api_key)
- Legacy /predict endpoint
- V3 /api/v1/analyze-url endpoint (static analysis only)
- Invalid request handling
- SSRF via API endpoint
"""

import pytest
from fastapi.testclient import TestClient

# Import the app — models may not load in test environment,
# so we handle that gracefully.
from app.main import app

client = TestClient(app)

API_KEY = "test-key-for-ci"


@pytest.fixture(autouse=True)
def patch_api_key(monkeypatch):
    """Override the ML_API_KEY for all tests."""
    import app.config as cfg
    import app.security as sec
    cfg.settings.ml_api_key = API_KEY
    yield


# ── Authentication tests ──────────────────────────────────────────────────────

class TestAuthentication:

    def test_no_key_returns_503_or_403(self):
        """No API key should be rejected."""
        resp = client.post(
            "/predict",
            json={"url": "https://example.com"},
        )
        assert resp.status_code in (403, 503)

    def test_wrong_key_returns_403(self):
        resp = client.post(
            "/predict",
            headers={"X-API-Key": "wrong-key"},
            json={"url": "https://example.com"},
        )
        assert resp.status_code == 403

    def test_health_live_no_auth(self):
        """Health endpoint must not require authentication."""
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_health_no_auth(self):
        resp = client.get("/health")
        assert resp.status_code == 200


# ── /predict endpoint tests ───────────────────────────────────────────────────

class TestPredictEndpoint:

    def _post(self, url: str):
        return client.post(
            "/predict",
            headers={"X-API-Key": API_KEY},
            json={"url": url},
        )

    def test_invalid_scheme_rejected(self):
        resp = self._post("ftp://example.com")
        assert resp.status_code == 422

    def test_missing_hostname_rejected(self):
        resp = self._post("https:///path")
        assert resp.status_code == 422

    def test_empty_url_rejected(self):
        resp = self._post("")
        assert resp.status_code == 422

    def test_valid_url_structure(self):
        """
        Valid URL should return a structured response.
        Model may not be loaded in CI, so accept 503 too.
        """
        resp = self._post("https://example.com")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "verdict" in data
            assert "risk_score" in data
            assert "ml_probability" in data


# ── /api/v1/analyze-url endpoint tests ───────────────────────────────────────

class TestAnalyzeURLEndpoint:

    def _post(self, url: str):
        return client.post(
            "/api/v1/analyze-url",
            headers={"X-API-Key": API_KEY},
            json={"url": url},
        )

    def test_invalid_url_returns_invalid(self):
        resp = self._post("not-a-url")
        # Should either be 422 from Pydantic or 200 with INVALID_URL verdict
        if resp.status_code == 200:
            data = resp.json()
            assert data["verdict"] == "INVALID_URL"
        else:
            assert resp.status_code == 422

    def test_ftp_scheme_returns_invalid(self):
        resp = self._post("ftp://example.com")
        if resp.status_code == 200:
            assert resp.json()["verdict"] == "INVALID_URL"
        else:
            assert resp.status_code in (200, 422)

    def test_localhost_returns_ssrf_blocked_or_invalid(self):
        """
        CRITICAL: submitting the ML server itself must never succeed.
        """
        resp = self._post("http://127.0.0.1:8001/predict")
        assert resp.status_code == 200
        data = resp.json()
        # Either SSRF_BLOCKED verdict or UNREACHABLE — NEVER SAFE or LEGITIMATE
        assert data["verdict"] not in ("SAFE", "legitimate", "LEGITIMATE")

    def test_private_ip_not_safe(self):
        resp = self._post("http://192.168.1.1/admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] not in ("SAFE", "legitimate")

    def test_response_schema(self):
        """Response must contain all required top-level keys."""
        resp = self._post("https://example.com")
        assert resp.status_code == 200
        data = resp.json()
        required_keys = {
            "url", "normalized_url", "website_status",
            "verdict", "risk_score", "ml_probability",
            "detections", "evidence", "domain", "dns",
            "ssl", "redirects", "website", "explanation",
            "recommendation",
        }
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_no_api_key_rejected(self):
        resp = client.post(
            "/api/v1/analyze-url",
            json={"url": "https://example.com"},
        )
        assert resp.status_code in (403, 503)

    def test_explanation_present(self):
        resp = self._post("https://example.com")
        assert resp.status_code == 200
        data = resp.json()
        expl = data.get("explanation", {})
        assert "summary" in expl
        assert "reasons" in expl
        assert "recommendation" in expl
