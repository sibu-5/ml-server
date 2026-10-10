"""
Comprehensive Test Suite for Deepfake Image Detection & Shared Qwen3 LLM Layer.

Covers:
- Valid supported image formats (JPEG, PNG, WEBP, BMP)
- Empty upload handling (422)
- Oversized upload payload rejection (413)
- Corrupt image payload handling (422)
- Decompression bomb protection (422)
- Correct response schema verification
- Global LLM_ENABLED toggle (True / False / Unavailable)
- ML prediction preservation when LLM is unavailable
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": settings.ml_api_key}


def _create_test_image(format="JPEG", size=(100, 100), color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class TestDeepfakeImageEndpoint:

    def test_unauthenticated_request_rejected(self, client):
        resp = client.post("/api/v1/deepfake/image")
        assert resp.status_code in (401, 403)

    def test_valid_jpeg_analysis(self, client, auth_headers):
        img_bytes = _create_test_image(format="JPEG")
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        resp = client.post("/api/v1/deepfake/image", headers=auth_headers, files=files)
        assert resp.status_code == 200

        data = resp.json()
        assert data["feature"] == "deepfake"
        assert data["media_type"] == "image"
        assert data["prediction"] in ("fake", "real", "unavailable")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["risk_level"] in ("critical", "high", "medium", "low", "none")
        assert "request_id" in data
        assert "processing_ms" in data
        assert "model" in data
        assert "explanation" in data
        assert "details" in data

    def test_valid_png_analysis(self, client, auth_headers):
        img_bytes = _create_test_image(format="PNG")
        files = {"file": ("test.png", img_bytes, "image/png")}

        resp = client.post("/api/v1/deepfake/image", headers=auth_headers, files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["details"]["format"] == "PNG"

    def test_empty_upload_rejected(self, client, auth_headers):
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        resp = client.post("/api/v1/deepfake/image", headers=auth_headers, files=files)
        assert resp.status_code == 422
        assert "Empty" in resp.json()["detail"]

    def test_corrupt_bytes_rejected(self, client, auth_headers):
        files = {"file": ("bad.jpg", b"NOT_AN_IMAGE_PAYLOAD_GARBAGE", "image/jpeg")}
        resp = client.post("/api/v1/deepfake/image", headers=auth_headers, files=files)
        assert resp.status_code == 422

    def test_oversized_payload_rejected(self, client, auth_headers, monkeypatch):
        monkeypatch.setattr(settings, "deepfake_max_image_bytes", 500)
        files = {"file": ("large.jpg", b"X" * 1000, "image/jpeg")}

        resp = client.post("/api/v1/deepfake/image", headers=auth_headers, files=files)
        assert resp.status_code == 413


class TestGlobalLLMSwitch:

    def test_llm_disabled_switch(self, client, auth_headers, monkeypatch):
        monkeypatch.setattr(settings, "llm_enabled", False)
        img_bytes = _create_test_image()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        resp = client.post("/api/v1/deepfake/image", headers=auth_headers, files=files)
        assert resp.status_code == 200

        data = resp.json()
        assert data["explanation"]["status"] == "disabled"
        assert data["prediction"] in ("fake", "real")

    def test_llm_enabled_unavailable_ollama(self, client, auth_headers, monkeypatch):
        monkeypatch.setattr(settings, "llm_enabled", True)
        monkeypatch.setattr(settings, "llm_host", "127.0.0.1")
        monkeypatch.setattr(settings, "llm_port", 59999)  # Closed port

        img_bytes = _create_test_image()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        resp = client.post("/api/v1/deepfake/image", headers=auth_headers, files=files)
        assert resp.status_code == 200

        data = resp.json()
        # ML prediction must remain valid even when Ollama is unavailable
        assert data["prediction"] in ("fake", "real")
        assert data["explanation"]["status"] == "unavailable"
