"""
Tests for POST /api/predict and related system endpoints.
"""
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)


def _make_jpeg_bytes(width: int = 300, height: int = 300, color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

def test_root_returns_online():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "online"
    assert "predict" in data


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "device" in data


def test_classes_endpoint():
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert data["classes"] == settings.CLASSES
    assert data["num_classes"] == len(settings.CLASSES)


# ---------------------------------------------------------------------------
# /api/predict — error cases
# ---------------------------------------------------------------------------

def test_predict_rejects_non_image():
    r = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"plain text content", "text/plain")},
    )
    assert r.status_code == 400
    assert "not a valid image" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /api/predict — happy path (standard, no TTA)
# ---------------------------------------------------------------------------

def test_predict_returns_valid_schema():
    r = client.post(
        "/api/predict",
        files={"file": ("skin.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()

    assert data["success"] is True
    assert data["tta_enabled"] is False
    assert "filename" in data
    assert "low_confidence" in data

    pred = data["prediction"]
    assert "label" in pred
    assert pred["label"] in settings.CLASSES
    assert 0.0 <= pred["confidence"] <= 1.0

    probs = data["probabilities"]
    assert set(probs.keys()) == set(settings.CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_predict_probabilities_sum_to_one():
    r = client.post(
        "/api/predict",
        files={"file": ("skin.jpg", _make_jpeg_bytes(color="blue"), "image/jpeg")},
    )
    assert r.status_code == 200
    probs = r.json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# /api/predict — TTA mode
# ---------------------------------------------------------------------------

def test_predict_tta_returns_std_per_class():
    r = client.post(
        "/api/predict?tta=true",
        files={"file": ("skin.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()

    assert data["tta_enabled"] is True
    assert "std_per_class" in data
    assert set(data["std_per_class"].keys()) == set(settings.CLASSES)
    pred = data["prediction"]
    assert pred["label"] in settings.CLASSES
