"""
Unit tests for model and preprocessing services.
These tests do NOT require the real checkpoint — they verify shapes and contracts.
"""
import io
import torch
from PIL import Image
from app.core.config import settings
from app.services.model import EfficientNetB3Classifier
from app.services.preprocess import preprocess_image


def _dummy_image_bytes(width: int = 300, height: int = 300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="green").save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

def test_preprocess_output_shape():
    tensor = preprocess_image(_dummy_image_bytes())
    assert tensor.shape == (1, 3, settings.IMG_SIZE, settings.IMG_SIZE)


def test_preprocess_handles_small_image():
    """Resize should work regardless of input resolution."""
    tensor = preprocess_image(_dummy_image_bytes(width=64, height=64))
    assert tensor.shape == (1, 3, settings.IMG_SIZE, settings.IMG_SIZE)


def test_preprocess_is_float_tensor():
    tensor = preprocess_image(_dummy_image_bytes())
    assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# EfficientNetB3Classifier
# ---------------------------------------------------------------------------

def test_model_forward_output_shape():
    model = EfficientNetB3Classifier(num_classes=len(settings.CLASSES))
    model.eval()
    dummy = torch.randn(1, 3, settings.IMG_SIZE, settings.IMG_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, len(settings.CLASSES))


def test_model_output_num_classes():
    """Output size must match number of target classes."""
    n = len(settings.CLASSES)
    model = EfficientNetB3Classifier(num_classes=n)
    model.eval()
    dummy = torch.randn(1, 3, settings.IMG_SIZE, settings.IMG_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[1] == n


def test_softmax_sums_to_one():
    model = EfficientNetB3Classifier(num_classes=len(settings.CLASSES))
    model.eval()
    dummy = torch.randn(1, 3, settings.IMG_SIZE, settings.IMG_SIZE)
    with torch.no_grad():
        logits = model(dummy)
        probs = torch.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
