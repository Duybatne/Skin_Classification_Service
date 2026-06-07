"""
Test-Time Augmentation (TTA) service.
Ports tta_predict() from the training notebook into a production-ready module.
"""
import io
import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from app.core.config import settings


def _get_base_transform():
    return transforms.Compose([
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.MEAN, std=settings.STD),
    ])


def _get_augmented_tensors(img: Image.Image) -> list:
    """
    Generates 10 augmented versions of the input PIL image.
    Matches the notebook's tta_predict augmentation set exactly.
    """
    base_tf = _get_base_transform()
    crop_tf = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.MEAN, std=settings.STD),
    ])

    enhancer_b = ImageEnhance.Brightness(img)
    enhancer_c = ImageEnhance.Contrast(img)
    bright = enhancer_b.enhance(1.2)
    contrast = enhancer_c.enhance(1.2)

    augments = [
        base_tf(img),                                                          # (1) original
        base_tf(img.transpose(Image.FLIP_LEFT_RIGHT)),                         # (2) h-flip
        base_tf(img.transpose(Image.FLIP_TOP_BOTTOM)),                         # (3) v-flip
        base_tf(img.rotate(90)),                                               # (4) rotate 90
        base_tf(img.rotate(180)),                                              # (5) rotate 180
        crop_tf(img),                                                          # (6) center crop
        base_tf(bright),                                                       # (7) brightness +20%
        base_tf(contrast),                                                     # (8) contrast +20%
        base_tf(bright.transpose(Image.FLIP_LEFT_RIGHT)),                      # (9) flip + bright
        base_tf(contrast.transpose(Image.FLIP_LEFT_RIGHT)),                    # (10) flip + contrast
    ]
    return augments


def tta_predict(model, image_bytes: bytes, device: torch.device) -> dict:
    """
    Runs TTA inference on image bytes.
    Returns:
      {
        label: str,
        confidence: float,
        probabilities: { class_name: float },
        std_per_class: { class_name: float },
      }
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensors = _get_augmented_tensors(img)

    batch = torch.stack(tensors).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)                        # [10, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # [10, num_classes]

    avg_probs = np.mean(probs, axis=0)    # [num_classes]
    std_probs = np.std(probs, axis=0)     # [num_classes]
    pred_idx = int(np.argmax(avg_probs))
    class_names = settings.CLASSES

    return {
        "label": class_names[pred_idx],
        "confidence": float(avg_probs[pred_idx]),
        "probabilities": {class_names[i]: float(avg_probs[i]) for i in range(len(class_names))},
        "std_per_class": {class_names[i]: float(std_probs[i]) for i in range(len(class_names))},
    }
