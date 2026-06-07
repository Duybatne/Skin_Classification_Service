import traceback
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.core.config import settings
from app.services.preprocess import preprocess_image
from app.services.model import SkinClassifier

router = APIRouter()

# Singleton classifier — loaded once at startup
try:
    classifier = SkinClassifier()
except Exception as e:
    print(f"[predict] Error initialising classifier: {e}")
    classifier = None


@router.post("/predict", tags=["prediction"])
async def predict(
    file: UploadFile = File(...),
    tta: bool = Query(False, description="Enable Test-Time Augmentation (10x slower, slightly more accurate)"),
):
    """
    Classify a skin lesion image.

    - **file**: JPEG/PNG image of the skin lesion
    - **tta**: set `true` to enable Test-Time Augmentation (ensemble over 10 augmented views)

    Returns the predicted class, confidence, per-class probabilities,
    and a `low_confidence` flag when the top probability is below the configured threshold.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    if classifier is None:
        raise HTTPException(status_code=503, detail="Classification model is unavailable.")

    try:
        image_bytes = await file.read()

        if tta:
            from app.services.tta import tta_predict
            result = tta_predict(classifier.model, image_bytes, classifier.device)
            label = result["label"]
            confidence = result["confidence"]
            probabilities_map = result["probabilities"]
            std_per_class = result["std_per_class"]
        else:
            tensor = preprocess_image(image_bytes)
            class_idx, confidence, probabilities = classifier.predict(tensor)
            if class_idx < 0 or class_idx >= len(settings.CLASSES):
                raise ValueError(f"Model returned out-of-range class index: {class_idx}")
            label = settings.CLASSES[class_idx]
            probabilities_map = {settings.CLASSES[i]: float(p) for i, p in enumerate(probabilities)}
            std_per_class = None

        response = {
            "success": True,
            "filename": file.filename,
            "tta_enabled": tta,
            "prediction": {
                "label": label,
                "confidence": round(confidence, 4),
            },
            "probabilities": {k: round(v, 4) for k, v in probabilities_map.items()},
            "low_confidence": confidence < settings.CONFIDENCE_THRESHOLD,
        }
        if std_per_class is not None:
            response["std_per_class"] = {k: round(v, 4) for k, v in std_per_class.items()}

        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
