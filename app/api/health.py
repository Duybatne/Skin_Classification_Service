from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check():
    """
    Returns service health status and whether the model weights are loaded.
    """
    from app.api.predict import classifier
    return {
        "status": "ok",
        "model_loaded": classifier.model_loaded if classifier else False,
        "model_path": settings.MODEL_PATH,
        "device": str(classifier.device) if classifier else "unavailable",
    }


@router.get("/classes", tags=["system"])
def list_classes():
    """
    Returns the list of skin condition classes the model can predict.
    """
    return {
        "classes": settings.CLASSES,
        "num_classes": len(settings.CLASSES),
    }
