import uvicorn
from fastapi import FastAPI
from app.core.config import settings
from app.api import predict
from app.api import health

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "REST API for classifying skin lesion images using a fine-tuned EfficientNet-B3 model. "
        "Supports 4 classes: acne, cancer, scar, normal."
    ),
    version="1.0.0",
)

app.include_router(predict.router, prefix="/api")
app.include_router(health.router)


@app.get("/", tags=["system"])
def read_root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /api/predict",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
