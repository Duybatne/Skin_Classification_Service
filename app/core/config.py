import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Skin Classification API"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "weights/best_model.pth")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CLASSES: list = ["acne", "cancer", "scar", "normal"]

    # Model input parameters — must match training notebook exactly
    IMG_SIZE: int = 300
    MEAN: list = [0.485, 0.456, 0.406]
    STD: list = [0.229, 0.224, 0.225]

    # Inference thresholds
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

settings = Settings()
