import io
from PIL import Image
import torchvision.transforms as transforms
from app.core.config import settings

# Inference transform — matches notebook's get_transforms('val')
transform = transforms.Compose([
    transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=settings.MEAN, std=settings.STD),
])

def preprocess_image(image_bytes: bytes):
    """
    Reads image bytes, converts to RGB, resizes to 300x300, normalizes,
    and adds batch dimension. Output shape: [1, 3, 300, 300].
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)
