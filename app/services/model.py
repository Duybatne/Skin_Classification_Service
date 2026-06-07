import os
import torch
import torch.nn as nn
import torchvision.models as models
from app.core.config import settings


class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 with custom classifier head.
    Architecture must match the training notebook exactly:
      Dropout(0.4) -> Linear(1536->512) -> ReLU -> Dropout(0.2) -> Linear(512->num_classes)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_last_n_blocks(self, n: int):
        """Mirror of the training notebook method — not used during inference."""
        blocks = list(self.backbone.features.children())
        for i in range(len(blocks) - n, len(blocks)):
            for param in blocks[i].parameters():
                param.requires_grad = True


class SkinClassifier:
    def __init__(self):
        num_classes = len(settings.CLASSES)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNetB3Classifier(num_classes=num_classes)
        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = False
        self._load_weights()

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def _load_weights(self):
        path = settings.MODEL_PATH
        if not os.path.exists(path):
            print(f"[SkinClassifier] Weights not found at '{path}'. Running with random weights.")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)
            # Notebook saves: { epoch, model_state, val_acc, val_f1, cfg }
            state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            self._model_loaded = True
            epoch = checkpoint.get("epoch", "?") if isinstance(checkpoint, dict) else "?"
            val_f1 = checkpoint.get("val_f1", "?") if isinstance(checkpoint, dict) else "?"
            print(f"[SkinClassifier] Loaded '{path}' (epoch={epoch}, val_f1={val_f1})")
        except Exception as e:
            print(f"[SkinClassifier] Failed to load weights: {e}")

    def predict(self, image_tensor: torch.Tensor):
        """
        Single-pass inference.
        Returns: (class_idx: int, confidence: float, probabilities: list[float])
        """
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
        return class_idx.item(), confidence.item(), probs[0].tolist()
