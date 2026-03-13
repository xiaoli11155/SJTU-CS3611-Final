import json
from pathlib import Path
from typing import List

import torch

from src.config import LABEL_PATH, MODEL_PATH, NUM_CLASSES, SEQ_LEN
from src.model.cnn1d import CNN1DClassifier


class OnlineTrafficClassifier:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        label_path: Path = LABEL_PATH,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = CNN1DClassifier(seq_len=SEQ_LEN, num_classes=NUM_CLASSES).to(self.device)
        self.labels: List[str] = ["Unknown"] * NUM_CLASSES
        self.enabled = False

        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                self.labels = json.load(f)

        if model_path.exists():
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.enabled = True

    def predict(self, feature_vector) -> str:
        if not self.enabled:
            return "ModelNotReady"

        x = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            pred = int(torch.argmax(logits, dim=1).item())
        if 0 <= pred < len(self.labels):
            return self.labels[pred]
        return "Unknown"
