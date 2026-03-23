import json
from pathlib import Path
from typing import Dict

import torch

from src.model.cnn1d import CNN1DClassifier


class OnlineMTCByteClassifier:
    """Inference loader for scripts/train_mtc_byte.py artifacts."""

    def __init__(
        self,
        model_path: Path = Path("artifacts/cnn1d_mtc_byte.pth"),
        label_path: Path = Path("artifacts/labels_mtc_byte.json"),
        seq_len: int = 1500,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.labels: Dict[int, str] = {}
        self.enabled = False

        if not label_path.exists() or not model_path.exists():
            return

        with label_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        idx_to_orig = {
            int(k): int(v) for k, v in meta.get("model_index_to_original_id", {}).items()
        }
        orig_to_name = {
            int(k): str(v) for k, v in meta.get("original_id_to_name", {}).items()
        }
        self.labels = {
            model_idx: orig_to_name.get(orig_id, f"class_{orig_id}")
            for model_idx, orig_id in idx_to_orig.items()
        }

        num_classes = max(len(self.labels), 1)
        self.model = CNN1DClassifier(seq_len=self.seq_len, num_classes=num_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.enabled = True

    def predict(self, feature_vector) -> str:
        if not self.enabled:
            return "ModelNotReady"
        x = (
            torch.tensor(feature_vector, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        with torch.no_grad():
            logits = self.model(x)
            pred = int(torch.argmax(logits, dim=1).item())
        return self.labels.get(pred, "Unknown")

