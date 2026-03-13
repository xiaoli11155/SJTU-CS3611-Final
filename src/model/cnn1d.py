import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    def __init__(self, seq_len: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        reduced = seq_len // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * reduced, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
