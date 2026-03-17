import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C]
        out, _ = self.lstm(x)
        if lengths is None:
            pooled = out.mean(dim=1)
            return self.classifier(pooled)

        # Mask padded timesteps and pool only valid sequence region.
        lengths = lengths.to(out.device).clamp_min(1)
        max_t = out.size(1)
        time_idx = torch.arange(max_t, device=out.device).unsqueeze(0)
        mask = (time_idx < lengths.unsqueeze(1)).unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        pooled = summed / lengths.unsqueeze(1).float()
        return self.classifier(pooled)
