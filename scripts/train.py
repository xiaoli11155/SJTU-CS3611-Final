import argparse
import json
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Allow running this file directly: `python scripts/train.py ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_PATH, MODEL_PATH, NUM_CLASSES, SEQ_LEN
from src.model.cnn1d import CNN1DClassifier


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--seq-len",
        default=None,
        type=int,
        help="Override sequence length. If omitted, inferred from CSV columns f0..fN.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    if args.seq_len is not None:
        if args.seq_len <= 0:
            raise ValueError("--seq-len must be positive.")
        seq_len = args.seq_len
        feature_cols = [f"f{i}" for i in range(seq_len)]
    else:
        feature_cols = sorted(
            [c for c in df.columns if c.startswith("f")],
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 10**9,
        )
        if not feature_cols:
            raise ValueError("No feature columns found. Expected f0..fN in CSV.")
        seq_len = len(feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing[:5]} (total {len(missing)})")
    print(f"Using SEQ_LEN={seq_len} from {'--seq-len' if args.seq_len is not None else 'CSV columns'}.")

    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y_raw = df["label"].astype(str).values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_raw)
    y = torch.tensor(y_enc, dtype=torch.long)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    class_counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0))
    class_weights = class_weights / class_weights.mean()

    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = CNN1DClassifier(seq_len=seq_len, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_losses = []
    val_accs = []
    best_state = None
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            bx = bx + 0.01 * torch.randn_like(bx)
            bx = bx.clamp(0.0, 1.0).unsqueeze(1)
            logits = model(bx)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()

        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx.unsqueeze(1))
                pred = torch.argmax(logits, dim=1)
                all_pred.extend(pred.cpu().tolist())
                all_true.extend(by.cpu().tolist())

        avg_loss = total_loss / max(len(train_loader), 1)
        acc = accuracy_score(all_true, all_pred)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_losses.append(avg_loss)
        val_accs.append(acc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f} - "
            f"val_acc: {acc:.4f} - lr: {current_lr:.6f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, MODEL_PATH)
    with LABEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, ensure_ascii=True, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(val_accs)
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(MODEL_PATH.parent / "training_curves.png", dpi=150)

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(MODEL_PATH.parent / "confusion_matrix.png", dpi=150)

    print(f"Best val_acc: {best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABEL_PATH}")


if __name__ == "__main__":
    main()
