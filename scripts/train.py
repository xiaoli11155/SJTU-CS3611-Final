import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.config import LABEL_PATH, MODEL_PATH, NUM_CLASSES, SEQ_LEN
from src.model.cnn1d import CNN1DClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    feature_cols = [f"f{i}" for i in range(SEQ_LEN)]

    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y_raw = df["label"].astype(str).values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_raw)
    y = torch.tensor(y_enc, dtype=torch.long)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = CNN1DClassifier(seq_len=SEQ_LEN, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_accs = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx = bx.unsqueeze(1)
            logits = model(bx)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.unsqueeze(1))
                pred = torch.argmax(logits, dim=1)
                all_pred.extend(pred.tolist())
                all_true.extend(by.tolist())

        avg_loss = total_loss / max(len(train_loader), 1)
        acc = accuracy_score(all_true, all_pred)
        train_losses.append(avg_loss)
        val_accs.append(acc)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f} - val_acc: {acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
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

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABEL_PATH}")


if __name__ == "__main__":
    main()
