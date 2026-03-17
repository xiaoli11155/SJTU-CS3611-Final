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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor | None = None,
    gamma: float = 1.5,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none", weight=class_weight)
    pt = torch.exp(-ce).clamp_min(1e-8)
    return (((1.0 - pt) ** gamma) * ce).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--focal-gamma",
        default=0,
        type=float,
        help="Use focal loss when > 0. Set 0 to disable.",
    )
    parser.add_argument(
        "--boost-class",
        default="",
        type=str,
        help="Class name to up-weight in loss, e.g. Web.",
    )
    parser.add_argument(
        "--boost-factor",
        default=1.0,
        type=float,
        help="Multiplier for --boost-class, e.g. 1.3.",
    )
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
    num_classes = len(encoder.classes_)
    print("Label mapping (index -> class):")
    for idx, name in enumerate(encoder.classes_):
        print(f"  {idx}: {name}")
    if NUM_CLASSES != num_classes:
        print(
            f"[WARN] src.config.NUM_CLASSES={NUM_CLASSES}, but CSV has {num_classes} classes. "
            "Training will use CSV classes."
        )

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0))
    class_weights = class_weights / class_weights.mean()
    if args.boost_class and args.boost_factor > 0:
        class_name_to_idx = {name: idx for idx, name in enumerate(encoder.classes_)}
        if args.boost_class in class_name_to_idx:
            idx = class_name_to_idx[args.boost_class]
            class_weights[idx] = class_weights[idx] * args.boost_factor
            class_weights = class_weights / class_weights.mean()
            print(f"Boost class: {args.boost_class} x{args.boost_factor:.3f}")
        else:
            print(f"[WARN] --boost-class '{args.boost_class}' not in {encoder.classes_.tolist()}")
    print("Train class distribution:")
    for idx, c in enumerate(class_counts.tolist()):
        print(f"  {idx} ({encoder.classes_[idx]}): {int(c)}")

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

    model = CNN1DClassifier(seq_len=seq_len, num_classes=num_classes).to(device)
    class_weights_device = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_device,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_losses = []
    val_accs = []
    val_macro_f1s = []
    best_state = None
    best_acc = 0.0
    best_macro_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            bx = bx + 0.01 * torch.randn_like(bx)
            bx = bx.clamp(0.0, 1.0).unsqueeze(1)
            logits = model(bx)
            if args.focal_gamma > 0:
                loss = focal_cross_entropy(
                    logits=logits,
                    targets=by,
                    class_weight=class_weights_device,
                    gamma=args.focal_gamma,
                )
            else:
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
        macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_losses.append(avg_loss)
        val_accs.append(acc)
        val_macro_f1s.append(macro_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f} - "
            f"val_acc: {acc:.4f} - val_macro_f1: {macro_f1:.4f} - lr: {current_lr:.6f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, MODEL_PATH)
    model.load_state_dict(best_state)
    model.eval()
    with LABEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, ensure_ascii=True, indent=2)

    best_pred = []
    best_true = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx.unsqueeze(1))
            pred = torch.argmax(logits, dim=1)
            best_pred.extend(pred.cpu().tolist())
            best_true.extend(by.cpu().tolist())

    print("Classification report (best model):")
    print(classification_report(best_true, best_pred, target_names=encoder.classes_, digits=4, zero_division=0))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(train_losses)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(val_accs)
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(val_macro_f1s)
    axes[2].set_title("Val Macro-F1")
    axes[2].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(MODEL_PATH.parent / "training_curves.png", dpi=150)

    cm = confusion_matrix(best_true, best_pred, labels=list(range(num_classes)))
    cm_norm = confusion_matrix(best_true, best_pred, labels=list(range(num_classes)), normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(MODEL_PATH.parent / "confusion_matrix.png", dpi=150)

    print(f"Best val_macro_f1: {best_macro_f1:.4f}")
    print(f"Best val_acc: {best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABEL_PATH}")


if __name__ == "__main__":
    main()
