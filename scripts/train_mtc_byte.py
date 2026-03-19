import argparse
import json
import pickle
import random
import signal
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Allow running this file directly: `python scripts/train_mtc_byte.py ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.cnn1d import CNN1DClassifier


INTERRUPTED = False


def _handle_sigint(signum, frame) -> None:
    del signum, frame
    global INTERRUPTED
    INTERRUPTED = True
    print("\n[INTERRUPT] Ctrl+C received. Stopping training now...")
    raise KeyboardInterrupt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(path: Path) -> list[dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def cap_rows_per_class(rows: list[dict], max_per_class: int, seed: int) -> list[dict]:
    if max_per_class <= 0:
        return rows

    rng = random.Random(seed)
    by_class: dict[int, list[dict]] = {}
    for row in rows:
        label = int(row["traffic_label"])
        by_class.setdefault(label, []).append(row)

    capped: list[dict] = []
    for label, items in by_class.items():
        if len(items) > max_per_class:
            capped.extend(rng.sample(items, max_per_class))
        else:
            capped.extend(items)

    rng.shuffle(capped)
    return capped


def rows_to_tensors(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for row in rows:
        if "feature" not in row or "traffic_label" not in row:
            raise ValueError("Each row must contain 'feature' and 'traffic_label'.")
        features.append(np.asarray(row["feature"], dtype=np.float32))
        labels.append(int(row["traffic_label"]))

    x = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(np.array(labels, dtype=np.int64), dtype=torch.long)
    return x, y


def build_label_names(class_ids: list[int]) -> list[str]:
    # Keep compatibility with current non-VPN 5-class setup.
    base = {
        0: "chat",
        1: "email",
        2: "file_transfer",
        3: "streaming",
        4: "voip",
    }
    out = []
    for cid in class_ids:
        out.append(base.get(cid, f"class_{cid}"))
    return out


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    all_pred: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for bx, by in loader:
            if INTERRUPTED:
                break
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx.unsqueeze(1))
            pred = torch.argmax(logits, dim=1)
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(by.cpu().tolist())

    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return acc, macro_f1, all_true, all_pred


def main() -> None:
    global INTERRUPTED
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Train CNN1D on MTC-style byte features from pkl rows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/mtc_byte_7class"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=0, help="Cap training rows per class (0 means no cap).")
    parser.add_argument("--max-per-class-val", type=int, default=0, help="Cap validation rows per class (0 means no cap).")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/cnn1d_mtc_byte.pth"))
    parser.add_argument("--label-out", type=Path, default=Path("artifacts/labels_mtc_byte.json"))
    parser.add_argument(
        "--cm-out",
        type=Path,
        default=Path("artifacts/confusion_matrix_mtc_byte.png"),
        help="Output path for confusion matrix figure.",
    )
    parser.add_argument(
        "--curves-out",
        type=Path,
        default=Path("artifacts/training_curves_mtc_byte.png"),
        help="Output path for training curves figure.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.max_per_class < 0:
        raise ValueError("--max-per-class must be >= 0.")
    if args.max_per_class_val < 0:
        raise ValueError("--max-per-class-val must be >= 0.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = args.data_dir / "train_data_rows.pkl"
    val_path = args.data_dir / "val_data_rows.pkl"
    test_path = args.data_dir / "test_data_rows.pkl"
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train_rows = load_rows(train_path)
    val_rows = load_rows(val_path)
    test_rows = load_rows(test_path)
    if args.max_per_class > 0:
        before_counts: dict[int, int] = {}
        for row in train_rows:
            k = int(row["traffic_label"])
            before_counts[k] = before_counts.get(k, 0) + 1
        train_rows = cap_rows_per_class(train_rows, max_per_class=args.max_per_class, seed=args.seed)
        after_counts: dict[int, int] = {}
        for row in train_rows:
            k = int(row["traffic_label"])
            after_counts[k] = after_counts.get(k, 0) + 1
        print(f"Applied training class cap: max_per_class={args.max_per_class}")
        print("Train counts before cap:", dict(sorted(before_counts.items())))
        print("Train counts after cap:", dict(sorted(after_counts.items())))
    if args.max_per_class_val > 0:
        before_counts_val: dict[int, int] = {}
        for row in val_rows:
            k = int(row["traffic_label"])
            before_counts_val[k] = before_counts_val.get(k, 0) + 1
        val_rows = cap_rows_per_class(val_rows, max_per_class=args.max_per_class_val, seed=args.seed)
        after_counts_val: dict[int, int] = {}
        for row in val_rows:
            k = int(row["traffic_label"])
            after_counts_val[k] = after_counts_val.get(k, 0) + 1
        print(f"Applied validation class cap: max_per_class_val={args.max_per_class_val}")
        print("Val counts before cap:", dict(sorted(before_counts_val.items())))
        print("Val counts after cap:", dict(sorted(after_counts_val.items())))

    x_train_raw, y_train_raw = rows_to_tensors(train_rows)
    x_val_raw, y_val_raw = rows_to_tensors(val_rows)
    x_test_raw, y_test_raw = rows_to_tensors(test_rows)

    class_ids = sorted(set(y_train_raw.tolist()) | set(y_val_raw.tolist()) | set(y_test_raw.tolist()))
    id_to_new = {cid: i for i, cid in enumerate(class_ids)}
    new_to_id = {v: k for k, v in id_to_new.items()}
    class_names = build_label_names(class_ids)

    y_train = torch.tensor([id_to_new[int(y)] for y in y_train_raw.tolist()], dtype=torch.long)
    y_val = torch.tensor([id_to_new[int(y)] for y in y_val_raw.tolist()], dtype=torch.long)
    y_test = torch.tensor([id_to_new[int(y)] for y in y_test_raw.tolist()], dtype=torch.long)

    seq_len = x_train_raw.size(1)
    num_classes = len(class_ids)
    print(f"Using device: {device}")
    print(f"SEQ_LEN={seq_len}, classes={num_classes}")
    print("Class mapping (model_index -> original_id -> name):")
    for idx, cid in enumerate(class_ids):
        print(f"  {idx} -> {cid} -> {class_names[idx]}")

    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0))
    class_weights = class_weights / class_weights.mean()
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        TensorDataset(x_train_raw, y_train),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_raw, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(x_test_raw, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CNN1DClassifier(seq_len=seq_len, num_classes=num_classes).to(device)
    class_weights_device = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_device, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_state = None
    best_val_f1 = -1.0
    best_val_acc = 0.0
    train_losses: list[float] = []
    val_accs: list[float] = []
    val_f1s: list[float] = []

    try:
        for epoch in range(args.epochs):
            if INTERRUPTED:
                break

            model.train()
            running_loss = 0.0

            if tqdm is not None:
                train_iter = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{args.epochs}",
                    unit="batch",
                    leave=False,
                )
            else:
                train_iter = train_loader

            for bx, by in train_iter:
                if INTERRUPTED:
                    break
                bx = bx.to(device)
                by = by.to(device)

                bx = bx + 0.01 * torch.randn_like(bx)
                bx = bx.clamp(0.0, 1.0)
                logits = model(bx.unsqueeze(1))
                loss = criterion(logits, by)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += float(loss.item())

                if tqdm is not None:
                    train_iter.set_postfix(loss=f"{float(loss.item()):.4f}")

            if tqdm is not None:
                train_iter.close()
            if INTERRUPTED:
                break

            scheduler.step()

            avg_loss = running_loss / max(len(train_loader), 1)
            val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            train_losses.append(float(avg_loss))
            val_accs.append(float(val_acc))
            val_f1s.append(float(val_f1))

            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"loss: {avg_loss:.4f} - val_acc: {val_acc:.4f} - val_macro_f1: {val_f1:.4f}"
            )
    except KeyboardInterrupt:
        INTERRUPTED = True
        print("\n[INTERRUPT] KeyboardInterrupt captured. Stopping training now...")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.label_out.parent.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, args.model_out)

    label_meta = {
        "model_index_to_original_id": new_to_id,
        "original_id_to_name": {cid: name for cid, name in zip(class_ids, class_names)},
    }
    with args.label_out.open("w", encoding="utf-8") as f:
        json.dump(label_meta, f, ensure_ascii=True, indent=2)

    if INTERRUPTED:
        print("[INTERRUPT] Saved best checkpoint and exiting without test evaluation.")
        print(f"Model saved to: {args.model_out}")
        print(f"Label meta saved to: {args.label_out}")
        return

    model.load_state_dict(best_state)
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"Best val_macro_f1: {best_val_f1:.4f}")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Test macro_f1: {test_f1:.4f}")
    print("Test classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    print(f"Model saved to: {args.model_out}")
    print(f"Label meta saved to: {args.label_out}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize="true")
    args.cm_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        args.curves_out.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(train_losses)
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[1].plot(val_accs)
        axes[1].set_title("Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[2].plot(val_f1s)
        axes[2].set_title("Val Macro-F1")
        axes[2].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(args.curves_out, dpi=160)
        print(f"Training curves saved to: {args.curves_out}")
    except Exception as e:
        print(f"[WARN] Failed to draw training curves: {e}")

    try:
        import seaborn as sns

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("MTC Byte Confusion Matrix (Test)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(args.cm_out, dpi=160)
        print(f"Confusion matrix saved to: {args.cm_out}")
    except Exception as e:
        print(f"[WARN] Failed to draw confusion matrix figure: {e}")


if __name__ == "__main__":
    main()
