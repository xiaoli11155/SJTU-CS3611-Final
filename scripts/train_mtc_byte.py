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
from sklearn.metrics import accuracy_score, classification_report, f1_score
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/cnn1d_mtc_byte.pth"))
    parser.add_argument("--label-out", type=Path, default=Path("artifacts/labels_mtc_byte.json"))
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")

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


if __name__ == "__main__":
    main()
