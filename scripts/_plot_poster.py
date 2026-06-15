"""生成 Poster 所需的对比图"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import csv

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# ── 中文字体设置 ──
# 尝试常见中文字体
zh_fonts = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"]
found = None
for f in zh_fonts:
    try:
        fm.findfont(f, fallback_to_default=False)
        found = f
        break
    except Exception:
        continue
if found is None:
    # fallback: any sans-serif
    found = "sans-serif"
plt.rcParams["font.family"] = found
plt.rcParams["axes.unicode_minus"] = False


def read_best(log_path: Path):
    """Read best macro_f1 and acc from training_log.csv"""
    if not log_path.exists():
        return None, None
    best_f1 = 0
    best_acc = 0
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            f1 = float(row["val_macro_f1"])
            acc = float(row["val_acc"])
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
    return best_f1, best_acc


def collect_results():
    results = {}
    for d in sorted(ARTIFACTS.glob("*")):
        if not d.is_dir():
            continue
        log = d / "training_log.csv"
        f1, acc = read_best(log)
        if f1 is None:
            continue
        name = d.name  # e.g. tcp3class_nodrop_cnn_seq32
        results[name] = {"macro_f1": f1, "acc": acc, "dir": d}
        # Also read labels
        labels_file = d / "labels.json"
        if labels_file.exists():
            with open(labels_file) as fh:
                results[name]["labels"] = json.load(fh)
    return results


# ═══════════════════════════════════════════════════
# FIG 1: seq 长度对比 (tcp3class & tcp4class, CNN)
# ═══════════════════════════════════════════════════
def plot_seq_comparison(results):
    schemes = ["tcp3class", "tcp4class"]
    seqs = [32, 64, 100]
    scheme_labels = {"tcp3class": "TCP 3类", "tcp4class": "TCP 4类"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seqs))
    width = 0.35
    colors = ["#2E86AB", "#A23B72"]

    for i, scheme in enumerate(schemes):
        vals = []
        for s in seqs:
            key = f"{scheme}_nodrop_cnn_seq{s}"
            vals.append(results.get(key, {}).get("macro_f1", 0))
        bars = ax.bar(x + i * width, vals, width, label=scheme_labels[scheme], color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_xlabel("序列长度 (seq_len)", fontsize=12)
    ax.set_title("不同序列长度 CNN 分类效果对比", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"seq{s}" for s in seqs])
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, 0.85)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "poster_seq_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved: poster_seq_comparison.png")


# ═══════════════════════════════════════════════════
# FIG 2: CNN vs LSTM (3/4/5 class, seq100)
# ═══════════════════════════════════════════════════
def plot_cnn_vs_lstm(results):
    schemes = ["tcp3class", "tcp4class", "tcp5class"]
    scheme_labels = ["TCP 3类", "TCP 4类", "TCP 5类"]

    cnn_vals = []
    lstm_vals = []
    for s in schemes:
        key_cnn = f"{s}_nodrop_cnn_seq100"
        key_lstm = f"{s}_nodrop_lstm_seq100"
        cnn_vals.append(results.get(key_cnn, {}).get("macro_f1", 0))
        lstm_vals.append(results.get(key_lstm, {}).get("macro_f1", 0))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(schemes))
    width = 0.35
    bars1 = ax.bar(x - width / 2, cnn_vals, width, label="CNN", color="#2E86AB")
    bars2 = ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#D33F49")

    for bar, v in zip(bars1, cnn_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    for bar, v in zip(bars2, lstm_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("CNN vs LSTM 分类效果对比 (seq100)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scheme_labels, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "poster_cnn_vs_lstm.png", dpi=200)
    plt.close(fig)
    print("Saved: poster_cnn_vs_lstm.png")


# ═══════════════════════════════════════════════════
# FIG 3: 类数影响对比 (CNN, seq100)
# ═══════════════════════════════════════════════════
def plot_class_count(results):
    schemes = ["tcp3class", "tcp4class", "tcp5class"]
    labels_zh = ["TCP 3类", "TCP 4类", "TCP 5类"]
    f1_vals = []
    acc_vals = []
    for s in schemes:
        key = f"{s}_nodrop_cnn_seq100"
        f1_vals.append(results.get(key, {}).get("macro_f1", 0))
        acc_vals.append(results.get(key, {}).get("acc", 0))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(schemes))
    ax.plot(x, f1_vals, "o-", color="#2E86AB", linewidth=2, markersize=10, label="Macro F1")
    ax.plot(x, acc_vals, "s--", color="#A23B72", linewidth=2, markersize=10, label="Accuracy")
    for i, (f1, acc) in enumerate(zip(f1_vals, acc_vals)):
        ax.annotate(f"{f1:.4f}", (i, f1), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
        ax.annotate(f"{acc:.4f}", (i, acc), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_zh, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("类别粒度对分类效果的影响 (CNN, seq100)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 0.85)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "poster_class_count.png", dpi=200)
    plt.close(fig)
    print("Saved: poster_class_count.png")


# ═══════════════════════════════════════════════════
# FIG 4: 最佳模型混淆矩阵 (tcp3class CNN seq64)
# ═══════════════════════════════════════════════════
def plot_best_confusion(results):
    best_dir = ARTIFACTS / "tcp3class_nodrop_cnn_seq64"
    cm_path = best_dir / "confusion_matrix.png"
    if cm_path.exists():
        import shutil
        shutil.copy(cm_path, ARTIFACTS / "poster_best_confusion.png")
        print("Copied: poster_best_confusion.png")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    results = collect_results()
    print(f"Collected {len(results)} experiment results")
    for k, v in sorted(results.items()):
        print(f"  {k}: macro_f1={v['macro_f1']:.4f}, acc={v['acc']:.4f}")

    plot_seq_comparison(results)
    plot_cnn_vs_lstm(results)
    plot_class_count(results)
    plot_best_confusion(results)
    print("\nAll poster plots generated in artifacts/")
