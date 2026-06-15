"""临时脚本：检查各数据集类别分布 & 原始 PCAP 统计"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- 1. 训练 CSV 类别分布 ---
training_csvs = sorted((PROJECT_ROOT / "data/processed/training").glob("*.csv"))
for csv_path in training_csvs:
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    print(f"\n=== {csv_path.name} ===")
    print(f"总样本数: {len(df)}, 类别数: {len(counts)}")
    for label, cnt in counts.items():
        print(f"  {label}: {cnt}")

# --- 2. 原始 PCAP 文件统计 ---
raw_dir = PROJECT_ROOT / "data/raw"
print(f"\n=== 原始 PCAP 文件 ===")
pcap_files = list(raw_dir.rglob("*.pcap")) + list(raw_dir.rglob("*.pcapng"))
print(f"PCAP 文件总数: {len(pcap_files)}")
# 按目录分组
dir_counts = Counter(f.parent.relative_to(raw_dir) for f in pcap_files)
for d, cnt in dir_counts.most_common():
    print(f"  {d}: {cnt} 个 PCAP")

# --- 3. 检查处理参数：去掉重复特征的影响 ---
# 选一个 PCAP 体验一下
if pcap_files:
    test_pcap = pcap_files[0]
    print(f"\n=== 测试 PCAP: {test_pcap.name} ===")
    # 模拟 pcap_preprocess 的处理
    from src.data.pcap_preprocess import pcap_to_csv
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "nodrop.csv"
        out2 = Path(tmpdir) / "drop.csv"
        pcap_to_csv(test_pcap, out1, seq_len=32, min_packets=1,
                    window_stride=0, max_windows_per_flow=0,
                    signed_lengths=False, directional=False,
                    keep_duplicate_features=True)
        pcap_to_csv(test_pcap, out2, seq_len=32, min_packets=1,
                    window_stride=0, max_windows_per_flow=0,
                    signed_lengths=False, directional=False,
                    keep_duplicate_features=False)
        df1 = pd.read_csv(out1) if out1.exists() else pd.DataFrame()
        df2 = pd.read_csv(out2) if out2.exists() else pd.DataFrame()
        print(f"  保留重复: {len(df1)} 条, 去除重复: {len(df2)} 条")

print("\n完成。")
