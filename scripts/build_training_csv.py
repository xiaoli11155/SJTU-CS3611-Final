import argparse
from glob import glob
from pathlib import Path
import sys

# Allow running this file directly: `python scripts/build_training_csv.py ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.pcap_preprocess import pcap_to_csv
from src.config import SEQ_LEN


CLASS_MAP = {
    "video": "Video",
    "chat": "Chat",
    "email": "Email",
    "audio": "Audio",
    "file": "FileTransfer",
    "filetransfer": "FileTransfer",
    "file_transform": "FileTransfer",
    "filetransform": "FileTransfer",
    "web": "Web",
}


def infer_source_group_name(spec: str) -> str:
    """Infer a stable output folder name from file/dir/glob input."""
    spec = spec.strip()
    path = Path(spec)

    if path.is_dir():
        return path.name

    if path.is_file():
        return path.parent.name

    if any(ch in spec for ch in "*?[]"):
        first_wildcard = min(spec.find(ch) for ch in "*?[]" if ch in spec)
        static_prefix = spec[:first_wildcard].rstrip("/\\")
        if static_prefix:
            prefix_path = Path(static_prefix)
            if prefix_path.suffix:
                return prefix_path.parent.name or "pcap_inputs"
            return prefix_path.name

    return "pcap_inputs"


def resolve_pcap_paths(spec: str) -> list[Path]:
    spec = spec.strip()
    path = Path(spec)

    if path.is_file():
        return [path]

    if path.is_dir():
        out = sorted(path.rglob("*.pcap")) + sorted(path.rglob("*.pcapng"))
        return out

    # Support shell-like patterns on all platforms.
    if any(ch in spec for ch in "*?[]"):
        return [Path(p) for p in sorted(glob(spec))]

    raise FileNotFoundError(f"PCAP path not found: {spec}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged training CSV from class-labeled PCAP files")
    parser.add_argument(
        "--class-pcap",
        action="append",
        default=[],
        help=(
            "Class and pcap path pair, format: class_name=path/to/file.pcap. "
            "Path can be a single file, a folder, or a wildcard pattern. "
            "Repeat this option for 3-4 classes."
        ),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/training/flows.csv"))
    parser.add_argument(
        "--seq-len",
        action="append",
        type=int,
        default=[],
        help=(
            "Sequence length for generated features. Repeat for multiple outputs, "
            "e.g., --seq-len 32 --seq-len 64 --seq-len 100. "
            "If omitted, defaults to src.config.SEQ_LEN."
        ),
    )
    args = parser.parse_args()

    if not args.class_pcap:
        print("No class PCAP provided.")
        print("Example: --class-pcap video=data/extracted/video.pcap --class-pcap chat=data/extracted/chat.pcap")
        return

    seq_lens = sorted(set(args.seq_len)) if args.seq_len else [SEQ_LEN]
    for seq_len in seq_lens:
        if seq_len <= 0:
            raise ValueError(f"Invalid --seq-len: {seq_len}. It must be positive.")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    out_base = args.out
    out_stem = out_base.stem
    out_suffix = out_base.suffix or ".csv"

    for seq_len in seq_lens:
        csv_files = []
        seq_work_dir = args.work_dir / f"seq{seq_len}"
        seq_work_dir.mkdir(parents=True, exist_ok=True)

        for pair in args.class_pcap:
            if "=" not in pair:
                raise ValueError(f"Invalid --class-pcap format: {pair}")
            cls, pcap = pair.split("=", 1)
            cls_norm = cls.strip().lower()
            label = CLASS_MAP.get(cls_norm, cls.strip())
            source_group = infer_source_group_name(pcap)
            group_work_dir = seq_work_dir / source_group
            pcap_paths = resolve_pcap_paths(pcap)
            if not pcap_paths:
                print(f"Skip class '{cls}': no pcap found from '{pcap}'")
                continue

            class_parts = []
            for pcap_path in pcap_paths:
                # Keep each pcap's features in its own directory for easier incremental data prep.
                pcap_dir = group_work_dir / pcap_path.stem
                part_csv = pcap_dir / f"{cls_norm}.csv"
                pcap_to_csv(pcap_path, part_csv, label, seq_len=seq_len)
                class_parts.append(part_csv)

            out_csv = group_work_dir / f"{cls_norm}.csv"
            pd.concat([pd.read_csv(f) for f in class_parts], ignore_index=True).to_csv(out_csv, index=False)
            print(f"[SEQ_LEN={seq_len}] Merged class '{cls_norm}' from {len(class_parts)} file(s) -> {out_csv}")
            csv_files.append(out_csv)

        if not csv_files:
            raise ValueError(f"[SEQ_LEN={seq_len}] No valid class CSV files were generated.")

        merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        out_path = out_base if len(seq_lens) == 1 else out_base.with_name(f"{out_stem}_seq{seq_len}{out_suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"[SEQ_LEN={seq_len}] Merged {len(csv_files)} class files into: {out_path}")
        print(f"[SEQ_LEN={seq_len}] Total flow samples: {len(merged)}")


if __name__ == "__main__":
    main()
