import argparse
from pathlib import Path

import pandas as pd

from src.data.pcap_preprocess import pcap_to_csv


CLASS_MAP = {
    "video": "Video",
    "chat": "Chat",
    "file": "FileTransfer",
    "web": "Web",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged training CSV from class-labeled PCAP files")
    parser.add_argument(
        "--class-pcap",
        action="append",
        default=[],
        help=(
            "Class and pcap path pair, format: class_name=path/to/file.pcap. "
            "Repeat this option for 3-4 classes."
        ),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/flows.csv"))
    args = parser.parse_args()

    if not args.class_pcap:
        print("No class PCAP provided.")
        print("Example: --class-pcap video=data/extracted/video.pcap --class-pcap chat=data/extracted/chat.pcap")
        return

    args.work_dir.mkdir(parents=True, exist_ok=True)
    csv_files = []

    for pair in args.class_pcap:
        if "=" not in pair:
            raise ValueError(f"Invalid --class-pcap format: {pair}")
        cls, pcap = pair.split("=", 1)
        cls_norm = cls.strip().lower()
        label = CLASS_MAP.get(cls_norm, cls.strip())
        pcap_path = Path(pcap.strip())

        out_csv = args.work_dir / f"{cls_norm}.csv"
        pcap_to_csv(pcap_path, out_csv, label)
        csv_files.append(out_csv)

    merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Merged {len(csv_files)} class files into: {args.out}")
    print(f"Total flow samples: {len(merged)}")


if __name__ == "__main__":
    main()
