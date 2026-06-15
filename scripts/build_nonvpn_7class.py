import argparse
from pathlib import Path
import sys

import pandas as pd

# Allow running this file directly: `python scripts/build_nonvpn_7class.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SEQ_LEN
from src.data.pcap_preprocess import pcap_to_csv


CLASS_RULES_TCP7 = [
    (
        "TextChat",
        [
            "aim",
            "icq",
            "gmailchat",
            "facebookchat",
            "facebook_chat",
            "hangouts_chat",
            "hangout_chat",
            "skype_chat",
        ],
    ),
    ("Email", ["email"]),
    (
        "Audio",
        [
            "facebook_audio",
            "hangouts_audio",
            "skype_audio",
            "voipbuster",
            "spotify",
        ],
    ),
    ("StreamingVideo", ["youtube", "netflix", "vimeo"]),
    (
        "VideoCall",
        [
            "facebook_video",
            "hangouts_video",
            "skype_video",
        ],
    ),
    (
        "FileDownload",
        [
            "ftps_down",
            "scpdown",
            "scp_down",
            "sftpdown",
            "sftp_down",
        ],
    ),
    (
        "FileUpload",
        [
            "ftps_up",
            "scpup",
            "scp_up",
            "sftpup",
            "sftp_up",
        ],
    ),
]


CLASS_RULES_TCP8 = [
    (
        "TextChat",
        [
            "aim",
            "icq",
            "gmailchat",
            "facebookchat",
            "facebook_chat",
            "hangouts_chat",
            "hangout_chat",
            "skype_chat",
        ],
    ),
    ("Email", ["email"]),
    (
        "AudioCall",
        [
            "facebook_audio",
            "hangouts_audio",
            "skype_audio",
            "voipbuster",
        ],
    ),
    ("Music", ["spotify"]),
    ("StreamingVideo", ["youtube", "netflix", "vimeo"]),
    (
        "VideoCall",
        [
            "facebook_video",
            "hangouts_video",
            "skype_video",
        ],
    ),
    (
        "FileDownload",
        [
            "ftps_down",
            "scpdown",
            "scp_down",
            "sftpdown",
            "sftp_down",
        ],
    ),
    (
        "FileUpload",
        [
            "ftps_up",
            "scpup",
            "scp_up",
            "sftpup",
            "sftp_up",
        ],
    ),
]


CLASS_RULES_TCP5 = [
    (
        "Communication",
        [
            "aim",
            "icq",
            "gmailchat",
            "facebookchat",
            "facebook_chat",
            "hangouts_chat",
            "hangout_chat",
            "skype_chat",
            "facebook_audio",
            "hangouts_audio",
            "skype_audio",
            "voipbuster",
        ],
    ),
    ("Email", ["email"]),
    (
        "StreamingMedia",
        [
            "spotify",
            "youtube",
            "netflix",
            "vimeo",
        ],
    ),
    (
        "VideoCall",
        [
            "facebook_video",
            "hangouts_video",
            "skype_video",
        ],
    ),
    (
        "FileTransfer",
        [
            "ftps_down",
            "ftps_up",
            "scpdown",
            "scp_down",
            "scpup",
            "scp_up",
            "scp",
            "sftpdown",
            "sftp_down",
            "sftpup",
            "sftp_up",
            "sftp",
            "skype_file",
        ],
    ),
]


CLASS_RULES_TCP4 = [
    (
        "Communication",
        [
            "aim",
            "icq",
            "gmailchat",
            "facebookchat",
            "facebook_chat",
            "hangouts_chat",
            "hangout_chat",
            "skype_chat",
            "facebook_audio",
            "hangouts_audio",
            "skype_audio",
            "voipbuster",
        ],
    ),
    ("Email", ["email"]),
    (
        "StreamingMedia",
        [
            "spotify",
            "youtube",
            "netflix",
            "vimeo",
            "facebook_video",
            "hangouts_video",
            "skype_video",
        ],
    ),
    (
        "FileTransfer",
        [
            "ftps_down",
            "ftps_up",
            "scpdown",
            "scp_down",
            "scpup",
            "scp_up",
            "scp",
            "sftpdown",
            "sftp_down",
            "sftpup",
            "sftp_up",
            "sftp",
            "skype_file",
        ],
    ),
]

CLASS_RULES_TCP3 = [
    (
        "Communication",
        [
            "aim",
            "icq",
            "gmailchat",
            "facebookchat",
            "facebook_chat",
            "hangouts_chat",
            "hangout_chat",
            "skype_chat",
            "facebook_audio",
            "hangouts_audio",
            "skype_audio",
            "voipbuster",
            "email",
        ],
    ),
    (
        "StreamingMedia",
        [
            "spotify",
            "youtube",
            "netflix",
            "vimeo",
            "facebook_video",
            "hangouts_video",
            "skype_video",
        ],
    ),
    (
        "FileTransfer",
        [
            "ftps_down",
            "ftps_up",
            "scpdown",
            "scp_down",
            "scpup",
            "scp_up",
            "scp",
            "sftpdown",
            "sftp_down",
            "sftpup",
            "sftp_up",
            "sftp",
            "skype_file",
        ],
    ),
]


def get_class_rules(scheme: str) -> list[tuple[str, list[str]]]:
    if scheme == "tcp7class":
        return CLASS_RULES_TCP7
    if scheme == "tcp8class":
        return CLASS_RULES_TCP8
    if scheme == "tcp3class":
        return CLASS_RULES_TCP3
    if scheme == "tcp4class":
        return CLASS_RULES_TCP4
    if scheme == "tcp5class":
        return CLASS_RULES_TCP5
    raise ValueError(f"Unknown class scheme: {scheme}")


def infer_label(path: Path, class_rules: list[tuple[str, list[str]]] = CLASS_RULES_TCP7) -> str | None:
    name = path.stem.lower()
    for label, prefixes in class_rules:
        if any(name.startswith(prefix) for prefix in prefixes):
            return label
    return None


def iter_nonvpn_pcaps(raw_dir: Path, subset: str | None = None) -> list[Path]:
    patterns = ["*.pcap", "*.pcapng"]
    out: list[Path] = []
    if subset:
        dirs = [raw_dir / subset]
    else:
        dirs = sorted(raw_dir.glob("NonVPN-PCAPs-*"))
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in patterns:
            out.extend(d.glob(pattern))
    return sorted(set(out))


def read_csv_rows(path: Path, label: str, seq_len: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if not is_valid_part_csv(path, label, seq_len):
        print(f"[WARN] Skip invalid CSV during merge: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def is_valid_part_csv(path: Path, label: str, seq_len: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    feature_cols = [f"f{i}" for i in range(seq_len)]
    required_cols = feature_cols + ["label"]
    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    if list(df.columns) != required_cols:
        return False
    if df.empty:
        return True
    return bool((df["label"].astype(str) == label).all())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a NonVPN packet-length training CSV from data/raw."
    )
    parser.add_argument(
        "--scheme",
        choices=["tcp3class", "tcp4class", "tcp5class", "tcp7class", "tcp8class"],
        default="tcp7class",
        help="Class mapping scheme. tcp3class: Comm+Email / Media / File. "
             "tcp4class: Comm / Email / Media+VideoCall / File. "
             "tcp5class: Comm / Email / Media / VideoCall / File. "
             "tcp7class: fine-grained 7 classes. "
             "tcp8class: tcp7class + Spotify split as Music.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--subset",
        default=None,
        type=str,
        help="Only use a specific NonVPN subset, e.g. NonVPN-PCAPs-01.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/processed/nonvpn_tcp7class"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/training/flows_nonvpn_tcp7class.csv"))
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--min-packets",
        type=int,
        default=10,
        help="Drop flows with fewer than this many packets before padding.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=0,
        help=(
            "Create extra samples from long flows using sliding windows. "
            "0 keeps only the first window per flow."
        ),
    )
    parser.add_argument(
        "--max-windows-per-flow",
        type=int,
        default=0,
        help="Limit sliding-window samples per flow. 0 means no limit.",
    )
    parser.add_argument(
        "--signed-lengths",
        action="store_true",
        help=(
            "Encode packet direction by signed lengths. The first observed endpoint "
            "of a flow is positive, the opposite direction is negative."
        ),
    )
    parser.add_argument(
        "--keep-duplicate-features",
        action="store_true",
        help="Keep identical feature rows. By default they are removed per PCAP.",
    )
    parser.add_argument(
        "--directional",
        action="store_true",
        help="Use directional 5-tuples. Default is bidirectional flow aggregation.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optionally cap rows per class in the final merged CSV. 0 means no cap.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all per-PCAP CSV files, ignoring resumable outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the file-to-class mapping without building CSV files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-PCAP tqdm packet progress bars.",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="Stop reading each PCAP after N packets. 0 means read everything.",
    )
    args = parser.parse_args()

    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")
    if args.min_packets <= 0:
        raise ValueError("--min-packets must be positive.")
    if args.window_stride < 0:
        raise ValueError("--window-stride must be >= 0.")
    if args.max_windows_per_flow < 0:
        raise ValueError("--max-windows-per-flow must be >= 0.")
    if args.max_per_class < 0:
        raise ValueError("--max-per-class must be >= 0.")

    pcaps = iter_nonvpn_pcaps(args.raw_dir, args.subset)
    if not pcaps:
        raise FileNotFoundError(f"No NonVPN PCAP files found under {args.raw_dir}")

    class_rules = get_class_rules(args.scheme)
    by_label: dict[str, list[Path]] = {label: [] for label, _ in class_rules}
    skipped: list[Path] = []

    for pcap in pcaps:
        label = infer_label(pcap, class_rules)
        if label is None:
            skipped.append(pcap)
        else:
            by_label[label].append(pcap)

    print(f"{args.scheme} NonVPN mapping:")
    for label, _ in class_rules:
        files = by_label[label]
        size_mb = sum(p.stat().st_size for p in files) / 1024 / 1024
        print(f"  {label}: {len(files)} file(s), {size_mb:.1f} MB")

    if skipped:
        print(f"Skipped files without a safe {args.scheme} label:")
        for p in skipped:
            print(f"  {p}")

    if args.dry_run:
        return

    part_csvs: list[tuple[Path, str]] = []
    reused = 0
    rebuilt = 0
    for label, files in by_label.items():
        if not files:
            print(f"[WARN] No files for class {label}")
            continue

        for pcap in files:
            part_csv = args.work_dir / f"seq{args.seq_len}" / label / f"{pcap.stem}.csv"
            if not args.force and is_valid_part_csv(part_csv, label, args.seq_len):
                print(f"Reuse valid CSV: {part_csv}")
                reused += 1
            else:
                if part_csv.exists() and not args.force:
                    print(f"Rebuild invalid or incomplete CSV: {part_csv}")
                print(f"Build {label}: {pcap} -> {part_csv}")
                pcap_to_csv(
                    pcap,
                    part_csv,
                    label,
                    seq_len=args.seq_len,
                    min_packets=args.min_packets,
                    drop_duplicate_features=not args.keep_duplicate_features,
                    bidirectional=not args.directional,
                    window_stride=args.window_stride,
                    max_windows_per_flow=args.max_windows_per_flow,
                    signed_lengths=args.signed_lengths,
                    show_progress=not args.no_progress,
                    max_packets=args.max_packets,
                )
                rebuilt += 1
            part_csvs.append((part_csv, label))

    if not part_csvs:
        raise ValueError("No class CSV files were generated.")

    print(f"Per-PCAP CSV summary: reused={reused}, rebuilt={rebuilt}")

    frames = []
    for p, label in part_csvs:
        if not p.exists():
            print(f"[WARN] Missing expected CSV after build: {p}")
            continue
        frames.append(read_csv_rows(p, label, args.seq_len))
    non_empty_frames = [df for df in frames if not df.empty]
    if not non_empty_frames:
        raise ValueError("All generated CSV files are empty. Try lowering --min-packets.")
    merged = pd.concat(non_empty_frames, ignore_index=True)

    if args.max_per_class > 0:
        before = merged["label"].value_counts().sort_index()
        sampled_frames = []
        for _, group in merged.groupby("label", sort=False):
            sampled_frames.append(
                group.sample(n=min(len(group), args.max_per_class), random_state=42)
            )
        merged = pd.concat(sampled_frames, ignore_index=True)
        after = merged["label"].value_counts().sort_index()
        print("Applied class cap:")
        for label in sorted(before.index):
            print(f"  {label}: {int(before[label])} -> {int(after.get(label, 0))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"Saved merged 7-class CSV: {args.out}")
    print("Final class distribution:")
    for label, count in merged["label"].value_counts().sort_index().items():
        print(f"  {label}: {int(count)}")


if __name__ == "__main__":
    main()
