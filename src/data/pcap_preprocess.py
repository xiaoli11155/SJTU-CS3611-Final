import argparse
from pathlib import Path
import sys
from typing import Callable

import pandas as pd
from scapy.all import IP, TCP, PcapReader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from src.config import MAX_PACKET_LEN, SEQ_LEN


def flow_key(pkt, bidirectional: bool = True):
    ip = pkt[IP]
    tcp = pkt[TCP]
    if not bidirectional:
        return (ip.src, ip.dst, tcp.sport, tcp.dport, "TCP")

    left = (ip.src, tcp.sport)
    right = (ip.dst, tcp.dport)
    first, second = sorted([left, right])
    return (first[0], second[0], first[1], second[1], "TCP")


def packet_endpoint(pkt) -> tuple[str, int]:
    ip = pkt[IP]
    tcp = pkt[TCP]
    return (ip.src, int(tcp.sport))


def _reader_position(reader) -> int:
    handle = getattr(reader, "f", None)
    if handle is None:
        handle = getattr(reader, "fp", None)
    tell = getattr(handle, "tell", None)
    if tell is None:
        return 0
    try:
        return int(tell())
    except Exception:
        return 0


def pcap_to_csv(
    pcap_path: Path,
    output_csv: Path,
    label: str,
    seq_len: int = SEQ_LEN,
    max_packet_len: int = MAX_PACKET_LEN,
    min_packets: int = 1,
    drop_duplicate_features: bool = False,
    bidirectional: bool = True,
    window_stride: int = 0,
    max_windows_per_flow: int = 0,
    signed_lengths: bool = False,
    show_progress: bool = True,
    progress_cb: Callable[[int], None] | None = None,
    max_packets: int = 0,
) -> None:
    flows = {}
    flow_forward_endpoint = {}
    fallback_progress_step = 200000

    with PcapReader(str(pcap_path)) as reader:
        progress_bar = None
        last_pos = 0
        if show_progress and tqdm is not None:
            total_bytes = pcap_path.stat().st_size
            progress_bar = tqdm(
                total=total_bytes,
                desc=pcap_path.name,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stdout,
                leave=True,
                disable=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
            last_pos = _reader_position(reader)
            if last_pos > 0:
                progress_bar.update(min(last_pos, total_bytes))

        idx = 0
        for idx, pkt in enumerate(reader, start=1):
            if max_packets > 0 and idx > max_packets:
                print(f"Reached --max-packets={max_packets}, stopping early.")
                break
            if IP in pkt and TCP in pkt:
                key = flow_key(pkt, bidirectional=bidirectional)
                pkt_len = len(pkt)
                if signed_lengths:
                    if key not in flow_forward_endpoint:
                        flow_forward_endpoint[key] = packet_endpoint(pkt)
                    sign = 1 if packet_endpoint(pkt) == flow_forward_endpoint[key] else -1
                    pkt_len *= sign
                flows.setdefault(key, []).append(pkt_len)
            if progress_cb is not None and idx % 20000 == 0:
                progress_cb(idx)
            elif show_progress and tqdm is None and idx % fallback_progress_step == 0:
                print(f"{pcap_path.name}: processed {idx} packets")
            if progress_bar is not None:
                current_pos = _reader_position(reader)
                if current_pos > last_pos:
                    progress_bar.update(current_pos - last_pos)
                    last_pos = current_pos

        if progress_bar is not None:
            if last_pos < progress_bar.total:
                progress_bar.update(progress_bar.total - last_pos)
            progress_bar.close()

    if progress_cb is not None:
        progress_cb(idx)

    rows = []
    for _, lengths in flows.items():
        if len(lengths) < min_packets:
            continue

        starts = [0]
        if window_stride > 0 and len(lengths) > seq_len:
            starts = list(range(0, len(lengths) - seq_len + 1, window_stride))
            if max_windows_per_flow > 0:
                starts = starts[:max_windows_per_flow]

        for start in starts:
            window = lengths[start : start + seq_len]
            vec = [0] * seq_len
            for i, v in enumerate(window):
                if signed_lengths:
                    clipped = max(-max_packet_len, min(v, max_packet_len))
                else:
                    clipped = min(v, max_packet_len)
                vec[i] = clipped / float(max_packet_len)
            row = {f"f{i}": vec[i] for i in range(seq_len)}
            row["label"] = label
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"f{i}" for i in range(seq_len)]
    df = pd.DataFrame(rows, columns=feature_cols + ["label"])
    if drop_duplicate_features and not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
        print(f"Dropped {before - len(df)} duplicate feature rows from {pcap_path}")
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} flows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", required=True, type=str)
    parser.add_argument("--seq-len", default=SEQ_LEN, type=int)
    parser.add_argument("--max-packet-len", default=MAX_PACKET_LEN, type=int)
    parser.add_argument("--min-packets", default=1, type=int)
    parser.add_argument("--window-stride", default=0, type=int)
    parser.add_argument("--max-windows-per-flow", default=0, type=int)
    parser.add_argument("--signed-lengths", action="store_true")
    parser.add_argument("--drop-duplicate-features", action="store_true")
    parser.add_argument("--max-packets", default=0, type=int, help="Stop reading after N packets (0 = unlimited).")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--directional",
        action="store_true",
        help="Use directional 5-tuples. Default is bidirectional flow aggregation.",
    )
    args = parser.parse_args()

    pcap_to_csv(
        args.pcap,
        args.out,
        args.label,
        seq_len=args.seq_len,
        max_packet_len=args.max_packet_len,
        min_packets=args.min_packets,
        drop_duplicate_features=args.drop_duplicate_features,
        bidirectional=not args.directional,
        window_stride=args.window_stride,
        max_windows_per_flow=args.max_windows_per_flow,
        signed_lengths=args.signed_lengths,
        show_progress=not args.no_progress,
        max_packets=args.max_packets,
    )


if __name__ == "__main__":
    main()
