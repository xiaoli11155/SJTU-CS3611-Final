import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
from scapy.all import IP, TCP, rdpcap

from src.config import MAX_PACKET_LEN, SEQ_LEN


def flow_key(pkt):
    ip = pkt[IP]
    tcp = pkt[TCP]
    return (ip.src, ip.dst, tcp.sport, tcp.dport, "TCP")


def pcap_to_csv(
    pcap_path: Path,
    output_csv: Path,
    label: str,
    seq_len: int = SEQ_LEN,
    max_packet_len: int = MAX_PACKET_LEN,
    progress_cb: Callable[[int], None] | None = None,
) -> None:
    packets = rdpcap(str(pcap_path))
    flows = {}

    for idx, pkt in enumerate(packets, start=1):
        if IP in pkt and TCP in pkt:
            key = flow_key(pkt)
            flows.setdefault(key, []).append(len(pkt))
        if progress_cb is not None and idx % 20000 == 0:
            progress_cb(idx)
    if progress_cb is not None:
        progress_cb(len(packets))

    rows = []
    for _, lengths in flows.items():
        vec = [0] * seq_len
        for i, v in enumerate(lengths[:seq_len]):
            vec[i] = min(v, max_packet_len) / float(max_packet_len)
        row = {f"f{i}": vec[i] for i in range(seq_len)}
        row["label"] = label
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} flows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", required=True, type=str)
    parser.add_argument("--seq-len", default=SEQ_LEN, type=int)
    parser.add_argument("--max-packet-len", default=MAX_PACKET_LEN, type=int)
    args = parser.parse_args()

    pcap_to_csv(
        args.pcap,
        args.out,
        args.label,
        seq_len=args.seq_len,
        max_packet_len=args.max_packet_len,
    )


if __name__ == "__main__":
    main()
