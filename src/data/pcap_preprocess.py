import argparse
from pathlib import Path

import pandas as pd
from scapy.all import IP, TCP, rdpcap

from src.config import MAX_PACKET_LEN, SEQ_LEN


def flow_key(pkt):
    ip = pkt[IP]
    tcp = pkt[TCP]
    return (ip.src, ip.dst, tcp.sport, tcp.dport, "TCP")


def pcap_to_csv(pcap_path: Path, output_csv: Path, label: str) -> None:
    packets = rdpcap(str(pcap_path))
    flows = {}

    for pkt in packets:
        if IP in pkt and TCP in pkt:
            key = flow_key(pkt)
            flows.setdefault(key, []).append(len(pkt))

    rows = []
    for _, lengths in flows.items():
        vec = [0] * SEQ_LEN
        for i, v in enumerate(lengths[:SEQ_LEN]):
            vec[i] = min(v, MAX_PACKET_LEN) / float(MAX_PACKET_LEN)
        row = {f"f{i}": vec[i] for i in range(SEQ_LEN)}
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
    args = parser.parse_args()

    pcap_to_csv(args.pcap, args.out, args.label)


if __name__ == "__main__":
    main()
