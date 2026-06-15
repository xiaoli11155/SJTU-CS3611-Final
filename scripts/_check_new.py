import csv
from collections import Counter
from pathlib import Path

for f in ["flows_nonvpn_tcp5class_01.csv", "flows_nonvpn_tcp5class_01_seq64.csv", "flows_nonvpn_tcp4class_01.csv"]:
    path = Path("data/processed/training") / f
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    cnt = Counter(r["label"] for r in rows)
    print(f"=== {f} ===")
    print(f"  total: {len(rows)}")
    for label, n in cnt.most_common():
        print(f"  {label}: {n}")
    print()
