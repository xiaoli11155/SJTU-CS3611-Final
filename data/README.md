# Public Dataset Guide

This project requires public encrypted-traffic data for offline training.

## Recommended Sources

1. ISCX VPN-nonVPN (official)
- https://www.unb.ca/cic/datasets/vpn.html

2. CIC-IDS2017 (official)
- https://www.unb.ca/cic/datasets/ids-2017.html

Notes:
- Official websites may require manual navigation and acceptance before download.
- Once you get direct file URLs, use the download script below.

## Step 1: Download Raw Files

Example:
```bash
python scripts/download_dataset.py --url "DIRECT_FILE_URL_1" --url "DIRECT_FILE_URL_2"
```

Downloaded files are saved to data/raw.
Zip files are extracted to data/extracted.

## Step 2: Build 3-4 Class Training CSV

Example:

```bash
python scripts/build_training_csv.py --class-pcap "chat=data/raw/NonVPN-PCAPs-01/*chat*.pcap*" --class-pcap "video=data/raw/NonVPN-PCAPs-01/*video*.pcap*" --class-pcap "file=data/raw/NonVPN-PCAPs-01/*email*.pcap*" --class-pcap "web=data/raw/NonVPN-PCAPs-01/*audio*.pcap*" --out data/flows.csv
```

The output data/flows.csv is directly used by training.
