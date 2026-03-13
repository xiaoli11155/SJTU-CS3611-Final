# Project Structure (Simple Version)

This project implements a minimal end-to-end pipeline for encrypted traffic identification:

1. TCP proxy forwarding with Python socket.
2. Online packet-length sequence extraction.
3. Offline 1D-CNN training with PyTorch.
4. Online inference integrated into proxy.
5. Real-time dashboard with Streamlit.

## Directory Layout

- `src/config.py`: Global config values (sequence length, model path, etc.).
- `src/features/packet_sequence.py`: Packet length feature buffer and normalization.
- `src/model/cnn1d.py`: 1D-CNN network definition.
- `src/model/inference.py`: Online classifier wrapper.
- `src/data/pcap_preprocess.py`: Optional PCAP-to-CSV preprocessing script.
- `src/proxy/tcp_proxy.py`: TCP/HTTP proxy server with feature capture.
- `src/utils/monitor_store.py`: Save online predictions for dashboard.
- `scripts/train.py`: Offline training entry.
- `dashboard/app.py`: Streamlit monitoring dashboard.
- `requirements.txt`: Python dependencies.

## Minimal Workflow

1. Prepare data: Generate CSV features from PCAP using `src/data/pcap_preprocess.py` or your own script.
2. Train model: `python scripts/train.py --csv data/flows.csv`.
3. Start dashboard: `streamlit run dashboard/app.py`.
4. Start proxy: `python -m src.proxy.tcp_proxy`.
5. Configure browser/system proxy to `127.0.0.1:8080` and generate traffic.

## Data Format for Training

CSV columns:

- `f0 ... f(N-1)`: normalized packet length sequence.
- `label`: class name, e.g., `Video`, `Chat`, `FileTransfer`.

The sequence length `N` is controlled in `src/config.py`.
