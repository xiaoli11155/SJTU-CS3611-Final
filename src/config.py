from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SEQ_LEN = 32
MAX_PACKET_LEN = 1500
NUM_CLASSES = 5
SIGNED_LENGTHS = True

MODEL_DIR = BASE_DIR / "artifacts"
MODEL_PATH = MODEL_DIR / "cnn1d.pth"
LABEL_PATH = MODEL_DIR / "labels.json"
MONITOR_LOG_PATH = BASE_DIR / "artifacts" / "predictions.jsonl"

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8080
