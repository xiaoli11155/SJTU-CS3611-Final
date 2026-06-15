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


def get_model_dir(tag: str | None = None) -> Path:
    """Return artifact directory for an experiment tag.

    When tag is None or empty, returns MODEL_DIR (backward compatible).
    Otherwise returns MODEL_DIR / tag.
    """
    if not tag:
        return MODEL_DIR
    return MODEL_DIR / tag


def get_model_path(tag: str | None = None) -> Path:
    """Return model checkpoint path for a given tag."""
    return get_model_dir(tag) / "cnn1d.pth"


def get_label_path(tag: str | None = None) -> Path:
    """Return label JSON path for a given tag."""
    return get_model_dir(tag) / "labels.json"


def get_monitor_log_path(tag: str | None = None) -> Path:
    """Return prediction log path for a given tag."""
    return get_model_dir(tag) / "predictions.jsonl"


PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8080
