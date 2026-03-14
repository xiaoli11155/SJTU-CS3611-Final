import json
from datetime import datetime
from pathlib import Path


def reset_prediction_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite existing file so each proxy run starts with a fresh log.
    log_path.write_text("", encoding="utf-8")


def append_prediction(log_path: Path, flow_id: str, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.utcnow().isoformat(),
        "flow_id": flow_id,
        "label": label,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
