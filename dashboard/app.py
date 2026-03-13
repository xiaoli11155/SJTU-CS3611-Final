import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MONITOR_LOG_PATH


st.set_page_config(page_title="Traffic Monitor", layout="wide")
st.title("Encrypted Traffic Real-time Monitor")


def load_recent_predictions(path: Path, seconds: int = 300):
    if not path.exists():
        return []

    cutoff = datetime.utcnow() - timedelta(seconds=seconds)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ts = datetime.fromisoformat(obj["time"])
            if ts >= cutoff:
                rows.append(obj)
    return rows


window_sec = st.slider("Statistics window (seconds)", min_value=30, max_value=1800, value=300, step=30)
rows = load_recent_predictions(MONITOR_LOG_PATH, window_sec)

if not rows:
    st.info("No inference data yet. Start proxy and generate traffic first.")
else:
    labels = [r["label"] for r in rows]
    counter = Counter(labels)

    col1, col2 = st.columns([2, 1])
    with col1:
        pie_df = pd.DataFrame({"label": list(counter.keys()), "count": list(counter.values())})
        st.subheader("Application Distribution")
        st.pyplot(pie_df.set_index("label").plot.pie(y="count", autopct="%1.1f%%", legend=False).figure)

    with col2:
        st.subheader("Latest Records")
        st.dataframe(pd.DataFrame(rows).tail(20), use_container_width=True)

st.caption(f"Prediction log: {MONITOR_LOG_PATH}")
