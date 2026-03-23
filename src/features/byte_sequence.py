from typing import Optional

import numpy as np


class ByteSequenceBuffer:
    """Collect a fixed-length byte vector from relayed stream bytes."""

    def __init__(self, feature_len: int = 1500) -> None:
        self.feature_len = feature_len
        self.buf = bytearray()

    def add_bytes(self, data: bytes) -> None:
        if not data or self.is_ready():
            return
        need = self.feature_len - len(self.buf)
        self.buf.extend(data[:need])

    def is_ready(self) -> bool:
        return len(self.buf) >= self.feature_len

    def to_normalized_vector(self) -> Optional[np.ndarray]:
        if not self.buf:
            return None
        arr = np.frombuffer(bytes(self.buf), dtype=np.uint8).astype(np.float32)
        if len(arr) < self.feature_len:
            arr = np.pad(arr, (0, self.feature_len - len(arr)), constant_values=0)
        arr = arr[: self.feature_len] / 255.0
        return arr

