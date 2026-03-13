from typing import List

import numpy as np


class PacketSequenceBuffer:
    def __init__(self, seq_len: int, max_packet_len: int = 1500) -> None:
        self.seq_len = seq_len
        self.max_packet_len = max_packet_len
        self.lengths: List[int] = []

    def add_packet_len(self, packet_len: int) -> None:
        if packet_len <= 0:
            return
        if len(self.lengths) < self.seq_len:
            self.lengths.append(packet_len)

    def is_ready(self) -> bool:
        return len(self.lengths) >= self.seq_len

    def to_normalized_vector(self) -> np.ndarray:
        arr = np.zeros(self.seq_len, dtype=np.float32)
        if self.lengths:
            trimmed = self.lengths[: self.seq_len]
            arr[: len(trimmed)] = np.array(trimmed, dtype=np.float32)
        arr = np.clip(arr / float(self.max_packet_len), 0.0, 1.0)
        return arr
