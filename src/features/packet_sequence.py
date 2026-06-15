from typing import List

import numpy as np


class PacketSequenceBuffer:
    def __init__(
        self,
        seq_len: int,
        max_packet_len: int = 1500,
        signed_lengths: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.max_packet_len = max_packet_len
        self.signed_lengths = signed_lengths
        self.lengths: List[int] = []

    def add_packet_len(self, packet_len: int, direction: int = 1) -> None:
        if packet_len == 0:
            return
        if len(self.lengths) < self.seq_len:
            sign = 1 if direction >= 0 else -1
            value = packet_len * sign if self.signed_lengths else packet_len
            self.lengths.append(value)

    def is_ready(self) -> bool:
        return len(self.lengths) >= self.seq_len

    def to_normalized_vector(self) -> np.ndarray:
        arr = np.zeros(self.seq_len, dtype=np.float32)
        if self.lengths:
            trimmed = self.lengths[: self.seq_len]
            arr[: len(trimmed)] = np.array(trimmed, dtype=np.float32)
        if self.signed_lengths:
            arr = np.clip(arr / float(self.max_packet_len), -1.0, 1.0)
        else:
            arr = np.clip(arr / float(self.max_packet_len), 0.0, 1.0)
        return arr
