"""Bounded circular RAM buffer for volatile video frame retention.

RESPONSIBILITIES:
- Maintain a strictly bounded FIFO buffer in volatile memory (RAM).
- Prevent memory unbounded growth during continuous video streams.
- Facilitate sliding-window evaporation where overwriting or evicting a frame
  triggers explicit memory clearing rather than waiting solely on GC.

EXECUTION FLOW INTEGRATION:
    Decoded in-memory frame -> VolatileBuffer.push(frame) -> sliding window -> Model

STATUS:
    Architectural interface stub. Production implementation pending.
"""

from collections import deque
from typing import Any
import numpy as np


class VolatileBuffer:
    """Bounded circular buffer residing in volatile memory (RAM).

    Attributes:
        capacity: Maximum number of frames retained at any moment (default: 16).
    """

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self._buffer: deque[np.ndarray] = deque(maxlen=capacity)
        self._total_pushed: int = 0

    def push(self, frame: np.ndarray) -> None:
        """Push a new frame into the volatile circular buffer.

        When capacity is reached, the oldest frame is discarded.
        """
        if frame is None:
            return
        self._buffer.append(frame)
        self._total_pushed += 1

    def is_ready(self) -> bool:
        """Return True if buffer has accumulated sufficient frames for a full sequence window."""
        return len(self._buffer) == self.capacity

    def get_window(self) -> list[np.ndarray]:
        """Return a copy of the current frame window for inference."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all buffered frames."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

