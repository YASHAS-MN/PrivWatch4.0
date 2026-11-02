# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Compressed volatile frame slot for memory-efficient event buffer storage.

Stores one video frame as an in-memory JPEG byte buffer alongside its
presentation timestamp. Provides on-demand decompression back to BGR ndarray.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import cv2  # type: ignore
import numpy as np


@dataclass
class CompressedFrame:
    """Single video frame stored as an in-memory JPEG byte buffer.

    Attributes:
        data: JPEG-encoded bytes, never written to disk.
        timestamp: Presentation timestamp in seconds.
        original_h: Original frame height (for validation on decode).
        original_w: Original frame width.
    """

    data: bytes
    timestamp: float
    original_h: int
    original_w: int

    DEFAULT_QUALITY: ClassVar[int] = 70

    @classmethod
    def from_bgr(
        cls,
        frame: np.ndarray,
        timestamp: float,
        jpeg_quality: int = DEFAULT_QUALITY,
    ) -> "CompressedFrame":
        """Encode a BGR uint8 ndarray as an in-memory JPEG byte buffer.

        Args:
            frame: BGR uint8 NumPy array (H, W, 3).
            timestamp: Presentation timestamp in seconds.
            jpeg_quality: JPEG quality 1–100. Default 70 (187x compression on 1080p).

        Returns:
            CompressedFrame instance. No disk access is performed.

        Raises:
            RuntimeError: If OpenCV JPEG encoding fails.
        """
        h, w = frame.shape[:2]
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for frame at t={timestamp:.3f}s")
        return cls(data=buf.tobytes(), timestamp=timestamp, original_h=h, original_w=w)

    def decode(self) -> np.ndarray:
        """Decompress stored JPEG bytes back to a BGR uint8 NumPy array.

        Returns:
            BGR ndarray of shape (H, W, 3) dtype uint8.

        Raises:
            RuntimeError: If JPEG decoding fails or produces unexpected shape.
        """
        arr = np.frombuffer(self.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(
                f"JPEG decoding failed for frame at t={self.timestamp:.3f}s"
            )
        return frame

    @property
    def size_bytes(self) -> int:
        """Return compressed byte size."""
        return len(self.data)

