# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Frame preprocessing and temporal clip generation for PrivWatch."""

from collections.abc import Iterable, Iterator
from pathlib import Path
import sys

_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import cv2  # type: ignore
import numpy as np
import torch

try:
    from privwatch.config import (
        DEVICE,
        IMG_SIZE,
        NORMALIZATION_MEAN,
        NORMALIZATION_STD,
        SEQ_LEN,
    )
except ImportError:
    from PrivWatch_Final.privwatch.config import (  # type: ignore
        DEVICE,
        IMG_SIZE,
        NORMALIZATION_MEAN,
        NORMALIZATION_STD,
        SEQ_LEN,
    )


def resize_frame(frame: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize a video frame to the spatial dimensions expected by the model.

    Args:
        frame: Input frame as a NumPy array (H, W, C).
        size: Target width and height (default: IMG_SIZE=112).

    Returns:
        Resized frame of shape (size, size, C).
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot resize empty frame")
    return cv2.resize(frame, (size, size))


def preprocess_clip(
    frames: list[np.ndarray] | np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a sequence of 16 frames into a normalized model input tensor.

    Pipeline:
        1. Stack frames into array of shape (SEQ_LEN, H, W, C)
        2. Permute to (SEQ_LEN, C, H, W)
        3. Convert float and scale to [0.0, 1.0]
        4. Normalize using ImageNet mean & std
        5. Add batch dimension -> (1, SEQ_LEN, C, H, W)

    Args:
        frames: Sequence of 16 pre-resized (IMG_SIZE, IMG_SIZE, 3) frames.
        device: Target torch device (defaults to config.DEVICE).

    Returns:
        Tensor of shape (1, 16, 3, 112, 112) on the target device.
    """
    target_device = device or DEVICE

    if isinstance(frames, list):
        if len(frames) != SEQ_LEN:
            raise ValueError(f"Expected sequence of {SEQ_LEN} frames, got {len(frames)}")
        clip_array = np.array(frames)
    else:
        if frames.shape[0] != SEQ_LEN:
            raise ValueError(f"Expected sequence length {SEQ_LEN}, got {frames.shape[0]}")
        clip_array = frames

    clip = torch.from_numpy(clip_array).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(NORMALIZATION_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZATION_STD, dtype=torch.float32).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    return clip.unsqueeze(0).to(target_device)


def stream_sliding_windows(
    frame_stream: Iterable[np.ndarray],
    seq_len: int = SEQ_LEN,
    img_size: int = IMG_SIZE,
) -> Iterator[list[np.ndarray]]:
    """Yield sliding windows of frames from a continuous frame stream.

    Args:
        frame_stream: Iterable providing raw video frames.
        seq_len: Number of frames per window (default: 16).
        img_size: Resize dimension (default: 112).

    Yields:
        list of `seq_len` resized frames.
    """
    buffer: list[np.ndarray] = []
    for frame in frame_stream:
        if frame is None:
            continue
        resized = resize_frame(frame, img_size)
        buffer.append(resized)
        if len(buffer) == seq_len:
            yield list(buffer)
            buffer.pop(0)

