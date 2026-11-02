"""Unit tests for video frame preprocessing and tensor transformation."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

# Ensure package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privwatch.config import (
    IMG_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    SEQ_LEN,
)
from privwatch.preprocessing import (
    preprocess_clip,
    resize_frame,
    stream_sliding_windows,
)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing steps and tensor shapes."""

    def test_resize_frame(self):
        """Verify frame is resized to (112, 112, 3)."""
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_frame(raw_frame)
        self.assertEqual(resized.shape, (IMG_SIZE, IMG_SIZE, 3))

    def test_resize_invalid_frame_raises(self):
        """Verify error raised on empty frame."""
        with self.assertRaises(ValueError):
            resize_frame(None)
        with self.assertRaises(ValueError):
            resize_frame(np.array([]))

    def test_preprocess_clip_shape_and_dtype(self):
        """Verify 16-frame list converts to (1, 16, 3, 112, 112) normalized float tensor."""
        frames = [np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8) for _ in range(SEQ_LEN)]
        tensor = preprocess_clip(frames, device=torch.device("cpu"))

        self.assertEqual(tensor.shape, (1, SEQ_LEN, 3, IMG_SIZE, IMG_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_preprocess_clip_normalization(self):
        """Verify mathematical normalization aligns with (val / 255 - mean) / std."""
        pixel_val = 200
        frames = [np.full((IMG_SIZE, IMG_SIZE, 3), pixel_val, dtype=np.uint8) for _ in range(SEQ_LEN)]
        tensor = preprocess_clip(frames, device=torch.device("cpu"))

        expected_c0 = ((pixel_val / 255.0) - NORMALIZATION_MEAN[0]) / NORMALIZATION_STD[0]
        actual_c0 = tensor[0, 0, 0, 0, 0].item()
        self.assertAlmostEqual(actual_c0, expected_c0, places=4)

    def test_preprocess_clip_invalid_length(self):
        """Verify ValueError raised when sequence length does not match SEQ_LEN."""
        short_frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) for _ in range(10)]
        with self.assertRaises(ValueError):
            preprocess_clip(short_frames)

    def test_stream_sliding_windows(self):
        """Verify stream yields sliding windows correctly."""
        total_frames = 20
        stream = (np.full((100, 100, 3), i, dtype=np.uint8) for i in range(total_frames))
        windows = list(stream_sliding_windows(stream, seq_len=16, img_size=112))

        # 20 frames with window size 16 produces 20 - 16 + 1 = 5 sliding windows
        self.assertEqual(len(windows), 5)
        self.assertEqual(len(windows[0]), 16)
        self.assertEqual(windows[0][0].shape, (112, 112, 3))


if __name__ == "__main__":
    unittest.main()

