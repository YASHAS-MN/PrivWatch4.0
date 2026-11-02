"""Unit tests for sliding-window inference and decision thresholds."""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np
import torch

# Ensure package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privwatch.config import (
    COLLAPSE_THRESHOLD,
    DEVICE,
    FIGHT_THRESHOLD,
    IMG_SIZE,
    SEQ_LEN,
)
from privwatch.inference import load_model, run_inference, run_inference_detailed


class SyntheticVideoCapture:
    """Mock OpenCV VideoCapture that generates a fixed number of synthetic frames."""

    def __init__(self, frame_count: int = 20):
        self.frame_count = frame_count
        self.current_frame = 0
        self.released = False

    def read(self):
        if self.current_frame < self.frame_count:
            self.current_frame += 1
            frame = np.full((120, 160, 3), 100, dtype=np.uint8)
            return True, frame
        return False, None

    def release(self):
        self.released = True

    def isOpened(self):
        return not self.released


class TestInference(unittest.TestCase):
    """Test inference runner, decision rules, and output formats."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = load_model(device=DEVICE)

    def test_run_inference_synthetic_stream(self):
        """Verify inference runs across a synthetic video stream without error."""
        cap = SyntheticVideoCapture(frame_count=20)
        result = run_inference(cap, model=self.model, device=DEVICE, verbose=False)

        self.assertIn(
            result,
            ["ALERT - Fight detected", "ALERT - Collapse detected", "Normal activity"],
        )
        self.assertTrue(cap.released)

    def test_run_inference_detailed(self):
        """Verify run_inference_detailed returns full metrics dictionary."""
        cap = SyntheticVideoCapture(frame_count=20)
        details = run_inference_detailed(cap, model=self.model, device=DEVICE)

        self.assertIn("decision", details)
        self.assertIn("predicted_class", details)
        self.assertIn("confidences", details)
        self.assertIn("windows_processed", details)
        self.assertEqual(details["windows_processed"], 5)  # 20 - 16 + 1
        self.assertIn("latency_seconds", details)
        self.assertTrue(cap.released)

    def test_empty_capture_returns_no_frames(self):
        """Verify capture with zero frames returns 'No frames processed'."""
        cap = SyntheticVideoCapture(frame_count=0)
        result = run_inference(cap, model=self.model, device=DEVICE, verbose=False)
        self.assertEqual(result, "No frames processed")

    def test_sample_video_inference(self):
        """Test inference on a real sample video from the outer workspace if available."""
        sample_path = Path("t/n/a (1).mp4")
        if not sample_path.exists():
            sample_path = Path("../t/n/a (1).mp4")

        if sample_path.exists():
            cap = cv2.VideoCapture(str(sample_path))
            self.assertTrue(cap.isOpened(), "Could not open sample video")
            result = run_inference(cap, model=self.model, device=DEVICE, verbose=False)
            self.assertIn(
                result,
                ["Normal activity", "ALERT - Fight detected", "ALERT - Collapse detected"],
            )


if __name__ == "__main__":
    unittest.main()

