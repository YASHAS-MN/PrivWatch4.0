"""Unit and integration tests for Sterile Room in-memory video decoding pipeline."""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import av
import cv2
import numpy as np

# Ensure package is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privwatch.config import DEVICE
from privwatch.inference import (
    decode_frames_from_bytes,
    load_model,
    run_inference_detailed,
    run_inference_on_bytes,
)
from sterile_room.sterile_session import SterileSession
from sterile_room.streamlit_integration import StreamlitSterileAdapter


def create_test_mp4_bytes(num_frames: int = 32, width: int = 112, height: int = 112, fps: int = 25) -> bytes:
    """Generate a valid in-memory MP4 byte stream without touching the filesystem."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        arr = np.full((height, width, 3), (i * 7) % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


class TestSterileInMemory(unittest.TestCase):
    """Test suite validating in-memory PyAV decoding and zero-disk persistence."""

    @classmethod
    def setUpClass(cls):
        cls.sample_bytes = create_test_mp4_bytes(num_frames=32)
        cls.model, _ = load_model(device=DEVICE)

    def test_in_memory_decoding_yields_bgr_frames(self):
        """Verify PyAV decodes video bytes in memory to numpy uint8 BGR frames."""
        frames = []
        for frame in decode_frames_from_bytes(self.sample_bytes):
            frames.append(frame)
            if len(frames) >= 10:
                break

        self.assertEqual(len(frames), 10)
        self.assertIsInstance(frames[0], np.ndarray)
        self.assertEqual(frames[0].dtype, np.uint8)
        self.assertEqual(len(frames[0].shape), 3)
        self.assertEqual(frames[0].shape[2], 3)  # BGR channels

    def test_prediction_fidelity_pyav_vs_opencv(self):
        """Verify in-memory decoding produces identical predictions and confidences to OpenCV."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(self.sample_bytes)
            tmp.close()

            # 1. OpenCV from disk
            cap = cv2.VideoCapture(tmp.name)
            cv2_results = run_inference_detailed(cap, model=self.model, device=DEVICE)
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

        # 2. PyAV from memory
        mem_results = run_inference_on_bytes(self.sample_bytes, model=self.model, device=DEVICE)

        self.assertEqual(mem_results["decision"], cv2_results["decision"])
        self.assertEqual(mem_results["predicted_class"], cv2_results["predicted_class"])
        self.assertEqual(mem_results["total_frames"], cv2_results["total_frames"])
        self.assertEqual(mem_results["windows_processed"], cv2_results["windows_processed"])

        for cls_name in ("Normal", "Fight", "Collapse"):
            self.assertAlmostEqual(
                mem_results["confidences"][cls_name],
                cv2_results["confidences"][cls_name],
                places=4,
            )

    def test_zero_disk_writes_in_sterile_mode(self):
        """Verify no temporary video files are created during Sterile in-memory processing."""
        temp_dir = Path(tempfile.gettempdir())
        before_temp_files = set(temp_dir.glob("*.mp4")) | set(temp_dir.glob("*.tmp"))
        before_proj_files = set(PROJECT_ROOT.rglob("*.mp4"))

        # Run sterile processing
        results = StreamlitSterileAdapter.process_uploaded_file(
            self.sample_bytes, model=self.model, device=DEVICE
        )

        after_temp_files = set(temp_dir.glob("*.mp4")) | set(temp_dir.glob("*.tmp"))
        after_proj_files = set(PROJECT_ROOT.rglob("*.mp4"))

        new_temp_mp4 = after_temp_files - before_temp_files
        new_proj_mp4 = after_proj_files - before_proj_files

        self.assertEqual(len(new_temp_mp4), 0, f"Unexpected temp files created: {new_temp_mp4}")
        self.assertEqual(len(new_proj_mp4), 0, f"Unexpected project files created: {new_proj_mp4}")
        self.assertIn("decision", results)

    def test_empty_bytes_raises_value_error(self):
        """Verify empty byte input raises ValueError immediately."""
        with self.assertRaises(ValueError) as ctx:
            list(decode_frames_from_bytes(b""))
        self.assertIn("empty", str(ctx.exception).lower())

    def test_corrupt_bytes_raises_value_error(self):
        """Verify invalid or corrupted byte payload raises ValueError."""
        # 1. Container without video stream
        stream_without_video = b"RIFF\x00\x00\x00\x00AVI LISTTHISISNOTVALIDVIDEO"
        with self.assertRaises(ValueError) as ctx1:
            list(decode_frames_from_bytes(stream_without_video))
        self.assertTrue(
            any(m in str(ctx1.exception).lower() for m in ["no video stream", "failed to open"])
        )

        # 2. Total garbage data
        garbage = b"\x00\x01\x02\x03\x04\x05GARBAGEPAYLOAD12345"
        with self.assertRaises(ValueError) as ctx2:
            list(decode_frames_from_bytes(garbage))
        self.assertIn("failed to open video container", str(ctx2.exception).lower())

    def test_sterile_session_lifecycle(self):
        """Verify SterileSession enforces lifecycle and buffer teardown."""
        session = SterileSession(session_id="lifecycle_test", model=self.model, device=DEVICE)
        self.assertTrue(session.is_active)

        session.close()
        self.assertFalse(session.is_active)
        self.assertEqual(len(session.buffer), 0)

        with self.assertRaises(RuntimeError):
            session.process_video_bytes(self.sample_bytes)


if __name__ == "__main__":
    unittest.main()
