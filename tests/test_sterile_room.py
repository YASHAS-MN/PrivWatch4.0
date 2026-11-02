"""Unit tests for Sterile Room component interfaces and data structures."""

from pathlib import Path
import sys
import unittest

import numpy as np

# Ensure package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sterile_room.alert_registry import AlertEvent, AlertRegistry
from sterile_room.inference_worker import MsgType, SterileInferenceWorker
from sterile_room.memory_sanitizer import MemorySanitizer
from sterile_room.sterile_session import SterileSession
from sterile_room.volatile_buffer import VolatileBuffer


class TestSterileRoom(unittest.TestCase):
    """Test volatile buffer, memory sanitizer, and alert registry."""

    def test_volatile_buffer_capacity(self):
        """Verify volatile buffer enforces bounded capacity."""
        buf = VolatileBuffer(capacity=16)
        self.assertFalse(buf.is_ready())

        for i in range(16):
            buf.push(np.full((10, 10, 3), i, dtype=np.uint8))

        self.assertTrue(buf.is_ready())
        self.assertEqual(len(buf), 16)

        # Push one more frame (should evict first frame, len stays 16)
        buf.push(np.full((10, 10, 3), 99, dtype=np.uint8))
        self.assertEqual(len(buf), 16)
        window = buf.get_window()
        self.assertEqual(window[-1][0, 0, 0], 99)
        self.assertEqual(window[0][0, 0, 0], 1)  # Frame 0 evicted

        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertFalse(buf.is_ready())

    def test_memory_sanitizer_wipe_array(self):
        """Verify wipe_array overwrites array values with zeros."""
        arr = np.ones((5, 5), dtype=np.float32)
        MemorySanitizer.wipe_array(arr)
        self.assertTrue(np.all(arr == 0))

    def test_memory_sanitizer_wipe_buffer(self):
        """Verify wipe_buffer zeroes out multiple arrays and clears list."""
        arr1 = np.ones((4, 4), dtype=np.uint8)
        arr2 = np.ones((4, 4), dtype=np.uint8)
        buf = [arr1, arr2]

        MemorySanitizer.wipe_buffer(buf)
        self.assertEqual(len(buf), 0)
        self.assertTrue(np.all(arr1 == 0))
        self.assertTrue(np.all(arr2 == 0))

    def test_memory_sanitizer_purge_caches(self):
        """Verify purge_runtime_caches runs without crashing."""
        MemorySanitizer.purge_runtime_caches()

    def test_alert_registry(self):
        """Verify AlertRegistry records metadata without storing images."""
        registry = AlertRegistry()
        alert = AlertEvent.create(
            event_type="Fight",
            confidence=0.88,
            latency_ms=120.5,
            details={"source": "camera_01"},
        )

        self.assertEqual(alert.event_type, "Fight")
        self.assertEqual(alert.confidence, 0.88)
        self.assertTrue(len(alert.event_id) > 10)
        self.assertIn("T", alert.timestamp)  # ISO timestamp format

        registry.record_alert(alert)
        self.assertEqual(len(registry), 1)

        alerts = registry.get_alerts()
        self.assertEqual(alerts[0].event_type, "Fight")

        registry.clear()
        self.assertEqual(len(registry), 0)

    def test_inference_worker_starts_and_fails_gracefully(self):
        """Verify SterileInferenceWorker handles corrupt bytes by putting ERROR + DONE onto queue."""
        # Use a dummy model=None — worker will call load_model internally
        # and fail on corrupt bytes gracefully (puts ERROR message, never crashes)
        worker = SterileInferenceWorker(video_bytes=b"corrupt", model=None)
        worker.start()
        worker.wait()
        msgs = worker.drain_alerts()
        types = [m.msg_type for m in msgs]
        self.assertIn(MsgType.DONE, types)
        # ERROR or RESULT must appear; never hangs
        self.assertTrue(MsgType.ERROR in types or MsgType.RESULT in types)

    def test_sterile_session_invalid_bytes(self):
        """Verify SterileSession validates bytes and raises ValueError on corrupt data."""
        session = SterileSession(session_id="test_session")
        with self.assertRaises(ValueError):
            session.process_video_bytes(b"dummy")


if __name__ == "__main__":
    unittest.main()

