# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Unit and integration tests for Phase 2A: Real-Time Anomaly Detection."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import unittest

import av
import numpy as np

# Ensure package root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privwatch.config import DEVICE
from privwatch.inference import load_model, run_inference_on_bytes
from sterile_room.alert_registry import AlertEvent, AlertRegistry
from sterile_room.event_manager import (
    EventState,
    RealTimeEvent,
    RealTimeEventManager,
    WindowPrediction,
)
from sterile_room.sterile_session import SterileSession


def create_mock_frame(val: int = 128) -> np.ndarray:
    """Generate a dummy BGR frame."""
    return np.full((112, 112, 3), val, dtype=np.uint8)


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


class TestRealTimeAnomalyDetection(unittest.TestCase):
    """Test suite covering Phase 2A real-time anomaly detection, event management, and volatile buffers."""

    def setUp(self):
        self.registry = AlertRegistry()
        self.manager = RealTimeEventManager(
            pre_event_seconds=2.0,  # 2s for fast unit test
            post_event_seconds=2.0,  # 2s for fast unit test
            fps=25.0,
            fight_threshold=0.35,
            collapse_threshold=0.35,
            camera_id="cam-01",
            alert_callback=self.registry.record_alert,
        )

    def test_normal_windows_do_not_trigger_alerts(self):
        """Verify normal probability streams produce no events and zero alerts."""
        for w in range(20):
            t = w * 0.04
            self.manager.push_frame(create_mock_frame(w), timestamp=t)
            pred = self.manager.process_window(
                window_index=w,
                timestamp=t,
                normal_prob=0.90,
                fight_prob=0.05,
                collapse_prob=0.05,
            )
            self.assertIsNone(pred.anomaly_type)

        self.assertEqual(len(self.manager.get_events()), 0)
        self.assertEqual(len(self.registry), 0)
        self.assertEqual(self.manager.state, EventState.IDLE)

    def test_anomaly_detected_before_eof(self):
        """Verify anomaly alert is dispatched immediately during stream processing, not waiting for EOF."""
        alerts_received = []
        self.manager.alert_callback = lambda a: alerts_received.append(a)

        # 5 normal windows
        for w in range(5):
            t = w * 0.04
            self.manager.push_frame(create_mock_frame(), timestamp=t)
            self.manager.process_window(w, t, 0.90, 0.05, 0.05)

        self.assertEqual(len(alerts_received), 0)

        # Anomaly at window 5 (t = 0.20s)
        self.manager.push_frame(create_mock_frame(), timestamp=0.20)
        self.manager.process_window(5, 0.20, normal_prob=0.20, fight_prob=0.75, collapse_prob=0.05)

        # Alert must be present IMMEDIATELY before calling on_eof()
        self.assertEqual(len(alerts_received), 1)
        self.assertEqual(alerts_received[0].event_type, "Fight")
        self.assertEqual(alerts_received[0].confidence, 0.75)
        self.assertEqual(alerts_received[0].details["window_index"], "5")

    def test_fight_threshold_behavior(self):
        """Verify strict Fight threshold at 0.35."""
        # 0.34 -> Not anomalous
        pred1 = self.manager.process_window(0, 0.0, normal_prob=0.60, fight_prob=0.34, collapse_prob=0.06)
        self.assertIsNone(pred1.anomaly_type)
        self.assertEqual(len(self.manager.get_events()), 0)

        # 0.35 -> Anomalous (Fight)
        pred2 = self.manager.process_window(1, 0.04, normal_prob=0.55, fight_prob=0.35, collapse_prob=0.10)
        self.assertEqual(pred2.anomaly_type, "Fight")
        self.assertEqual(len(self.manager.get_events()), 1)

    def test_collapse_threshold_behavior(self):
        """Verify strict Collapse threshold at 0.35 and decision precedence."""
        # 0.34 Collapse -> Not anomalous
        pred1 = self.manager.process_window(0, 0.0, normal_prob=0.60, fight_prob=0.06, collapse_prob=0.34)
        self.assertIsNone(pred1.anomaly_type)

        # 0.35 Collapse -> Anomalous (Collapse)
        pred2 = self.manager.process_window(1, 0.04, normal_prob=0.55, fight_prob=0.10, collapse_prob=0.35)
        self.assertEqual(pred2.anomaly_type, "Collapse")

        # Both >= 0.35 -> Fight takes precedence (matches model decision order)
        manager2 = RealTimeEventManager(fight_threshold=0.35, collapse_threshold=0.35)
        pred3 = manager2.process_window(0, 0.0, normal_prob=0.20, fight_prob=0.40, collapse_prob=0.40)
        self.assertEqual(pred3.anomaly_type, "Fight")

    def test_consecutive_anomalous_windows_merge_into_one_event(self):
        """Verify consecutive anomalous windows merge and do not create duplicate events or alerts."""
        alerts = []
        self.manager.alert_callback = lambda a: alerts.append(a)

        # Feed 10 consecutive anomalous windows
        for w in range(10):
            t = 1.0 + w * 0.04
            self.manager.push_frame(create_mock_frame(), timestamp=t)
            self.manager.process_window(w, t, normal_prob=0.10, fight_prob=0.60 + (w * 0.02), collapse_prob=0.05)

        # Must merge into exactly 1 event and 1 alert
        self.assertEqual(len(self.manager.get_events()), 1)
        self.assertEqual(len(alerts), 1)

        event = self.manager.get_events()[0]
        self.assertEqual(event.event_type, "Fight")
        self.assertEqual(event.state, EventState.ACTIVE)
        self.assertEqual(event.first_detected_window, 0)
        self.assertAlmostEqual(event.last_anomalous_timestamp, 1.0 + 9 * 0.04)
        self.assertAlmostEqual(event.confidence, 0.60 + 9 * 0.02)  # Peak confidence tracked

    def test_event_closes_after_post_event_interval(self):
        """Verify active event transitions to CLOSED and is marked ready after post-event duration."""
        # Post event is set to 2.0s
        self.manager.post_event_seconds = 2.0
        self.manager.fps = 25.0

        # Anomaly at t = 1.0s
        self.manager.push_frame(create_mock_frame(), timestamp=1.0)
        self.manager.process_window(0, 1.0, 0.10, 0.80, 0.10)

        event = self.manager.get_events()[0]
        self.assertEqual(event.state, EventState.ACTIVE)
        self.assertFalse(event.is_ready)

        # Feed 1.5s of normal frames (time_since = 1.5s < 2.0s target)
        for i in range(1, 38):
            t = 1.0 + i * 0.04
            self.manager.push_frame(create_mock_frame(), timestamp=t)
            self.manager.process_window(i, t, 0.90, 0.05, 0.05)

        self.assertEqual(event.state, EventState.ACTIVE)

        # Feed remaining frames up to t = 3.2s (> 2.0s post-event)
        for i in range(38, 55):
            t = 1.0 + i * 0.04
            self.manager.push_frame(create_mock_frame(), timestamp=t)
            self.manager.process_window(i, t, 0.90, 0.05, 0.05)

        # Event must be CLOSED and is_ready True
        self.assertEqual(event.state, EventState.CLOSED)
        self.assertTrue(event.is_ready)
        self.assertEqual(self.manager.state, EventState.IDLE)

    def test_pre_event_frames_are_retained(self):
        """Verify pre-event time-based buffer retains frames within the configured window duration."""
        # manager has pre_event_seconds=2.0; push 80 frames at 25fps (dt=0.04s)
        # Frames span t=0..3.16s; only frames in [1.16s, 3.16s] stay => ~50 frames
        for f in range(80):
            self.manager.push_frame(create_mock_frame(val=f % 255), timestamp=f * 0.04)

        # TimedPreBuffer evicts by PTS; ~50 frames should remain (allow 48-52)
        retained = len(self.manager.pre_buffer)
        self.assertGreaterEqual(retained, 48)
        self.assertLessEqual(retained, 52)

        # Trigger anomaly to snapshot pre-event frames into event
        self.manager.process_window(0, 80 * 0.04, 0.10, 0.80, 0.10)

        event = self.manager.get_events()[0]
        # Pre-event frames are CompressedFrame objects; count must match buffer size
        self.assertGreaterEqual(len(event.pre_event_frames), 48)
        self.assertLessEqual(len(event.pre_event_frames), 52)

    def test_no_video_files_written_to_disk(self):
        """Verify end-to-end SterileSession with RealTimeEventManager creates zero disk files."""
        model, _ = load_model(device=DEVICE)
        session = SterileSession(model=model, device=DEVICE)

        test_bytes = create_test_mp4_bytes(num_frames=32)

        temp_dir = Path(tempfile.gettempdir())
        before_temp = set(temp_dir.glob("*.mp4")) | set(temp_dir.glob("*.tmp"))
        before_proj = set(PROJECT_ROOT.rglob("*.mp4"))

        results = session.process_video_bytes(test_bytes)
        session.close()

        after_temp = set(temp_dir.glob("*.mp4")) | set(temp_dir.glob("*.tmp"))
        after_proj = set(PROJECT_ROOT.rglob("*.mp4"))

        self.assertEqual(len(after_temp - before_temp), 0)
        self.assertEqual(len(after_proj - before_proj), 0)
        self.assertIn("events", results)
        self.assertIn("window_predictions", results)

    def test_original_final_video_level_prediction_preserved(self):
        """Verify final video-level aggregation and classification remain 100% identical."""
        model, _ = load_model(device=DEVICE)
        test_bytes = create_test_mp4_bytes(num_frames=32)

        # Baseline Phase 1 inference
        res_baseline = run_inference_on_bytes(test_bytes, model=model, device=DEVICE)

        # Phase 2A SterileSession inference
        session = SterileSession(model=model, device=DEVICE)
        res_phase2 = session.process_video_bytes(test_bytes)
        session.close()

        self.assertEqual(res_phase2["decision"], res_baseline["decision"])
        self.assertEqual(res_phase2["predicted_class"], res_baseline["predicted_class"])
        self.assertEqual(res_phase2["total_frames"], res_baseline["total_frames"])
        self.assertEqual(res_phase2["windows_processed"], res_baseline["windows_processed"])

        for cls_name in ("Normal", "Fight", "Collapse"):
            self.assertAlmostEqual(
                res_phase2["confidences"][cls_name],
                res_baseline["confidences"][cls_name],
                places=5,
            )


if __name__ == "__main__":
    unittest.main()

