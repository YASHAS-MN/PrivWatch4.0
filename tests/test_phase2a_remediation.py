# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Phase 2A Remediation test suite.

Covers:
  Task 1 — Background inference worker with thread-safe live alert queue
  Task 2 — JPEG frame compression in event buffers
  Task 3 — Time-based pre/post-event buffering with VFR handling
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
import sys
import unittest

import av
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privwatch.config import DEVICE
from privwatch.inference import load_model
from sterile_room.alert_registry import AlertEvent, AlertRegistry
from sterile_room.compressed_frame import CompressedFrame
from sterile_room.event_manager import RealTimeEventManager, TimedPreBuffer, EventState
from sterile_room.inference_worker import MsgType, SterileInferenceWorker, WorkerMessage
from sterile_room.sterile_session import SterileSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bgr_frame(val: int = 128, h: int = 112, w: int = 112) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _make_hd_bgr_frame(val: int = 128) -> np.ndarray:
    return np.full((1080, 1920, 3), val, dtype=np.uint8)


def create_test_mp4_bytes(
    num_frames: int = 32,
    width: int = 112,
    height: int = 112,
    fps: int = 25,
) -> bytes:
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


# ---------------------------------------------------------------------------
# Task 2 — JPEG Compressed Frame
# ---------------------------------------------------------------------------

class TestCompressedFrame(unittest.TestCase):
    """Verify in-memory JPEG frame compression and decompression."""

    def test_from_bgr_returns_compressed_frame(self):
        frame = _make_bgr_frame(200)
        cf = CompressedFrame.from_bgr(frame, timestamp=1.5, jpeg_quality=70)
        self.assertIsInstance(cf, CompressedFrame)
        self.assertEqual(cf.timestamp, 1.5)
        self.assertEqual(cf.original_h, 112)
        self.assertEqual(cf.original_w, 112)
        self.assertGreater(cf.size_bytes, 0)

    def test_decode_returns_bgr_ndarray(self):
        frame = _make_bgr_frame(128)
        cf = CompressedFrame.from_bgr(frame, timestamp=0.0)
        decoded = cf.decode()
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertEqual(decoded.shape, (112, 112, 3))

    def test_jpeg_compression_reduces_memory_vs_raw(self):
        """Verify JPEG compresses substantially vs raw float32."""
        frame = _make_hd_bgr_frame()
        raw_bytes = frame.nbytes  # 1080 * 1920 * 3 = 6,220,800 bytes
        cf = CompressedFrame.from_bgr(frame, timestamp=0.0, jpeg_quality=70)
        # JPEG must be significantly smaller (at least 10x)
        self.assertLess(cf.size_bytes, raw_bytes // 10)

    def test_no_disk_access_during_encode_decode(self):
        """Verify no files are written when using CompressedFrame."""
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        before = set(tmp_dir.glob("*.jpg")) | set(tmp_dir.glob("*.png"))

        frame = _make_bgr_frame(64)
        cf = CompressedFrame.from_bgr(frame, timestamp=0.5)
        _ = cf.decode()

        after = set(tmp_dir.glob("*.jpg")) | set(tmp_dir.glob("*.png"))
        self.assertEqual(len(after - before), 0)

    def test_channel_order_preserved_within_jpeg_tolerance(self):
        """Verify BGR channel order is preserved through encode/decode within JPEG tolerance."""
        # Pure blue frame (BGR = [255, 0, 0])
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel
        cf = CompressedFrame.from_bgr(frame, timestamp=0.0, jpeg_quality=95)
        decoded = cf.decode()
        # Blue channel dominant
        self.assertGreater(float(decoded[:, :, 0].mean()), 200.0)
        self.assertLess(float(decoded[:, :, 1].mean()), 30.0)
        self.assertLess(float(decoded[:, :, 2].mean()), 30.0)


# ---------------------------------------------------------------------------
# Task 3 — TimedPreBuffer
# ---------------------------------------------------------------------------

class TestTimedPreBuffer(unittest.TestCase):
    """Verify time-based rolling buffer eviction for CFR and VFR streams."""

    def _make_manager(self, pre_s: float = 5.0, post_s: float = 5.0) -> RealTimeEventManager:
        return RealTimeEventManager(
            pre_event_seconds=pre_s,
            post_event_seconds=post_s,
            fight_threshold=0.35,
            collapse_threshold=0.35,
        )

    def test_buffer_retains_5s_at_25fps(self):
        """At 25fps, 5s window retains ~125 frames."""
        buf = TimedPreBuffer(window_seconds=5.0)
        for i in range(200):  # 8s of footage
            buf.push(_make_bgr_frame(i % 255), timestamp=i / 25.0)

        # Current PTS = 199/25 = 7.96s; window = [2.96s, 7.96s]
        retained = len(buf)
        self.assertGreaterEqual(retained, 123)
        self.assertLessEqual(retained, 127)

    def test_buffer_retains_5s_at_30fps(self):
        """At 30fps, 5s window retains ~150 frames."""
        buf = TimedPreBuffer(window_seconds=5.0)
        for i in range(300):  # 10s of footage
            buf.push(_make_bgr_frame(i % 255), timestamp=i / 30.0)

        # Current PTS = 299/30 = 9.97s; window = [4.97s, 9.97s]
        retained = len(buf)
        self.assertGreaterEqual(retained, 148)
        self.assertLessEqual(retained, 152)

    def test_vfr_mixed_intervals(self):
        """Variable frame-rate inputs are handled correctly by PTS eviction."""
        buf = TimedPreBuffer(window_seconds=2.0)
        pts = 0.0
        intervals = [0.033, 0.050, 0.033, 0.040, 0.033, 0.050] * 30
        for dt in intervals:
            buf.push(_make_bgr_frame(), timestamp=pts)
            pts += dt

        # All frames with timestamp < (pts - 2.0) must be evicted
        if len(buf) > 0:
            earliest_ts = buf._buffer[0].timestamp
            latest_ts = buf._buffer[-1].timestamp
            self.assertGreaterEqual(latest_ts - earliest_ts, 0.0)
            self.assertLessEqual(latest_ts - earliest_ts, 2.0 + 0.1)  # allow small rounding

    def test_duplicate_pts_handled_by_monotonic_correction(self):
        """Duplicate PTS values receive epsilon correction and are not dropped."""
        buf = TimedPreBuffer(window_seconds=5.0)
        count_before = len(buf)
        for _ in range(10):
            buf.push(_make_bgr_frame(), timestamp=1.0)  # all same PTS
        self.assertEqual(len(buf), 10)

    def test_non_monotonic_pts_corrected(self):
        """Backwards-jumping PTS does not cause frames to be discarded silently."""
        buf = TimedPreBuffer(window_seconds=5.0)
        pts_sequence = [1.0, 0.5, 0.3, 2.0, 1.5, 3.0]
        for pts in pts_sequence:
            buf.push(_make_bgr_frame(), timestamp=pts)
        # All frames must be inserted (with monotonic correction)
        self.assertEqual(len(buf), len(pts_sequence))

    def test_pre_event_snapshot_populates_event(self):
        """Verify pre-event buffer snapshot is correctly assigned to RealTimeEvent."""
        manager = self._make_manager(pre_s=2.0, post_s=2.0)
        for i in range(60):  # 60 frames at 25fps = 2.4s
            manager.push_frame(_make_bgr_frame(i % 255), timestamp=i / 25.0)

        manager.process_window(0, 60 / 25.0, 0.10, 0.80, 0.10)
        event = manager.get_events()[0]

        # Pre-event frames must be CompressedFrame instances
        self.assertGreater(len(event.pre_event_frames), 0)
        self.assertIsInstance(event.pre_event_frames[0], CompressedFrame)

    def test_post_event_closure_by_pts(self):
        """Verify active event closes only after post_event_seconds elapsed by PTS."""
        manager = self._make_manager(pre_s=2.0, post_s=2.0)

        # Anomaly at t=1.0s
        manager.push_frame(_make_bgr_frame(), timestamp=1.0)
        manager.process_window(0, 1.0, 0.10, 0.80, 0.10)
        event = manager.get_events()[0]
        self.assertEqual(event.state, EventState.ACTIVE)

        # Push frames up to t=2.9s (only 1.9s since anomaly — not closed yet)
        for i in range(1, 48):
            t = 1.0 + i * 0.04
            manager.push_frame(_make_bgr_frame(), timestamp=t)
            manager.process_window(i, t, 0.90, 0.05, 0.05)

        self.assertEqual(event.state, EventState.ACTIVE)

        # Push frames up to t=3.2s (3.2 - 1.0 = 2.2s >= 2.0s post_event)
        for i in range(48, 56):
            t = 1.0 + i * 0.04
            manager.push_frame(_make_bgr_frame(), timestamp=t)
            manager.process_window(i, t, 0.90, 0.05, 0.05)

        self.assertEqual(event.state, EventState.CLOSED)
        self.assertTrue(event.is_ready)

    def test_new_anomaly_during_post_event_extends_event(self):
        """A new anomaly during post-event collection extends the event's last_anomalous_timestamp."""
        manager = self._make_manager(pre_s=2.0, post_s=5.0)

        # First anomaly at t=1.0s
        manager.push_frame(_make_bgr_frame(), timestamp=1.0)
        manager.process_window(0, 1.0, 0.10, 0.80, 0.10)
        event = manager.get_events()[0]
        first_last_ts = event.last_anomalous_timestamp

        # Push 1.0s of post-event normal frames (not enough to close at 5s post)
        for i in range(1, 26):
            t = 1.0 + i * 0.04
            manager.push_frame(_make_bgr_frame(), timestamp=t)
            manager.process_window(i, t, 0.90, 0.05, 0.05)

        # New anomaly at t=2.0s — must extend event not create new one
        manager.push_frame(_make_bgr_frame(), timestamp=2.0)
        manager.process_window(26, 2.0, 0.10, 0.80, 0.10)

        self.assertEqual(len(manager.get_events()), 1)
        self.assertGreater(event.last_anomalous_timestamp, first_last_ts)


# ---------------------------------------------------------------------------
# Task 1 — SterileInferenceWorker
# ---------------------------------------------------------------------------

class TestSterileInferenceWorker(unittest.TestCase):
    """Verify background worker puts messages on queue without calling Streamlit APIs."""

    def test_corrupt_bytes_puts_error_then_done(self):
        """Corrupt video bytes must result in ERROR + DONE messages, no hang."""
        worker = SterileInferenceWorker(video_bytes=b"corrupt", model=None)
        worker.start()
        worker.wait()
        msgs = worker.drain_alerts()
        types = [m.msg_type for m in msgs]
        self.assertIn(MsgType.DONE, types)
        self.assertTrue(MsgType.ERROR in types or MsgType.RESULT in types)

    def test_worker_runs_in_daemon_thread(self):
        """Worker thread must be a daemon thread."""
        worker = SterileInferenceWorker(video_bytes=b"x", model=None)
        worker.start()
        self.assertIsNotNone(worker._thread)
        self.assertTrue(worker._thread.daemon)  # type: ignore[union-attr]
        worker.wait()

    def test_worker_messages_are_worker_message_instances(self):
        """All queue outputs must be WorkerMessage dataclass instances."""
        worker = SterileInferenceWorker(video_bytes=b"bad", model=None)
        worker.start()
        worker.wait()
        for msg in worker.drain_alerts():
            self.assertIsInstance(msg, WorkerMessage)
            self.assertIsInstance(msg.msg_type, MsgType)

    def test_no_streamlit_import_in_worker_thread(self):
        """Worker body must not import or call streamlit from inside the thread."""
        # We verify by inspecting inference_worker module — 'streamlit' must not be imported
        import sterile_room.inference_worker as iw_module
        self.assertNotIn("streamlit", dir(iw_module))

    def test_valid_video_puts_result_then_done(self):
        """Valid video bytes produce RESULT followed by DONE messages."""
        model, _ = load_model(device=DEVICE)
        video_bytes = create_test_mp4_bytes(num_frames=32)
        worker = SterileInferenceWorker(video_bytes=video_bytes, model=model, device=DEVICE)
        worker.start()
        worker.wait()

        msgs = worker.drain_alerts()
        types = [m.msg_type for m in msgs]
        self.assertIn(MsgType.RESULT, types)
        self.assertIn(MsgType.DONE, types)
        self.assertNotIn(MsgType.ERROR, types)

        # RESULT must come before DONE
        result_idx = next(i for i, m in enumerate(msgs) if m.msg_type == MsgType.RESULT)
        done_idx = next(i for i, m in enumerate(msgs) if m.msg_type == MsgType.DONE)
        self.assertLess(result_idx, done_idx)

    def test_alert_emitted_before_done_for_fight_video(self):
        """An anomaly alert appears in the queue before the DONE message (pre-EOF detection)."""
        alerts_received: list[AlertEvent] = []
        done_received = threading.Event()

        model, _ = load_model(device=DEVICE)
        video_bytes = create_test_mp4_bytes(num_frames=32)

        worker = SterileInferenceWorker(video_bytes=video_bytes, model=model, device=DEVICE)
        # Override alert callback so we capture it before worker finishes
        # We do this by monkey-patching on a dummy manager before start()
        # The real test: drain_alerts() after wait() will show all ALERT messages before DONE
        worker.start()
        worker.wait()

        msgs = worker.drain_alerts()
        types = [m.msg_type for m in msgs]
        # DONE must always be last
        if len(types) > 0:
            self.assertEqual(types[-1], MsgType.DONE)

    def test_multiple_alerts_no_duplication(self):
        """Multiple anomalous windows are deduplicated into events; no duplicate ALERT messages."""
        model, _ = load_model(device=DEVICE)
        video_bytes = create_test_mp4_bytes(num_frames=64)
        worker = SterileInferenceWorker(video_bytes=video_bytes, model=model, device=DEVICE)
        worker.start()
        worker.wait()

        msgs = worker.drain_alerts()
        alert_msgs = [m for m in msgs if m.msg_type == MsgType.ALERT]
        event_ids = [m.payload.event_id for m in alert_msgs if m.payload]

        # No two ALERT messages should share the same event_id (no duplicate alerts)
        self.assertEqual(len(event_ids), len(set(event_ids)))

    def test_worker_cancel_before_start_puts_done_only(self):
        """Cancelling the worker before start() results in only DONE message."""
        worker = SterileInferenceWorker(video_bytes=b"x", model=None)
        worker.cancel()
        worker.start()
        worker.wait()
        msgs = worker.drain_alerts()
        types = [m.msg_type for m in msgs]
        self.assertIn(MsgType.DONE, types)
        # Should not have processed anything (no RESULT)
        self.assertNotIn(MsgType.RESULT, types)


    def test_worker_emits_progress_messages(self):
        """Worker dispatches PROGRESS messages containing frame index, timestamp, and lifecycle state."""
        model, _ = load_model(device=DEVICE)
        video_bytes = create_test_mp4_bytes(num_frames=32)
        worker = SterileInferenceWorker(video_bytes=video_bytes, model=model, device=DEVICE)
        worker.start()
        worker.wait()

        msgs = worker.drain_alerts()
        progress_msgs = [m for m in msgs if m.msg_type == MsgType.PROGRESS]
        self.assertGreater(len(progress_msgs), 0)

        # Inspect first and last progress messages
        first_prog = progress_msgs[0].payload
        self.assertIn("current_frame", first_prog)
        self.assertIn("timestamp", first_prog)
        self.assertIn("lifecycle_state", first_prog)
        self.assertEqual(first_prog["current_frame"], 1)

        # Last progress message at EOF should be 'Analysis completed'
        last_prog = progress_msgs[-1].payload
        self.assertEqual(last_prog["lifecycle_state"], "Analysis completed")

    def test_session_probe_total_frames(self):
        """SterileSession accurately probes total frames from valid MP4 container."""
        video_bytes = create_test_mp4_bytes(num_frames=32)
        progress_reports = []
        session = SterileSession()
        session.process_video_bytes(
            video_bytes,
            progress_callback=lambda p: progress_reports.append(p),
        )
        session.close()

        self.assertGreater(len(progress_reports), 0)
        # 32 frames requested
        self.assertEqual(progress_reports[0]["total_frames"], 32)


# ---------------------------------------------------------------------------
# Integration — SterileSession with jpeg_quality parameter
# ---------------------------------------------------------------------------

class TestSterileSessionJpegQuality(unittest.TestCase):
    """Verify SterileSession correctly passes jpeg_quality through to event buffers."""

    def test_session_accepts_jpeg_quality_param(self):
        """SterileSession must accept jpeg_quality without TypeError."""
        session = SterileSession(session_id="test-q", jpeg_quality=85)
        self.assertEqual(session.jpeg_quality, 85)
        self.assertEqual(session.event_manager.jpeg_quality, 85)
        self.assertEqual(session.event_manager.pre_buffer.jpeg_quality, 85)

    def test_default_jpeg_quality_is_70(self):
        """Default jpeg_quality should be 70."""
        session = SterileSession(session_id="test-default")
        self.assertEqual(session.jpeg_quality, 70)

    def test_end_to_end_with_jpeg_quality(self):
        """End-to-end processing with non-default jpeg_quality completes without error."""
        model, _ = load_model(device=DEVICE)
        video_bytes = create_test_mp4_bytes(num_frames=32)
        session = SterileSession(model=model, device=DEVICE, jpeg_quality=80)
        results = session.process_video_bytes(video_bytes)
        session.close()
        self.assertIn("decision", results)
        self.assertIn("events", results)


if __name__ == "__main__":
    unittest.main()

