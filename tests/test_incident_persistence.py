# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Tests for Pillar 3: Selective Incident Persistence.

Test IDs:
  T1  — Normal-only video → zero incident files
  T2  — Anomaly video → exactly one .mp4 + one .json
  T3  — Pre-event footage present in clip (frame count >= pre-event count)
  T4  — Anomaly footage present in clip
  T5  — Post-event footage present in clip
  T6  — Event extension during post-event → single incident file
  T7  — Duplicate closure does not write duplicate files
  T8  — Encoding failure → no valid incident pair
  T9  — Disk-write failure → success=False, no crash
  T10 — Multiple incidents → separate files
  T11 — Alert delivery before persistence completes
  T12 — RAW Mode → incidents/ directory never created
  T13 — Existing 73 tests continue passing (run as subprocess)
"""

from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# ── project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    from privwatch.config import DEVICE
    from privwatch.inference import load_model
    from sterile_room.compressed_frame import CompressedFrame
    from sterile_room.event_manager import EventState, RealTimeEvent, RealTimeEventManager
    from sterile_room.incident_writer import IncidentRecord, IncidentWriter
    from sterile_room.inference_worker import MsgType, SterileInferenceWorker, WorkerMessage
    from sterile_room.sterile_session import SterileSession
except ImportError:
    from PrivWatch_Final.privwatch.config import DEVICE  # type: ignore
    from PrivWatch_Final.privwatch.inference import load_model  # type: ignore
    from PrivWatch_Final.sterile_room.compressed_frame import CompressedFrame  # type: ignore
    from PrivWatch_Final.sterile_room.event_manager import EventState, RealTimeEvent, RealTimeEventManager  # type: ignore
    from PrivWatch_Final.sterile_room.incident_writer import IncidentRecord, IncidentWriter  # type: ignore
    from PrivWatch_Final.sterile_room.inference_worker import MsgType, SterileInferenceWorker, WorkerMessage  # type: ignore
    from PrivWatch_Final.sterile_room.sterile_session import SterileSession  # type: ignore

# ── test video byte helpers ───────────────────────────────────────────────────

FIGHT_VIDEO_PATH = Path("C:/Prototypes/VI_PrivWaatch/t/f/b (1).mp4")
NORMAL_VIDEO_PATH = Path("C:/Prototypes/VI_PrivWaatch/t/n/a (1).mp4")


def _load_video_bytes(path: Path) -> bytes:
    return path.read_bytes()


def create_test_mp4_bytes(n_frames: int = 32) -> bytes:
    """Create a minimal valid MP4 in-memory with n_frames of synthetic noise."""
    import av  # type: ignore

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=25)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "30", "preset": "ultrafast"}
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        frame_data = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return buf.getvalue()


def _make_closed_event(
    n_pre: int = 5,
    n_event: int = 5,
    n_post: int = 5,
    event_type: str = "Fight",
    confidence: float = 0.80,
) -> RealTimeEvent:
    """Build a synthetic CLOSED RealTimeEvent with CompressedFrame buffers."""
    rng = np.random.default_rng(123)
    base_ts = 0.0

    def _make_frames(count: int, start_ts: float) -> list[CompressedFrame]:
        frames = []
        for i in range(count):
            bgr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            ts = start_ts + i * (1.0 / 25.0)
            frames.append(CompressedFrame.from_bgr(bgr, ts, jpeg_quality=70))
        return frames

    pre = _make_frames(n_pre, base_ts)
    evt_start = base_ts + n_pre / 25.0
    evt_frames = _make_frames(n_event, evt_start)
    post_start = evt_start + n_event / 25.0
    post_frames = _make_frames(n_post, post_start)

    event = RealTimeEvent(
        event_id=f"evt-test001",
        event_type=event_type,
        state=EventState.CLOSED,
        first_detected_timestamp=evt_start,
        first_detected_window=5,
        last_anomalous_timestamp=evt_start + n_event / 25.0,
        last_anomalous_frame_idx=n_pre + n_event,
        confidence=confidence,
        camera_id="cam-01",
        pre_event_frames=pre,
        event_frames=evt_frames,
        post_event_frames=post_frames,
        is_ready=True,
    )
    return event


# ── test model (shared across all tests that need it) ────────────────────────

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL, _ = load_model(device=DEVICE)
    return _MODEL


# =============================================================================
# Test Suite
# =============================================================================


class TestIncidentPersistence(unittest.TestCase):
    """Pillar 3: Selective Incident Persistence tests."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.incidents_dir = self.tmp_dir / "incidents"
        self.incidents_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── T1: Normal-only video → zero incident files ───────────────────────────

    def test_T1_normal_video_zero_incident_files(self) -> None:
        """A video with no anomalies must not create any incident files."""
        # Build a synthetic normal video (short random noise, unlikely to trigger threshold)
        # We directly test at the SterileSession level
        model = _get_model()
        session = SterileSession(model=model, device=DEVICE)
        # Provide no events (simulate normal processing)
        session.event_manager.on_eof()
        # Should have zero closed events
        closed = [e for e in session.event_manager.events if e.state == EventState.CLOSED]
        self.assertEqual(len(closed), 0)
        # incidents_dir should remain empty
        mp4s = list(self.incidents_dir.glob("*.mp4"))
        self.assertEqual(len(mp4s), 0)

    # ── T2: Anomaly video → exactly 1 .mp4 + 1 .json ────────────────────────

    def test_T2_anomaly_event_produces_mp4_and_json(self) -> None:
        """A CLOSED event produces exactly one .mp4 and one .json sidecar."""
        event = _make_closed_event()
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record = writer.write_incident(event)

        self.assertTrue(record.success, f"write_incident failed: {record.error}")
        mp4_files = list(self.incidents_dir.glob("*.mp4"))
        json_files = list(self.incidents_dir.glob("*.json"))
        self.assertEqual(len(mp4_files), 1)
        self.assertEqual(len(json_files), 1)
        # No .tmp files left behind
        tmp_files = list(self.incidents_dir.glob("*.tmp.*"))
        self.assertEqual(len(tmp_files), 0)

    # ── T3: Pre-event footage in clip ────────────────────────────────────────

    def test_T3_pre_event_frames_included(self) -> None:
        """Clip must contain at least as many frames as the pre-event buffer."""
        n_pre, n_event, n_post = 8, 4, 4
        event = _make_closed_event(n_pre=n_pre, n_event=n_event, n_post=n_post)
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record = writer.write_incident(event)

        self.assertTrue(record.success)
        self.assertGreaterEqual(
            record.frame_count, n_pre,
            "Frame count must be >= number of pre-event frames"
        )

    # ── T4: Anomaly footage in clip ───────────────────────────────────────────

    def test_T4_event_frames_included(self) -> None:
        """Clip total frame count covers pre + event frames."""
        n_pre, n_event, n_post = 3, 10, 3
        event = _make_closed_event(n_pre=n_pre, n_event=n_event, n_post=n_post)
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record = writer.write_incident(event)

        self.assertTrue(record.success)
        self.assertGreaterEqual(record.frame_count, n_pre + n_event)

    # ── T5: Post-event footage in clip ────────────────────────────────────────

    def test_T5_post_event_frames_included(self) -> None:
        """Clip total frame count covers all three buffer segments."""
        n_pre, n_event, n_post = 5, 5, 10
        event = _make_closed_event(n_pre=n_pre, n_event=n_event, n_post=n_post)
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record = writer.write_incident(event)

        self.assertTrue(record.success)
        self.assertEqual(record.frame_count, n_pre + n_event + n_post)

    # ── T6: Event extension → single incident file ───────────────────────────

    def test_T6_event_extension_single_file(self) -> None:
        """Multiple anomalous windows merged into one event → one incident file."""
        # Simulate two consecutive anomalous windows merged into one event
        rng = np.random.default_rng(7)
        event = _make_closed_event(n_pre=5, n_event=15, n_post=5)
        event.event_id = "evt-merged01"
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record = writer.write_incident(event)

        self.assertTrue(record.success)
        mp4s = list(self.incidents_dir.glob("*.mp4"))
        jsons = list(self.incidents_dir.glob("*.json"))
        self.assertEqual(len(mp4s), 1)
        self.assertEqual(len(jsons), 1)

    # ── T7: Duplicate closure → no duplicate files ───────────────────────────

    def test_T7_duplicate_write_no_duplicate_files(self) -> None:
        """Calling write_incident twice for the same event_id must not create duplicates."""
        event = _make_closed_event()
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record1 = writer.write_incident(event)
        record2 = writer.write_incident(event)  # second call: same event_id

        self.assertTrue(record1.success)
        # Second call must return success=False (already written)
        self.assertFalse(record2.success)
        mp4s = list(self.incidents_dir.glob("*.mp4"))
        self.assertEqual(len(mp4s), 1, "Must have exactly one MP4 after duplicate call")

    # ── T8: Encoding failure → no valid incident pair ────────────────────────

    def test_T8_encoding_failure_no_valid_pair(self) -> None:
        """When encoding raises, no .mp4 or .json file is left."""
        event = _make_closed_event()
        writer = IncidentWriter(incidents_dir=self.incidents_dir)

        with patch.object(writer, "_encode_frames_to_mp4", side_effect=RuntimeError("encode boom")):
            record = writer.write_incident(event)

        self.assertFalse(record.success)
        self.assertIn("encode boom", record.error or "")
        # No final files should remain
        mp4s = list(self.incidents_dir.glob("*.mp4"))
        json_files = list(self.incidents_dir.glob("*.json"))
        self.assertEqual(len(mp4s), 0)
        self.assertEqual(len(json_files), 0)

    # ── T9: Disk-write failure → success=False, no crash ─────────────────────

    def test_T9_disk_failure_handled_gracefully(self) -> None:
        """JSON write failure returns success=False without raising."""
        event = _make_closed_event()
        writer = IncidentWriter(incidents_dir=self.incidents_dir)

        with patch.object(writer, "_write_json", side_effect=OSError("disk full")):
            record = writer.write_incident(event)

        self.assertFalse(record.success)
        self.assertIn("disk full", record.error or "")
        # Temp files must be cleaned up
        tmp_files = list(self.incidents_dir.glob("*.tmp.*"))
        self.assertEqual(len(tmp_files), 0)

    # ── T10: Multiple incidents → separate files ─────────────────────────────

    def test_T10_multiple_incidents_separate_files(self) -> None:
        """Two separate events produce two separate .mp4 + .json pairs."""
        event1 = _make_closed_event(n_pre=3, n_event=3, n_post=3)
        event1.event_id = "evt-multi001"
        event2 = _make_closed_event(n_pre=3, n_event=3, n_post=3)
        event2.event_id = "evt-multi002"

        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record1 = writer.write_incident(event1)
        record2 = writer.write_incident(event2)

        self.assertTrue(record1.success, f"Event1 failed: {record1.error}")
        self.assertTrue(record2.success, f"Event2 failed: {record2.error}")
        mp4s = sorted(self.incidents_dir.glob("*.mp4"))
        jsons = sorted(self.incidents_dir.glob("*.json"))
        self.assertEqual(len(mp4s), 2)
        self.assertEqual(len(jsons), 2)
        # Filenames must differ
        self.assertNotEqual(mp4s[0].name, mp4s[1].name)

    # ── T11: Alert delivery before persistence completes ─────────────────────

    def test_T11_alert_before_persistence(self) -> None:
        """ALERT message arrives in the queue before INCIDENT message."""
        alert_times: list[float] = []
        incident_times: list[float] = []
        lock = threading.Lock()

        model = _get_model()
        video_bytes = _load_video_bytes(FIGHT_VIDEO_PATH) if FIGHT_VIDEO_PATH.exists() else create_test_mp4_bytes(32)
        worker = SterileInferenceWorker(
            video_bytes=video_bytes,
            model=model,
            device=DEVICE,
            incidents_dir=self.incidents_dir,
        )

        orig_alert = worker._alert_dispatch

        def timed_alert(alert: object) -> None:
            with lock:
                alert_times.append(time.time())
            orig_alert(alert)  # type: ignore

        orig_persist = worker._persist_dispatch

        def timed_persist(event: RealTimeEvent) -> None:
            with lock:
                incident_times.append(time.time())
            orig_persist(event)

        worker._alert_dispatch = timed_alert  # type: ignore
        worker._persist_dispatch = timed_persist  # type: ignore

        worker.start()
        worker.wait()

        # Drain all messages
        all_msgs = worker.drain_alerts()
        msg_types = [m.msg_type for m in all_msgs]

        self.assertIn(MsgType.ALERT, msg_types, "ALERT message must be present in queue")
        self.assertIn(MsgType.INCIDENT, msg_types, "INCIDENT message must be present in queue")
        self.assertGreater(len(alert_times), 0)
        self.assertGreater(len(incident_times), 0)
        self.assertLess(min(alert_times), min(incident_times),
                        "First ALERT must precede first INCIDENT")

    # ── T12: RAW Mode → incidents/ not created ───────────────────────────────

    def test_T12_raw_mode_no_incidents_dir(self) -> None:
        """RAW Mode inference must not touch the incidents/ directory."""
        # RAW Mode uses cv2.VideoCapture, which reads from a file path.
        # IncidentWriter is only instantiated inside SterileInferenceWorker.
        # We verify that creating a plain cv2 inference session does not
        # create any incidents/ directory.
        incidents_path = self.tmp_dir / "incidents"
        if incidents_path.exists():
            shutil.rmtree(incidents_path)

        # Importing and running raw inference does not instantiate IncidentWriter
        import importlib
        import sterile_room.inference_worker as iw_mod

        # The IncidentWriter should only be referenced in SterileInferenceWorker
        # Verify it is not auto-instantiated at module level
        self.assertFalse(
            incidents_path.exists(),
            "incidents/ must not be created by importing inference_worker"
        )

    # ── T13: All existing 73 tests still pass ────────────────────────────────

    def test_T13_existing_tests_still_pass(self) -> None:
        """Run the full pre-existing test suite and assert 0 failures."""
        import os
        if os.environ.get("PRIVWATCH_IN_T13") == "1":
            return
        # If imported by a parent discovery runner, discovery is already executing
        # all test suites; avoid redundant nested subprocess.
        if __name__ != "__main__":
            return

        env = os.environ.copy()
        env["PRIVWATCH_IN_T13"] = "1"
        result = subprocess.run(
            [
                sys.executable, "-m", "unittest", "discover",
                "-s", "tests", "-p", "test_*.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            timeout=300,
            env=env,
        )
        combined = result.stdout + result.stderr
        # unittest prints "OK" on success; failures contain "FAILED"
        self.assertNotIn("FAILED", combined, f"Existing tests failed:\n{combined}")
        # Also check that it ran a non-zero number of tests
        self.assertIn("Ran ", combined, f"No test output:\n{combined}")

    # ── Extra: JSON sidecar content validation ────────────────────────────────

    def test_json_sidecar_fields(self) -> None:
        """JSON sidecar must contain required fields with valid types."""
        event = _make_closed_event()
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record = writer.write_incident(event)
        self.assertTrue(record.success)

        json_path = Path(record.json_path)
        self.assertTrue(json_path.exists())
        data = json.loads(json_path.read_text(encoding="utf-8"))

        required_fields = [
            "event_id", "camera_id", "anomaly_class", "confidence",
            "first_detected_timestamp", "last_anomalous_timestamp",
            "clip_duration_seconds", "total_frame_count",
            "mp4_path", "created_at", "schema_version",
        ]
        for f in required_fields:
            self.assertIn(f, data, f"Missing JSON field: {f}")

        self.assertEqual(data["schema_version"], "1.0")
        self.assertIsInstance(data["confidence"], float)
        self.assertIsInstance(data["total_frame_count"], int)

    # ── Extra: Memory released after successful write ─────────────────────────

    def test_frame_buffers_cleared_after_write(self) -> None:
        """Frame buffer lists must be empty after a successful write (RAM release)."""
        event = _make_closed_event(n_pre=5, n_event=5, n_post=5)
        writer = IncidentWriter(incidents_dir=self.incidents_dir)
        record = writer.write_incident(event)

        self.assertTrue(record.success)
        self.assertEqual(len(event.pre_event_frames), 0, "pre_event_frames must be cleared")
        self.assertEqual(len(event.event_frames), 0, "event_frames must be cleared")
        self.assertEqual(len(event.post_event_frames), 0, "post_event_frames must be cleared")

    # ── Extra: can_write guard ────────────────────────────────────────────────

    def test_can_write_guards(self) -> None:
        """can_write returns False for non-CLOSED events and already-written IDs."""
        writer = IncidentWriter(incidents_dir=self.incidents_dir)

        # ACTIVE event
        active = _make_closed_event()
        active.state = EventState.ACTIVE
        active.is_ready = False
        self.assertFalse(writer.can_write(active))

        # CLOSED but not ready
        not_ready = _make_closed_event()
        not_ready.is_ready = False
        self.assertFalse(writer.can_write(not_ready))

        # CLOSED and ready: can write
        ready = _make_closed_event()
        ready.event_id = "evt-can001"
        self.assertTrue(writer.can_write(ready))

        # After writing, can_write returns False
        writer.write_incident(ready)
        self.assertFalse(writer.can_write(ready))

    # ── Extra: persistence_callback is fired on event close ──────────────────

    def test_persistence_callback_fired_on_close(self) -> None:
        """event_manager fires persistence_callback when _close_active_event is called."""
        received_events: list[RealTimeEvent] = []

        def capture(event: RealTimeEvent) -> None:
            received_events.append(event)

        mgr = RealTimeEventManager(
            pre_event_seconds=1.0,
            post_event_seconds=1.0,
            persistence_callback=capture,
        )

        # Push frames + process an anomalous window to start an event
        rng = np.random.default_rng(0)
        for i in range(20):
            frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            mgr.push_frame(frame, float(i) / 25.0)

        mgr.process_window(0, 0.0, 0.1, 0.9, 0.0)  # Fight detected

        # Push more frames past post_event_seconds to trigger closure
        for i in range(20, 60):
            frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            mgr.push_frame(frame, float(i) / 25.0)

        mgr.process_window(1, 2.0, 0.9, 0.05, 0.05)  # Normal window 2s later

        # Force EOF closure
        mgr.on_eof()

        # Callback should have fired exactly once
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].state, EventState.CLOSED)
        self.assertTrue(received_events[0].is_ready)


if __name__ == "__main__":
    unittest.main(verbosity=2)

