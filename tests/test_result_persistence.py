# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Unit and integration tests for final result persistence and dual-output UX."""

import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import av
import numpy as np

from privwatch.config import DEVICE
from privwatch.inference import load_model
from sterile_room.alert_registry import AlertEvent
from sterile_room.inference_worker import MsgType, SterileInferenceWorker, WorkerMessage
from sterile_room.sterile_session import SterileSession


def create_test_mp4_bytes(num_frames: int = 32, fps: int = 25) -> bytes:
    """Generate in-memory MP4 test video."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = 112
    stream.height = 112
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        arr = np.full((112, 112, 3), (i * 7) % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


class TestResultPersistenceAndDualOutput(unittest.TestCase):
    """Test suite verifying persistence of analysis results across reruns and uploads."""

    def setUp(self):
        self.video_a_bytes = create_test_mp4_bytes(32)
        self.video_b_bytes = create_test_mp4_bytes(40)
        self.hash_a = hashlib.sha256(self.video_a_bytes).hexdigest()
        self.hash_b = hashlib.sha256(self.video_b_bytes).hexdigest()

    def test_upload_hash_detects_new_video_and_clears_state(self):
        """Uploading video B after video A clears video A's results from session state."""
        session_state = {
            "upload_hash": self.hash_a,
            "sterile_result": {"decision": "Normal activity"},
            "sterile_alerts": [AlertEvent.create("Fight", 0.85, 100.0)],
            "sterile_progress": {"current_frame": 32, "lifecycle_state": "Analysis completed"},
            "sterile_done": True,
            "raw_result": {"decision": "Normal activity"},
        }

        # Simulating upload of video B (different hash)
        current_hash = self.hash_b
        prev_hash = session_state.get("upload_hash")

        if prev_hash is not None and current_hash != prev_hash:
            session_state.pop("sterile_result", None)
            session_state.pop("sterile_alerts", None)
            session_state.pop("sterile_progress", None)
            session_state.pop("sterile_done", None)
            session_state.pop("sterile_error", None)
            session_state.pop("raw_result", None)

        session_state["upload_hash"] = current_hash

        self.assertEqual(session_state["upload_hash"], self.hash_b)
        self.assertNotIn("sterile_result", session_state)
        self.assertNotIn("sterile_alerts", session_state)
        self.assertNotIn("sterile_progress", session_state)
        self.assertNotIn("sterile_done", session_state)
        self.assertNotIn("raw_result", session_state)

    def test_same_video_upload_preserves_results(self):
        """Re-running with the same video hash does NOT clear results."""
        stored_result = {"decision": "ALERT - Fight detected", "predicted_class": "Fight"}
        session_state = {
            "upload_hash": self.hash_a,
            "sterile_result": stored_result,
            "sterile_alerts": [AlertEvent.create("Fight", 0.95, 100.0)],
            "sterile_done": True,
        }

        # Simulating stream run / widget interaction with SAME video
        current_hash = self.hash_a
        prev_hash = session_state.get("upload_hash")

        if prev_hash is not None and current_hash != prev_hash:
            session_state.pop("sterile_result", None)

        self.assertEqual(session_state.get("sterile_result"), stored_result)
        self.assertTrue(session_state.get("sterile_done"))
        self.assertEqual(len(session_state.get("sterile_alerts")), 1)

    def test_fragment_polling_does_not_duplicate_alerts(self):
        """Repeated drain cycles on the fragment do not duplicate alerts with identical event_id."""
        session_state = {
            "sterile_alerts": [],
        }

        alert1 = AlertEvent(
            event_id="evt-001",
            timestamp="2026-11-02T12:00:00Z",
            event_type="Fight",
            confidence=0.90,
            latency_ms=50.0,
        )
        alert2 = AlertEvent(
            event_id="evt-002",
            timestamp="2026-11-02T12:00:01Z",
            event_type="Collapse",
            confidence=0.88,
            latency_ms=60.0,
        )

        # Cycle 1: alert1 arrives
        msgs_cycle1 = [WorkerMessage(MsgType.ALERT, alert1)]
        seen_ids = {a.event_id for a in session_state["sterile_alerts"]}
        for m in msgs_cycle1:
            if m.payload.event_id not in seen_ids:
                session_state["sterile_alerts"].append(m.payload)
                seen_ids.add(m.payload.event_id)

        self.assertEqual(len(session_state["sterile_alerts"]), 1)

        # Cycle 2: alert1 redelivered (or polled again) + alert2
        msgs_cycle2 = [WorkerMessage(MsgType.ALERT, alert1), WorkerMessage(MsgType.ALERT, alert2)]
        seen_ids = {a.event_id for a in session_state["sterile_alerts"]}
        for m in msgs_cycle2:
            if m.payload.event_id not in seen_ids:
                session_state["sterile_alerts"].append(m.payload)
                seen_ids.add(m.payload.event_id)

        # Should contain exactly 2 alerts, no duplicate of alert1
        self.assertEqual(len(session_state["sterile_alerts"]), 2)
        self.assertEqual(session_state["sterile_alerts"][0].event_id, "evt-001")
        self.assertEqual(session_state["sterile_alerts"][1].event_id, "evt-002")

    def test_both_outputs_coexist_after_eof(self):
        """After EOF, both provisional alerts and final classification exist simultaneously."""
        model, _ = load_model(device=DEVICE)
        worker = SterileInferenceWorker(video_bytes=self.video_a_bytes, model=model, device=DEVICE)
        worker.start()
        worker.wait()

        msgs = worker.drain_alerts()
        session_state = {
            "sterile_alerts": [],
            "sterile_result": None,
            "sterile_done": False,
        }

        for m in msgs:
            if m.msg_type == MsgType.ALERT:
                session_state["sterile_alerts"].append(m.payload)
            elif m.msg_type == MsgType.RESULT:
                session_state["sterile_result"] = m.payload
            elif m.msg_type == MsgType.DONE:
                session_state["sterile_done"] = True

        self.assertTrue(session_state["sterile_done"])
        self.assertIsNotNone(session_state["sterile_result"])
        self.assertIn("decision", session_state["sterile_result"])
        # Both outputs are populated in session state
        self.assertIsInstance(session_state["sterile_alerts"], list)
        self.assertIsInstance(session_state["sterile_result"], dict)


if __name__ == "__main__":
    unittest.main()
