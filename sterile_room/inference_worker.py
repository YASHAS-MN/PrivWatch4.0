# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Background inference worker for live alert streaming in Sterile Mode.

DESIGN (Task 1):
- Inference runs in a daemon Thread completely isolated from Streamlit's main thread.
- Communication uses queue.Queue (thread-safe, zero Streamlit API calls from worker).
- Streamlit polls a shared queue via st.fragment with auto-rerun to update the UI
  while inference is still executing.

GUARANTEES:
- No Streamlit API is called from the worker thread.
- Worker exceptions are forwarded via the queue and re-raised in the UI thread.
- Worker is daemon=True so it does not prevent process exit on Streamlit restart.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
from typing import Any

import torch

_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from privwatch.config import DEVICE
    from privwatch.model import ActionModel
    from sterile_room.alert_registry import AlertEvent
    from sterile_room.event_manager import RealTimeEvent
    from sterile_room.incident_writer import IncidentRecord, IncidentWriter
    from sterile_room.sterile_session import SterileSession
except ImportError:
    from PrivWatch_Final.privwatch.config import DEVICE  # type: ignore
    from PrivWatch_Final.privwatch.model import ActionModel  # type: ignore
    from PrivWatch_Final.sterile_room.alert_registry import AlertEvent  # type: ignore
    from PrivWatch_Final.sterile_room.event_manager import RealTimeEvent  # type: ignore
    from PrivWatch_Final.sterile_room.incident_writer import IncidentRecord, IncidentWriter  # type: ignore
    from PrivWatch_Final.sterile_room.sterile_session import SterileSession  # type: ignore


# ---------------------------------------------------------------------------
# Queue message types
# ---------------------------------------------------------------------------


class MsgType(str, Enum):
    ALERT = "ALERT"         # Real-time anomaly alert (before EOF)
    PROGRESS = "PROGRESS"   # Real-time frame & lifecycle progress update
    RESULT = "RESULT"       # Final inference result dict (at EOF)
    INCIDENT = "INCIDENT"   # Incident persistence result (IncidentRecord)
    ERROR = "ERROR"         # Worker exception
    DONE = "DONE"           # Sentinel: worker finished (success or failure)


@dataclass
class WorkerMessage:
    msg_type: MsgType
    payload: Any = None     # AlertEvent, result dict, or Exception


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class SterileInferenceWorker:
    """Runs SterileSession.process_video_bytes() in a background daemon thread.

    Communication contract:
    - The worker puts WorkerMessage objects onto `out_queue`.
    - Callers (Streamlit UI) drain out_queue in the main thread.
    - No Streamlit API is called from inside the worker thread.

    Thread safety:
    - out_queue is a stdlib queue.Queue (thread-safe FIFO).
    - cancel_event is a threading.Event set by the caller to request abort.
      The worker does not currently poll cancel_event during frame iteration
      (PyAV generators do not support cooperative cancellation), but setting it
      before the worker starts prevents the thread from starting at all.
    """

    def __init__(
        self,
        video_bytes: bytes,
        model: ActionModel,
        device: torch.device = DEVICE,
        jpeg_quality: int = 70,
        camera_id: str = "cam-01",
        incidents_dir: Path = Path("incidents"),
    ):
        self.video_bytes = video_bytes
        self.model = model
        self.device = device
        self.jpeg_quality = jpeg_quality
        self.camera_id = camera_id
        self._incident_writer = IncidentWriter(incidents_dir=incidents_dir)
        self._persistence_threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()

        self.out_queue: queue.Queue[WorkerMessage] = queue.Queue()
        self.cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: SterileSession | None = None

    def _alert_dispatch(self, alert: AlertEvent) -> None:
        """Called by RealTimeEventManager from the worker thread.

        Puts an ALERT message onto out_queue so the UI thread can drain it.
        Must NOT call any Streamlit API.
        """
        self.out_queue.put(WorkerMessage(MsgType.ALERT, alert))

    def _progress_dispatch(self, progress_data: dict[str, Any]) -> None:
        """Called by SterileSession during frame decoding loop.

        Puts a PROGRESS message onto out_queue for non-blocking UI updates.
        Must NOT call any Streamlit API.
        """
        self.out_queue.put(WorkerMessage(MsgType.PROGRESS, progress_data))

    def _persist_dispatch(self, event: RealTimeEvent) -> None:
        """Called by RealTimeEventManager when an event transitions to CLOSED.

        Spawns a daemon thread to encode frames to disk without blocking
        the frame decoding loop.  The result is posted to out_queue as
        WorkerMessage(MsgType.INCIDENT, IncidentRecord).
        Must NOT call any Streamlit API.
        """
        writer = self._incident_writer
        q = self.out_queue

        def _encode() -> None:
            record = writer.write_incident(event)
            q.put(WorkerMessage(MsgType.INCIDENT, record))

        t = threading.Thread(target=_encode, daemon=True, name=f"incident-{event.event_id}")
        with self._threads_lock:
            self._persistence_threads.append(t)
        t.start()

    def _run(self) -> None:
        """Worker body executed in the background thread."""
        if self.cancel_event.is_set():
            self.out_queue.put(WorkerMessage(MsgType.DONE, None))
            return

        session = SterileSession(
            model=self.model,
            device=self.device,
            jpeg_quality=self.jpeg_quality,
            camera_id=self.camera_id,
        )
        self._session = session
        # Patch alert callback to use the thread-safe queue
        session.event_manager.alert_callback = self._alert_dispatch
        # Patch persistence callback: fires on event CLOSED (after alert)
        session.event_manager.persistence_callback = self._persist_dispatch

        try:
            result = session.process_video_bytes(
                self.video_bytes,
                progress_callback=self._progress_dispatch,
            )
            self.out_queue.put(WorkerMessage(MsgType.RESULT, result))
        except Exception as exc:
            self.out_queue.put(WorkerMessage(MsgType.ERROR, exc))
        finally:
            # Wait for all background persistence encoding tasks to finish
            # before releasing session memory and sending DONE.
            with self._threads_lock:
                threads = list(self._persistence_threads)
            for t in threads:
                t.join(timeout=30.0)
            session.close()
            self.out_queue.put(WorkerMessage(MsgType.DONE, None))

    def start(self) -> None:
        """Start the background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="sterile-inference")
        self._thread.start()

    def cancel(self) -> None:
        """Signal the worker to stop (best-effort; active decoding cannot be interrupted)."""
        self.cancel_event.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def drain_alerts(self, timeout: float = 0.0) -> list[WorkerMessage]:
        """Non-blocking drain: return all messages currently in the queue.

        Args:
            timeout: If > 0, block up to timeout seconds waiting for the first
                message. Subsequent reads are non-blocking.

        Returns:
            List of WorkerMessage objects. May include ALERT, RESULT, ERROR, and DONE.
        """
        messages: list[WorkerMessage] = []
        try:
            if timeout > 0:
                msg = self.out_queue.get(timeout=timeout)
                messages.append(msg)
        except queue.Empty:
            return messages
        # Drain remaining without blocking
        while True:
            try:
                messages.append(self.out_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def wait(self) -> None:
        """Block until the worker thread completes."""
        if self._thread is not None:
            self._thread.join()
