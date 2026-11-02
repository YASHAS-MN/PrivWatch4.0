# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Real-time anomaly detection and volatile event management for PrivWatch Sterile Room.

RESPONSIBILITIES:
- Track per-window prediction outputs with presentation timestamps.
- Evaluate real-time threshold conditions (Fight >= 0.35, Collapse >= 0.35) immediately per window.
- Implement stateful event lifecycle: IDLE -> ACTIVE -> CLOSED.
- Merge consecutive anomalous windows into a single incident event (prevent duplicate alerts).
- Maintain a time-based rolling pre-event buffer using PTS (not fixed frame count).
- Retain the configured 5s before first detection and 5s after last anomaly, purely by PTS.
- Collect frames as JPEG-compressed CompressedFrame objects to drastically reduce RAM footprint.
- Emit immediate alerts upon first detection without waiting for EOF or post-event completion.
- Guarantee ZERO disk persistence.

TASK 2 (Memory): Frames in all three buffers (pre-event, active, post-event) are stored as
  in-memory JPEG bytes via CompressedFrame, not as raw BGR arrays.

TASK 3 (Timing): Pre-event retention and post-event close are gated by PTS elapsed time, not
  by fixed frame-count deque capacity. Handles VFR, duplicate PTS, and non-monotonic PTS.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import time as _time
from typing import Any, Callable
import uuid

import numpy as np

_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from sterile_room.alert_registry import AlertEvent
from sterile_room.compressed_frame import CompressedFrame


# ---------------------------------------------------------------------------
# Enums and data types
# ---------------------------------------------------------------------------


class EventState(str, Enum):
    """Lifecycle states for real-time anomaly detection incidents."""

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


@dataclass
class WindowPrediction:
    """Prediction output for a single 16-frame sliding temporal window."""

    window_index: int
    timestamp: float
    normal_prob: float
    fight_prob: float
    collapse_prob: float
    anomaly_type: str | None = None  # "Fight", "Collapse", or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "timestamp": round(self.timestamp, 3),
            "normal_prob": round(self.normal_prob, 4),
            "fight_prob": round(self.fight_prob, 4),
            "collapse_prob": round(self.collapse_prob, 4),
            "anomaly_type": self.anomaly_type,
        }


@dataclass
class RealTimeEvent:
    """Stateful anomaly event record with JPEG-compressed incident frames in RAM."""

    event_id: str
    event_type: str                     # "Fight" or "Collapse"
    state: EventState
    first_detected_timestamp: float
    first_detected_window: int
    last_anomalous_timestamp: float
    last_anomalous_frame_idx: int
    confidence: float
    camera_id: str = "cam-01"
    pre_event_frames: list[CompressedFrame] = field(default_factory=list)
    event_frames: list[CompressedFrame] = field(default_factory=list)
    post_event_frames: list[CompressedFrame] = field(default_factory=list)
    is_ready: bool = False

    @property
    def total_frames_count(self) -> int:
        return len(self.pre_event_frames) + len(self.event_frames) + len(self.post_event_frames)

    @property
    def compressed_size_bytes(self) -> int:
        total = 0
        for pool in (self.pre_event_frames, self.event_frames, self.post_event_frames):
            total += sum(cf.size_bytes for cf in pool)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "state": self.state.value,
            "first_detected_timestamp": round(self.first_detected_timestamp, 3),
            "first_detected_window": self.first_detected_window,
            "last_anomalous_timestamp": round(self.last_anomalous_timestamp, 3),
            "confidence": round(self.confidence, 4),
            "camera_id": self.camera_id,
            "pre_event_frames_count": len(self.pre_event_frames),
            "event_frames_count": len(self.event_frames),
            "post_event_frames_count": len(self.post_event_frames),
            "total_frames_count": self.total_frames_count,
            "compressed_size_bytes": self.compressed_size_bytes,
            "is_ready": self.is_ready,
        }


# ---------------------------------------------------------------------------
# Time-based rolling pre-event buffer
# ---------------------------------------------------------------------------


class TimedPreBuffer:
    """Time-based rolling buffer retaining frames within a configurable duration window.

    Unlike a fixed-capacity deque, this buffer evicts frames whose PTS falls
    earlier than (current_pts - window_seconds). This is correct for VFR
    streams where frame count does not reliably proxy elapsed time.

    Stored objects are CompressedFrame instances (in-memory JPEG bytes + PTS).

    Fallback for non-monotonic PTS: frames with a PTS not greater than the
    previous frame's PTS are assigned PTS = prev_pts + epsilon (1ms). This
    preserves insertion order without silently discarding frames.
    """

    _MONOTONIC_EPSILON = 0.001  # seconds added on non-monotonic PTS

    def __init__(self, window_seconds: float, jpeg_quality: int = 70):
        self.window_seconds = window_seconds
        self.jpeg_quality = jpeg_quality
        self._buffer: deque[CompressedFrame] = deque()
        self._last_pts: float = -1.0

    def push(self, frame: np.ndarray, timestamp: float) -> float:
        """Compress frame, insert, evict expired entries, return sanitised PTS.

        Args:
            frame: Full-resolution BGR uint8 ndarray.
            timestamp: PTS in seconds. Non-monotonic values are sanitised.

        Returns:
            Sanitised (monotonic) PTS used for storage.
        """
        if timestamp <= self._last_pts:
            timestamp = self._last_pts + self._MONOTONIC_EPSILON
        self._last_pts = timestamp

        cf = CompressedFrame.from_bgr(frame, timestamp, jpeg_quality=self.jpeg_quality)
        self._buffer.append(cf)
        self._evict(timestamp)
        return timestamp

    def _evict(self, current_pts: float) -> None:
        """Remove frames older than (current_pts - window_seconds)."""
        cutoff = current_pts - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def snapshot(self) -> list[CompressedFrame]:
        """Return a shallow list of current buffer contents (no frame copy)."""
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def duration(self) -> float:
        """Return elapsed PTS duration currently retained in the buffer."""
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].timestamp - self._buffer[0].timestamp


# ---------------------------------------------------------------------------
# Real-time event manager
# ---------------------------------------------------------------------------


class RealTimeEventManager:
    """Manages per-window evaluation, event state machine, and time-based memory buffers.

    Task 2: Frames stored as CompressedFrame (JPEG in-memory bytes), not raw BGR arrays.
    Task 3: Pre-event and post-event collection are governed by PTS elapsed time.
    """

    def __init__(
        self,
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        fps: float = 25.0,            # advisory; used only for post-event fallback
        fight_threshold: float = 0.35,
        collapse_threshold: float = 0.35,
        jpeg_quality: int = 70,
        camera_id: str = "cam-01",
        alert_callback: Callable[[AlertEvent], None] | None = None,
        persistence_callback: Callable[["RealTimeEvent"], None] | None = None,
    ):
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self._fps = fps if fps > 0 else 25.0
        self.fight_threshold = fight_threshold
        self.collapse_threshold = collapse_threshold
        self.jpeg_quality = jpeg_quality
        self.camera_id = camera_id
        self.alert_callback = alert_callback
        # Fired after _close_active_event() with the CLOSED event (is_ready=True).
        # Used by inference_worker to trigger background persistence without
        # blocking alert delivery or frame decoding.
        self.persistence_callback: Callable[["RealTimeEvent"], None] | None = persistence_callback

        self.state = EventState.IDLE
        self.pre_buffer = TimedPreBuffer(
            window_seconds=pre_event_seconds,
            jpeg_quality=jpeg_quality,
        )
        self.active_event: RealTimeEvent | None = None
        self.events: list[RealTimeEvent] = []
        self.window_predictions: list[WindowPrediction] = []
        self.total_frames_pushed = 0
        # Monotonic-safe last known PTS (used for post-event fallback)
        self._last_pts: float = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    @fps.setter
    def fps(self, value: float) -> None:
        if value > 0:
            self._fps = value

    def push_frame(self, frame: np.ndarray, timestamp: float) -> float:
        """Compress and push an incoming frame into the rolling pre-event buffer.

        If an event is ACTIVE, also appends the CompressedFrame to the active
        event's incident buffer, classifying it as event or post-event frame
        based on PTS relative to last_anomalous_timestamp.

        Args:
            frame: Full-resolution BGR uint8 ndarray (any resolution).
            timestamp: PTS in seconds.

        Returns:
            Sanitised (monotonic) PTS actually used.
        """
        self.total_frames_pushed += 1
        # Push to time-based rolling pre-buffer; returns monotonic PTS
        sanitised_pts = self.pre_buffer.push(frame, timestamp)
        self._last_pts = sanitised_pts

        if self.active_event is not None and self.active_event.state == EventState.ACTIVE:
            # Classify as active-event frame or post-event frame
            cf = self.pre_buffer._buffer[-1]  # the frame just pushed (no copy)
            if sanitised_pts <= self.active_event.last_anomalous_timestamp:
                self.active_event.event_frames.append(cf)
            else:
                self.active_event.post_event_frames.append(cf)

            # Post-event closure: check using PTS elapsed time (Task 3)
            time_since = sanitised_pts - self.active_event.last_anomalous_timestamp
            if time_since >= self.post_event_seconds:
                self._close_active_event()

        return sanitised_pts

    def process_window(
        self,
        window_index: int,
        timestamp: float,
        normal_prob: float,
        fight_prob: float,
        collapse_prob: float,
    ) -> WindowPrediction:
        """Evaluate a completed 16-frame window for real-time anomaly detection.

        Alert is dispatched immediately on first anomaly; subsequent consecutive
        anomalies are merged into the existing event (no duplicate alerts).
        A new anomaly during post-event collection extends the same event and
        resets the post-event interval.
        """
        # 1. Threshold evaluation — exact order: Fight >= 0.35, Collapse >= 0.35, Normal
        if fight_prob >= self.fight_threshold:
            anomaly_type = "Fight"
            anomaly_conf = fight_prob
        elif collapse_prob >= self.collapse_threshold:
            anomaly_type = "Collapse"
            anomaly_conf = collapse_prob
        else:
            anomaly_type = None
            anomaly_conf = 0.0

        pred = WindowPrediction(
            window_index=window_index,
            timestamp=timestamp,
            normal_prob=normal_prob,
            fight_prob=fight_prob,
            collapse_prob=collapse_prob,
            anomaly_type=anomaly_type,
        )
        self.window_predictions.append(pred)

        # 2. State machine
        if anomaly_type is not None:
            if self.state == EventState.IDLE or self.active_event is None:
                # IDLE -> ACTIVE: create new incident
                self.state = EventState.ACTIVE
                event = RealTimeEvent(
                    event_id=f"evt-{uuid.uuid4().hex[:8]}",
                    event_type=anomaly_type,
                    state=EventState.ACTIVE,
                    first_detected_timestamp=timestamp,
                    first_detected_window=window_index,
                    last_anomalous_timestamp=timestamp,
                    last_anomalous_frame_idx=self.total_frames_pushed,
                    confidence=anomaly_conf,
                    camera_id=self.camera_id,
                    pre_event_frames=self.pre_buffer.snapshot(),  # shallow ref copy
                    event_frames=[],
                    post_event_frames=[],
                    is_ready=False,
                )
                self.active_event = event
                self.events.append(event)

                # Emit alert IMMEDIATELY — no wait for EOF or post-event
                if self.alert_callback is not None:
                    alert = AlertEvent.create(
                        event_type=anomaly_type,
                        confidence=anomaly_conf,
                        latency_ms=0.0,
                        details={
                            "event_id": event.event_id,
                            "window_index": str(window_index),
                            "timestamp": f"{timestamp:.3f}s",
                            "camera_id": self.camera_id,
                            "state": "ACTIVE",
                        },
                    )
                    self.alert_callback(alert)
            else:
                # Consecutive/returning anomaly: merge into active event
                self.active_event.last_anomalous_timestamp = timestamp
                self.active_event.last_anomalous_frame_idx = self.total_frames_pushed
                if anomaly_conf > self.active_event.confidence:
                    self.active_event.confidence = anomaly_conf
                # Re-classify any post-event frames accumulated during lull
                if self.active_event.post_event_frames:
                    self.active_event.event_frames.extend(self.active_event.post_event_frames)
                    self.active_event.post_event_frames.clear()
        else:
            # Non-anomalous window — check post-event timeout by PTS (Task 3)
            if self.active_event is not None and self.active_event.state == EventState.ACTIVE:
                time_since = timestamp - self.active_event.last_anomalous_timestamp
                if time_since >= self.post_event_seconds:
                    self._close_active_event()

        return pred

    def _close_active_event(self) -> None:
        """Close the active event, mark buffer ready, reset to IDLE.

        Fires persistence_callback (if set) AFTER is_ready is True, so the
        callback receives a fully-assembled CLOSED event.
        """
        if self.active_event is not None:
            self.active_event.state = EventState.CLOSED
            self.active_event.is_ready = True
            closed_event = self.active_event  # hold reference before clearing
            self.active_event = None
            self.state = EventState.IDLE
            # Fire persistence callback AFTER state is fully committed
            if self.persistence_callback is not None:
                self.persistence_callback(closed_event)

    def on_eof(self) -> None:
        """Handle video EOF, concluding any in-flight active event."""
        if self.active_event is not None:
            self._close_active_event()

    def get_events(self) -> list[RealTimeEvent]:
        return list(self.events)

    def get_window_predictions(self) -> list[WindowPrediction]:
        return list(self.window_predictions)

    def clear(self) -> None:
        """Evaporate all compressed frame references from all buffers."""
        self.pre_buffer.clear()
        for evt in self.events:
            evt.pre_event_frames.clear()
            evt.event_frames.clear()
            evt.post_event_frames.clear()
        self.active_event = None
        self.state = EventState.IDLE
