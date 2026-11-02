# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Session lifecycle and resource manager for Sterile Room processing.

RESPONSIBILITIES:
- Orchestrate session initialization, in-memory decoding, buffer feeding,
  inference dispatch, and post-session sanitation.
- Track transient memory allocations and enforce session teardown.
- Perform real-time anomaly detection and event management during stream decoding.
- Retain rolling pre-event (5s) and post-event (5s) incident frames in volatile RAM.
- Ensure that no raw video frames remain in user-accessible scope after session end.

EXECUTION FLOW:
    Video bytes -> in-memory PyAV decoding -> RealTimeEventManager (RAM) + VolatileBuffer (RAM) ->
    RGB preprocessing -> MobileNetV2 + LSTM -> real-time alert + per-window prediction ->
    video-level aggregation -> cleanup.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
import time
from typing import Any, Callable
import uuid

# Ensure PrivWatch root is on sys.path
_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import av
import numpy as np
import torch

try:
    from privwatch.config import (
        COLLAPSE_THRESHOLD,
        DEVICE,
        FIGHT_THRESHOLD,
        IMG_SIZE,
        SEQ_LEN,
    )
    from privwatch.inference import (
        decode_frames_from_bytes,
        decode_frames_with_timestamps,
        load_model,
    )
    from privwatch.model import ActionModel
    from privwatch.preprocessing import preprocess_clip, resize_frame
    from sterile_room.alert_registry import AlertEvent, AlertRegistry
    from sterile_room.event_manager import (
        EventState,
        RealTimeEvent,
        RealTimeEventManager,
        WindowPrediction,
    )
    from sterile_room.memory_sanitizer import MemorySanitizer
    from sterile_room.volatile_buffer import VolatileBuffer
except ImportError:
    from PrivWatch_Final.privwatch.config import (  # type: ignore
        COLLAPSE_THRESHOLD,
        DEVICE,
        FIGHT_THRESHOLD,
        IMG_SIZE,
        SEQ_LEN,
    )
    from PrivWatch_Final.privwatch.inference import (  # type: ignore
        decode_frames_from_bytes,
        decode_frames_with_timestamps,
        load_model,
    )
    from PrivWatch_Final.privwatch.model import ActionModel  # type: ignore
    from PrivWatch_Final.privwatch.preprocessing import preprocess_clip, resize_frame  # type: ignore
    from PrivWatch_Final.sterile_room.alert_registry import AlertEvent, AlertRegistry  # type: ignore
    from PrivWatch_Final.sterile_room.event_manager import (  # type: ignore
        EventState,
        RealTimeEvent,
        RealTimeEventManager,
        WindowPrediction,
    )
    from PrivWatch_Final.sterile_room.memory_sanitizer import MemorySanitizer  # type: ignore
    from PrivWatch_Final.sterile_room.volatile_buffer import VolatileBuffer  # type: ignore


class SterileSession:
    """Manages an active sterile processing session in volatile memory."""

    def __init__(
        self,
        session_id: str | None = None,
        buffer_capacity: int = SEQ_LEN,
        model: ActionModel | None = None,
        device: torch.device = DEVICE,
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        jpeg_quality: int = 70,
        camera_id: str = "cam-01",
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.jpeg_quality = jpeg_quality
        self.buffer = VolatileBuffer(capacity=buffer_capacity)
        self.registry = AlertRegistry()
        self.event_manager = RealTimeEventManager(
            pre_event_seconds=pre_event_seconds,
            post_event_seconds=post_event_seconds,
            jpeg_quality=jpeg_quality,
            fight_threshold=FIGHT_THRESHOLD,
            collapse_threshold=COLLAPSE_THRESHOLD,
            camera_id=camera_id,
            alert_callback=self.registry.record_alert,
        )
        self.model = model
        self.device = device
        self.is_active = True

    def process_video_bytes(
        self,
        video_bytes: bytes,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Decode in-memory video stream, feed volatile buffer, and run inference.

        Decodes frames entirely in-memory using PyAV without touching disk storage.
        Accumulates temporal sequences in a circular volatile buffer and performs
        per-window real-time anomaly detection and zero-disk activity classification.

        Args:
            video_bytes: In-memory byte stream of uploaded video.
            progress_callback: Optional non-blocking callback for streaming progress updates.

        Returns:
            Detailed metrics, real-time events, and classification decision dictionary.
        """
        if not self.is_active:
            raise RuntimeError("Cannot process video on an inactive or closed SterileSession.")

        if self.model is None:
            self.model, metadata = load_model(device=self.device)
        else:
            metadata = {}

        # Probe container metadata for reliable total frame count
        total_expected_frames: int | None = None
        try:
            with av.open(io.BytesIO(video_bytes)) as probe_container:
                if probe_container.streams.video:
                    vs = probe_container.streams.video[0]
                    if vs.frames and vs.frames > 0:
                        total_expected_frames = int(vs.frames)
        except Exception:
            total_expected_frames = None

        start_time = time.time()
        fight_probs: list[float] = []
        normal_probs: list[float] = []
        collapse_probs: list[float] = []
        total_frames = 0
        current_lifecycle_state = "Monitoring"

        try:
            for frame, timestamp in decode_frames_with_timestamps(video_bytes):
                total_frames += 1

                # 1. Feed frame into real-time rolling buffer
                self.event_manager.push_frame(frame, timestamp)

                # 2. Resize and push into sliding window buffer
                resized = resize_frame(frame, IMG_SIZE)
                self.buffer.push(resized)

                # 3. Sliding-window inference when buffer reaches SEQ_LEN
                if self.buffer.is_ready():
                    window_idx = total_frames - SEQ_LEN
                    window = self.buffer.get_window()
                    clip = preprocess_clip(window, device=self.device)

                    with torch.no_grad():
                        outputs = self.model(clip)
                        probs = torch.softmax(outputs, dim=1)[0]
                        p_normal = probs[0].item()
                        p_fight = probs[1].item()
                        p_collapse = probs[2].item()

                        fight_probs.append(p_fight)
                        normal_probs.append(p_normal)
                        collapse_probs.append(p_collapse)

                    # 4. Immediate real-time window evaluation & alert trigger
                    pred = self.event_manager.process_window(
                        window_index=window_idx,
                        timestamp=timestamp,
                        normal_prob=p_normal,
                        fight_prob=p_fight,
                        collapse_prob=p_collapse,
                    )

                    # Update event lifecycle state
                    if pred.anomaly_type is not None:
                        current_lifecycle_state = "Anomaly detected"
                    elif self.event_manager.active_event is not None and self.event_manager.active_event.state == EventState.ACTIVE:
                        current_lifecycle_state = "Capturing post-event footage"
                    elif any(e.is_ready for e in self.event_manager.get_events()):
                        current_lifecycle_state = "Incident buffer ready"
                    else:
                        current_lifecycle_state = "Monitoring"

                # 5. Dispatch non-blocking progress update
                if progress_callback:
                    progress_callback({
                        "current_frame": total_frames,
                        "total_frames": total_expected_frames,
                        "timestamp": round(timestamp, 3),
                        "lifecycle_state": current_lifecycle_state,
                    })

            # Signal video EOF to event manager
            self.event_manager.on_eof()

            if progress_callback:
                progress_callback({
                    "current_frame": total_frames,
                    "total_frames": total_expected_frames,
                    "timestamp": round(timestamp if 'timestamp' in locals() else 0.0, 3),
                    "lifecycle_state": "Analysis completed",
                })

        finally:
            self.buffer.clear()
            MemorySanitizer.purge_runtime_caches()

        latency = time.time() - start_time

        if fight_probs:
            mean_fight = float(sum(fight_probs) / len(fight_probs))
            mean_normal = float(sum(normal_probs) / len(normal_probs))
            mean_collapse = float(sum(collapse_probs) / len(collapse_probs))

            if mean_fight >= FIGHT_THRESHOLD:
                decision = "ALERT - Fight detected"
                predicted_class = "Fight"
            elif mean_collapse >= COLLAPSE_THRESHOLD:
                decision = "ALERT - Collapse detected"
                predicted_class = "Collapse"
            else:
                decision = "Normal activity"
                predicted_class = "Normal"

            confidences = {
                "Normal": mean_normal,
                "Fight": mean_fight,
                "Collapse": mean_collapse,
            }

            return {
                "session_id": self.session_id,
                "decision": decision,
                "predicted_class": predicted_class,
                "confidences": confidences,
                "windows_processed": len(fight_probs),
                "total_frames": total_frames,
                "latency_seconds": round(latency, 2),
                "model_metadata": metadata,
                "alerts": self.registry.get_alerts(),
                "events": [evt.to_dict() for evt in self.event_manager.get_events()],
                "raw_events": self.event_manager.get_events(),
                "window_predictions": [pred.to_dict() for pred in self.event_manager.get_window_predictions()],
            }

        return {
            "session_id": self.session_id,
            "decision": "No frames processed",
            "predicted_class": None,
            "confidences": {"Normal": 0.0, "Fight": 0.0, "Collapse": 0.0},
            "windows_processed": 0,
            "total_frames": total_frames,
            "latency_seconds": round(latency, 2),
            "model_metadata": metadata,
            "alerts": [],
            "events": [],
            "raw_events": [],
            "window_predictions": [],
        }

    def close(self) -> None:
        """Evaporate session buffers, event frames, and purge runtime caches."""
        self.buffer.clear()
        self.event_manager.clear()
        MemorySanitizer.purge_runtime_caches()
        self.is_active = False
