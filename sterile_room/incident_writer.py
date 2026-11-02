# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Incident writer: encodes a CLOSED RealTimeEvent's frame buffers to MP4 + JSON sidecar.

Design:
- Atomic write: all output goes to <name>.tmp.* files first, then both are renamed
  to their final names only after both succeed.  A partial write leaves no
  valid-looking incident pair.
- Deduplication: once an event_id has been written it is never written again,
  even if write_incident() is called concurrently from multiple threads.
- Never writes before event.state == EventState.CLOSED and event.is_ready is True.
- On encoding failure the .tmp files are removed and an IncidentRecord with
  success=False is returned — no crash, no silent data loss.
- Memory release: on successful write the three frame-buffer lists on the event
  object are cleared to free JPEG bytes from RAM.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    import av  # type: ignore
    import numpy as np
    from sterile_room.compressed_frame import CompressedFrame
    from sterile_room.event_manager import EventState, RealTimeEvent
except ImportError:
    import av  # type: ignore
    import numpy as np
    from PrivWatch_Final.sterile_room.compressed_frame import CompressedFrame  # type: ignore
    from PrivWatch_Final.sterile_room.event_manager import EventState, RealTimeEvent  # type: ignore


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class IncidentRecord:
    """Result of a single incident persistence attempt."""

    event_id: str
    camera_id: str
    anomaly_class: str
    confidence: float
    start_timestamp: float
    end_timestamp: float
    clip_duration: float
    pre_event_duration: float
    post_event_duration: float
    frame_count: int
    created_at: str
    mp4_path: str
    json_path: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "anomaly_class": self.anomaly_class,
            "confidence": round(self.confidence, 4),
            "start_timestamp": round(self.start_timestamp, 3),
            "end_timestamp": round(self.end_timestamp, 3),
            "clip_duration": round(self.clip_duration, 3),
            "pre_event_duration": round(self.pre_event_duration, 3),
            "post_event_duration": round(self.post_event_duration, 3),
            "frame_count": self.frame_count,
            "created_at": self.created_at,
            "mp4_path": self.mp4_path,
            "json_path": self.json_path,
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Incident writer
# ---------------------------------------------------------------------------

_DEFAULT_INCIDENTS_DIR = Path("incidents")


class IncidentWriter:
    """Encodes CLOSED event frame buffers to incident MP4 + JSON sidecar.

    Thread-safe.  Uses an internal lock + a written-IDs set to deduplicate.
    """

    def __init__(
        self,
        incidents_dir: Path = _DEFAULT_INCIDENTS_DIR,
        fps: float = 25.0,
    ) -> None:
        self.incidents_dir = Path(incidents_dir)
        self.fps = fps if fps > 0 else 25.0
        self._written_ids: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_write(self, event: RealTimeEvent) -> bool:
        """Return True when the event is CLOSED, ready, and not yet written."""
        if event.state != EventState.CLOSED:
            return False
        if not event.is_ready:
            return False
        with self._lock:
            return event.event_id not in self._written_ids

    def write_incident(self, event: RealTimeEvent) -> IncidentRecord:
        """Encode and persist incident clip + JSON for a CLOSED event.

        Thread-safe. Atomic: writes to .tmp files, validates, then renames.
        Clears frame buffers from the event on success to release RAM.

        Returns:
            IncidentRecord with success=True on success, success=False on failure.
        """
        # --- guard: only CLOSED events ----------------------------------------
        if event.state != EventState.CLOSED or not event.is_ready:
            return IncidentRecord(
                event_id=event.event_id,
                camera_id=event.camera_id,
                anomaly_class=event.event_type,
                confidence=event.confidence,
                start_timestamp=event.first_detected_timestamp,
                end_timestamp=event.last_anomalous_timestamp,
                clip_duration=0.0,
                pre_event_duration=0.0,
                post_event_duration=0.0,
                frame_count=0,
                created_at=_utc_now(),
                mp4_path="",
                json_path="",
                success=False,
                error="Event is not CLOSED or not ready",
            )

        # --- deduplication guard ----------------------------------------------
        with self._lock:
            if event.event_id in self._written_ids:
                return IncidentRecord(
                    event_id=event.event_id,
                    camera_id=event.camera_id,
                    anomaly_class=event.event_type,
                    confidence=event.confidence,
                    start_timestamp=event.first_detected_timestamp,
                    end_timestamp=event.last_anomalous_timestamp,
                    clip_duration=0.0,
                    pre_event_duration=0.0,
                    post_event_duration=0.0,
                    frame_count=0,
                    created_at=_utc_now(),
                    mp4_path="",
                    json_path="",
                    success=False,
                    error="Already written (duplicate)",
                )
            # Mark as in-progress by adding to written set early
            # so concurrent callers do not also try to write this event.
            self._written_ids.add(event.event_id)

        # --- assemble chronological frame list --------------------------------
        # Shallow snapshot copy to guarantee thread safety
        pre_frames = list(event.pre_event_frames)
        evt_frames = list(event.event_frames)
        post_frames = list(event.post_event_frames)

        all_frames: list[CompressedFrame] = sorted(
            pre_frames + evt_frames + post_frames,
            key=lambda cf: cf.timestamp,
        )

        if not all_frames:
            # Edge case: event closed with zero frames (e.g. very short video)
            all_frames = []

        # --- timing stats -----------------------------------------------------
        pre_ts = [cf.timestamp for cf in pre_frames]
        post_ts = [cf.timestamp for cf in post_frames]
        pre_dur = (max(pre_ts) - min(pre_ts)) if len(pre_ts) >= 2 else 0.0
        post_dur = (max(post_ts) - min(post_ts)) if len(post_ts) >= 2 else 0.0
        start_ts = all_frames[0].timestamp if all_frames else event.first_detected_timestamp
        end_ts = all_frames[-1].timestamp if all_frames else event.last_anomalous_timestamp
        clip_dur = end_ts - start_ts

        # --- file paths -------------------------------------------------------
        eid = event.event_id
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        tmp_mp4 = self.incidents_dir / f"incident_{eid}.tmp.mp4"
        tmp_json = self.incidents_dir / f"incident_{eid}.tmp.json"
        final_mp4 = self.incidents_dir / f"incident_{eid}.mp4"
        final_json = self.incidents_dir / f"incident_{eid}.json"
        created_at = _utc_now()

        # --- encode -----------------------------------------------------------
        try:
            self._encode_frames_to_mp4(all_frames, tmp_mp4)
        except Exception as exc:  # noqa: BLE001
            _safe_remove(tmp_mp4)
            _safe_remove(tmp_json)
            return IncidentRecord(
                event_id=event.event_id,
                camera_id=event.camera_id,
                anomaly_class=event.event_type,
                confidence=event.confidence,
                start_timestamp=event.first_detected_timestamp,
                end_timestamp=event.last_anomalous_timestamp,
                clip_duration=clip_dur,
                pre_event_duration=pre_dur,
                post_event_duration=post_dur,
                frame_count=len(all_frames),
                created_at=created_at,
                mp4_path=str(final_mp4),
                json_path=str(final_json),
                success=False,
                error=f"MP4 encoding failed: {exc}",
            )

        # --- validate MP4 exists and is non-empty ----------------------------
        if not tmp_mp4.exists() or tmp_mp4.stat().st_size == 0:
            _safe_remove(tmp_mp4)
            _safe_remove(tmp_json)
            return IncidentRecord(
                event_id=event.event_id,
                camera_id=event.camera_id,
                anomaly_class=event.event_type,
                confidence=event.confidence,
                start_timestamp=event.first_detected_timestamp,
                end_timestamp=event.last_anomalous_timestamp,
                clip_duration=clip_dur,
                pre_event_duration=pre_dur,
                post_event_duration=post_dur,
                frame_count=len(all_frames),
                created_at=created_at,
                mp4_path=str(final_mp4),
                json_path=str(final_json),
                success=False,
                error="MP4 temp file is empty after encoding",
            )

        # --- write JSON sidecar -----------------------------------------------
        record_data: dict[str, Any] = {
            "event_id": event.event_id,
            "camera_id": event.camera_id,
            "anomaly_class": event.event_type,
            "confidence": round(event.confidence, 4),
            "first_detected_timestamp": round(event.first_detected_timestamp, 3),
            "last_anomalous_timestamp": round(event.last_anomalous_timestamp, 3),
            "start_timestamp": round(start_ts, 3),
            "end_timestamp": round(end_ts, 3),
            "clip_duration_seconds": round(clip_dur, 3),
            "pre_event_duration_seconds": round(pre_dur, 3),
            "post_event_duration_seconds": round(post_dur, 3),
            "pre_event_frame_count": len(pre_frames),
            "event_frame_count": len(evt_frames),
            "post_event_frame_count": len(post_frames),
            "total_frame_count": len(all_frames),
            "mp4_path": str(final_mp4),
            "created_at": created_at,
            "schema_version": "1.0",
        }

        try:
            self._write_json(record_data, tmp_json)
        except Exception as exc:  # noqa: BLE001
            _safe_remove(tmp_mp4)
            _safe_remove(tmp_json)
            return IncidentRecord(
                event_id=event.event_id,
                camera_id=event.camera_id,
                anomaly_class=event.event_type,
                confidence=event.confidence,
                start_timestamp=event.first_detected_timestamp,
                end_timestamp=event.last_anomalous_timestamp,
                clip_duration=clip_dur,
                pre_event_duration=pre_dur,
                post_event_duration=post_dur,
                frame_count=len(all_frames),
                created_at=created_at,
                mp4_path=str(final_mp4),
                json_path=str(final_json),
                success=False,
                error=f"JSON sidecar write failed: {exc}",
            )

        # --- atomic rename (both files must succeed before either is exposed) -
        try:
            os.replace(str(tmp_mp4), str(final_mp4))
            os.replace(str(tmp_json), str(final_json))
        except Exception as exc:  # noqa: BLE001
            _safe_remove(tmp_mp4)
            _safe_remove(tmp_json)
            _safe_remove(final_mp4)
            _safe_remove(final_json)
            return IncidentRecord(
                event_id=event.event_id,
                camera_id=event.camera_id,
                anomaly_class=event.event_type,
                confidence=event.confidence,
                start_timestamp=event.first_detected_timestamp,
                end_timestamp=event.last_anomalous_timestamp,
                clip_duration=clip_dur,
                pre_event_duration=pre_dur,
                post_event_duration=post_dur,
                frame_count=len(all_frames),
                created_at=created_at,
                mp4_path=str(final_mp4),
                json_path=str(final_json),
                success=False,
                error=f"Atomic rename failed: {exc}",
            )

        # --- success: release frame RAM ---------------------------------------
        event.pre_event_frames.clear()
        event.event_frames.clear()
        event.post_event_frames.clear()

        return IncidentRecord(
            event_id=event.event_id,
            camera_id=event.camera_id,
            anomaly_class=event.event_type,
            confidence=event.confidence,
            start_timestamp=event.first_detected_timestamp,
            end_timestamp=event.last_anomalous_timestamp,
            clip_duration=clip_dur,
            pre_event_duration=pre_dur,
            post_event_duration=post_dur,
            frame_count=len(all_frames),
            created_at=created_at,
            mp4_path=str(final_mp4),
            json_path=str(final_json),
            success=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_frames_to_mp4(
        self,
        all_frames: list[CompressedFrame],
        output_path: Path,
    ) -> None:
        """Encode a list of CompressedFrame objects to an MP4 at output_path using PyAV.

        Args:
            all_frames: Chronologically ordered CompressedFrame list.
            output_path: Destination path for the .tmp.mp4 file.

        Raises:
            RuntimeError: If av.open() fails or if any frame cannot be decoded.
        """
        if not all_frames:
            # Write a valid empty-ish MP4 (0-frame container) so callers still
            # get a file to validate.  This is an edge case; the size check
            # downstream will catch it.
            container = av.open(str(output_path), mode="w")
            container.close()
            return

        # Determine frame dimensions from the first decodable frame
        first_bgr = all_frames[0].decode()
        h, w = first_bgr.shape[:2]

        container = av.open(str(output_path), mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(self.fps))
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            # Ensure wide compatibility
            stream.options = {"crf": "23", "preset": "fast"}

            for cf in all_frames:
                bgr = cf.decode()
                # av.VideoFrame expects RGB
                rgb = bgr[:, :, ::-1]
                av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for pkt in stream.encode(av_frame):
                    container.mux(pkt)

            # Flush encoder
            for pkt in stream.encode():
                container.mux(pkt)
        finally:
            container.close()

    def _write_json(self, record_data: dict[str, Any], path: Path) -> None:
        """Write record_data as a pretty-printed JSON file at path."""
        path.write_text(json.dumps(record_data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_remove(path: Path) -> None:
    """Remove a file without raising if it does not exist."""
    try:
        if path.exists():
            path.unlink()
    except Exception:  # noqa: BLE001
        pass

