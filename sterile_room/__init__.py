# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Sterile Room: Privacy-preserving volatile execution boundary for PrivWatch."""

from pathlib import Path
import sys

_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from sterile_room.alert_registry import AlertEvent, AlertRegistry
    from sterile_room.compressed_frame import CompressedFrame
    from sterile_room.event_manager import (
        EventState,
        RealTimeEvent,
        RealTimeEventManager,
        TimedPreBuffer,
        WindowPrediction,
    )
    from sterile_room.incident_writer import IncidentRecord, IncidentWriter
    from sterile_room.inference_worker import MsgType, SterileInferenceWorker, WorkerMessage
    from sterile_room.memory_sanitizer import MemorySanitizer
    from sterile_room.sterile_session import SterileSession
    from sterile_room.streamlit_integration import StreamlitSterileAdapter
    from sterile_room.volatile_buffer import VolatileBuffer
except ImportError:
    from PrivWatch_Final.sterile_room.alert_registry import AlertEvent, AlertRegistry  # type: ignore
    from PrivWatch_Final.sterile_room.compressed_frame import CompressedFrame  # type: ignore
    from PrivWatch_Final.sterile_room.event_manager import (  # type: ignore
        EventState,
        RealTimeEvent,
        RealTimeEventManager,
        TimedPreBuffer,
        WindowPrediction,
    )
    from PrivWatch_Final.sterile_room.incident_writer import IncidentRecord, IncidentWriter  # type: ignore
    from PrivWatch_Final.sterile_room.inference_worker import MsgType, SterileInferenceWorker, WorkerMessage  # type: ignore
    from PrivWatch_Final.sterile_room.memory_sanitizer import MemorySanitizer  # type: ignore
    from PrivWatch_Final.sterile_room.sterile_session import SterileSession  # type: ignore
    from PrivWatch_Final.sterile_room.streamlit_integration import StreamlitSterileAdapter  # type: ignore
    from PrivWatch_Final.sterile_room.volatile_buffer import VolatileBuffer  # type: ignore

__all__ = [
    "AlertEvent",
    "AlertRegistry",
    "CompressedFrame",
    "EventState",
    "IncidentRecord",
    "IncidentWriter",
    "MemorySanitizer",
    "MsgType",
    "RealTimeEvent",
    "RealTimeEventManager",
    "SterileInferenceWorker",
    "SterileSession",
    "StreamlitSterileAdapter",
    "TimedPreBuffer",
    "VolatileBuffer",
    "WindowPrediction",
    "WorkerMessage",
]
