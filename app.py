# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""PrivWatch Streamlit Demonstration Application.

Integrates the MobileNetV2 + LSTM video anomaly detection model with real-time
confidence visualization, forensic metadata inspection, and disk I/O monitoring.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from typing import Any

# Handle SIGINT gracefully if running in the main thread
try:
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
except (ValueError, AttributeError):
    pass

# Ensure package directory is on sys.path for standalone or subfolder execution
_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import cv2  # type: ignore # noqa: E402
import psutil  # type: ignore # noqa: E402
import streamlit as st  # type: ignore # noqa: E402

try:
    from privwatch.config import (  # type: ignore # noqa: E402
        COLLAPSE_THRESHOLD,
        DEVICE,
        FIGHT_THRESHOLD,
    )
    from privwatch.inference import (  # type: ignore # noqa: E402
        load_model,
        run_inference_detailed,
    )
    from privwatch.paths import RAW_MODEL_PATH  # type: ignore # noqa: E402
    from sterile_room.inference_worker import MsgType, SterileInferenceWorker  # type: ignore # noqa: E402
except ImportError:
    from PrivWatch_Final.privwatch.config import (  # type: ignore # noqa: E402
        COLLAPSE_THRESHOLD,
        DEVICE,
        FIGHT_THRESHOLD,
    )
    from PrivWatch_Final.privwatch.inference import (  # type: ignore # noqa: E402
        load_model,
        run_inference_detailed,
    )
    from PrivWatch_Final.privwatch.paths import RAW_MODEL_PATH  # type: ignore # noqa: E402
    from PrivWatch_Final.sterile_room.inference_worker import MsgType, SterileInferenceWorker  # type: ignore # noqa: E402

WATCH_DIR = Path("./temp_monitor")
WATCH_DIR.mkdir(exist_ok=True)


def remove_temp_file(path: str | None) -> None:
    """Safely remove a temporary file with retry handling for Windows file locks."""
    if not path:
        return
    for _ in range(10):
        try:
            if os.path.exists(path):
                os.remove(path)
                break
        except PermissionError:
            time.sleep(0.1)
        except Exception:
            break


def list_files() -> set[str]:
    """List absolute paths of active monitoring files."""
    return {str(p.resolve()) for p in WATCH_DIR.glob("*") if p.is_file()}


def snapshot() -> dict:
    """Capture snapshot of monitored directory."""
    return {"files": list_files(), "time": time.time()}


def diff_files(before: dict, after: dict) -> list[str]:
    """Return new files created between snapshots."""
    return sorted(list(after["files"] - before["files"]))


def io_snapshot() -> dict:
    """Capture disk I/O counters."""
    io = psutil.disk_io_counters()
    if io is None:
        return {"read_bytes": 0, "write_bytes": 0}
    return {"read_bytes": io.read_bytes, "write_bytes": io.write_bytes}


def io_diff(before: dict, after: dict) -> dict:
    """Calculate disk I/O byte change."""
    return {
        "read_bytes": after["read_bytes"] - before["read_bytes"],
        "write_bytes": after["write_bytes"] - before["write_bytes"],
    }


@st.cache_resource
def get_cached_model():
    """Cache loaded model in Streamlit session memory to avoid re-instantiation."""
    return load_model(RAW_MODEL_PATH, device=DEVICE)


def _render_final_results(results: dict, is_sterile: bool, new_files: list, io_change: dict) -> None:
    """Render final inference results, anomaly events, and storage audit."""
    decision = results["decision"]
    confidences = results["confidences"]

    if "Fight" in decision:
        st.error(f"### 🚨 {decision}")
    elif "Collapse" in decision:
        st.warning(f"### ⚠️ {decision}")
    else:
        st.success(f"### ✅ {decision}")

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Normal", f"{confidences['Normal']:.2%}")
    mcol2.metric("Fight", f"{confidences['Fight']:.2%}")
    mcol3.metric("Collapse", f"{confidences['Collapse']:.2%}")

    st.write(
        f"**Latency:** `{results['latency_seconds']:.2f}s` | "
        f"**Windows Processed:** `{results['windows_processed']}` | "
        f"**Total Frames:** `{results['total_frames']}`"
    )

    if is_sterile and results.get("events"):
        st.divider()
        st.subheader("⏱️ Real-Time Anomaly Incidents (Phase 2A)")
        for evt in results["events"]:
            icon = "🚨" if evt["event_type"] == "Fight" else "⚠️"
            st.markdown(
                f"• **{icon} {evt['event_type']} Anomaly** (`{evt['event_id']}`) — "
                f"First Detected: `{evt['first_detected_timestamp']}s` (Window #{evt['first_detected_window']}) | "
                f"Peak Confidence: `{evt['confidence']:.2%}` | "
                f"RAM Incident Frames: `{evt['total_frames_count']}` ({evt['pre_event_frames_count']} pre-event) | "
                f"Status: `{evt['state']}`"
            )

    st.divider()
    st.subheader("🛡️ Privacy & Storage Audit")
    if is_sterile:
        st.success("🔒 **Sterile Execution Mode:** Video decoded directly in volatile RAM via PyAV. Zero temporary files created on disk.")
    else:
        st.info("💾 **Raw Execution Mode:** Persistent disk retention active. Video file retained on disk.")

    st.write(f"• **New Files Created:** `{len(new_files)}`")
    st.write(f"• **Disk Write:** `{io_change['write_bytes']:,} bytes`")
    st.write(f"• **Disk Read:** `{io_change['read_bytes']:,} bytes`")


def render_dual_output(
    alerts: list[Any],
    prog: dict[str, Any] | None,
    is_done: bool,
    result: dict[str, Any] | None = None,
    new_files: list[str] | None = None,
    io_change: dict[str, int] | None = None,
    incidents: list[Any] | None = None,
) -> None:
    """Render distinct outputs for Live Anomaly Monitoring and Final Video Analysis.

    Strictly separates provisional window-level detections from aggregated final classification.
    Keeps both visible after video EOF until a different video is uploaded.
    """
    # -----------------------------------------------------------------------
    # Section A: Live Anomaly Monitoring ("Provisional Real-Time Detection")
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📡 Live Anomaly Monitoring")
    st.info("**Output Category:** Provisional Real-Time Detection")
    st.caption(
        "⚠️ *Provisional detections identify instantaneous anomalous activity in localized "
        "16-frame sliding windows during active stream decoding. They provide immediate early warning "
        "and do not alter or overwrite the final video-level mean classification.*"
    )

    # 1. Event Lifecycle Status
    lifecycle_state = (
        "Analysis completed"
        if is_done
        else (prog.get("lifecycle_state", "Monitoring") if prog else "Monitoring")
    )
    state_icons = {
        "Monitoring": "🟢",
        "Anomaly detected": "🚨",
        "Capturing post-event footage": "⏳",
        "Incident buffer ready": "📦",
        "Analysis completed": "🏁",
    }
    icon = state_icons.get(lifecycle_state, "ℹ️")
    st.markdown(f"**Event Lifecycle Status:** {icon} `{lifecycle_state}`")

    # 2. Progress Indicator (during active decoding)
    if not is_done:
        if prog:
            cur_f = prog.get("current_frame", 0)
            tot_f = prog.get("total_frames")
            cur_ts = prog.get("timestamp", 0.0)
            if tot_f and tot_f > 0:
                pct = min(max(cur_f / tot_f, 0.0), 1.0)
                st.progress(
                    pct,
                    text=f"Frame {cur_f} / {tot_f} | Video Timestamp: {cur_ts:.2f}s ({pct:.0%})",
                )
            else:
                st.info(f"⏳ Processing Frame {cur_f} (Video Timestamp: {cur_ts:.2f}s) — Total frames indeterminate")
        else:
            st.info("⏳ Initializing volatile pipeline and loading first video frames...")

    # 3. Provisional Alerts List
    if alerts:
        st.markdown("**Detected Real-Time Anomalies:**")
        for alert in alerts:
            a_icon = "🚨" if alert.event_type == "Fight" else "⚠️"
            det = alert.details or {}
            st.markdown(
                f"• {a_icon} **{alert.event_type} (Provisional)** | "
                f"Camera: `{det.get('camera_id', 'cam-01')}` | "
                f"Timestamp: `{det.get('timestamp', '?')}s` | "
                f"Confidence: `{alert.confidence:.2%}` | "
                f"Window #{det.get('window_index', '?')} | "
                f"Event ID: `{det.get('event_id', alert.event_id)[:8]}`"
            )
    else:
        if is_done:
            st.success("✅ No provisional anomalies detected in temporal sliding windows.")

    # -----------------------------------------------------------------------
    # Section B: Final Video Analysis ("Final Video-Level Classification")
    # -----------------------------------------------------------------------
    if is_done and result is not None:
        st.markdown("---")
        st.subheader("📋 Final Video Analysis")
        st.success("**Output Category:** Final Video-Level Classification")
        st.caption(
            "🔒 *Final classification is determined strictly by video-level mean probability "
            "aggregation across all sliding windows, evaluated against established decision thresholds.*"
        )
        _render_final_results(
            result,
            is_sterile=True,
            new_files=new_files or [],
            io_change=io_change or {"read_bytes": 0, "write_bytes": 0},
        )

    # -----------------------------------------------------------------------
    # Section C: Incident Persistence Status
    # -----------------------------------------------------------------------
    if incidents:
        st.markdown("---")
        st.subheader("💾 Incident Persistence")
        for record in incidents:
            if record.success:
                mp4_name = Path(record.mp4_path).name
                st.success(
                    f"✅ Incident saved: `{mp4_name}` | "
                    f"{record.anomaly_class} | conf: {record.confidence:.2%} | "
                    f"duration: {record.clip_duration:.2f}s | frames: {record.frame_count}"
                )
            else:
                st.error(
                    f"❌ Persistence failed for event `{record.event_id[:8]}`: {record.error}"
                )


@st.fragment(run_every=0.5)
def render_sterile_live_view() -> None:
    """Streamlit fragment polling active worker queue and updating session state."""
    w: SterileInferenceWorker | None = st.session_state.get("sterile_worker")
    if w is not None:
        msgs = w.drain_alerts()
        seen_ids = {a.event_id for a in st.session_state.get("sterile_alerts", [])}
        for msg in msgs:
            if msg.msg_type == MsgType.ALERT:
                if msg.payload.event_id not in seen_ids:
                    st.session_state.setdefault("sterile_alerts", []).append(msg.payload)
                    seen_ids.add(msg.payload.event_id)
            elif msg.msg_type == MsgType.PROGRESS:
                st.session_state["sterile_progress"] = msg.payload
            elif msg.msg_type == MsgType.RESULT:
                st.session_state["sterile_result"] = msg.payload
            elif msg.msg_type == MsgType.INCIDENT:
                incidents = st.session_state.setdefault("sterile_incidents", [])
                incidents.append(msg.payload)
            elif msg.msg_type == MsgType.ERROR:
                st.session_state["sterile_error"] = str(msg.payload)
            elif msg.msg_type == MsgType.DONE:
                st.session_state["sterile_done"] = True
                # Clean up worker reference
                st.session_state.pop("sterile_worker", None)
                # Compute disk/file snapshot delta
                io_after = io_snapshot()
                files_after = snapshot()
                st.session_state["sterile_new_files"] = diff_files(
                    st.session_state.get("sterile_files_before", {"files": set(), "time": 0}),
                    files_after,
                )
                st.session_state["sterile_io_change"] = io_diff(
                    st.session_state.get("sterile_io_before", {"read_bytes": 0, "write_bytes": 0}),
                    io_after,
                )
                st.rerun()

    alerts = st.session_state.get("sterile_alerts", [])
    prog = st.session_state.get("sterile_progress")
    is_done = st.session_state.get("sterile_done", False)
    result = st.session_state.get("sterile_result")
    err = st.session_state.get("sterile_error")

    if err:
        st.error(f"Inference error: {err}")
        return

    render_dual_output(
        alerts=alerts,
        prog=prog,
        is_done=is_done,
        result=result,
        new_files=st.session_state.get("sterile_new_files", []),
        io_change=st.session_state.get("sterile_io_change", {"read_bytes": 0, "write_bytes": 0}),
        incidents=st.session_state.get("sterile_incidents", []),
    )


def main():
    st.set_page_config(
        page_title="PrivWatch - Privacy-Preserving Surveillance",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ PrivWatch: Video Anomaly Detection")
    st.markdown(
        "**RGB Video → MobileNetV2 → LSTM → Activity Classification (Normal, Fight, Collapse)**"
    )

    model = None
    metadata = {}

    # Sidebar: System & Forensic Metadata
    with st.sidebar:
        st.header("🔬 Model Forensics")
        try:
            model, metadata = get_cached_model()
            st.success("Model loaded successfully")
            st.metric("Architecture", "MobileNetV2 + LSTM")
            st.metric("Validation Accuracy", f"{metadata.get('val_accuracy', 0.8658):.2%}")
            st.caption(f"Trained Epoch: {metadata.get('epoch', 5)}")
            st.caption(f"Trained Timestamp: {metadata.get('timestamp', 'N/A')}")
            st.caption(f"Target Device: {DEVICE}")
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            model = None

        st.divider()
        st.header("⚙️ Detection Thresholds")
        st.write(f"• **Fight Threshold:** `{FIGHT_THRESHOLD:.2f}`")
        st.write(f"• **Collapse Threshold:** `{COLLAPSE_THRESHOLD:.2f}`")
        st.info("Averages probabilities across sliding 16-frame temporal windows.")

    # Main UI Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Video Ingestion")
        uploaded_file = st.file_uploader(
            "Upload surveillance video clip (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
        )
        execution_mode = st.radio(
            "Execution Pipeline Mode:",
            options=["RAW Pipeline (Standard Archiving)", "Sterile Pipeline (Transient RAM-Only)"],
            help="Raw simulates standard persistent video pipelines. Sterile simulates volatile, non-persistent processing.",
        )

    if uploaded_file is not None and model is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()
        prev_hash = st.session_state.get("upload_hash")
        current_path = st.session_state.get("video_path")
        is_sterile = "Sterile" in execution_mode

        # -------------------------------------------------------------------
        # Requirement 1: Clear previous results when genuinely a NEW upload
        # -------------------------------------------------------------------
        if prev_hash is not None and current_hash != prev_hash:
            # Cancel running background worker if any
            old_worker: SterileInferenceWorker | None = st.session_state.pop("sterile_worker", None)
            if old_worker is not None and old_worker.is_alive():
                old_worker.cancel()

            # Clear previous video's results, alerts, progress and event state
            st.session_state.pop("sterile_result", None)
            st.session_state.pop("sterile_alerts", None)
            st.session_state.pop("sterile_progress", None)
            st.session_state.pop("sterile_done", None)
            st.session_state.pop("sterile_error", None)
            st.session_state.pop("sterile_new_files", None)
            st.session_state.pop("sterile_io_change", None)
            st.session_state.pop("sterile_incidents", None)
            st.session_state.pop("raw_result", None)
            st.session_state.pop("raw_new_files", None)
            st.session_state.pop("raw_io_change", None)

        if is_sterile:
            # Sterile Mode: Eliminate persistent disk storage entirely
            if current_path:
                remove_temp_file(current_path)
                st.session_state.pop("video_path", None)

            st.session_state["video_bytes"] = file_bytes
            st.session_state["upload_hash"] = current_hash
            video_preview_source = file_bytes
        else:
            # RAW Mode: Standard persistent archiving (saves file to disk)
            if current_hash != prev_hash or not current_path or not os.path.exists(current_path):
                if current_path:
                    remove_temp_file(current_path)

                suffix = Path(uploaded_file.name).suffix or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(file_bytes)
                tmp.close()

                st.session_state["video_path"] = tmp.name
                st.session_state["video_bytes"] = file_bytes
                st.session_state["upload_hash"] = current_hash

            video_preview_source = st.session_state["video_path"]

        with col1:
            st.video(video_preview_source)

        with col2:
            st.subheader("2. Inference Execution")
            analyze_clicked = st.button("🚀 Analyze Video Feed", use_container_width=True)

            if analyze_clicked:
                # Clean monitor directory
                for f in WATCH_DIR.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass

                files_before = snapshot()
                io_before = io_snapshot()

                if is_sterile:
                    # Cancel any previously running worker
                    old_worker = st.session_state.pop("sterile_worker", None)
                    if old_worker is not None and old_worker.is_alive():
                        old_worker.cancel()

                    # Reset per-run analysis state
                    st.session_state["sterile_alerts"] = []
                    st.session_state["sterile_progress"] = None
                    st.session_state["sterile_result"] = None
                    st.session_state["sterile_done"] = False
                    st.session_state["sterile_error"] = None
                    st.session_state["sterile_incidents"] = []
                    st.session_state["sterile_files_before"] = files_before
                    st.session_state["sterile_io_before"] = io_before
                    st.session_state["sterile_new_files"] = []
                    st.session_state["sterile_io_change"] = {"read_bytes": 0, "write_bytes": 0}

                    worker = SterileInferenceWorker(
                        video_bytes=file_bytes,
                        model=model,
                        device=DEVICE,
                    )
                    st.session_state["sterile_worker"] = worker
                    worker.start()
                else:
                    # RAW Mode — synchronous disk inference
                    with st.spinner("Analyzing temporal frame sequences (Disk Video)..."):
                        raw_path = st.session_state.get("video_path")
                        if not raw_path or not os.path.exists(raw_path):
                            st.error("Persistent video file missing.")
                            return
                        cap = cv2.VideoCapture(raw_path)
                        if not cap.isOpened():
                            st.error("Failed to read video file.")
                            return
                        results = run_inference_detailed(cap, model=model, device=DEVICE)

                    files_after = snapshot()
                    io_after = io_snapshot()
                    st.session_state["raw_result"] = results
                    st.session_state["raw_new_files"] = diff_files(files_before, files_after)
                    st.session_state["raw_io_change"] = io_diff(io_before, io_after)

            # ---------------------------------------------------------------
            # Render Outputs: Retained across reruns until a new video is uploaded
            # ---------------------------------------------------------------
            if is_sterile:
                # If worker is actively running, render via auto-refreshing fragment
                if st.session_state.get("sterile_worker") is not None:
                    render_sterile_live_view()
                # If analysis completed or has stored results, render persistently
                elif st.session_state.get("sterile_done") and st.session_state.get("sterile_result"):
                    render_dual_output(
                        alerts=st.session_state.get("sterile_alerts", []),
                        prog=st.session_state.get("sterile_progress"),
                        is_done=True,
                        result=st.session_state.get("sterile_result"),
                        new_files=st.session_state.get("sterile_new_files", []),
                        io_change=st.session_state.get("sterile_io_change", {"read_bytes": 0, "write_bytes": 0}),
                        incidents=st.session_state.get("sterile_incidents", []),
                    )
                else:
                    st.info("ℹ️ Video loaded in Sterile Mode. Click **'🚀 Analyze Video Feed'** above to begin in-memory analysis.")
            else:
                # RAW Mode persistent output display
                raw_res = st.session_state.get("raw_result")
                if raw_res is not None:
                    st.markdown("---")
                    st.subheader("📋 Final Video Analysis")
                    st.success("**Output Category:** Final Video-Level Classification")
                    st.caption(
                        "💾 *Standard archiving pipeline with persistent video file on disk. "
                        "Final classification is determined by video-level mean probability aggregation.*"
                    )
                    _render_final_results(
                        raw_res,
                        is_sterile=False,
                        new_files=st.session_state.get("raw_new_files", []),
                        io_change=st.session_state.get("raw_io_change", {"read_bytes": 0, "write_bytes": 0}),
                    )
                else:
                    st.info("ℹ️ Video loaded in RAW Mode. Click **'🚀 Analyze Video Feed'** above to begin disk-based analysis.")

    elif uploaded_file is None:
        # Clear upload state and clean any lingering temporary file if file was removed
        old_path = st.session_state.pop("video_path", None)
        remove_temp_file(old_path)
        st.session_state.pop("video_bytes", None)
        st.session_state.pop("upload_hash", None)
        # Cancel any running sterile worker
        old_worker = st.session_state.pop("sterile_worker", None)
        if old_worker is not None and old_worker.is_alive():
            old_worker.cancel()

        # Clean all analysis results
        st.session_state.pop("sterile_result", None)
        st.session_state.pop("sterile_alerts", None)
        st.session_state.pop("sterile_progress", None)
        st.session_state.pop("sterile_done", None)
        st.session_state.pop("sterile_error", None)
        st.session_state.pop("sterile_incidents", None)
        st.session_state.pop("sterile_new_files", None)
        st.session_state.pop("sterile_io_change", None)
        st.session_state.pop("raw_result", None)
        st.session_state.pop("raw_new_files", None)
        st.session_state.pop("raw_io_change", None)


if __name__ == "__main__":
    main()
