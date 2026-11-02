# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Inference engine for PrivWatch video activity classification."""

import argparse
import io
from pathlib import Path
import sys
import time
from typing import Any, Iterator

# Ensure package root is on sys.path
_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import av  # type: ignore
import cv2  # type: ignore
import numpy as np
import torch

try:
    from privwatch.config import (
        CLASS_MAP,
        COLLAPSE_THRESHOLD,
        DEVICE,
        FIGHT_THRESHOLD,
        IMG_SIZE,
        SEQ_LEN,
    )
    from privwatch.model import ActionModel
    from privwatch.paths import RAW_MODEL_PATH
    from privwatch.preprocessing import preprocess_clip, resize_frame
except ImportError:
    from PrivWatch_Final.privwatch.config import (  # type: ignore
        CLASS_MAP,
        COLLAPSE_THRESHOLD,
        DEVICE,
        FIGHT_THRESHOLD,
        IMG_SIZE,
        SEQ_LEN,
    )
    from PrivWatch_Final.privwatch.model import ActionModel  # type: ignore
    from PrivWatch_Final.privwatch.paths import RAW_MODEL_PATH  # type: ignore
    from PrivWatch_Final.privwatch.preprocessing import preprocess_clip, resize_frame  # type: ignore


def load_model(
    model_path: Path | str = RAW_MODEL_PATH,
    device: torch.device = DEVICE,
) -> tuple[ActionModel, dict[str, Any]]:
    """Load the trained ActionModel from checkpoint.

    Args:
        model_path: Path to the .pth checkpoint file.
        device: Device to map model and weights to.

    Returns:
        Tuple of (model in eval mode, metadata dictionary).
    """
    model = ActionModel(pretrained_backbone=False).to(device)
    checkpoint = torch.load(str(model_path), map_location=device)

    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "timestamp": checkpoint.get("timestamp"),
            "dataset_summary": checkpoint.get("dataset_summary"),
        }
        print("=== MODEL FORENSIC INFO ===")
        print(f"epoch: {metadata['epoch']}")
        print(f"validation accuracy: {metadata['val_accuracy']}")
        print(f"timestamp: {metadata['timestamp']}")
        print(f"dataset summary: {metadata['dataset_summary']}")
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def run_inference(
    cap: cv2.VideoCapture,
    model: ActionModel | None = None,
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> str:
    """Run sliding-window activity inference across a video capture stream.

    Args:
        cap: OpenCV VideoCapture object.
        model: Optional pre-loaded ActionModel instance. If None, loaded automatically.
        device: Torch execution device.
        verbose: Whether to print confidence scores and decisions to console.

    Returns:
        Decision string matching original format:
        - "ALERT - Fight detected"
        - "ALERT - Collapse detected"
        - "Normal activity"
        - "No frames processed"
    """
    if model is None:
        model, _ = load_model(device=device)

    buffer: list[np.ndarray] = []
    fight_probs: list[float] = []
    normal_probs: list[float] = []
    collapse_probs: list[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue

            resized = resize_frame(frame, IMG_SIZE)
            buffer.append(resized)

            if len(buffer) == SEQ_LEN:
                clip = preprocess_clip(buffer, device=device)

                with torch.no_grad():
                    outputs = model(clip)
                    probs = torch.softmax(outputs, dim=1)[0]

                    fight_probs.append(probs[1].item())
                    normal_probs.append(probs[0].item())
                    collapse_probs.append(probs[2].item())

                buffer.pop(0)  # sliding window
    finally:
        cap.release()

    if len(fight_probs) > 0:
        final_fight = float(sum(fight_probs) / len(fight_probs))
        final_normal = float(sum(normal_probs) / len(normal_probs))
        final_collapse = float(sum(collapse_probs) / len(collapse_probs))

        if final_fight >= FIGHT_THRESHOLD:
            final_output = "ALERT - Fight detected"
        elif final_collapse >= COLLAPSE_THRESHOLD:
            final_output = "ALERT - Collapse detected"
        else:
            final_output = "Normal activity"

        if verbose:
            print("\n===== FINAL VIDEO CONFIDENCE =====")
            print(f"Fight: {final_fight:.2f}")
            print(f"Normal: {final_normal:.2f}")
            print(f"Collapse: {final_collapse:.2f}")
            print("===============================")
            print("\n===== FINAL DECISION =====")
            print(f"Fight threshold: {FIGHT_THRESHOLD}")
            print(f"Collapse threshold: {COLLAPSE_THRESHOLD}")
            print(f"Selected output: {final_output}")
            print("====================")

        return final_output

    return "No frames processed"


def run_inference_detailed(
    cap: cv2.VideoCapture,
    model: ActionModel | None = None,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """Run inference returning a detailed dictionary of metrics and per-window probabilities."""
    if model is None:
        model, metadata = load_model(device=device)
    else:
        metadata = {}

    start_time = time.time()
    buffer: list[np.ndarray] = []
    window_probs: list[dict[str, float]] = []
    fight_probs: list[float] = []
    normal_probs: list[float] = []
    collapse_probs: list[float] = []
    total_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            total_frames += 1
            resized = resize_frame(frame, IMG_SIZE)
            buffer.append(resized)

            if len(buffer) == SEQ_LEN:
                clip = preprocess_clip(buffer, device=device)
                with torch.no_grad():
                    outputs = model(clip)
                    probs = torch.softmax(outputs, dim=1)[0]
                    p_normal = probs[0].item()
                    p_fight = probs[1].item()
                    p_collapse = probs[2].item()

                    fight_probs.append(p_fight)
                    normal_probs.append(p_normal)
                    collapse_probs.append(p_collapse)
                    window_probs.append({
                        "Normal": p_normal,
                        "Fight": p_fight,
                        "Collapse": p_collapse,
                    })

                buffer.pop(0)
    finally:
        cap.release()

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

        return {
            "decision": decision,
            "predicted_class": predicted_class,
            "confidences": {
                "Normal": mean_normal,
                "Fight": mean_fight,
                "Collapse": mean_collapse,
            },
            "windows_processed": len(fight_probs),
            "total_frames": total_frames,
            "latency_seconds": round(latency, 2),
            "model_metadata": metadata,
        }

    return {
        "decision": "No frames processed",
        "predicted_class": None,
        "confidences": {"Normal": 0.0, "Fight": 0.0, "Collapse": 0.0},
        "windows_processed": 0,
        "total_frames": total_frames,
        "latency_seconds": round(latency, 2),
        "model_metadata": metadata,
    }


def decode_frames_from_bytes(
    video_bytes: bytes,
    with_timestamps: bool = False,
) -> Iterator[np.ndarray | tuple[np.ndarray, float]]:
    """Decode video frames sequentially from in-memory video bytes using PyAV.

    Args:
        video_bytes: In-memory byte stream of uploaded video.
        with_timestamps: If True, yields (frame_ndarray, timestamp_seconds).
            If False (default), yields frame_ndarray for backwards compatibility.

    Yields:
        np.ndarray | tuple[np.ndarray, float]: Decoded video frames in native BGR format
        (H, W, 3) as uint8, optionally paired with presentation timestamp in seconds.

    Raises:
        ValueError: If video bytes are empty, container cannot be opened, or
            no video stream is present.
    """
    if not video_bytes:
        raise ValueError("Video bytes stream is empty.")

    stream = io.BytesIO(video_bytes)
    try:
        container = av.open(stream, mode="r")
    except Exception as exc:
        raise ValueError(f"Failed to open video container from memory: {exc}") from exc

    try:
        if not container.streams.video:
            raise ValueError("No video stream found in the provided media bytes.")

        fps = 25.0
        vstream = container.streams.video[0]
        if vstream.average_rate and float(vstream.average_rate) > 0:
            fps = float(vstream.average_rate)

        frame_idx = 0
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="bgr24")
            pts_time = frame.time
            if pts_time is None:
                pts_time = frame_idx / fps
            frame_idx += 1

            if with_timestamps:
                yield arr, float(pts_time)
            else:
                yield arr
    finally:
        container.close()


def decode_frames_with_timestamps(
    video_bytes: bytes,
) -> Iterator[tuple[np.ndarray, float]]:
    """Convenience generator yielding (bgr_frame, timestamp_seconds) from video bytes."""
    for item in decode_frames_from_bytes(video_bytes, with_timestamps=True):
        yield item  # type: ignore


def run_inference_on_bytes(
    video_bytes: bytes,
    model: ActionModel | None = None,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """Run sliding-window activity inference directly on in-memory video bytes.

    Decodes frames entirely in RAM using PyAV without writing temporary files to disk.

    Args:
        video_bytes: Raw bytes of the video file.
        model: Optional pre-loaded ActionModel instance. If None, loaded automatically.
        device: Torch execution device.

    Returns:
        Dictionary with identical structure and metrics to run_inference_detailed:
        - decision: Decision string.
        - predicted_class: Winning class label.
        - confidences: Mean confidence per class across all temporal windows.
        - windows_processed: Total number of 16-frame sliding windows evaluated.
        - total_frames: Total number of video frames decoded from memory.
        - latency_seconds: Execution latency.
        - model_metadata: Checkpoint forensic metadata.
    """
    if model is None:
        model, metadata = load_model(device=device)
    else:
        metadata = {}

    start_time = time.time()
    buffer: list[np.ndarray] = []
    window_probs: list[dict[str, float]] = []
    fight_probs: list[float] = []
    normal_probs: list[float] = []
    collapse_probs: list[float] = []
    total_frames = 0

    try:
        for frame in decode_frames_from_bytes(video_bytes):
            total_frames += 1
            resized = resize_frame(frame, IMG_SIZE)
            buffer.append(resized)

            if len(buffer) == SEQ_LEN:
                clip = preprocess_clip(buffer, device=device)
                with torch.no_grad():
                    outputs = model(clip)
                    probs = torch.softmax(outputs, dim=1)[0]
                    p_normal = probs[0].item()
                    p_fight = probs[1].item()
                    p_collapse = probs[2].item()

                    fight_probs.append(p_fight)
                    normal_probs.append(p_normal)
                    collapse_probs.append(p_collapse)
                    window_probs.append({
                        "Normal": p_normal,
                        "Fight": p_fight,
                        "Collapse": p_collapse,
                    })

                buffer.pop(0)
    finally:
        buffer.clear()

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

        return {
            "decision": decision,
            "predicted_class": predicted_class,
            "confidences": {
                "Normal": mean_normal,
                "Fight": mean_fight,
                "Collapse": mean_collapse,
            },
            "windows_processed": len(fight_probs),
            "total_frames": total_frames,
            "latency_seconds": round(latency, 2),
            "model_metadata": metadata,
        }

    return {
        "decision": "No frames processed",
        "predicted_class": None,
        "confidences": {"Normal": 0.0, "Fight": 0.0, "Collapse": 0.0},
        "windows_processed": 0,
        "total_frames": total_frames,
        "latency_seconds": round(latency, 2),
        "model_metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Run PrivWatch raw inference on a video.")
    parser.add_argument("video_path", help="Path to an input video file")
    args = parser.parse_args()

    print("Model loaded. Running inference...")
    start_time = time.time()

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    result = run_inference(cap)
    end_time = time.time()

    print(f"\nFINAL RESULT: {result}")
    print(f"\nLatency: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

