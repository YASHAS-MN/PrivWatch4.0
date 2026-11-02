# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""PrivWatch: Privacy-Preserving Video Anomaly Detection Framework."""

from pathlib import Path
import sys

_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from privwatch.config import (
        CLASS_MAP,
        CLASS_NAMES,
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
        run_inference,
        run_inference_detailed,
        run_inference_on_bytes,
    )
    from privwatch.model import ActionModel
    from privwatch.paths import MODELS_DIR, PROJECT_ROOT, RAW_MODEL_PATH
    from privwatch.preprocessing import preprocess_clip, resize_frame
except ImportError:
    from PrivWatch_Final.privwatch.config import (  # type: ignore
        CLASS_MAP,
        CLASS_NAMES,
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
        run_inference,
        run_inference_detailed,
        run_inference_on_bytes,
    )
    from PrivWatch_Final.privwatch.model import ActionModel  # type: ignore
    from PrivWatch_Final.privwatch.paths import MODELS_DIR, PROJECT_ROOT, RAW_MODEL_PATH  # type: ignore
    from PrivWatch_Final.privwatch.preprocessing import preprocess_clip, resize_frame  # type: ignore

__all__ = [
    "ActionModel",
    "load_model",
    "run_inference",
    "run_inference_detailed",
    "preprocess_clip",
    "resize_frame",
    "SEQ_LEN",
    "IMG_SIZE",
    "CLASS_MAP",
    "CLASS_NAMES",
    "FIGHT_THRESHOLD",
    "COLLAPSE_THRESHOLD",
    "DEVICE",
    "PROJECT_ROOT",
    "MODELS_DIR",
    "RAW_MODEL_PATH",
]

