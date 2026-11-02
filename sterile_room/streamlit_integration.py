# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=import-error
"""Streamlit integration adapter for RAM-first video inference.

RESPONSIBILITIES:
- Handle file upload directly from st.file_uploader as in-memory bytes.
- In Sterile Mode: completely eliminate writing uploaded video to disk via tempfile.
- Feed decoded frames directly to SterileSession and return prediction metrics.
- Enforce cleanup and cache purging upon completion.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

# Ensure PrivWatch root is on sys.path
_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch

try:
    from privwatch.config import DEVICE
    from privwatch.model import ActionModel
    from sterile_room.sterile_session import SterileSession
except ImportError:
    from PrivWatch_Final.privwatch.config import DEVICE  # type: ignore
    from PrivWatch_Final.privwatch.model import ActionModel  # type: ignore
    from PrivWatch_Final.sterile_room.sterile_session import SterileSession  # type: ignore


class StreamlitSterileAdapter:
    """Adapter bridging Streamlit file upload events to SterileSession."""

    @staticmethod
    def process_uploaded_file(
        uploaded_file: Any,
        model: ActionModel | None = None,
        device: torch.device = DEVICE,
    ) -> dict[str, Any]:
        """Ingest bytes from uploaded_file without saving to disk.

        Args:
            uploaded_file: Streamlit UploadedFile object or raw video bytes.
            model: Optional pre-loaded ActionModel.
            device: Execution device.

        Returns:
            Dictionary containing prediction metrics and decision.
        """
        if hasattr(uploaded_file, "getvalue"):
            video_bytes = uploaded_file.getvalue()
        elif isinstance(uploaded_file, (bytes, bytearray)):
            video_bytes = bytes(uploaded_file)
        else:
            raise ValueError(f"Unsupported uploaded_file input type: {type(uploaded_file)}")

        session = SterileSession(model=model, device=device)
        try:
            return session.process_video_bytes(video_bytes)
        finally:
            session.close()
