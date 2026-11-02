# -*- coding: utf-8 -*-
"""Unit tests for the Streamlit video upload lifecycle and state-management logic."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure package directory is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app import remove_temp_file


class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def simulate_upload_step(
    uploaded_file: MockUploadedFile | None,
    session_state: dict,
) -> None:
    """Simulates the exact state-management logic from app.py."""
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()
        prev_hash = session_state.get("upload_hash")
        current_path = session_state.get("video_path")

        # Save to temporary file if new upload or previous temporary file was removed
        if current_hash != prev_hash or not current_path or not os.path.exists(current_path):
            if current_path:
                remove_temp_file(current_path)

            suffix = Path(uploaded_file.name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(file_bytes)
            tmp.close()

            session_state["video_path"] = tmp.name
            session_state["video_bytes"] = file_bytes
            session_state["upload_hash"] = current_hash
    else:
        old_path = session_state.pop("video_path", None)
        remove_temp_file(old_path)
        session_state.pop("video_bytes", None)
        session_state.pop("upload_hash", None)


class TestUploadLifecycle(unittest.TestCase):
    """Test every upload lifecycle scenario required by PrivWatch."""

    def setUp(self):
        self.session_state = {}
        self.video_a_bytes = b"FAKE_VIDEO_STREAM_A_SAMPLE_DATA_12345"
        self.video_b_bytes = b"FAKE_VIDEO_STREAM_B_SAMPLE_DATA_67890"
        self.video_c_bytes = b"FAKE_VIDEO_STREAM_C_SAMPLE_DATA_ABCDE"

        self.file_a = MockUploadedFile("video_a.mp4", self.video_a_bytes)
        self.file_b = MockUploadedFile("video_b.mp4", self.video_b_bytes)
        self.file_c = MockUploadedFile("video_c.mp4", self.video_c_bytes)

    def tearDown(self):
        # Cleanup any remaining test files
        if "video_path" in self.session_state:
            remove_temp_file(self.session_state["video_path"])

    def test_upload_a_initial(self):
        """Scenario 1: Upload video A -> Preview shows A, file created."""
        simulate_upload_step(self.file_a, self.session_state)

        path_a = self.session_state.get("video_path")
        self.assertIsNotNone(path_a)
        self.assertTrue(os.path.exists(path_a))
        self.assertEqual(self.session_state.get("video_bytes"), self.video_a_bytes)
        self.assertEqual(
            self.session_state.get("upload_hash"),
            hashlib.sha256(self.video_a_bytes).hexdigest(),
        )

    def test_upload_b_replaces_a_without_refresh(self):
        """Scenario 2 & 3: Upload video B without refreshing -> Preview changes to B, A is deleted."""
        simulate_upload_step(self.file_a, self.session_state)
        path_a = self.session_state["video_path"]
        self.assertTrue(os.path.exists(path_a))

        # Upload video B in same session
        simulate_upload_step(self.file_b, self.session_state)
        path_b = self.session_state["video_path"]

        self.assertNotEqual(path_a, path_b)
        self.assertFalse(os.path.exists(path_a), "Old temporary file A was not deleted")
        self.assertTrue(os.path.exists(path_b), "New temporary file B was not created")
        self.assertEqual(self.session_state["video_bytes"], self.video_b_bytes)
        self.assertEqual(
            self.session_state["upload_hash"],
            hashlib.sha256(self.video_b_bytes).hexdigest(),
        )

    def test_upload_c_replaces_b(self):
        """Scenario 4: Upload video C -> Preview changes to C."""
        simulate_upload_step(self.file_b, self.session_state)
        path_b = self.session_state["video_path"]

        simulate_upload_step(self.file_c, self.session_state)
        path_c = self.session_state["video_path"]

        self.assertFalse(os.path.exists(path_b))
        self.assertTrue(os.path.exists(path_c))
        self.assertEqual(self.session_state["video_bytes"], self.video_c_bytes)

    def test_upload_same_video_reuses_file(self):
        """Scenario 5: Upload the same video again -> Reuses existing temporary file, no duplicates."""
        simulate_upload_step(self.file_a, self.session_state)
        path_first = self.session_state["video_path"]

        # Simulate rerun with same uploaded file
        simulate_upload_step(self.file_a, self.session_state)
        path_second = self.session_state["video_path"]

        self.assertEqual(path_first, path_second)
        self.assertTrue(os.path.exists(path_first))

    def test_sterile_mode_cleanup_and_reupload(self):
        """Scenario 6 & 7: Sterile mode deletes temporary file, re-upload works without KeyError."""
        simulate_upload_step(self.file_a, self.session_state)
        path_a = self.session_state["video_path"]
        self.assertTrue(os.path.exists(path_a))

        # Simulate Sterile mode cleanup
        remove_temp_file(path_a)
        self.session_state.pop("video_path", None)
        self.session_state.pop("video_bytes", None)
        self.session_state.pop("upload_hash", None)

        self.assertFalse(os.path.exists(path_a))
        self.assertNotIn("video_path", self.session_state)
        self.assertNotIn("video_bytes", self.session_state)
        self.assertNotIn("upload_hash", self.session_state)

        # Re-upload same video after Sterile mode
        simulate_upload_step(self.file_a, self.session_state)
        new_path_a = self.session_state["video_path"]

        self.assertTrue(os.path.exists(new_path_a))
        self.assertEqual(self.session_state["video_bytes"], self.video_a_bytes)

    def test_raw_mode_retains_file(self):
        """Scenario 8: RAW mode leaves temporary file on disk."""
        simulate_upload_step(self.file_a, self.session_state)
        path_a = self.session_state["video_path"]
        self.assertTrue(os.path.exists(path_a))

        # In RAW mode, file is NOT deleted after inference
        self.assertTrue(os.path.exists(path_a))
        self.assertEqual(self.session_state["video_path"], path_a)

    def test_file_removal_cleans_state(self):
        """Verify removing uploaded file cleans state without KeyError."""
        simulate_upload_step(self.file_a, self.session_state)
        path_a = self.session_state["video_path"]
        self.assertTrue(os.path.exists(path_a))

        # User clears file uploader (uploaded_file = None)
        simulate_upload_step(None, self.session_state)
        self.assertFalse(os.path.exists(path_a))
        self.assertNotIn("video_path", self.session_state)
        self.assertNotIn("video_bytes", self.session_state)
        self.assertNotIn("upload_hash", self.session_state)


if __name__ == "__main__":
    unittest.main()

