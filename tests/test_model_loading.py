"""Unit tests for model instantiation, checkpoint verification, and forward pass."""

from pathlib import Path
import sys
import unittest

import torch

# Ensure package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privwatch.config import DEVICE, IMG_SIZE, SEQ_LEN
from privwatch.inference import load_model
from privwatch.model import ActionModel
from privwatch.paths import RAW_MODEL_PATH


class TestModelLoading(unittest.TestCase):
    """Test model creation and checkpoint loading."""

    def test_checkpoint_exists(self):
        """Verify checkpoint file exists and has expected size."""
        self.assertTrue(RAW_MODEL_PATH.exists(), f"Checkpoint missing at {RAW_MODEL_PATH}")
        file_size = RAW_MODEL_PATH.stat().st_size
        self.assertEqual(file_size, 12028684, f"Checkpoint size mismatch: {file_size}")

    def test_instantiate_model(self):
        """Verify ActionModel can be instantiated with custom/default parameters."""
        model = ActionModel(pretrained_backbone=False)
        self.assertIsInstance(model, ActionModel)
        self.assertEqual(model.fc.out_features, 3)
        self.assertEqual(model.lstm.hidden_size, 128)

    def test_load_checkpoint_weights(self):
        """Verify weights load into model cleanly with metadata intact."""
        model, metadata = load_model(RAW_MODEL_PATH, device=torch.device("cpu"))
        self.assertIsInstance(model, ActionModel)
        self.assertEqual(metadata.get("epoch"), 5)
        self.assertAlmostEqual(metadata.get("val_accuracy", 0.0), 0.8658, places=3)
        self.assertIn("train_fight", metadata.get("dataset_summary", {}))

    def test_forward_pass_shape(self):
        """Verify forward pass on synthetic input produces (batch_size, 3) logits."""
        model, _ = load_model(RAW_MODEL_PATH, device=DEVICE)
        model.eval()

        dummy_input = torch.zeros(1, SEQ_LEN, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (1, 3))
        self.assertFalse(torch.isnan(output).any().item())
        self.assertFalse(torch.isinf(output).any().item())


if __name__ == "__main__":
    unittest.main()

