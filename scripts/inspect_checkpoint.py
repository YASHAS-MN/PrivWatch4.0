"""Inspect checkpoint metadata and model weight integrity for PrivWatch."""

import argparse
import hashlib
from pathlib import Path
import sys

import torch

# Ensure package root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privwatch.model import ActionModel
from privwatch.paths import RAW_MODEL_PATH


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_checkpoint(checkpoint_path: Path) -> None:
    """Print comprehensive forensic information for a model checkpoint."""
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    file_size = checkpoint_path.stat().st_size
    sha256 = compute_sha256(checkpoint_path)

    print("=" * 60)
    print("PRIVWATCH CHECKPOINT FORENSIC REPORT")
    print("=" * 60)
    print(f"Path:      {checkpoint_path.resolve()}")
    print(f"File Size: {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)")
    print(f"SHA-256:   {sha256}")
    print("-" * 60)

    # Load safely to CPU
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Checkpoint object type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"Top-level keys: {list(checkpoint.keys())}")
        for key in checkpoint:
            if key != "model_state_dict":
                print(f"  {key}: {checkpoint[key]}")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    print("-" * 60)
    print(f"Model parameters (state dict entries): {len(state_dict)}")
    first_layers = list(state_dict.items())[:3]
    last_layers = list(state_dict.items())[-3:]

    print("Initial parameter tensors:")
    for name, tensor in first_layers:
        print(f"  {name:30} : {tuple(tensor.shape)}")
    print("  ...")
    print("Final parameter tensors:")
    for name, tensor in last_layers:
        print(f"  {name:30} : {tuple(tensor.shape)}")

    print("-" * 60)
    print("Verifying architecture compatibility with ActionModel...")
    try:
        model = ActionModel(pretrained_backbone=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print(f"SUCCESS: Checkpoint matches ActionModel architecture perfectly.")
        print(f"Missing keys:    {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"FAILURE: Checkpoint state dict mismatch: {e}")
        sys.exit(1)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect PrivWatch PyTorch checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RAW_MODEL_PATH,
        help="Path to checkpoint file (default: models/best_model.pth)",
    )
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()

