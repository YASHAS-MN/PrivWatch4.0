"""Configuration constants for PrivWatch inference and preprocessing."""

import torch

# Temporal and Spatial Dimensions
SEQ_LEN: int = 16
IMG_SIZE: int = 112

# Inference Decision Thresholds
FIGHT_THRESHOLD: float = 0.35
COLLAPSE_THRESHOLD: float = 0.35

# Classification Classes
CLASS_MAP: dict[int, str] = {
    0: "Normal",
    1: "Fight",
    2: "Collapse",
}
CLASS_NAMES: list[str] = ["Normal", "Fight", "Collapse"]

# Normalization parameters (ImageNet defaults used in original pipeline)
NORMALIZATION_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORMALIZATION_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Runtime Execution Device
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

