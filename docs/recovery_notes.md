# PrivWatch Repository Recovery Notes

## 1. Migration Summary

This document records the recovery, triage, and migration of the verified RGB video anomaly detection system into the clean `PrivWatch_Final/` repository.

Migration Date: 2026-11-01  
Lead Systems Engineer: Antigravity (AI Pair Programmer)  
Primary Architecture: RGB Video → MobileNetV2 → LSTM → 3-Class Classification → Sterile Room

---

## 2. Inventory of Migrated Files

| Destination Path | Source Path | Status / Modification Notes |
| :--- | :--- | :--- |
| `models/best_model.pth` | `PrivWatch/models/best_model.pth` | Verified byte-for-byte copy (12,028,684 bytes). SHA-256: `ac2fe829a0147af98450cb584112eb5685bf4dc864407a35e3dedcc3be4fb0fc`. |
| `privwatch/model.py` | `PrivWatch/privwatch/model.py` | Preserved exact `ActionModel` neural network definition. Added `pretrained_backbone=True` default parameter to support offline weight loading. |
| `privwatch/inference.py` | `PrivWatch/privwatch/inference.py` | Refactored into clean modular runner; preserves exact sliding-window logic, probabilities, threshold decisions, and console output. Added detailed metrics dictionary helper. |
| `privwatch/paths.py` | `PrivWatch/privwatch/paths.py` | Updated `PROJECT_ROOT` relative path resolution for clean standalone repository imports. |
| `privwatch/preprocessing.py` | Extracted from `inference.py` & `scripts/preprocess.py` | Modularized frame resizing (`112x112`), sequence stacking (`T=16`), and ImageNet normalization. |
| `privwatch/config.py` | Extracted from `inference.py` | Centralized hyperparameter constants (`SEQ_LEN=16`, `IMG_SIZE=112`, thresholds `0.35`, device selection). |
| `privwatch/__init__.py` | `PrivWatch/privwatch/__init__.py` | Modernized package exports. |
| `app.py` | `PrivWatch/app.py` | Modernized Streamlit dashboard with model caching, forensic metadata inspection, and disk I/O monitoring. |
| `requirements.txt` | `PrivWatch/requirements.txt` | Cleaned and bounded dependencies (`torch`, `torchvision`, `opencv-python`, `numpy`, `streamlit`, `psutil`, `av`). |
| `.gitignore` | `PrivWatch/.gitignore` | Replaced legacy ignore file with streamlined Python/.gitignore retaining model checkpoints. |

---

## 3. Inventory of Excluded Files and Rationale

The original workspace contained mixed legacy experiments and exploratory prototypes that violated the RGB MobileNetV2 + LSTM objective:

| Excluded Item | Location | Rationale for Exclusion |
| :--- | :--- | :--- |
| `models/action_model.pth` | Outer `models/` | 173 KB file containing a 51-input keypoint LSTM trained on YOLO pose coordinates (4 classes: Normal, Fight, Collapse, Intrusion). Incompatible with RGB video. |
| `yolov8n-pose.pt` | Outer root & `PrivWatch/Priv2.0/` | YOLOv8 nano pose model weights (skeletal extraction). Specifically prohibited by task constraints. |
| `Priv1.0/` | `PrivWatch/Priv1.0/` | Deprecated initial iteration (MediaPipe Lite landmarker, ResNet/CNN frame extractors, archived downloader scripts). |
| `Priv2.0/` | Outer root & `PrivWatch/Priv2.0/` | Legacy pose-based exploration experimenting with skeletal temporal extraction and coordinate alignment. |
| `src/data_prep/extract_poses.py` | Outer `src/` | Pose extraction script using YOLO keypoints. |
| `src/data_prep/split_dataset.py` | Outer `src/` | Legacy split script for pose dataset. |
| `src/models/train_lstm.py` | Outer `src/` | Legacy 51-keypoint LSTM training script. |
| `src/models/evaluate_model.py` | Outer `src/` | Legacy keypoint evaluation script. |
| `align_features.py` | Outer root & `PrivWatch/Priv2.0/` | Script for aligning skeletal pose feature vectors. |
| `Ignore_this` | `PrivWatch/Ignore_this` | Unused text placeholder file. |
| `__pycache__/` | Various | Compiled Python bytecode cache files. |
| Sample videos (`t/c/`, `t/n/`) | Outer `t/` | Kept untouched in outer workspace for evaluation testing; excluded from Git repository to keep repository lightweight. |

---

## 4. Technical Findings and Compatibility Solutions

### 1. Pretrained Weights vs Offline Checkpoint Loading
In `PrivWatch/privwatch/model.py`, `ActionModel.__init__()` instantiated `mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)`. When running on an offline edge device or within a restricted sandbox without external network access, downloading torchvision weights throws a connection timeout before `model.load_state_dict()` can overwrite the weights.  
**Resolution:** Parameterized `ActionModel(pretrained_backbone=True)`. When `load_model()` loads `best_model.pth`, it sets `pretrained_backbone=False` so initialization is instantaneous and fully offline.

### 2. OpenCV Color Space Consistency
OpenCV's `cv2.VideoCapture.read()` outputs frames in BGR channel order. In the training script (`PrivWatch/scripts/preprocess.py`), frames were saved directly as tensors and normalized using standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` without an intermediate `cv2.cvtColor(BGR2RGB)`.  
**Resolution:** To strictly preserve inference fidelity and avoid modifying trained weights, the inference pipeline preserves this exact color channel pipeline.

### 3. Windows File Locking During Streamlit Video Uploads
In `PrivWatch/app.py`, uploaded files were written to a temporary `.mp4` file via `tempfile.NamedTemporaryFile` so OpenCV could read them. On Windows, if `cv2.VideoCapture` does not completely release the file handle before `os.remove()`, a `PermissionError` is raised.  
**Resolution:** Wrapped file removal in a safe retry loop with short sleep intervals and explicit `cap.release()`.

---

## 5. Sterile Room Implementation Status

- **Status:** Architectural specification and interface stubs recovered.
- **Completed Components:**
  - `VolatileBuffer`: Bounded circular buffer (`maxlen=16`) implemented and tested.
  - `MemorySanitizer`: In-place array zeroing (`arr.fill(0)`) and cache purge routines implemented and tested.
  - `AlertRegistry` & `AlertEvent`: Non-visual metadata event logging implemented and tested.
- **Pending Components (Stubs):**
  - `InferenceWorker`: Subprocess isolation boundary (raises `NotImplementedError`).
  - `SterileSession` / `StreamlitSterileAdapter`: Direct in-memory video decoding via `PyAV` (raises `NotImplementedError`).
- **Transparency Statement:** The Sterile Room is NOT claimed to be a fully operational production enclave. Full end-to-end in-memory decoding via PyAV is planned for the subsequent engineering phase.

