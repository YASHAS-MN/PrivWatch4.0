# PrivWatch: Privacy-Preserving Video Anomaly Detection

**RGB Video → MobileNetV2 → LSTM → Activity Classification → RAM-First Sterile Room**

PrivWatch is an intelligent surveillance framework engineered to detect safety-critical activities while minimizing the persistence and retention of sensitive visual data.

---

## 1. Project Objective

Conventional CCTV systems stream and archive hundreds of hours of surveillance footage to permanent storage disks by default. In the vast majority of scenarios, this footage captures ordinary, harmless daily activities, creating a large, sensitive data retention footprint vulnerable to leaks, breaches, and misuse.

**PrivWatch** demonstrates an alternative:
- Perform real-time activity recognition to identify emergency incidents.
- Treat data lifetime in memory as an essential systems constraint.
- Avoid persistent video archives through a volatile, RAM-first processing procedure.

---

## 2. Final System Architecture

The evaluated system uses raw RGB surveillance video:

```text
RGB Video Stream (MP4/AVI)
         │
         ▼
Frame Ingestion & Resizing (112 x 112)
         │
         ▼
Temporal Sequence Buffer (16 Frames)
         │
         ▼
Normalization (ImageNet Mean & Std)
         │
         ▼
MobileNetV2 CNN Feature Backbone (1280-dim representation)
         │
         ▼
Adaptive Average Pooling (1280 x 1 x 1)
         │
         ▼
LSTM Recurrent Temporal Head (Hidden Size: 128)
         │
         ▼
Linear Classification Projection (3 Classes)
         │
         ▼
Activity Decision (Normal / Fight / Collapse)
         │
         ▼
Sterile Room Procedure (Frame Evaporation, Memory Sanitization, Event-Only Alert)
```

> **Note:** Skeletal pose extraction (YOLO/MediaPipe keypoints) is excluded from this architecture. Features are extracted directly from RGB pixel dynamics.

---

## 3. Classification Classes

The model classifies video sequences into three categories:

1. **Normal Activity** (Class Index 0)
2. **Fight** (Class Index 1) — Safety anomaly alert
3. **Collapse** (Class Index 2) — Medical anomaly alert

### Decision Thresholds
Surveillance systems prioritize high anomaly recall over symmetric accuracy. Decisions are made using tuned confidence cutoffs:
- **Fight Threshold:** `0.35`
- **Collapse Threshold:** `0.35`

If average Fight confidence $\ge 0.35 \implies$ `ALERT - Fight detected`  
Else if average Collapse confidence $\ge 0.35 \implies$ `ALERT - Collapse detected`  
Else $\implies$ `Normal activity`

---

## 4. Model Checkpoint Location & Forensics

The evaluated, pre-trained weights are located at:
```text
models/best_model.pth
```

### Verified Forensic Information
- **File Size:** `12,028,684 bytes` (`11.47 MB`)
- **SHA-256 Checksum:** `ac2fe829a0147af98450cb584112eb5685bf4dc864407a35e3dedcc3be4fb0fc`
- **Validation Accuracy:** `86.58%` (`0.8658269441401971`)
- **Trained Epoch:** 5
- **Parameters:** 318 state dictionary layers

To inspect this checkpoint independently:
```bash
python scripts/inspect_checkpoint.py
```

---

## 5. Sterile Room Design Objective

The **Sterile Room** wraps inference in a volatile execution boundary:
- **RAM-First Execution:** Video frames reside in volatile RAM rather than written to persistent disk archives.
- **Bounded Buffering:** Only 16 frames exist concurrently in memory (`VolatileBuffer`).
- **Sliding Evaporation:** Overwritten or evicted frames undergo in-place memory clearing.
- **Process Isolation:** Worker processes terminate after inference, enabling the operating system to reclaim memory allocators.
- **Event-Only Persistence:** When an anomaly is detected, only metadata (UUID, timestamp, event type, confidence) is recorded in [`AlertRegistry`](sterile_room/alert_registry.py). No raw pixels, face crops, or visual embeddings are stored.
- **PyAV In-Memory Ingestion:** Investigating PyAV (`av.open(io.BytesIO(video_bytes))`) to eliminate `tempfile` disk writes during web uploads.

*Limitations:* RAM-first execution does not guarantee complete erasure against OS-level swap/paging or kernel-level memory inspection. It enforces procedural minimization of intentional persistence.

---

## 6. Repository Layout

```text
PrivWatch_Final/
├── app.py                      # Streamlit demonstration dashboard
├── requirements.txt            # Minimal required dependencies
├── README.md                   # Repository documentation
├── .gitignore                  # Git exclusions
│
├── models/
│   └── best_model.pth          # Evaluated RGB MobileNetV2 + LSTM checkpoint
│
├── privwatch/                  # Core package
│   ├── __init__.py             # Public API exports
│   ├── model.py                # ActionModel (MobileNetV2 + LSTM)
│   ├── inference.py            # Sliding-window inference runner
│   ├── preprocessing.py        # Frame resizing and normalization
│   ├── paths.py                # Project filesystem paths
│   └── config.py               # Hyperparameters and thresholds
│
├── sterile_room/               # Volatile execution layer (stubs & specs)
│   ├── __init__.py
│   ├── volatile_buffer.py      # Circular bounded RAM buffer
│   ├── memory_sanitizer.py     # In-place array zeroing & cache cleanup
│   ├── inference_worker.py     # Process isolation boundary
│   ├── sterile_session.py      # Session lifecycle manager
│   ├── streamlit_integration.py# In-memory Streamlit upload adapter
│   ├── alert_registry.py       # Non-visual metadata event logging
│   └── README.md               # Detailed Sterile Room architecture & PyAV notes
│
├── tests/                      # Automated test suite (20 unit tests)
│   ├── test_model_loading.py   # Checkpoint integrity & forward pass
│   ├── test_preprocessing.py   # Resizing, normalization, sliding window
│   ├── test_inference.py       # Decision logic & sample video inference
│   └── test_sterile_room.py    # Buffer, memory sanitizer, alert registry
│
├── scripts/
│   └── inspect_checkpoint.py   # Standalone CLI checkpoint inspector
│
└── docs/
    ├── architecture.md         # Full system and layer specifications
    ├── recovery_notes.md       # Migration log and exclusion inventory
    └── evaluation.md           # Checkpoint metrics and validation rules
```

---

## 7. Setup & Execution Instructions

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ with TorchVision
- OpenCV Python

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Inspect Model Checkpoint
```bash
python scripts/inspect_checkpoint.py
```

### Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Run CLI Inference on a Video
```bash
python -m privwatch.inference "path/to/surveillance_video.mp4"
```

### Launch Interactive Demonstration UI
```bash
streamlit run app.py
```

---

## 8. Current Implementation Status

| Milestone | Status | Notes |
| :--- | :--- | :--- |
| **MobileNetV2 + LSTM Architecture** | ✅ Verified | Instantiated, tested, matches original model.py. |
| **Model Weights Checkpoint** | ✅ Verified | Matched byte-for-byte (SHA-256 verified, 86.58% val accuracy). |
| **Sliding-Window Inference Engine** | ✅ Verified | Functional across synthetic streams and sample videos. |
| **Unit Test Suite** | ✅ Passed | 20 of 20 unit tests passing. |
| **Streamlit Interface** | ✅ Verified | Interactive mode with forensic & disk audit widgets. |
| **Volatile Buffer & Memory Sanitizer** | ✅ Verified | Implemented and validated via unit tests. |
| **Metadata Alert Registry** | ✅ Verified | Non-visual event logging active. |
| **PyAV In-Memory Decoding** | 📋 Specification | Architecture defined in `sterile_room/README.md`. |
| **Subprocess Worker Isolation** | 📋 Specification | Architecture stub defined in `sterile_room/inference_worker.py`. |

