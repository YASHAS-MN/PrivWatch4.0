# PrivWatch Final System Architecture

## 1. High-Level Pipeline

The PrivWatch evaluated pipeline processes RGB surveillance video without relying on skeletal keypoint detection or pose estimation:

```text
RGB Video Stream / Clip
           │
           ▼
Transient Frame Decoding (OpenCV / PyAV)
           │
           ▼
Spatial Frame Resizing (112 x 112)
           │
           ▼
Sliding Window Buffer (T = 16 frames)
           │
           ▼
Tensor Conversion & Normalization (ImageNet Mean/Std)
           │
           ▼
MobileNetV2 Spatial Feature Extractor (Backbone Features: 1280-d)
           │
           ▼
Adaptive Average Pooling (1280 x 1 x 1)
           │
           ▼
LSTM Temporal Sequence Model (Input: 1280, Hidden: 128)
           │
           ▼
Linear Classifier Head (128 -> 3 Logits)
           │
           ▼
Softmax Activation & Threshold Decision Logic
           │
           ▼
Classification Output: Normal (0) | Fight (1) | Collapse (2)
           │
           ▼
RAM-First Sterile Room (Evaporation, Memory Sanitization, Alert Registry)
```

---

## 2. Neural Network Architecture Details

The model is defined in [`ActionModel`](../privwatch/model.py) as a unified PyTorch `nn.Module`.

### Layer Specifications and Tensor Transformations

| Stage | Operation / Layer | Input Tensor Shape | Output Tensor Shape | Parameters / Details |
| :--- | :--- | :--- | :--- | :--- |
| **0. Input** | Video Batch Clip | - | `(B, 16, 3, 112, 112)` | 16 RGB frames, 112x112, normalized |
| **1. Reshape** | Flatten temporal batch | `(B, 16, 3, 112, 112)` | `(B * 16, 3, 112, 112)` | View operation |
| **2. CNN** | `self.cnn` (MobileNetV2 features) | `(B * 16, 3, 112, 112)` | `(B * 16, 1280, 4, 4)` | Inverted residual bottleneck blocks |
| **3. Pool** | `self.pool` (`AdaptiveAvgPool2d(1)`) | `(B * 16, 1280, 4, 4)` | `(B * 16, 1280, 1, 1)` | Spatial global average pooling |
| **4. Reshape** | Restore temporal sequence | `(B * 16, 1280, 1, 1)` | `(B, 16, 1280)` | View operation |
| **5. LSTM** | `self.lstm` (`nn.LSTM`) | `(B, 16, 1280)` | `(B, 16, 128)` | `input_size=1280`, `hidden_size=128`, `batch_first=True` |
| **6. Slicing** | Final timestep extraction `x[:, -1, :]` | `(B, 16, 128)` | `(B, 128)` | Temporal summary state |
| **7. Linear Head**| `self.fc` (`nn.Linear`) | `(B, 128)` | `(B, 3)` | Logits for Normal (0), Fight (1), Collapse (2) |
| **8. Activation** | `torch.softmax(dim=1)` | `(B, 3)` | `(B, 3)` | Probability distribution across 3 classes |

---

## 3. Preprocessing Specifications

- **Spatial Resolution:** `112 x 112` pixels.
- **Sequence Length:** `16` frames.
- **Color Ordering:** BGR / RGB consistent with OpenCV default reading (`cv2.VideoCapture`).
- **Scaling:** Float division by `255.0` mapping pixel values to `[0.0, 1.0]`.
- **Normalization:** Channel-wise standardization:
  $$\text{Normalized} = \frac{\text{Pixel} - \text{Mean}}{\text{Std}}$$
  $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$

---

## 4. Inference and Decision Thresholds

Surveillance safety monitoring is asymmetric: **missing a fight or collapse (false negative) is significantly more detrimental than a false alarm (false positive).**

To prioritize recall of critical safety anomalies, tuned confidence thresholds are applied to sliding window mean probabilities:

```python
FIGHT_THRESHOLD = 0.35
COLLAPSE_THRESHOLD = 0.35

if mean_fight_prob >= FIGHT_THRESHOLD:
    decision = "ALERT - Fight detected"
elif mean_collapse_prob >= COLLAPSE_THRESHOLD:
    decision = "ALERT - Collapse detected"
else:
    decision = "Normal activity"
```

---

## 5. Sterile Room Integration

The Sterile Room wraps the inference execution in a controlled volatile procedure:
1. **Volatile Bounded Buffer:** Only `T = 16` frames exist concurrently in RAM.
2. **Sliding Evaporation:** As new frames arrive, evicted frames are unlinked and subject to best-effort memory zeroing.
3. **In-Memory Decoding:** Planned PyAV integration to stream video packets from memory without writing temporary `.mp4` files to storage.
4. **Isolated Worker:** Short-lived subprocess boundary so OS process cleanup destroys allocator arenas and CUDA context upon termination.
5. **Alert Registry:** Emits non-visual event records (UUID, timestamp, confidence, event type). No image frames or embeddings are saved.

