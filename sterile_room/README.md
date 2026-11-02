# Sterile Room Architecture Specification

## 1. Overview and Design Objective

The **Sterile Room** is an execution wrapper designed to enforce a *Privacy by Procedure* lifecycle around the PrivWatch video activity recognition model.

In conventional video analytics workflows, surveillance video feeds or uploaded video files are routinely persisted to disk storage (e.g., NVR archives, temporary files created by web frameworks). PrivWatch investigates an alternative execution paradigm:
**Perform real-time safety-critical activity detection while eliminating intentional persistence of reconstructable video footage.**

The core design principle is:
> **Process transiently in volatile memory, emit only non-visual event metadata, and actively evict frame buffers from memory.**

---

## 2. Intended Execution Flow

```text
Uploaded Video Bytes / Video Stream
              │
              ▼
    In-Memory Decoding (PyAV / av.open(BytesIO))
              │
              ▼
    Bounded Circular RAM Buffer (VolatileBuffer, maxlen=16)
              │
              ▼
    RGB Video Clip Preprocessing (16 frames -> 112x112, normalized)
              │
              ▼
    Inference Worker Execution (MobileNetV2 -> LSTM)
              │
              ▼
    Activity Classification (Normal / Fight / Collapse)
              │
              ▼
    Metadata-Only Alert Logging (AlertRegistry)
              │
              ▼
    Memory Sanitization & Evaporation (MemorySanitizer, GC, CUDA cache purge)
```

---

## 3. Component Responsibilities

| Component | Responsibility |
| --- | --- |
| `volatile_buffer.py` | Implements a bounded circular FIFO RAM buffer (`maxlen=16`) that prevents unbounded memory accumulation and supports sliding-window evaporation. |
| `memory_sanitizer.py` | Executes best-effort in-place zeroing of mutable NumPy arrays, explicit garbage collection passes, and PyTorch CUDA cache evacuation. |
| `inference_worker.py` | Manages an isolated worker process for neural network inference so that operating-system process termination completely reclaims runtime allocator state. |
| `sterile_session.py` | Manages the full lifecycle of a transient processing session: session setup, buffer ingestion, inference triggering, and teardown cleanup. |
| `streamlit_integration.py` | Ingests `io.BytesIO` streams from `st.file_uploader` directly into memory without invoking `tempfile.NamedTemporaryFile`. |
| `alert_registry.py` | Records detection events with non-visual metadata only (UUID, timestamp, event type, confidence score, latency). Strictly forbids saving raw frames or visual embeddings. |

---

## 4. PyAV Investigation: In-Memory Video Decoding

### The Problem with OpenCV (`cv2.VideoCapture`)
Standard OpenCV bindings for Python (`cv2.VideoCapture`) primarily accept filesystem paths (e.g., `"video.mp4"`) or hardware device indices (`0`). When integrating with web interfaces like Streamlit:
```python
# Conventional (insecure) pattern:
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())  # Flushes raw surveillance video to disk!
cap = cv2.VideoCapture(tfile.name)
```
Even if deleted after inference, the video file has been committed to disk storage, creating a permanent forensic footprint and exposure to undelete/carving tools.

### The PyAV Solution
`PyAV` (`import av`) provides direct Pythonic bindings to FFmpeg's `libavformat`, `libavcodec`, and `libavutil`. Crucially, PyAV can open an in-memory byte buffer directly:
```python
import io
import av

def decode_video_in_memory(video_bytes: bytes):
    container = av.open(io.BytesIO(video_bytes))
    for frame in container.decode(video=0):
        # Convert directly to NumPy BGR array without disk touch
        img = frame.to_ndarray(format="bgr24")
        yield img
```
**Benefits:**
- Eliminates temporary disk files entirely during user uploads.
- Streamable directly into `VolatileBuffer`.
- Supports standard surveillance codecs (H.264, H.265, MPEG-4).

---

## 5. Threat Model and Realistic Boundaries

PrivWatch explicitly rejects exaggerated security or privacy claims. The system operates under the following transparent boundaries:

1. **No Absolute Memory Erasure Guarantee:**
   - Overwriting Python/NumPy arrays (`arr.fill(0)`) is best-effort user-space memory zeroing.
   - Operating system virtual memory mechanisms (e.g., Windows pagefile/swap, Linux swap partitions) may page dirty memory pages to disk before user-space zeroing occurs.
   - CPU caches, branch prediction buffers, and GPU driver allocators operate outside user-space Python control.
2. **Crash Dumps and Core Dumps:**
   - If the application crashes during inference, operating-system error reporting may dump process memory (containing recent frames) to disk.
3. **Kernel and Root Compromise:**
   - A privileged process or kernel rootkit running on the host OS can inspect RAM at any point. PrivWatch is an application-level architecture, not a trusted execution environment (TEE/SGX).
4. **Valid Claim:**
   - PrivWatch eliminates *intentional persistent storage* of surveillance video and minimizes the temporal lifetime of uncompressed frames in memory.

---

## 6. Implementation Status

- **Status:** Architectural specification and interface recovery stubs.
- **Next Phase:** Implementation of PyAV decoding worker and process isolation tests after core model validation is complete.

