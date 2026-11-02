# PrivWatch Model Evaluation & Forensic Metrics

## 1. Recovered Checkpoint Metadata

The final evaluated checkpoint [`models/best_model.pth`](../models/best_model.pth) was forensically inspected and yielded the following verified metadata recorded at training completion:

| Attribute | Verified Value |
| :--- | :--- |
| **Model Architecture** | MobileNetV2 Feature Extractor + LSTM Temporal Head |
| **Best Epoch** | 5 |
| **Validation Accuracy** | `86.58%` (`0.8658269441401971`) |
| **Training Timestamp** | `2026-05-14T02:44:04.878854` |
| **File Size** | `12,028,684` bytes (`11.47 MB`) |
| **SHA-256 Checksum** | `ac2fe829a0147af98450cb584112eb5685bf4dc864407a35e3dedcc3be4fb0fc` |
| **Parameter Tensors** | 318 state dictionary layers |

---

## 2. Dataset Distribution Summary

The model was evaluated against a balanced 3-class video clip dataset:

### Training Split (8,588 Clips Total)
- **Fight:** 2,800 clips (32.6%)
- **Collapse:** 2,800 clips (32.6%)
- **Normal:** 2,988 clips (34.8%)

### Validation Split (1,826 Clips Total)
- **Fight:** 600 clips (32.9%)
- **Collapse:** 600 clips (32.9%)
- **Normal:** 626 clips (34.2%)

---

## 3. Decision Threshold Analysis

Standard classification architectures use an argmax or 0.50 probability cutoff. However, for surveillance safety systems, an asymmetric risk profile applies:
- **False Negative (missed fight or collapse):** Potential physical harm or unaddressed medical emergency.
- **False Positive (normal misclassified as anomaly):** Human operator review or temporary alert.

PrivWatch tunes the anomaly thresholds to **0.35**:

$$\text{Fight Decision} \iff \bar{P}_{\text{fight}} \ge 0.35$$
$$\text{Collapse Decision} \iff \bar{P}_{\text{collapse}} \ge 0.35 \quad (\text{if } \bar{P}_{\text{fight}} < 0.35)$$
$$\text{Normal Decision} \iff \text{Otherwise}$$

This ensures elevated sensitivity toward safety-critical incidents across continuous video sliding windows.

---

## 4. Evaluation Metrics Reference Matrix

During experimental validation, the following confusion matrix structure is utilized:

```text
                     Predicted
               Normal    Fight    Collapse
Actual Normal    TN       FP        FP
Actual Fight     FN       TP        FN
Actual Collapse  FN       FN        TP
```

### Safety Metrics
- **Anomaly Recall:** $\frac{\text{TP}_{\text{Fight}} + \text{TP}_{\text{Collapse}}}{\text{Total Actual Anomalies}}$
- **Fight Recall:** $\frac{\text{TP}_{\text{Fight}}}{\text{Actual Fights}}$
- **Collapse Recall:** $\frac{\text{TP}_{\text{Collapse}}}{\text{Actual Collapses}}$
- **False Negative Rate:** $1 - \text{Anomaly Recall}$

---

## 5. Verification Protocol

To verify the model without retraining:

```bash
# 1. Inspect checkpoint metadata and hash
python scripts/inspect_checkpoint.py

# 2. Run all non-training unit and integration tests
python -m unittest discover -s tests -p "test_*.py"

# 3. Test inference on a sample video
python -m privwatch.inference "../t/n/a (1).mp4"
```

