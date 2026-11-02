"""Metadata-only alert registry for privacy-preserving event recording.

RESPONSIBILITIES:
- Record safety-critical events (Fight, Collapse, Anomaly) with metadata only.
- Prohibit storage of raw pixel frames, crops, facial images, or visual embeddings.
- Maintain an append-only in-memory log of structured detection alerts.

RECORDED METADATA FIELDS:
- event_id: Unique string identifier for the event.
- timestamp: ISO 8601 UTC timestamp.
- event_type: Classification label ("Fight", "Collapse", "Normal").
- confidence: Prediction probability (0.0 - 1.0).
- latency_ms: Processing latency in milliseconds.
- metadata: Optional non-visual diagnostic tags.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True)
class AlertEvent:
    """Immutable event record containing strictly non-visual metadata."""

    event_id: str
    timestamp: str
    event_type: str
    confidence: float
    latency_ms: float
    details: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        confidence: float,
        latency_ms: float = 0.0,
        details: dict[str, str] | None = None,
    ) -> "AlertEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            confidence=round(confidence, 4),
            latency_ms=round(latency_ms, 2),
            details=details or {},
        )


class AlertRegistry:
    """In-memory metadata registry storing detection events without visual data."""

    def __init__(self):
        self._alerts: list[AlertEvent] = []

    def record_alert(self, alert: AlertEvent) -> None:
        self._alerts.append(alert)

    def get_alerts(self) -> list[AlertEvent]:
        return list(self._alerts)

    def clear(self) -> None:
        self._alerts.clear()

    def __len__(self) -> int:
        return len(self._alerts)

