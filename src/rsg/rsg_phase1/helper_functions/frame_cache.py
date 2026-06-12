"""Small bounded frame/evidence caches used by the coordinator."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CachedFrame:
    """Frame stored while waiting for classifier output."""

    frame_id: str
    sequence: int
    received_monotonic: float
    received_stamp_sec: float
    msg: Any
    status: str = "received"
    # Monotonic timestamp when the coordinator dequeued this frame and
    # published it to the classifier. 0.0 means not sent yet.
    sent_to_classifier_monotonic: float = 0.0
    # Delay from coordinator receive time to classifier-send time.
    sent_to_classifier_delay_ms: float = 0.0


class BoundedFrameCache:
    """Bounded mapping from frame_id to frame data with drop-oldest behaviour."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max(1, int(max_size))
        self._items: "OrderedDict[str, CachedFrame]" = OrderedDict()

    def put(self, frame: CachedFrame) -> None:
        if frame.frame_id in self._items:
            self._items.pop(frame.frame_id)
        self._items[frame.frame_id] = frame
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def get(self, frame_id: str) -> Optional[CachedFrame]:
        item = self._items.get(frame_id)
        if item is not None:
            self._items.move_to_end(frame_id)
        return item

    def remove(self, frame_id: str) -> Optional[CachedFrame]:
        return self._items.pop(frame_id, None)

    def __len__(self) -> int:
        return len(self._items)


class EvidenceBuffer:
    """Bounded RAM metadata buffer for future risk-annotation frame retrieval."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max(1, int(max_size))
        self._items: deque[Dict[str, Any]] = deque(maxlen=self.max_size)

    def add(self, record: Dict[str, Any]) -> None:
        self._items.append(record)

    def latest(self, count: int = 5) -> list[Dict[str, Any]]:
        count = max(0, int(count))
        if count == 0:
            return []
        return list(self._items)[-count:]

    def __len__(self) -> int:
        return len(self._items)
