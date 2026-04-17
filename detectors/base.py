"""
Base types shared by all detectors.

Every detector subclasses ``BaseDetector`` and emits ``SignalFire``
objects when their criteria are met.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Direction(Enum):
    """Which side of the market the signal favours."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SignalFire:
    """A single detector firing on a specific market."""
    detector_name: str
    market: str            # condition_id
    token_id: str          # the YES token the signal applies to
    direction: Direction
    confidence: float      # 0.0 – 1.0
    timestamp: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)  # detector-specific context


class BaseDetector(ABC):
    """
    Interface that every detector must implement.

    Detectors come in two flavours:

    *  **Streaming** — fed websocket events via ``on_event()``.
    *  **Polling**   — periodically called via ``poll()``.

    A detector may implement one or both.  The default implementations
    are no-ops so subclasses only override what they need.
    """

    name: str = "base"

    def on_event(self, event: dict) -> List[SignalFire]:
        """Process a real-time websocket event.  Return any fires."""
        return []

    def poll(self) -> List[SignalFire]:
        """Called on a timer.  Return any fires."""
        return []

    def reset(self, market: str) -> None:
        """Clear internal state for *market* (e.g. after a position closes)."""
        pass
