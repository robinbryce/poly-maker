"""
Event bus.

Receives normalised events from the websocket layer, fans them out
to every registered detector, and forwards the resulting fires to
the coordinator.
"""

from __future__ import annotations

from typing import Callable, List

from detectors.base import BaseDetector, SignalFire


class EventBus:
    def __init__(self, detectors: List[BaseDetector]):
        self._detectors = detectors
        self._fire_callback: Callable[[List[SignalFire]], None] | None = None

    def set_fire_callback(self, cb: Callable[[List[SignalFire]], None]) -> None:
        """Register the coordinator's ingest function."""
        self._fire_callback = cb

    def dispatch(self, event: dict) -> None:
        """Fan out a single normalised event to all streaming detectors."""
        all_fires: List[SignalFire] = []
        for det in self._detectors:
            try:
                fires = det.on_event(event)
                if fires:
                    all_fires.extend(fires)
            except Exception as exc:
                print(f"[event_bus] {det.name} raised {exc}")
        if all_fires and self._fire_callback:
            self._fire_callback(all_fires)

    def poll_all(self) -> None:
        """Call poll() on every detector and forward any fires."""
        all_fires: List[SignalFire] = []
        for det in self._detectors:
            try:
                fires = det.poll()
                if fires:
                    all_fires.extend(fires)
            except Exception as exc:
                print(f"[event_bus] {det.name} poll raised {exc}")
        if all_fires and self._fire_callback:
            self._fire_callback(all_fires)
