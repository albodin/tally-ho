"""First-detection history backfill.

A sonde first heard mid-air - it launched outside receiver range, or the
daemon started mid-flight - begins with an empty wind profile and no
launch/burst context, so its predictions start far worse than they need to:
SondeHub already holds everything the sonde transmitted before we could hear
it. :class:`HistoryFetcher` pulls that history on a worker thread (network
must never stall frame processing); the app drains completed fetches on its
single consumer thread and rebuilds the flight by replaying the merged frame
list through the tracker (see ``App.apply_backfills``).
"""

from __future__ import annotations

import logging
import queue
import threading

from windfall.history import fetch_live_telemetry
from windfall.telemetry import try_parse_frame

from .config import IngestConfig
from .models import Frame

log = logging.getLogger(__name__)


def merge_history(raw: list[dict] | None, live: list[Frame], serial: str) -> list[Frame]:
    """Parse fetched history into Frames (oldest first, one per timestamp) and
    splice in the ``live`` frames heard while the fetch ran - only those newer
    than the history's tail, so nothing is double-counted. Pure, so the merge
    logic tests offline."""
    frames = [f for f in (try_parse_frame(m) for m in raw or [])
              if f is not None and f.serial == serial]
    frames.sort(key=lambda f: f.t)
    merged: list[Frame] = []
    for f in frames:
        if not merged or f.t > merged[-1].t:
            merged.append(f)
    last_t = merged[-1].t if merged else float("-inf")
    merged.extend(sorted((f for f in live if f.t > last_t), key=lambda f: f.t))
    return merged


class HistoryFetcher:
    """Serial in, raw frame history out, off the consumer thread.

    ``request()`` and ``drain()`` are meant for the app's single consumer
    thread; only the worker touches the network. Fetches are best-effort: any
    failure surfaces as a ``(serial, None)`` result (logged, never raised), so
    a flight whose history can't be fetched just keeps its live-only state.
    ``fetch_fn(serial, timeout=...)`` is injectable for offline tests."""

    def __init__(self, cfg: IngestConfig, fetch_fn=None):
        self.cfg = cfg
        self.fetch_fn = fetch_fn or fetch_live_telemetry
        self._pending: queue.Queue = queue.Queue(maxsize=64)
        self._results: queue.Queue = queue.Queue()
        self._in_flight: set[str] = set()   # consumer thread only
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def request(self, serial: str) -> bool:
        """Queue one history fetch. False if it is already pending (or the
        queue is saturated - only plausible with the worker dead)."""
        if serial in self._in_flight:
            return False
        try:
            self._pending.put_nowait(serial)
        except queue.Full:
            log.warning("backfill queue full; dropping request for %s", serial)
            return False
        self._in_flight.add(serial)
        return True

    def drain(self) -> list[tuple[str, list[dict] | None]]:
        """Completed fetches since the last drain (None = fetch failed/empty)."""
        out: list[tuple[str, list[dict] | None]] = []
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            self._in_flight.discard(item[0])
            out.append(item)
        return out

    # ---- worker thread ------------------------------------------------------
    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="backfill", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                serial = self._pending.get(timeout=1.0)
            except queue.Empty:
                continue
            self._results.put((serial, self.fetch_one(serial)))

    def run_pending(self) -> int:
        """Process every queued request inline - the worker loop, minus the
        thread. For tests (deterministic) and thread-less embedders."""
        n = 0
        while True:
            try:
                serial = self._pending.get_nowait()
            except queue.Empty:
                break
            self._results.put((serial, self.fetch_one(serial)))
            n += 1
        return n

    def fetch_one(self, serial: str) -> list[dict] | None:
        """One guarded fetch (the worker loop body; callable directly in tests
        for thread-free determinism)."""
        try:
            frames = self.fetch_fn(serial, timeout=self.cfg.backfill_timeout_seconds)
        except Exception:  # noqa: BLE001 - backfill is best-effort by contract
            log.exception("backfill fetch for %s failed", serial)
            return None
        if not frames:
            log.info("no SondeHub history for %s", serial)
            return None
        log.info("fetched %d frame(s) of history for %s", len(frames), serial)
        return frames
