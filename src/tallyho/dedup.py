"""De-duplication + reorder buffer.

SondeHub aggregates hundreds of receivers, so a frame arrives multiple times and
out of order. We:

* **De-dup** by ``(serial, frame)`` (or ``(serial, datetime)`` fallback).
* **Reorder** with a short hold window, emitting frames in sonde-``datetime``
  order so finite-differencing (kinematics, winds) sees a monotonic series.

The hold uses a per-serial *datetime watermark* rather than wall-clock, which
makes the buffer deterministic and equally correct for the live stream and the
offline replay harness: a frame is releasable once a later frame
(by sonde time) more than ``hold_seconds`` ahead has been seen.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

from .models import Frame


@dataclass(slots=True)
class _PerSerial:
    heap: list = field(default_factory=list)        # (t, seq, Frame)
    watermark: float = float("-inf")                # max sonde-time seen
    last_emitted_t: float = float("-inf")           # t of most recent emitted frame
    seen: dict = field(default_factory=dict)        # dedup_key -> last-seen t


class ReorderBuffer:
    """Holds frames briefly and emits them de-duplicated, in datetime order."""

    def __init__(self, hold_seconds: float = 3.0, dedup_ttl_seconds: float = 60.0):
        self.hold_seconds = hold_seconds
        self.dedup_ttl_seconds = dedup_ttl_seconds
        self._serials: dict[str, _PerSerial] = {}
        self._counter = itertools.count()  # tiebreaker → stable heap ordering

    def push(self, frame: Frame) -> list[Frame]:
        """Accept a frame; return the (possibly empty) list now safe to process,
        in ascending datetime order. Duplicates are dropped silently."""
        ps = self._serials.get(frame.serial)
        if ps is None:
            ps = self._serials[frame.serial] = _PerSerial()

        key = frame.dedup_key
        if key in ps.seen:
            return []  # duplicate
        ps.seen[key] = frame.t
        # We emit frames in non-decreasing t. A frame older than the most recent
        # one we already emitted can no longer be reordered into place → drop it
        # (rather than emit out of order and corrupt finite-differencing).
        if frame.t < ps.last_emitted_t:
            return []

        heapq.heappush(ps.heap, (frame.t, next(self._counter), frame))
        ps.watermark = max(ps.watermark, frame.t)
        return self._drain(ps, release_before=ps.watermark - self.hold_seconds)

    def flush(self, serial: str | None = None) -> list[Frame]:
        """Emit everything still held, in datetime order. Call on shutdown or
        when a flight ends. With ``serial`` flushes just that flight."""
        out: list[Frame] = []
        serials = [serial] if serial is not None else list(self._serials)
        for s in serials:
            ps = self._serials.get(s)
            if ps is None:
                continue
            out.extend(self._drain(ps, release_before=float("inf")))
        return out

    def _drain(self, ps: _PerSerial, release_before: float) -> list[Frame]:
        out: list[Frame] = []
        while ps.heap and ps.heap[0][0] <= release_before:
            _, _, frame = heapq.heappop(ps.heap)
            out.append(frame)
            ps.last_emitted_t = frame.t
        self._expire_dedup(ps)
        return out

    def _expire_dedup(self, ps: _PerSerial) -> None:
        """Forget dedup keys older than the TTL - duplicates only arrive within
        seconds, so this keeps the seen-set bounded without losing protection."""
        cutoff = ps.watermark - self.dedup_ttl_seconds
        if cutoff == float("-inf"):
            return
        stale = [k for k, t in ps.seen.items() if t < cutoff]
        for k in stale:
            del ps.seen[k]
