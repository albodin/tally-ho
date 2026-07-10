"""In-process pub/sub bridging non-async writer threads to SSE clients.

`publish(name)` is safe to call from any thread (it is what `Store.on_change`
is wired to). Names are coalesced over a short debounce window and fanned out to
each connected `/api/events` generator as a dirty-set + wake. Events carry no
payload: the browser refetches the affected endpoint, which stays canonical.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field


@dataclass(eq=False)
class _Client:
    # eq=False keeps identity hashing (a dataclass with eq=True is unhashable),
    # so instances can live in the `_clients` set keyed by connection identity.
    dirty: set[str] = field(default_factory=set)
    event: asyncio.Event = field(default_factory=asyncio.Event)


class EventBus:
    DEBOUNCE = 1.0  # collapses the ~1 Hz append_track_point firehose into 1 event

    def __init__(self, debounce: float = DEBOUNCE) -> None:
        self._debounce = debounce
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._flush_scheduled = False
        self._clients: set[_Client] = set()  # loop-thread only; no lock needed
        self.closed = False

    # -- loop thread: called from the web app's lifespan --
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def close(self) -> None:
        """Lifespan shutdown / tests: wake every generator so it exits."""
        self.closed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._wake_all)

    # -- any thread: Store.on_change --
    def publish(self, name: str) -> None:
        loop = self._loop
        with self._lock:
            self._pending.add(name)
            if self._flush_scheduled or loop is None:
                # No loop yet (web disabled / not started) → accumulate and drop;
                # a client resyncs on (re)connect anyway. `_pending` is bounded to
                # the handful of distinct event names, so this never grows.
                return
            self._flush_scheduled = True
        loop.call_soon_threadsafe(self._arm_flush)

    # -- loop thread --
    def _arm_flush(self) -> None:
        assert self._loop is not None
        self._loop.call_later(self._debounce, self._flush)

    def _flush(self) -> None:
        with self._lock:
            names, self._pending = self._pending, set()
            self._flush_scheduled = False
        for c in self._clients:
            c.dirty |= names
            c.event.set()

    def _wake_all(self) -> None:
        for c in self._clients:
            c.event.set()

    def register(self) -> _Client:
        c = _Client()
        self._clients.add(c)
        return c

    def unregister(self, c: _Client) -> None:
        self._clients.discard(c)
