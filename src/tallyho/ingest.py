"""Telemetry ingest.

Two layers:

* :class:`TelemetryProcessor` - pure pipeline (raw dict → parse → reorder/de-dup
  → callback). Fully testable offline by feeding it mocked frames.
* :class:`SondeHubStream` - thin live driver around ``pysondehub`` (lazy import).
  We drive the MQTT loop ourselves with exponential backoff + jitter and
  resubscribe on reconnect, being a polite client on a shared community broker.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

from .config import IngestConfig
from .dedup import ReorderBuffer
from .models import Frame
from windfall.telemetry import try_parse_frame

log = logging.getLogger(__name__)

FramesCallback = Callable[[list[Frame]], None]


def backoff_delay(
    attempt: int,
    base: float,
    maximum: float,
    jitter: float,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with +/- ``jitter`` fraction.

    ``attempt`` is 0-based. Deterministic when an ``rng`` is supplied (tests)."""
    rng = rng or random
    raw = min(maximum, base * (2.0 ** max(0, attempt)))
    if jitter > 0:
        factor = 1.0 + rng.uniform(-jitter, jitter)
        raw *= factor
    return max(0.0, raw)


class TelemetryProcessor:
    """Parse + de-dup + reorder. Feeds ordered frames to ``on_frames``."""

    def __init__(self, cfg: IngestConfig, on_frames: FramesCallback):
        self.cfg = cfg
        self.on_frames = on_frames
        self.buffer = ReorderBuffer(
            hold_seconds=cfg.reorder_hold_seconds,
            dedup_ttl_seconds=cfg.dedup_ttl_seconds,
        )
        self.parsed = 0
        self.rejected = 0

    def handle_raw(self, msg: dict) -> None:
        """Handle one raw telemetry dict from the stream."""
        frame = try_parse_frame(msg)
        if frame is None:
            self.rejected += 1
            return
        self.parsed += 1
        released = self.buffer.push(frame)
        if released:
            self.on_frames(released)

    def flush(self) -> None:
        released = self.buffer.flush()
        if released:
            self.on_frames(released)


class SondeHubStream:
    """Live MQTT driver around pysondehub with self-managed reconnect.

    Connection management only: each raw message is handed to ``on_raw`` (which
    the app enqueues), so all parsing/tracking/prediction runs on a single
    consumer thread - keeping rasterio DEM lookups thread-safe."""

    def __init__(self, on_raw: Callable[[dict], None], cfg: IngestConfig):
        self.on_raw = on_raw
        self.cfg = cfg
        self._stop = False

    def run_forever(self) -> None:  # pragma: no cover - needs network/broker
        """Connect and pump frames until :meth:`stop`. Reconnects with backoff."""
        try:
            import sondehub  # lazy: only needed for live ingest
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pysondehub not installed; `pip install '.[ingest]'` for live ingest"
            ) from exc

        attempt = 0
        while not self._stop:
            stream = None
            try:
                stream = sondehub.Stream(
                    on_message=self._on_message,
                    auto_start_loop=False,
                )
                stream.loop_start()
                log.info("connected to SondeHub MQTT")
                attempt = 0  # reset backoff on a clean connect
                while not self._stop:
                    time.sleep(1.0)
            except Exception:  # noqa: BLE001 - we genuinely want to retry anything
                log.exception("SondeHub stream error; will reconnect")
            finally:
                if stream is not None:
                    try:
                        stream.loop_stop()
                        stream.disconnect()
                    except Exception:  # pragma: no cover
                        pass
            if self._stop:
                break
            delay = backoff_delay(
                attempt,
                self.cfg.reconnect_base_seconds,
                self.cfg.reconnect_max_seconds,
                self.cfg.reconnect_jitter,
            )
            log.warning("reconnecting to SondeHub in %.1fs", delay)
            time.sleep(delay)
            attempt += 1

    def _on_message(self, msg: dict) -> None:  # pragma: no cover - callback thread
        self.on_raw(msg)

    def stop(self) -> None:
        self._stop = True
