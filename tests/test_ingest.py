"""Tests for the ingest pipeline + backoff."""

import random

import pytest

from tallyho.config import IngestConfig
from tallyho.ingest import TelemetryProcessor, backoff_delay


def test_backoff_is_exponential_and_capped():
    rng = random.Random(0)
    d0 = backoff_delay(0, base=1, maximum=120, jitter=0, rng=rng)
    d1 = backoff_delay(1, base=1, maximum=120, jitter=0, rng=rng)
    d2 = backoff_delay(2, base=1, maximum=120, jitter=0, rng=rng)
    assert d0 == 1
    assert d1 == 2
    assert d2 == 4
    # capped
    assert backoff_delay(20, base=1, maximum=120, jitter=0, rng=rng) == 120


def test_backoff_jitter_within_bounds():
    rng = random.Random(42)
    for _ in range(100):
        d = backoff_delay(3, base=1, maximum=120, jitter=0.3, rng=rng)
        assert 8 * 0.7 <= d <= 8 * 1.3


def test_processor_parses_and_dedups(flight):
    received = []
    proc = TelemetryProcessor(IngestConfig(reorder_hold_seconds=1.0),
                              on_frames=lambda fs: received.extend(fs))
    for msg in flight.frames:
        proc.handle_raw(msg)
        proc.handle_raw(dict(msg))   # duplicate of same frame
    proc.flush()
    assert proc.parsed == 2 * len(flight.frames)
    # duplicates collapsed: emitted count == unique frames
    assert len(received) == len(flight.frames)
    # emitted in datetime order
    ts = [f.t for f in received]
    assert ts == sorted(ts)


def test_processor_rejects_bad_frames():
    rejected = []
    proc = TelemetryProcessor(IngestConfig(), on_frames=lambda fs: None)
    proc.handle_raw({"serial": "x"})          # missing mandatory fields
    proc.handle_raw({"not": "a frame"})
    assert proc.rejected == 2
    assert proc.parsed == 0
