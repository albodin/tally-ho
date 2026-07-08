"""Tests for de-dup + reorder buffer."""

from datetime import datetime, timedelta, timezone

from tallyho.dedup import ReorderBuffer
from tallyho.models import Frame

BASE = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)


def mk(serial: str, frame: int, secs: float) -> Frame:
    dt = BASE + timedelta(seconds=secs)
    return Frame(serial=serial, lat=45.0, lon=7.0, alt=1000.0 + secs,
                 t=dt.timestamp(), dt=dt, frame=frame)


def test_dedup_drops_repeat_frame():
    buf = ReorderBuffer(hold_seconds=2.0)
    out1 = buf.push(mk("A", 1, 0))
    out2 = buf.push(mk("A", 1, 0))   # duplicate
    assert out2 == []
    # nothing released yet (within hold window); flush to inspect
    released = out1 + buf.flush()
    assert [f.frame for f in released] == [1]


def test_reorder_emits_in_datetime_order():
    # hold large enough that nothing releases until flush
    buf = ReorderBuffer(hold_seconds=10.0)
    out = []
    out += buf.push(mk("A", 3, 2.0))   # arrive out of order
    out += buf.push(mk("A", 1, 0.0))
    out += buf.push(mk("A", 2, 1.0))
    assert out == []                   # all still held
    out += buf.flush()
    assert [f.frame for f in out] == [1, 2, 3]


def test_hold_window_releases_progressively():
    buf = ReorderBuffer(hold_seconds=3.0)
    assert buf.push(mk("A", 1, 0.0)) == []     # held
    assert buf.push(mk("A", 2, 1.0)) == []     # held
    # advance watermark to t=5 → frames at t<=2 are now releasable
    out = buf.push(mk("A", 3, 5.0))
    assert [f.frame for f in out] == [1, 2]


def test_frame_older_than_last_emitted_is_dropped():
    buf = ReorderBuffer(hold_seconds=2.0)
    buf.push(mk("A", 1, 1.0))
    out = buf.push(mk("A", 5, 10.0))   # advances watermark, releases frame 1
    assert [f.frame for f in out] == [1]
    # a frame strictly before the last-emitted time can't be reordered → dropped
    late = buf.push(mk("A", 2, 0.5))
    assert late == []
    # but a late frame still after last-emitted is reordered/emitted in order
    ok = buf.push(mk("A", 3, 2.0))
    assert [f.frame for f in ok] == [3]


def test_serials_independent():
    buf = ReorderBuffer(hold_seconds=2.0)
    buf.push(mk("A", 1, 0.0))
    buf.push(mk("B", 1, 0.0))
    out = buf.flush()
    serials = sorted(f.serial for f in out)
    assert serials == ["A", "B"]


def test_dedup_keys_expire(monkeypatch):
    buf = ReorderBuffer(hold_seconds=1.0, dedup_ttl_seconds=5.0)
    buf.push(mk("A", 1, 0.0))
    buf.push(mk("A", 2, 100.0))   # far ahead → expires old dedup keys
    ps = buf._serials["A"]
    assert (("A", 1)) not in ps.seen
