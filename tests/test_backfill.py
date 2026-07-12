"""First-detection history backfill - fetch plumbing and flight rebuilds.

All offline: the SondeHub fetch is injected (``App(backfill_fetch_fn=...)``)
and driven synchronously via ``HistoryFetcher.run_pending()``.
"""

import threading
import time
from datetime import datetime, timezone

import pytest

from tallyho.app import App
from tallyho.backfill import HistoryFetcher, merge_history
from tallyho.config import IngestConfig
from tallyho.models import Subscriber
from tallyho.notify import FakeNtfySink
from tallyho.store import Store
from windfall.geo import haversine_km
from windfall.telemetry import parse_frame
from tests.conftest import fast_ensemble_cfg, simulate_flight


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def _app(store, sub_lat, sub_lon, fetch_fn, radius=40.0, cfg=None):
    store.add_subscriber(Subscriber(
        name="alice", lat=sub_lat, lon=sub_lon, radius_km=radius,
        ntfy_server="https://ntfy.sh", ntfy_topic="alice", ntfy_token_ref="NTFY"))
    sink = FakeNtfySink()
    app = App(cfg or fast_ensemble_cfg(), store=store, sink=sink,
              backfill_fetch_fn=fetch_fn)
    return app, sink


def _burst_index(frames):
    return max(range(len(frames)), key=lambda i: frames[i].alt)


# ---- merge_history ----------------------------------------------------------

def test_merge_history_parses_sorts_dedups_and_filters(flight):
    raw = list(flight.frames[:10])
    raw.reverse()                                   # arrives unordered
    raw.append(dict(flight.frames[3]))              # duplicate timestamp
    raw.append({"serial": "S1234567"})              # unparseable (no fix)
    raw.append(dict(flight.frames[4], serial="OTHER"))   # someone else's frame
    merged = merge_history(raw, [], "S1234567")
    want = [parse_frame(m) for m in flight.frames[:10]]
    assert [f.t for f in merged] == [f.t for f in want]
    assert all(f.serial == "S1234567" for f in merged)


def test_merge_history_splices_only_newer_live_frames(flight):
    frames = [parse_frame(m) for m in flight.frames]
    live = frames[8:14]                       # overlaps the history tail
    merged = merge_history(flight.frames[:10], live, "S1234567")
    assert [f.t for f in merged] == [f.t for f in frames[:14]]
    # no history at all → the live frames stand alone
    assert [f.t for f in merge_history(None, live, "S1234567")] == \
        [f.t for f in live]


# ---- HistoryFetcher ---------------------------------------------------------

def test_fetcher_dedups_pending_and_releases_on_drain():
    calls = []

    def fetch(serial, timeout=None):
        calls.append((serial, timeout))
        return [{"serial": serial}]

    cfg = IngestConfig()
    cfg.backfill_timeout_seconds = 7.0
    hf = HistoryFetcher(cfg, fetch_fn=fetch)
    assert hf.request("AAA") is True
    assert hf.request("AAA") is False          # already pending
    assert hf.run_pending() == 1
    assert calls == [("AAA", 7.0)]             # timeout knob reaches the fetch
    assert hf.drain() == [("AAA", [{"serial": "AAA"}])]
    assert hf.drain() == []
    assert hf.request("AAA") is True           # drained → requestable again


def test_fetcher_failures_and_empty_yield_none():
    def boom(serial, timeout=None):
        raise OSError("network down")

    hf = HistoryFetcher(IngestConfig(), fetch_fn=boom)
    hf.request("BAD")
    hf.run_pending()
    assert hf.drain() == [("BAD", None)]

    hf = HistoryFetcher(IngestConfig(), fetch_fn=lambda s, timeout=None: [])
    hf.request("EMPTY")
    hf.run_pending()
    assert hf.drain() == [("EMPTY", None)]


def test_fetcher_worker_thread_lifecycle():
    fetched = threading.Event()

    def fetch(serial, timeout=None):
        fetched.set()
        return [{"serial": serial}]

    hf = HistoryFetcher(IngestConfig(), fetch_fn=fetch)
    try:
        assert hf.start() is True
        assert hf.start() is False             # already running
        assert hf.request("AAA") is True
        assert fetched.wait(2.0)
        deadline = time.monotonic() + 2.0
        results = hf.drain()
        while not results and time.monotonic() < deadline:
            time.sleep(0.01)
            results = hf.drain()
        assert results == [("AAA", [{"serial": "AAA"}])]
    finally:
        hf.stop()
        hf._thread.join(timeout=3.0)
    assert not hf._thread.is_alive()


# ---- App-level rebuilds -----------------------------------------------------

def test_midair_first_detection_rebuilds_flight_from_history(store):
    f = simulate_flight(serial="MID1", burst_alt=20000)
    frames = [parse_frame(m) for m in f.frames]
    k = next(i for i, fr in enumerate(frames) if fr.alt > 8000)
    # history the archive knows: everything up to shortly after first hearing
    history = f.frames[:k + 2]
    calls = []

    def fetch(serial, timeout=None):
        calls.append(serial)
        return history

    app, _ = _app(store, f.land_lat, f.land_lon, fetch)
    app.on_frames(frames[k:k + 30])            # first heard at ~8 km
    before = app.tracker.get("MID1")
    assert before is not None and before.launch_lat is None
    assert calls == []                         # fetch runs on the worker...
    app._backfill.run_pending()                # ...driven inline here
    assert calls == ["MID1"]

    assert app.apply_backfills() == 1
    fl = app.tracker.get("MID1")
    assert fl is not before                    # rebuilt, not patched
    # the rebuilt flight knows its real launch and full ascent
    assert fl.launch_lat == pytest.approx(f.launch_lat, abs=0.01)
    assert fl.first_seen == frames[0].dt
    assert fl.profile.n_bins > before.profile.n_bins
    # live frames newer than the history tail were spliced back in
    assert fl.last_t == frames[k + 29].t
    row = store.get_flight("MID1", frames[0].dt.date())
    assert row is not None and row["launch_lat"] is not None


def test_backfill_not_requested_when_launch_heard(store):
    f = simulate_flight(serial="GND1", burst_alt=20000)
    frames = [parse_frame(m) for m in f.frames]
    calls = []
    app, _ = _app(store, f.land_lat, f.land_lon,
                  lambda s, timeout=None: calls.append(s))
    app.on_frames(frames[:50])                 # heard from the launch pad
    assert app.tracker.get("GND1").launch_lat is not None
    assert app._backfill.run_pending() == 0 and calls == []


def test_backfill_disabled_by_config(store):
    f = simulate_flight(serial="OFF1", burst_alt=20000)
    frames = [parse_frame(m) for m in f.frames]
    k = next(i for i, fr in enumerate(frames) if fr.alt > 8000)
    cfg = fast_ensemble_cfg()
    cfg.ingest.backfill_enabled = False
    app, _ = _app(store, f.land_lat, f.land_lon,
                  lambda s, timeout=None: f.frames, cfg=cfg)
    app.on_frames(frames[k:k + 10])
    assert app.tracker.get("OFF1") is not None
    assert app._backfill.run_pending() == 0


def test_first_heard_descending_predicts_right_after_backfill(store):
    # THE motivating case: a sonde first heard already falling has no ascent
    # profile, so its landing predictions are junk until the backfill lands.
    # After the rebuild it must predict immediately - and well.
    f = simulate_flight(serial="DESC1", burst_alt=24000)
    frames = [parse_frame(m) for m in f.frames]
    bi = _burst_index(frames)
    j = next(i for i in range(bi, len(frames)) if frames[i].alt < 20000)

    app, sink = _app(store, f.land_lat, f.land_lon,
                     lambda s, timeout=None: f.frames[:j + 5])
    app.on_frames(frames[j:j + 15])            # too few live frames to predict
    assert sink.sent == []
    app._backfill.run_pending()
    assert app.apply_backfills() == 1

    fl = app.tracker.get("DESC1")
    assert fl.state.value == "DESCENT"
    assert fl.burst_alt == pytest.approx(24000, abs=100)   # burst seen in replay
    day = frames[0].dt.date()
    pred = store.latest_prediction("DESC1", day)
    assert pred is not None
    assert haversine_km(pred["land_lat"], pred["land_lon"],
                        f.land_lat, f.land_lon) < 10.0
    assert any("inbound" in m.title.lower() for m in sink.sent)


def test_backfill_replaying_landed_flight_records_landing_and_alerts(store):
    # History can outrun the live frames: if it already shows touchdown, the
    # rebuild closes the flight out - landing truth + LANDED alert - now.
    f = simulate_flight(serial="LATE1", burst_alt=20000)
    frames = [parse_frame(m) for m in f.frames]
    bi = _burst_index(frames)
    j = next(i for i in range(bi, len(frames)) if frames[i].alt < 2500)

    app, sink = _app(store, f.land_lat, f.land_lon,
                     lambda s, timeout=None: f.frames)     # full flight
    app.on_frames(frames[j:j + 5])             # heard 5 frames at ~2.5 km
    app._backfill.run_pending()
    assert app.apply_backfills() == 1

    assert app.tracker.get("LATE1").state.value == "LANDED"
    lnd = store.get_landing("LATE1", frames[0].dt.date())
    assert lnd is not None and lnd["detected_by"] == "telemetry"
    assert haversine_km(lnd["land_lat"], lnd["land_lon"],
                        f.land_lat, f.land_lon) < 1.0
    assert any("LANDED" in m.title for m in sink.sent)


def test_backfill_skipped_when_flight_dropped_meanwhile(store):
    f = simulate_flight(serial="GONE1", burst_alt=20000)
    frames = [parse_frame(m) for m in f.frames]
    k = next(i for i, fr in enumerate(frames) if fr.alt > 8000)
    app, sink = _app(store, f.land_lat, f.land_lon,
                     lambda s, timeout=None: f.frames[:k])
    app.on_frames(frames[k:k + 10])
    app.tracker.drop(app.tracker.get("GONE1"))   # e.g. left the capture ROI
    app._backfill.run_pending()
    assert app.apply_backfills() == 0
    assert app.tracker.get("GONE1") is None      # stays dropped
    assert "GONE1" not in app._backfill_buffer   # buffer cleaned up
    assert sink.sent == []


def test_backfill_with_no_older_history_leaves_flight_alone(store):
    f = simulate_flight(serial="SAME1", burst_alt=20000)
    frames = [parse_frame(m) for m in f.frames]
    k = next(i for i, fr in enumerate(frames) if fr.alt > 8000)
    # the archive only has the very frames we already heard live
    app, _ = _app(store, f.land_lat, f.land_lon,
                  lambda s, timeout=None: f.frames[k:k + 10])
    app.on_frames(frames[k:k + 10])
    fl = app.tracker.get("SAME1")
    app._backfill.run_pending()
    assert app.apply_backfills() == 0
    assert app.tracker.get("SAME1") is fl        # same object: no rebuild


def test_rebuild_is_one_atomic_change_not_an_animation(store):
    # The replay must not re-commit (and SSE-ping) per historical frame - the
    # dashboard would show the sonde flying through its history. One batch:
    # readers see the provisional flight, then the fully rebuilt one.
    f = simulate_flight(serial="ATOM1", burst_alt=20000)
    frames = [parse_frame(m) for m in f.frames]
    k = next(i for i, fr in enumerate(frames) if fr.alt > 8000)
    app, _ = _app(store, f.land_lat, f.land_lon,
                  lambda s, timeout=None: f.frames[:k + 2])
    app.on_frames(frames[k:k + 10])
    app._backfill.run_pending()

    pings = []
    store.on_change = pings.append
    assert app.apply_backfills() == 1
    assert len(store.track_for("ATOM1", frames[0].dt.date())) > 50
    assert pings.count("flights") == 1


def test_backfill_moves_flight_to_its_true_launch_day(store):
    # Launched before midnight UTC, first heard after: the provisional flight
    # is keyed to the wrong launch_day. The rebuild re-keys it to the real
    # launch date and erases the provisional rows.
    f = simulate_flight(serial="MIDN1", burst_alt=12000,
                        start=datetime(2026, 6, 6, 23, 40, tzinfo=timezone.utc))
    frames = [parse_frame(m) for m in f.frames]
    k = next(i for i, fr in enumerate(frames) if fr.alt > 8000)
    assert frames[k].dt.date().isoformat() == "2026-06-07"

    app, _ = _app(store, f.land_lat, f.land_lon,
                  lambda s, timeout=None: f.frames[:k + 2])
    app.on_frames(frames[k:k + 10])
    assert store.get_flight("MIDN1", frames[k].dt.date()) is not None
    app._backfill.run_pending()
    assert app.apply_backfills() == 1

    fl = app.tracker.get("MIDN1")
    assert fl.launch_day.isoformat() == "2026-06-06"
    assert store.get_flight("MIDN1", frames[0].dt.date()) is not None
    assert store.get_flight("MIDN1", frames[k].dt.date()) is None   # ghost gone
