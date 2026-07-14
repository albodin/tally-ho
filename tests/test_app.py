"""End-to-end app pipeline tests - ingest→...→ntfy, offline."""

from datetime import datetime, timedelta, timezone

import pytest

from tallyho.app import App
from tallyho.config import Config
from tallyho.models import AlertType, FlightState, Subscriber
from tallyho.notify import FakeNtfySink
from tallyho.store import Store
from windfall.telemetry import parse_frame
from tests.conftest import fast_ensemble_cfg, simulate_flight


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def _app_with_sub(store, sub_lat, sub_lon, radius=40.0, gfs_source=None):
    store.add_subscriber(Subscriber(
        name="alice", lat=sub_lat, lon=sub_lon, radius_km=radius,
        ntfy_server="https://ntfy.sh", ntfy_topic="alice", ntfy_token_ref="NTFY"))
    sink = FakeNtfySink()
    # wiring-level tests: tiny ensemble keeps the code path hot but cheap
    app = App(fast_ensemble_cfg(), store=store, sink=sink, gfs_source=gfs_source)
    return app, sink


def _static_gfs():
    """A full-column GFS source so pre-burst predictions aren't gated off
    (early-ascent sondes have no measured winds above their current altitude)."""
    from windfall.gfs import StaticGFSSource
    from windfall.profile import FlightProfile
    prof = FlightProfile()
    for a in range(0, 36000, 1000):
        prof.add_sample(a, 8.0, 2.0)
    return StaticGFSSource(prof)


def test_end_to_end_inbound_and_landed(store):
    # Generate a flight; put the subscriber right at its true landing point.
    f = simulate_flight(serial="E2E1", burst_alt=28000)
    app, sink = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    frames = [parse_frame(m) for m in f.frames]
    app.on_frames(frames)

    titles = [m.title for m in sink.sent]
    assert any("inbound" in t.lower() for t in titles), titles
    assert any("LANDED" in t for t in titles), titles

    # predictions were persisted
    pred = store.latest_prediction("E2E1", frames[-1].dt.date())
    assert pred is not None
    # alert de-dup rows recorded
    subs = store.list_subscribers()
    assert store.get_alert(subs[0].id, "E2E1", frames[-1].dt.date(), AlertType.INBOUND)
    assert store.get_alert(subs[0].id, "E2E1", frames[-1].dt.date(), AlertType.LANDED)


def test_expired_flight_records_no_landing_truth(store):
    # A flight that goes silent mid-air is closed out on the maintenance tick,
    # but no landing ground-truth row is written (it would poison accuracy) and
    # no LANDED alert is pushed.
    f = simulate_flight(serial="LOST1", burst_alt=28000)
    app, sink = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    # bare timeout sweep: with backfill on, a silent flight is first held for
    # a history-recovery attempt (that flow is covered in test_backfill.py)
    app.cfg.ingest.backfill_enabled = False
    frames = [parse_frame(m) for m in f.frames]
    # feed only the early ascent, then let it go stale
    app.on_frames(frames[:100])
    flight = app.tracker.get("LOST1")
    assert flight is not None
    later = flight.last_seen + timedelta(
        seconds=app.cfg.tracker.stale_flight_seconds + 60)
    app.tick(now=later)
    assert store.get_flight("LOST1", frames[0].dt.date())["state"] == "LANDED"
    assert store.get_landing("LOST1", frames[0].dt.date()) is None
    assert sink.sent == []


def test_capture_roi_filters_distant_flight(store):
    # Subscriber far from the flight; flight never enters capture ROI → no alerts.
    f = simulate_flight(serial="FAR1", launch_lat=45.0, launch_lon=7.0, burst_alt=28000)
    app, sink = _app_with_sub(store, sub_lat=10.0, sub_lon=100.0, radius=30.0)
    app.on_frames([parse_frame(m) for m in f.frames])
    assert sink.sent == []
    # nothing tracked either (filtered before the tracker)
    assert app.tracker.get("FAR1") is None


def test_roi_change_drops_tracked_flight(store):
    # Moving/shrinking the watched area mid-flight drops flights that are now
    # outside the capture box: closed out silently (no landing truth, no
    # alert), and their later frames are gated out instead of re-tracking.
    f = simulate_flight(serial="ROI1", burst_alt=28000)
    app, sink = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    frames = [parse_frame(m) for m in f.frames]
    app.on_frames(frames[:100])
    assert app.tracker.get("ROI1") is not None

    sub = store.list_subscribers()[0]
    sub.lat, sub.lon = -33.9, 151.2   # watched location moved across the planet
    store.update_subscriber(sub)
    app.reload_subscribers()

    assert app.tracker.get("ROI1") is None
    assert store.get_flight("ROI1", frames[0].dt.date())["state"] == "LANDED"
    assert store.get_landing("ROI1", frames[0].dt.date()) is None
    assert sink.sent == []
    app.on_frames(frames[100:200])
    assert app.tracker.get("ROI1") is None


def test_flight_flying_out_of_roi_dropped_on_tick(store):
    # A tracked flight that drifts beyond the capture box is dropped by the
    # maintenance tick - flown away, not landed: no truth row, no alert.
    f = simulate_flight(serial="AWAY1", burst_alt=28000)
    frames = [parse_frame(m) for m in f.frames]
    store.add_subscriber(Subscriber(
        name="alice", lat=frames[0].lat, lon=frames[0].lon, radius_km=1.0,
        ntfy_server="https://ntfy.sh", ntfy_topic="alice", ntfy_token_ref="NTFY"))
    sink = FakeNtfySink()
    cfg = fast_ensemble_cfg()
    cfg.roi.capture_margin_km = 5.0   # tiny box: the simulated drift exits it
    app = App(cfg, store=store, sink=sink)

    app.on_frames(frames[:900])       # ~10 m/s eastward drift → well outside
    assert app.tracker.get("AWAY1") is not None
    app.tick(now=app.tracker.get("AWAY1").last_seen)

    assert app.tracker.get("AWAY1") is None
    assert store.get_landing("AWAY1", frames[0].dt.date()) is None
    assert sink.sent == []
    app.on_frames(frames[900:1000])   # still outside: stays gated out
    assert app.tracker.get("AWAY1") is None


def test_descent_predictions_throttled_not_per_frame(store):
    # ~1 Hz descent telemetry must NOT produce ~1 Hz predictions - at most one
    # per descent_predict_seconds of sonde time (plus the immediate burst one).
    f = simulate_flight(serial="THR1", burst_alt=24000)
    app, _ = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    frames = [parse_frame(m) for m in f.frames]
    app.on_frames(frames)
    day = frames[-1].dt.date()

    preds = store.predictions_for("THR1", day)
    assert preds  # the flight was predicted...
    descent_frames = sum(1 for fr in frames if fr.alt < 24000 and frames[0].alt < fr.alt)
    # ...but far less than once per frame: bounded by sonde-time / throttle (+1
    # for the burst-transition prediction, +1 for boundary slack)
    duration = frames[-1].t - frames[0].t
    assert len(preds) <= duration / app.cfg.predict.descent_predict_seconds + 2
    assert len(preds) >= 5
    # the burst transition itself predicted immediately (alt near burst alt)
    assert preds[0]["alt_at_pred"] > 22000


def test_inbound_only_once_across_many_frames(store):
    f = simulate_flight(serial="ONCE1", burst_alt=26000)
    app, sink = _app_with_sub(store, f.land_lat, f.land_lon, radius=50.0)
    app.on_frames([parse_frame(m) for m in f.frames])
    inbound = [m for m in sink.sent if "inbound" in m.title.lower()]
    assert len(inbound) == 1   # de-duped despite hundreds of descent predictions


def test_gfs_downloader_thread_gated_off_by_default(store):
    app, _ = _app_with_sub(store, 45.0, 7.0)
    # GFS disabled by default → no in-app downloader thread
    assert app.start_gfs_downloader() is False
    assert app._gfs_thread is None


def test_gfs_downloader_thread_runs_in_process(store, monkeypatch, tmp_path):
    # The test is about the timer-thread lifecycle, not downloading: poison
    # herbie so download_gfs_cycle no-ops instantly even where herbie IS
    # installed (otherwise the first pass does real network/file work and
    # outlives the join below).
    import sys
    monkeypatch.setitem(sys.modules, "herbie", None)
    cfg = Config()
    cfg.gfs.enabled = True
    cfg.gfs.download_in_process = True
    cfg.gfs.path = str(tmp_path / "gfs")
    store.add_subscriber(Subscriber(
        name="a", lat=45.0, lon=7.0, radius_km=30,
        ntfy_server="https://ntfy.sh", ntfy_topic="a"))
    app = App(cfg, store=store, sink=FakeNtfySink())
    try:
        assert app.start_gfs_downloader() is True
        assert app._gfs_thread is not None and app._gfs_thread.is_alive()
        # starting again is a no-op while already running
        assert app.start_gfs_downloader() is False
    finally:
        app.stop_gfs_downloader()
        app._gfs_thread.join(timeout=3)
    assert not app._gfs_thread.is_alive()


def test_dem_downloader_gated_off_when_disabled(store):
    cfg = Config()
    cfg.dem.enabled = False
    app = App(cfg, store=store, sink=FakeNtfySink())
    assert app.start_dem_downloader() is False
    assert app._dem_thread is None
    # enabled but opted out of in-process downloads → also no thread
    cfg2 = Config()
    cfg2.dem.enabled = True
    cfg2.dem.download_in_process = False
    app2 = App(cfg2, store=store, sink=FakeNtfySink())
    assert app2.start_dem_downloader() is False


def test_dem_downloader_thread_runs_in_process(store, monkeypatch, tmp_path):
    # Lifecycle test, not a download test: stub the tile fetch so the first
    # pass touches neither the network nor rasterio.
    import tallyho.app as app_mod
    monkeypatch.setattr(app_mod, "download_dem_tiles", lambda *a, **k: [])
    cfg = Config()
    cfg.dem.enabled = True
    cfg.dem.download_in_process = True
    cfg.dem.path = str(tmp_path / "dem")
    store.add_subscriber(Subscriber(
        name="a", lat=45.0, lon=7.0, radius_km=30,
        ntfy_server="https://ntfy.sh", ntfy_topic="a"))
    app = App(cfg, store=store, sink=FakeNtfySink())
    try:
        assert app.start_dem_downloader() is True
        assert app._dem_thread is not None and app._dem_thread.is_alive()
        # starting again is a no-op while already running
        assert app.start_dem_downloader() is False
    finally:
        app.stop_gfs_downloader()   # shared stop event ends all timer threads
        app._dem_thread.join(timeout=3)
    assert not app._dem_thread.is_alive()


def test_dem_download_pass_reloads_ground_model(store, monkeypatch, tmp_path):
    # When a pass downloads tiles, the app rebuilds the ground model so the
    # tracker/predictor (which captured ground_fn at init) see real terrain.
    import tallyho.app as app_mod
    cfg = Config()
    cfg.dem.path = str(tmp_path / "dem")
    cfg.dem.download_check_seconds = 3600.0
    store.add_subscriber(Subscriber(
        name="a", lat=45.0, lon=7.0, radius_km=30,
        ntfy_server="https://ntfy.sh", ntfy_topic="a"))
    app = App(cfg, store=store, sink=FakeNtfySink())

    downloads = iter([["Copernicus_DSM_COG_10_N45_00_E007_00_DEM"], []])

    def fake_download(*a, **k):
        # Stop after this pass - exactly one per loop call. Must go through the
        # public stop (not a raw _gfs_stop.set()): it also sets the kick events
        # the loop now sleeps on, so the wait wakes instead of running out the
        # download_check_seconds cadence.
        app.stop_gfs_downloader()
        return next(downloads)

    monkeypatch.setattr(app_mod, "download_dem_tiles", fake_download)
    reloads = []
    monkeypatch.setattr(app.ground_fn, "reload", lambda: reloads.append(1))
    app._dem_loop()
    assert reloads == [1]   # new tiles → reload
    app._gfs_stop.clear()
    app._dem_loop()
    assert reloads == [1]   # nothing new → no reload


def test_web_server_gated_off_when_disabled(store):
    cfg = Config()
    cfg.web.enabled = False
    app = App(cfg, store=store, sink=FakeNtfySink())
    assert app.start_web_server() is False
    assert app._web_thread is None


def test_web_server_runs_in_process(store):
    pytest.importorskip("fastapi")   # in-app dashboard needs the `api` extra
    pytest.importorskip("uvicorn")
    cfg = Config()
    cfg.web.enabled = True
    cfg.web.port = 0   # let the OS pick a free port (no fixed-port collision)
    app = App(cfg, store=store, sink=FakeNtfySink())
    try:
        assert app.start_web_server() is True
        assert app._web_thread is not None and app._web_thread.is_alive()
        # starting again is a no-op while already running
        assert app.start_web_server() is False
    finally:
        app.stop_web_server()
        app._web_thread.join(timeout=3)


def test_landing_recorded_on_telemetry(store):
    from windfall.geo import haversine_km

    f = simulate_flight(serial="LAND1", burst_alt=27000)
    app, _ = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    frames = [parse_frame(m) for m in f.frames]
    app.on_frames(frames)
    day = frames[-1].dt.date()

    lnd = store.get_landing("LAND1", day)
    assert lnd is not None
    assert lnd["detected_by"] == "telemetry"
    # recorded landing sits on the true landing (last telemetry ≈ touchdown)
    assert haversine_km(lnd["land_lat"], lnd["land_lon"], f.land_lat, f.land_lon) < 1.0


def test_landing_recorded_on_timeout(store):
    # Feed only through descent (no near-ground frame), then let the timeout
    # sweep declare LANDED and record the last-known position as truth.
    f = simulate_flight(serial="TOUT1", burst_alt=20000)
    app, _ = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    # bare timeout sweep: with backfill on, a silent flight is first held for
    # a history-recovery attempt (that flow is covered in test_backfill.py)
    app.cfg.ingest.backfill_enabled = False
    frames = [parse_frame(m) for m in f.frames]
    # Drop the final near-ground frames so the tracker never sees touchdown
    # directly, but leave it low enough (<2 km) to be inside the timeout band.
    descent_frames = [fr for fr in frames if fr.alt > 1000]
    app.on_frames(descent_frames)
    day = descent_frames[-1].dt.date()
    assert store.get_landing("TOUT1", day) is None     # not landed yet

    # advance well past the landed timeout → timeout sweep marks it LANDED
    app.tick(now=descent_frames[-1].dt + timedelta(seconds=10_000))
    lnd = store.get_landing("TOUT1", day)
    assert lnd is not None and lnd["detected_by"] == "timeout"


def test_timeout_landing_retracted_when_sonde_still_falling(store):
    # A timeout landing is provisional. When frames then arrive well below the
    # declared landing altitude (the silence was a reception gap), the
    # provisional truth row and LANDED alert are retracted, the descent
    # resumes, and the real landing re-records and re-alerts at the corrected
    # position (regression: the LANDED husk used to swallow the late frames).
    f = simulate_flight(serial="RELAND1", burst_alt=20000)
    app, sink = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    app.cfg.ingest.backfill_enabled = False
    frames = [parse_frame(m) for m in f.frames]
    for i, fr in enumerate(frames):
        app.process_frame(fr)
        fl = app.tracker.get("RELAND1")
        if fl is not None and fl.state == FlightState.DESCENT and fr.alt < 1000:
            break
    day = fr.dt.date()
    app.tick(now=fr.dt + timedelta(seconds=app.cfg.tracker.landed_timeout_seconds + 60))
    lnd = store.get_landing("RELAND1", day)
    assert lnd is not None and lnd["detected_by"] == "timeout"
    assert sum("LANDED" in m.title for m in sink.sent) == 1

    # reception returns: the sonde is still falling - feed the real descent tail
    app.on_frames(frames[i + 1:])
    lnd2 = store.get_landing("RELAND1", day)
    assert lnd2 is not None and lnd2["detected_by"] == "telemetry"
    assert lnd2["land_alt"] < lnd["land_alt"]
    # the stale LANDED alert row was retracted, so the corrected one re-sent
    assert sum("LANDED" in m.title for m in sink.sent) == 2


def test_timeout_landing_refined_by_late_ground_pings(store):
    # A timeout-landed sonde keeps pinging from the ground; those fixes are
    # better landing truth than the last frame heard before the silence. Fixes
    # that meaningfully move (or drop) refresh the recorded landing - without
    # reopening the flight or re-alerting. Fixes within GPS wander change nothing.
    from windfall.models import Frame

    f = simulate_flight(serial="REFINE1", burst_alt=20000)
    app, sink = _app_with_sub(store, f.land_lat, f.land_lon, radius=40.0)
    app.cfg.ingest.backfill_enabled = False
    frames = [parse_frame(m) for m in f.frames]
    for fr in frames:
        app.process_frame(fr)
        fl = app.tracker.get("REFINE1")
        if fl is not None and fl.state == FlightState.DESCENT and fr.alt < 1000:
            break
    day = fr.dt.date()
    app.tick(now=fr.dt + timedelta(seconds=app.cfg.tracker.landed_timeout_seconds + 60))
    lnd = store.get_landing("REFINE1", day)
    assert lnd is not None and lnd["detected_by"] == "timeout"

    lat, lon, t0 = fl.last_lat, fl.last_lon, fl.last_seen
    move = app.cfg.tracker.landing_refine_move_m
    drop = app.cfg.tracker.landing_refine_alt_m

    def ping(alt, secs, dlat=0.0):
        dt = t0 + timedelta(seconds=secs)
        return Frame(serial="REFINE1", lat=lat + dlat, lon=lon, alt=alt,
                     t=dt.timestamp(), dt=dt, frame=int(secs), type="RS41")

    # within GPS wander of the recorded fix: the row must not churn
    app.process_frame(ping(lnd["land_alt"] - drop / 2, 30))
    assert store.get_landing("REFINE1", day) == lnd

    # clearly lower (still inside the redescent noise band → no resume):
    # the recorded landing follows the fresher fix
    app.process_frame(ping(lnd["land_alt"] - drop - 20, 60))
    lnd2 = store.get_landing("REFINE1", day)
    assert lnd2["land_alt"] == pytest.approx(lnd["land_alt"] - drop - 20)
    assert lnd2["landed_at"] == lnd["landed_at"]   # same landing, better fix
    assert lnd2["detected_by"] == "timeout"

    # moved horizontally past the threshold: position refreshes too
    dlat = (move * 3) / 111_000.0
    app.process_frame(ping(lnd2["land_alt"], 90, dlat=dlat))
    lnd3 = store.get_landing("REFINE1", day)
    assert lnd3["land_lat"] == pytest.approx(lat + dlat)

    # still the same LANDED flight (no ghost, no resume), and no extra alert
    assert app.tracker.get("REFINE1").state == FlightState.LANDED
    assert sum("LANDED" in m.title for m in sink.sent) == 1


def test_predict_active_predicts_ascending_sonde(store):
    # An ascending sonde gets a pre-burst landing prediction + path on the sweep
    # (winds above the measured range must be available → GFS source wired).
    f = simulate_flight(serial="ASC1", burst_alt=30000)
    app, _ = _app_with_sub(store, 45.0, 7.0, radius=40.0, gfs_source=_static_gfs())
    frames = [parse_frame(m) for m in f.frames]
    ascent = [fr for fr in frames if fr.alt < 15000][:200]
    app.on_frames(ascent)
    fl = app.tracker.get("ASC1")
    assert fl.state.value == "ASCENT"

    n = app.predict_active()
    assert n == 1
    paths = store.latest_paths_for_active()
    assert len(paths) == 1 and paths[0]["serial"] == "ASC1"
    assert len(paths[0]["path"]) >= 2          # a real polyline, not a dot
    assert store.latest_prediction("ASC1", ascent[-1].dt.date()) is not None


def test_predict_active_gated_off(store):
    cfg = Config()
    cfg.predict.preburst_enabled = False
    f = simulate_flight(serial="ASC2", burst_alt=30000)
    store.add_subscriber(Subscriber(
        name="a", lat=45.0, lon=7.0, radius_km=40,
        ntfy_server="https://ntfy.sh", ntfy_topic="a"))
    app = App(cfg, store=store, sink=FakeNtfySink())
    frames = [parse_frame(m) for m in f.frames]
    app.on_frames([fr for fr in frames if fr.alt < 15000][:200])
    assert app.predict_active() == 0
    assert store.latest_paths_for_active() == []


def test_health(store):
    f = simulate_flight(serial="H1", burst_alt=20000)
    app, _ = _app_with_sub(store, f.land_lat, f.land_lon)
    frames = [parse_frame(m) for m in f.frames]
    app.on_frames(frames[:5])
    # last_frame_at is the frames' ARRIVAL wall-clock time, not frame.dt -
    # sonde clock skew (or one garbage future-dated frame) must not stretch
    # the health window past a real stall.
    assert app.last_frame_at is not None
    assert abs((datetime.now(timezone.utc) - app.last_frame_at).total_seconds()) < 5
    assert app.healthy()
    assert not app.healthy(now=app.last_frame_at + timedelta(seconds=10_000))


def test_roi_change_kicks_downloaders(store):
    """Adding the first location must wake the GFS/DEM downloader threads
    (which otherwise sleep out their cadence - 6 h for GFS) via the kick
    events their loops wait on."""
    app = App(Config(), store=store, sink=FakeNtfySink())
    # fresh install: no subscribers, no ROI, nothing to kick
    assert app._capture_box is None
    app._gfs_kick.clear(), app._dem_kick.clear()
    app.reload_subscribers()
    assert not app._gfs_kick.is_set() and not app._dem_kick.is_set()

    # first location appears → ROI exists → both downloaders kicked
    store.add_subscriber(Subscriber(
        name="a", lat=45.0, lon=7.0, radius_km=40,
        ntfy_server="https://ntfy.sh", ntfy_topic="a"))
    app.reload_subscribers()
    assert app._capture_box is not None
    assert app._gfs_kick.is_set() and app._dem_kick.is_set()

    # unchanged ROI on the next periodic reload → no kick
    app._gfs_kick.clear(), app._dem_kick.clear()
    app.reload_subscribers()
    assert not app._gfs_kick.is_set() and not app._dem_kick.is_set()


def test_landing_truth_requires_real_fall(store):
    # A "landing" whose fix sits just below the flight's apogee never saw a
    # real descent - e.g. GPS settling on the launch pad walked the tracker to
    # LANDED pre-launch (26004618, 2026-07-12). No truth row: a poisoned one
    # corrupts calibration and outlives rebuilds of the flight.
    from datetime import date

    from windfall.tracker import Flight

    app, _ = _app_with_sub(store, 39.1, -108.5)
    pad = Flight(serial="PAD1", launch_day=date(2026, 7, 12),
                 last_lat=39.12, last_lon=-108.52, last_alt=1460.0,
                 max_alt=1463.0)
    app._record_landing(pad, now=datetime.now(timezone.utc), detected_by="telemetry")
    assert store.get_landing("PAD1", pad.launch_day) is None

    real = Flight(serial="REAL1", launch_day=date(2026, 7, 12),
                  last_lat=39.4, last_lon=-109.3, last_alt=1800.0,
                  max_alt=34000.0)
    app._record_landing(real, now=datetime.now(timezone.utc), detected_by="telemetry")
    assert store.get_landing("REAL1", real.launch_day) is not None
