"""Tests for the flight state machine."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from windfall.config import Config
from windfall.models import Frame, FlightState
from tests.conftest import FakeFlightStore as Store
from windfall.telemetry import parse_frame
from windfall.tracker import FlightTracker, TrackerEvent
from tests.conftest import simulate_flight

BASE = datetime(2026, 6, 7, tzinfo=timezone.utc)


def run_flight(tracker, frames):
    states = []
    all_events = []
    for f in frames:
        flight, events = tracker.update(f)
        states.append(flight.state)
        all_events.extend(events)
    return flight, states, all_events


def test_full_flight_transitions(flight):
    cfg = Config()
    tracker = FlightTracker(cfg)
    frames = [parse_frame(m) for m in flight.frames]
    fl, states, events = run_flight(tracker, frames)
    assert TrackerEvent.NEW_FLIGHT in events
    assert TrackerEvent.BURST in events
    assert TrackerEvent.LANDED in events
    assert fl.state == FlightState.LANDED
    # ascent profile was built
    assert not fl.profile.is_empty()
    # descent samples collected
    assert len(fl.descent_samples) > 10
    # ordering: first ASCENT, eventually DESCENT, finally LANDED
    assert states[0] == FlightState.ASCENT
    assert FlightState.DESCENT in states


def test_robust_ascent_rate_and_windowed_descent_samples(flight):
    cfg = Config()
    tracker = FlightTracker(cfg)
    frames = [parse_frame(m) for m in flight.frames]
    fl, _, _ = run_flight(tracker, frames)
    # the synthetic flight climbs at 5 m/s → the windowed median recovers it
    assert fl.robust_ascent_rate() == pytest.approx(5.0, abs=0.3)
    # late descent samples (regression-based v_obs) are close to the true
    # ballistic speed at their altitude
    from windfall.atmosphere import isa_density
    for s in fl.descent_samples[-20:]:
        truth = 5.5 * isa_density(s.alt) ** -0.5
        assert s.v_obs == pytest.approx(truth, rel=0.15)


def test_descent_b_fitted_and_persisted_on_landing(flight):
    cfg = Config()
    store = Store(":memory:")
    try:
        tracker = FlightTracker(cfg, store=store)
        frames = [parse_frame(m) for m in flight.frames]
        fl, _, _ = run_flight(tracker, frames)
        assert fl.state == FlightState.LANDED
        # the simulated chute had B=5.5; the fit at landing recovers it
        assert fl.descent_b == pytest.approx(5.5, rel=0.1)
        row = store.get_flight(fl.serial, fl.launch_day)
        assert row["descent_b"] == pytest.approx(fl.descent_b)
    finally:
        store.close()


def test_burst_records_burst_alt(flight):
    cfg = Config()
    tracker = FlightTracker(cfg)
    frames = [parse_frame(m) for m in flight.frames]
    fl, _, _ = run_flight(tracker, frames)
    assert fl.burst_alt == pytest.approx(flight.burst_alt, abs=200)


def test_floater_does_not_burst():
    # A floating sonde: climbs to 18 km then holds level for a long time.
    cfg = Config()
    tracker = FlightTracker(cfg)
    frames = []
    t = 0.0
    alt = 200.0
    lat, lon = 45.0, 7.0
    fn = 0
    # ascent
    while alt < 18000:
        frames.append(_mk(lat, lon, alt, t, fn))
        alt += 5.0
        t += 1.0
        fn += 1
    # float: hold altitude (tiny noise, <1 m/s rate) for 10 minutes
    for _ in range(600):
        frames.append(_mk(lat, lon, alt + math.sin(t) * 0.3, t, fn))
        t += 1.0
        fn += 1
    fl, states, events = run_flight(tracker, frames)
    assert TrackerEvent.FLOAT in events
    assert TrackerEvent.BURST not in events
    assert fl.state == FlightState.FLOAT


def _frames_from_alts(serial, alts, lat=45.0, lon=7.0, t0=0.0, dt=1.0, type="RS41"):
    """Build a frame stream that walks a given altitude profile at a fixed
    position (position is irrelevant to burst/float/landed state logic)."""
    frames = []
    for i, alt in enumerate(alts):
        d = BASE + timedelta(seconds=t0 + i * dt)
        frames.append(Frame(serial=serial, lat=lat, lon=lon, alt=float(alt),
                            t=d.timestamp(), dt=d, frame=i, type=type))
    return frames


def test_first_heard_descending_records_no_burst():
    """A sonde first heard already falling (launched out of range, drifted into
    reception on the way down) must NOT be logged as a burst at its first-heard
    altitude - that bogus low 'burst' is what used to poison the per-site
    climatology. It is tracked as a genuine descent instead: state reaches
    DESCENT/LANDED, descent samples are collected, but burst_alt stays None and
    no launch site is claimed."""
    cfg = Config()
    tracker = FlightTracker(cfg)
    # first heard at 8 km, monotonic descent all the way to the ground
    alts = [a for a in range(8000, 240, -8)]
    frames = _frames_from_alts("FELLIN", alts)
    fl, states, events = run_flight(tracker, frames)
    assert FlightState.DESCENT in states
    assert fl.state == FlightState.LANDED
    assert fl.burst_alt is None           # never observed a burst
    assert fl.launch_lat is None          # first heard high → no launch site
    assert len(fl.descent_samples) > 10   # the descent itself is real data


def test_wind_dip_reverts_false_burst():
    """A strong downdraft (or a GPS spike poisoning max_alt) can shove a still-
    rising balloon >burst_drop_m below its running max and briefly look like a
    burst. When it climbs back above that apogee the false burst must be undone,
    so the REAL burst higher up is the one recorded - not the dip altitude."""
    cfg = Config()
    tracker = FlightTracker(cfg)
    alts, a = [], 200.0
    while a < 15000:            # climb to a first apogee at ~15 km
        alts.append(a); a += 5.0
    while a > 15000 - 600:      # downdraft shoves it 600 m below max (> 300 m)
        alts.append(a); a -= 10.0
    while a < 30000:            # recover, climb past the false apogee to real burst
        alts.append(a); a += 5.0
    real_top = a
    while a > 29000:            # then the real descent
        alts.append(a); a -= 8.0
    fl, states, events = run_flight(tracker, _frames_from_alts("DIP", alts))
    # the dip drove it to DESCENT, then it re-ascended (revert), then burst for real
    first_desc = states.index(FlightState.DESCENT)
    assert FlightState.ASCENT in states[first_desc:]        # reverted back to ASCENT
    assert fl.burst_alt == pytest.approx(real_top, abs=300)  # real apogee, not the dip
    assert fl.burst_alt > 25000


def test_serial_reuse_starts_new_flight():
    cfg = Config()
    tracker = FlightTracker(cfg)
    # Day 1 flight: quick up and down to LANDED.
    day1 = simulate_flight(serial="REUSE1", start=datetime(2026, 6, 1, tzinfo=timezone.utc),
                           burst_alt=12000)
    frames1 = [parse_frame(m) for m in day1.frames]
    run_flight(tracker, frames1)
    fl1 = tracker.get("REUSE1")
    assert fl1.state == FlightState.LANDED
    key1 = (fl1.serial, fl1.launch_day)

    # Day 2: same serial, a genuine new launch. The old flight is still in memory
    # (no timeout sweep ran here), so the reuse is recognised once the new sonde
    # has clearly climbed away from the old landing altitude - NOT on the first
    # ground-level frame, which is indistinguishable from a post-landing ping.
    day2 = simulate_flight(serial="REUSE1", start=datetime(2026, 6, 2, tzinfo=timezone.utc),
                           burst_alt=12000)
    frames2 = [parse_frame(m) for m in day2.frames]
    fl2, _, events2 = run_flight(tracker, frames2)
    assert TrackerEvent.NEW_FLIGHT in events2
    assert (fl2.serial, fl2.launch_day) != key1   # distinct flight identity
    assert fl2.launch_day != fl1.launch_day
    assert fl2.state == FlightState.LANDED


def test_landed_sonde_pinging_from_ground_not_resurrected():
    # A radiosonde keeps transmitting for hours after it lands. Its noisy ground
    # fixes - including the occasional high spike and the crossing of the UTC day
    # boundary - must NOT resurrect it into a ghost flight stuck in ASCENT. Only
    # a *sustained* climb is a real relaunch (regression for the post-landing
    # "ascending again" ghost).
    cfg = Config()
    tracker = FlightTracker(cfg)
    day1 = simulate_flight(serial="GNDPING",
                           start=datetime(2026, 6, 1, 22, tzinfo=timezone.utc),
                           burst_alt=12000)
    run_flight(tracker, [parse_frame(m) for m in day1.frames])
    landed = tracker.get("GNDPING")
    assert landed.state == FlightState.LANDED
    key = (landed.serial, landed.launch_day)
    base_alt, lat, lon, t0 = landed.last_alt, landed.last_lat, landed.last_lon, landed.last_seen

    def ping(alt, secs):
        dt = t0 + timedelta(seconds=secs)
        return Frame(serial="GNDPING", lat=lat, lon=lon, alt=alt, t=dt.timestamp(),
                     dt=dt, frame=int(secs), type="RS41")

    climb = cfg.tracker.new_ascent_climb_m
    pings = [
        ping(base_alt + 20, 30),
        ping(base_alt - 15, 60),
        ping(base_alt + climb + 700, 90),    # one wild ground fix above the climb bar
        ping(base_alt + 10, 120),            # ...not sustained → must not count
        ping(base_alt + 5, 7200),            # ~2 h on: now the next UTC day
        ping(base_alt + 25, 7230),
    ]
    for f in pings:
        fl, events = tracker.update(f)
        assert TrackerEvent.NEW_FLIGHT not in events
        assert fl.state == FlightState.LANDED
        assert (fl.serial, fl.launch_day) == key


def test_timeout_marks_landed():
    cfg = Config()
    tracker = FlightTracker(cfg)
    # build a descending flight but stop feeding frames while low
    f = simulate_flight(serial="GAP1", burst_alt=12000)
    frames = [parse_frame(m) for m in f.frames]
    # feed frames until the descent passes below 1500 m, then simulate signal loss
    seen_descent = False
    for fr in frames:
        flight, events = tracker.update(fr)
        if flight.state == FlightState.DESCENT:
            seen_descent = True
        if seen_descent and fr.alt < 1500:
            break
    assert flight.state == FlightState.DESCENT
    later = flight.last_seen + timedelta(seconds=cfg.tracker.landed_timeout_seconds + 10)
    out = tracker.check_timeouts(later)
    assert any(ev == TrackerEvent.LANDED for _, ev in out)
    assert flight.state == FlightState.LANDED


def _timeout_landed_flight(tracker, cfg, serial):
    """Drive a simulated flight into DESCENT, cut the frames while low, and let
    the timeout sweep declare a (provisional) landing. Returns the flight and
    the remaining, unfed frames of the real descent."""
    f = simulate_flight(serial=serial, burst_alt=12000)
    frames = [parse_frame(m) for m in f.frames]
    seen_descent = False
    for i, fr in enumerate(frames):
        flight, _ = tracker.update(fr)
        if flight.state == FlightState.DESCENT:
            seen_descent = True
        if seen_descent and fr.alt < 1500:
            break
    assert flight.state == FlightState.DESCENT
    later = flight.last_seen + timedelta(
        seconds=cfg.tracker.landed_timeout_seconds + 10)
    assert tracker.check_timeouts(later) == [(flight, TrackerEvent.LANDED)]
    assert flight.state == FlightState.LANDED
    assert flight.landed_by_timeout
    return flight, frames[i + 1:]


def test_timeout_landing_reopens_when_sonde_still_falling():
    # A DESCENT flight silent while low is declared LANDED by the timeout
    # sweep - provisionally. When frames then arrive well BELOW the declared
    # landing altitude, the silence was a reception gap and the sonde is still
    # falling: the descent must reopen (RESUMED) and the real landing land the
    # flight for real - not be swallowed by the LANDED husk (seen live:
    # timeout-landed at 1134 m, later frames at 832 m changed nothing).
    cfg = Config()
    tracker = FlightTracker(cfg)
    flight, rest = _timeout_landed_flight(tracker, cfg, "RELAND")
    anchor = flight.landed_alt
    assert anchor is not None

    events = []
    for fr in rest:
        fl, evs = tracker.update(fr)
        events.extend(evs)
    assert TrackerEvent.RESUMED in events
    assert TrackerEvent.NEW_FLIGHT not in events    # same flight, not a ghost
    assert TrackerEvent.LANDED in events            # the real landing
    assert fl.state == FlightState.LANDED
    assert not fl.landed_by_timeout                 # telemetry-confirmed now
    assert fl.landed_alt < anchor - cfg.tracker.redescent_drop_m
    # the close-out fit was retracted and refit over the full descent
    assert fl.descent_b == pytest.approx(5.5, rel=0.1)


def test_timeout_landing_not_reopened_by_ground_noise():
    # A genuinely landed sonde keeps pinging from the ground with noisy GPS
    # fixes. Fixes near the landing altitude - including an isolated deep
    # downward spike - must NOT reopen the descent; only a sustained run of
    # clearly-lower fixes does (the still-falling case above).
    cfg = Config()
    tracker = FlightTracker(cfg)
    flight, _ = _timeout_landed_flight(tracker, cfg, "GNDNOISE")
    anchor, lat, lon = flight.landed_alt, flight.last_lat, flight.last_lon
    t0 = flight.last_seen
    drop = cfg.tracker.redescent_drop_m

    def ping(alt, secs):
        dt = t0 + timedelta(seconds=secs)
        return Frame(serial="GNDNOISE", lat=lat, lon=lon, alt=alt,
                     t=dt.timestamp(), dt=dt, frame=int(secs), type="RS41")

    pings = [
        ping(anchor - 30, 30),               # inside the noise band
        ping(anchor + 20, 60),
        ping(anchor - drop - 200, 90),       # one wild low fix...
        ping(anchor - drop - 250, 120),      # ...even two in a row...
        ping(anchor - 10, 150),              # ...not sustained → must not count
        ping(anchor - 40, 180),
    ]
    for f in pings:
        fl, events = tracker.update(f)
        assert TrackerEvent.RESUMED not in events
        assert fl.state == FlightState.LANDED
    assert fl.landed_by_timeout                     # still just provisional


def test_descent_lost_above_ground_band_expires():
    # A descending sonde dropping below the radio horizon while still well
    # above the ground band must not linger as "active" for hours: after the
    # descent-lost timeout it is closed out - as EXPIRED, not as a landing,
    # since its last fix was high.
    cfg = Config()
    tracker = FlightTracker(cfg)
    f = simulate_flight(serial="LOWGONE", burst_alt=12000)
    frames = [parse_frame(m) for m in f.frames]
    seen_descent = False
    for fr in frames:
        flight, _ = tracker.update(fr)
        if flight.state == FlightState.DESCENT:
            seen_descent = True
        if seen_descent and fr.alt < 3500:
            break
    assert flight.state == FlightState.DESCENT
    assert flight.last_alt > 2500   # above the landed band (flat ground + 2000)
    # shortly after the landed timeout: too high to be a landing → still active
    t1 = flight.last_seen + timedelta(seconds=cfg.tracker.landed_timeout_seconds + 30)
    assert tracker.check_timeouts(t1) == []
    assert flight.state == FlightState.DESCENT
    # after the descent-lost timeout: closed out without a LANDED event
    t2 = flight.last_seen + timedelta(seconds=cfg.tracker.descent_lost_timeout_seconds + 30)
    assert tracker.check_timeouts(t2) == [(flight, TrackerEvent.EXPIRED)]
    assert flight.state == FlightState.LANDED


def test_launch_site_only_claimed_near_ground():
    cfg = Config()
    tracker = FlightTracker(cfg)

    def _frame(serial, alt, secs):
        dt = BASE + timedelta(seconds=secs)
        return Frame(serial=serial, lat=45.0, lon=7.0, alt=alt, t=dt.timestamp(),
                     dt=dt, frame=int(secs), type="RS41")

    # first heard mid-air: the launch site is unknown, no rocket on the map
    high, _ = tracker.update(_frame("MIDAIR", 5000.0, 0))
    assert high.first_seen is not None
    assert high.launch_lat is None and high.launch_lon is None
    # first heard near the ground: that IS the launch site
    low, _ = tracker.update(_frame("NEARGND", 300.0, 0))
    assert low.launch_lat == pytest.approx(45.0)
    assert low.launch_lon == pytest.approx(7.0)


def test_stale_airborne_flight_expires_without_landing():
    # Signal lost mid-ascent and never regained: the flight must eventually stop
    # counting as active (no ghost sonde on the map forever), but its mid-air
    # last position is NOT a landing - EXPIRED, not LANDED, is emitted.
    cfg = Config()
    tracker = FlightTracker(cfg)
    for s in range(0, 1200, 10):
        flight, _ = tracker.update(_mk(45.0, 7.0, 200.0 + 5 * s, s, s))
    assert flight.state == FlightState.ASCENT
    later = flight.last_seen + timedelta(seconds=cfg.tracker.stale_flight_seconds + 60)
    out = tracker.check_timeouts(later)
    assert out == [(flight, TrackerEvent.EXPIRED)]
    assert flight.state == FlightState.LANDED
    # the next sweep evicts the long-silent finished flight from memory
    assert tracker.check_timeouts(later) == []
    assert tracker.get("FLOATER") is None
    assert tracker.flights == {}


def test_lost_ascent_flight_expires_before_stale_sweep():
    # An ascending sonde that goes silent for good burst and landed unheard
    # (daemon stopped with sondes aloft, dead sonde): it must close out after
    # ascent_lost_timeout_seconds, not sit "mid-air" for stale_flight_seconds.
    cfg = Config()
    tracker = FlightTracker(cfg)
    for s in range(0, 1200, 10):
        flight, _ = tracker.update(_mk(45.0, 7.0, 200.0 + 5 * s, s, s))
    assert flight.state == FlightState.ASCENT
    # still within the ascent-lost window: nothing happens
    t1 = flight.last_seen + timedelta(seconds=cfg.tracker.ascent_lost_timeout_seconds - 60)
    assert tracker.check_timeouts(t1) == []
    assert flight.state == FlightState.ASCENT
    # past it (but well before the stale sweep): EXPIRED, no landing truth
    t2 = flight.last_seen + timedelta(seconds=cfg.tracker.ascent_lost_timeout_seconds + 60)
    assert t2 < flight.last_seen + timedelta(seconds=cfg.tracker.stale_flight_seconds)
    assert tracker.check_timeouts(t2) == [(flight, TrackerEvent.EXPIRED)]
    assert flight.state == FlightState.LANDED


def test_float_flight_is_exempt_from_ascent_lost_timeout():
    # Floaters legitimately stay aloft for hours; a silent FLOAT flight keeps
    # the long stale sweep, not the ascent-lost timeout.
    cfg = Config()
    tracker = FlightTracker(cfg)
    frames, t, alt, fn = [], 0.0, 200.0, 0
    while alt < 18000:
        frames.append(_mk(45.0, 7.0, alt, t, fn))
        alt += 5.0
        t += 1.0
        fn += 1
    for _ in range(600):
        frames.append(_mk(45.0, 7.0, alt + math.sin(t) * 0.3, t, fn))
        t += 1.0
        fn += 1
    flight, _, _ = run_flight(tracker, frames)
    assert flight.state == FlightState.FLOAT
    t1 = flight.last_seen + timedelta(seconds=cfg.tracker.ascent_lost_timeout_seconds + 60)
    assert tracker.check_timeouts(t1) == []
    assert flight.state == FlightState.FLOAT
    t2 = flight.last_seen + timedelta(seconds=cfg.tracker.stale_flight_seconds + 60)
    assert tracker.check_timeouts(t2) == [(flight, TrackerEvent.EXPIRED)]


def test_hold_defers_timeout_closeout():
    # A serial in `hold` (its history fetch is in flight) must survive the
    # sweep untouched; once released it closes out as usual.
    cfg = Config()
    tracker = FlightTracker(cfg)
    for s in range(0, 1200, 10):
        flight, _ = tracker.update(_mk(45.0, 7.0, 200.0 + 5 * s, s, s))
    assert flight.state == FlightState.ASCENT
    later = flight.last_seen + timedelta(seconds=cfg.tracker.stale_flight_seconds + 60)
    assert tracker.check_timeouts(later, hold={flight.serial}) == []
    assert flight.state == FlightState.ASCENT
    assert tracker.check_timeouts(later) == [(flight, TrackerEvent.EXPIRED)]


def test_drop_evicts_flight_without_landing_truth():
    # drop() = the caller decided this flight is no longer interesting (e.g.
    # it left the region of interest): closed out immediately, evicted from
    # memory, and the serial re-tracks as a brand-new flight if it reappears.
    cfg = Config()
    store = Store(":memory:")
    tracker = FlightTracker(cfg, store=store)
    for s in range(0, 600, 10):
        flight, _ = tracker.update(_mk(45.0, 7.0, 200.0 + 5 * s, s, s))
    assert flight.state == FlightState.ASCENT

    tracker.drop(flight)
    assert tracker.get("FLOATER") is None
    assert tracker.flights == {}
    assert store.get_flight("FLOATER", flight.launch_day)["state"] == "LANDED"
    # a later frame starts over rather than resurrecting the dropped flight
    fresh, events = tracker.update(_mk(45.0, 7.0, 3300.0, 620, 620))
    assert TrackerEvent.NEW_FLIGHT in events
    assert fresh is not flight


def test_forget_evicts_without_persisting():
    # forget() = the caller is about to rebuild the flight (e.g. replaying
    # fetched history): the in-memory state vanishes, but unlike drop() nothing
    # is written - the persisted row keeps whatever state it had.
    cfg = Config()
    store = Store(":memory:")
    tracker = FlightTracker(cfg, store=store)
    for s in range(0, 300, 10):
        flight, _ = tracker.update(_mk(45.0, 7.0, 200.0 + 5 * s, s, s))
    assert flight.state == FlightState.ASCENT

    gone = tracker.forget("FLOATER")
    assert gone is flight
    assert tracker.get("FLOATER") is None
    assert tracker.flights == {}
    # not closed out: the store still says ASCENT (cleanup is the caller's job)
    assert store.get_flight("FLOATER", flight.launch_day)["state"] == "ASCENT"
    # replaying starts a brand-new flight, ungated by the old one
    fresh, events = tracker.update(_mk(45.0, 7.0, 200.0, 0, 0))
    assert TrackerEvent.NEW_FLIGHT in events
    assert fresh is not flight

    assert tracker.forget("UNKNOWN") is None


def test_flown_track_captured_and_downsampled():
    cfg = Config()
    store = Store(":memory:")
    tracker = FlightTracker(cfg, store=store)
    sim = simulate_flight(serial="TRK1")
    frames = [parse_frame(m) for m in sim.frames]
    for fr in frames:
        tracker.update(fr)
    flight = tracker.get("TRK1")

    track = store.track_for("TRK1", flight.launch_day)
    # the first frame is the launch point and is always kept, at the launch site
    assert track[0]["lat"] == pytest.approx(sim.launch_lat, abs=1e-6)
    assert track[0]["lon"] == pytest.approx(sim.launch_lon, abs=1e-6)
    # times are strictly increasing (oldest → newest)
    ts = [r["t"] for r in track]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    # downsampled: a real shape, but far fewer points than the raw frames
    assert 5 < len(track) < len(frames)
    store.close()


def _mk(lat, lon, alt, secs, fn):
    dt = BASE + timedelta(seconds=secs)
    return Frame(serial="FLOATER", lat=lat, lon=lon, alt=alt, t=dt.timestamp(),
                 dt=dt, frame=fn, type="RS41")


def test_restart_resumes_flight_and_keeps_launch_site(flight):
    """A restart mid-flight must *resume* the airborne sonde, not treat the next
    frame as a fresh launch - which would move the launch marker to the sonde's
    current position and discard state/profile. Regression for the launch icon
    jumping next to the sonde after a container restart."""
    cfg = Config()
    store = Store(":memory:")
    frames = [parse_frame(m) for m in flight.frames]
    serial = frames[0].serial
    split = len(frames) // 2

    # First run: ingest the first half; the sonde is still airborne.
    t1 = FlightTracker(cfg, store=store)
    for fr in frames[:split]:
        t1.update(fr)
    fl1 = t1.get(serial)
    assert fl1.state != FlightState.LANDED
    assert fl1.launch_lat == pytest.approx(flight.launch_lat, abs=1e-6)
    assert fl1.launch_lon == pytest.approx(flight.launch_lon, abs=1e-6)
    saved_state, saved_max_alt = fl1.state, fl1.max_alt

    # Restart: brand-new tracker, same store. Rehydrate active flights.
    t2 = FlightTracker(cfg, store=store)
    assert t2.load_active() == 1
    resumed = t2.get(serial)
    assert resumed is not None
    # State, accumulated max altitude, and the ascent profile all survived.
    assert resumed.state == saved_state
    assert resumed.max_alt == pytest.approx(saved_max_alt)
    assert not resumed.profile.is_empty()

    # Ingest the rest through the restarted tracker.
    for fr in frames[split:]:
        t2.update(fr)
    final = t2.get(serial)
    # The launch marker is still the real launch site, NOT the mid-air resume point.
    assert final.launch_lat == pytest.approx(flight.launch_lat, abs=1e-6)
    assert final.launch_lon == pytest.approx(flight.launch_lon, abs=1e-6)
    assert final.state == FlightState.LANDED
    # The DB row the map reads from also holds the real launch site.
    row = store.get_flight(serial, final.launch_day)
    assert row["launch_lat"] == pytest.approx(flight.launch_lat, abs=1e-6)
    store.close()


def test_teleport_frame_rejected(flight):
    """A single corrupt frame (DFM-style teleport) must not poison ``max_alt``
    - which would degrade burst detection to rate-only forever and record a
    garbage burst_alt into climatology - and must not emit events."""
    from windfall.models import Frame as F

    cfg = Config()
    tracker = FlightTracker(cfg)
    frames = [parse_frame(m) for m in flight.frames]
    for fr in frames[:200]:
        fl, _ = tracker.update(fr)
    max_alt_before = fl.max_alt

    g = frames[200]
    bad = F(serial=g.serial, lat=g.lat, lon=g.lon, alt=g.alt + 20_000.0,
            t=g.t, dt=g.dt, frame=g.frame, type=g.type)
    fl2, events = tracker.update(bad)
    assert events == []
    assert fl2.max_alt == pytest.approx(max_alt_before, abs=10)

    # the flight resumes cleanly and still lives a normal full life cycle
    for fr in frames[200:]:
        fl3, _ = tracker.update(fr)
    assert fl3.state == FlightState.LANDED
    assert fl3.burst_alt == pytest.approx(flight.burst_alt, abs=200)


def test_consistent_teleports_eventually_accepted():
    """A sonde that is consistently *there* is not glitching - after
    ``glitch_accept_after`` consecutive rejections the new position wins,
    instead of the tracker fighting the data forever."""
    cfg = Config()
    tracker = FlightTracker(cfg)

    def mk(lat, alt, secs):
        dt = BASE + timedelta(seconds=secs)
        return Frame(serial="JUMP1", lat=lat, lon=7.0, alt=alt,
                     t=dt.timestamp(), dt=dt, frame=int(secs), type="RS41")

    for s in range(60):
        fl, _ = tracker.update(mk(45.0, 200.0 + 5 * s, s))
    # ~111 km northward in one second, then it stays there
    n_rejected = 0
    for s in range(60, 70):
        fl, _ = tracker.update(mk(46.0, 200.0 + 5 * s, s))
        if fl.last_lat < 45.5:
            n_rejected += 1
        else:
            break
    assert 1 <= n_rejected <= cfg.tracker.glitch_accept_after
    assert fl.last_lat == pytest.approx(46.0)


def test_restart_mid_descent_resumes_ballistic_fit(flight):
    """A daemon restart mid-descent keeps the checkpointed descent samples and
    the burst anchors, so the next prediction still fits B from real data
    instead of degrading to the single-point shortcut and re-discarding 45 s
    of fresh samples as 'post-burst transient'."""
    from windfall.descent import fit_descent

    cfg = Config()
    store = Store(":memory:")
    t1 = FlightTracker(cfg, store=store)
    frames = [parse_frame(m) for m in flight.frames]
    for fr in frames:
        fl, _ = t1.update(fr)
        if fl.state == FlightState.DESCENT and fr.alt < 20000:
            break
    assert fl.state == FlightState.DESCENT
    assert fl.burst_t is not None

    t2 = FlightTracker(cfg, store=store)
    assert t2.load_active() == 1
    resumed = t2.get(fl.serial)
    assert resumed.burst_t == pytest.approx(fl.burst_t)
    assert resumed.burst_alt == pytest.approx(fl.burst_alt)
    # samples survived up to the last ~30-sample checkpoint stride
    assert len(resumed.descent_samples) >= len(fl.descent_samples) - 30
    model = fit_descent(resumed.descent_samples, cfg.descent,
                        burst_t=resumed.burst_t, burst_alt=resumed.burst_alt)
    assert model is not None
    assert model.n_points >= cfg.descent.min_fit_points
    assert model.b == pytest.approx(5.5, rel=0.1)


def test_pad_gps_settling_never_enters_descent():
    # A sonde cold-starting on the launch pad can emit a sustained fake
    # "descent" while its GPS altitude settles downward. That used to walk the
    # state machine to DESCENT → LANDED before launch, minting a landing-truth
    # row and LANDED alert at the launch site (26004618, 2026-07-12). A flight
    # whose apogee never cleared min_airborne_agl_m must not enter DESCENT.
    cfg = Config()
    tracker = FlightTracker(cfg, ground_fn=lambda lat, lon: 1459.0)
    frames = []
    fn = 0
    for s in range(0, 90, 3):                      # settle 1900 → 1450 m
        frames.append(_mk(39.12, -108.52, 1900.0 - 5.0 * s, s, fn))
        fn += 1
    for s in range(90, 300, 3):                    # then sit at the pad
        frames.append(_mk(39.12, -108.52, 1459.0 + (s % 3), s, fn))
        fn += 1
    fl, states, events = run_flight(tracker, frames)
    assert FlightState.DESCENT not in states
    assert TrackerEvent.LANDED not in events
    assert fl.state == FlightState.ASCENT


def test_first_heard_falling_high_still_enters_descent():
    # The never-airborne gate must not block a genuine descent first heard
    # mid-air: its apogee (the first fix) is far above the ground.
    cfg = Config()
    tracker = FlightTracker(cfg, ground_fn=lambda lat, lon: 1459.0)
    frames = [_mk(39.3, -109.2, 9000.0 - 30.0 * (3 * i), 3 * i, i)
              for i in range(40)]
    fl, states, _ = run_flight(tracker, frames)
    assert fl.state == FlightState.DESCENT
    assert fl.burst_alt is None            # burst never observed


def test_expired_descent_reopens_and_lands_when_frames_resume():
    # A DESCENT flight closed out by the silence sweep (dropped below the
    # radio horizon) whose sonde later comes back into range: the close-out
    # was premature, so tracking must resume - not leave a LANDED husk that
    # swallows the rest of the flight (track kept updating but nothing
    # predicted or recorded the real landing - 26004618, 2026-07-13).
    cfg = Config()
    tracker = FlightTracker(cfg)
    frames = []
    t, fn = 0, 0
    for _ in range(120):                           # climb 200 → 6200 m
        frames.append(_mk(45.0, 7.0, 200.0 + 50.0 * (t / 10), t, fn))
        t += 10
        fn += 1
    alt = 6200.0
    while alt > 5000.0:                            # fall to 5 km, then silence
        frames.append(_mk(45.0, 7.0, alt, t, fn))
        alt -= 90.0
        t += 3
        fn += 1
    fl, _, _ = run_flight(tracker, frames)
    assert fl.state == FlightState.DESCENT
    lost = fl.last_seen + timedelta(
        seconds=cfg.tracker.descent_lost_timeout_seconds + 60)
    assert tracker.check_timeouts(lost) == [(fl, TrackerEvent.EXPIRED)]
    assert fl.state == FlightState.LANDED

    # back in range lower down: resumes DESCENT (no ghost NEW_FLIGHT), then
    # genuinely lands with a LANDED event once it reaches the ground band
    t += int(cfg.tracker.descent_lost_timeout_seconds) + 120
    alt, events = 3500.0, []
    while alt > 100.0:
        fl, evs = tracker.update(_mk(45.0, 7.0, alt, t, fn))
        events.extend(evs)
        alt -= 60.0
        t += 3
        fn += 1
    assert TrackerEvent.NEW_FLIGHT not in events
    assert TrackerEvent.LANDED in events
    assert fl.state == FlightState.LANDED
    assert fl.landed_alt is not None       # a real landing, not an expiry


def test_expired_ascent_reopens_and_bursts_on_return():
    # Lost mid-ascent past ascent_lost_timeout_seconds → EXPIRED; the sonde
    # reappears high and falling (its burst happened out of range): tracking
    # resumes and the burst machinery takes it to DESCENT.
    cfg = Config()
    tracker = FlightTracker(cfg)
    for s in range(0, 1200, 10):                   # climb 200 → 6150 m
        flight, _ = tracker.update(_mk(45.0, 7.0, 200.0 + 5 * s, s, s))
    assert flight.state == FlightState.ASCENT
    lost = flight.last_seen + timedelta(
        seconds=cfg.tracker.ascent_lost_timeout_seconds + 60)
    assert tracker.check_timeouts(lost) == [(flight, TrackerEvent.EXPIRED)]
    assert flight.state == FlightState.LANDED

    t0 = 1190 + int(cfg.tracker.ascent_lost_timeout_seconds) + 120
    alt, events = 5500.0, []
    for i in range(40):
        fl, evs = tracker.update(_mk(45.0, 7.0, alt, t0 + 3 * i, 2000 + i))
        events.extend(evs)
        alt -= 90.0
    assert TrackerEvent.NEW_FLIGHT not in events
    assert TrackerEvent.BURST in events
    assert fl.state == FlightState.DESCENT
