"""Tests for the GFS fallback wind source."""

import math
from datetime import datetime, timezone

import pytest

from windfall.atmosphere import isa_density
from windfall.config import Config
from windfall.gfs import (
    GFSLevel,
    StaticGFSSource,
    build_profile_from_levels,
    estimate_burst_alt,
    make_gfs_source,
    pressure_to_altitude,
)
from windfall.models import FlightState, PredictionSource
from windfall.predictor import Predictor
from windfall.tracker import DescentSample, Flight
from datetime import date

T0 = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
DAY = date(2026, 6, 7)


def test_pressure_to_altitude_roundtrips_isa():
    # 500 hPa ~ 5.5 km, 250 hPa ~ 10.4 km (standard atmosphere)
    assert pressure_to_altitude(50000) == pytest.approx(5570, abs=150)
    assert pressure_to_altitude(25000) == pytest.approx(10360, abs=300)


def test_build_profile_from_levels():
    levels = [
        GFSLevel(height_m=1000, u=5, v=1),
        GFSLevel(height_m=5000, u=20, v=2),
        GFSLevel(height_m=10000, u=30, v=0, pressure_pa=26500, temp_k=223),
    ]
    prof = build_profile_from_levels(levels)
    assert not prof.is_empty()
    u, v = prof.wind(5000)
    assert u == pytest.approx(20, abs=0.5)
    # density at the level with P/T comes from GFS, not ISA (exact at the bin
    # centre; nearby altitudes get the continuous ISA-shape scaling)
    centre = prof.bin_near(10000).alt
    assert prof.density(centre) == pytest.approx(26500 / (287.05 * 223), rel=1e-3)
    assert prof.density(10000) == pytest.approx(26500 / (287.05 * 223), rel=1e-2)


def test_estimate_burst_alt_from_timer():
    # 10 m/s ascent, 600 s remaining, from 20 km → ~26 km
    alt = estimate_burst_alt(current_alt=20000, ascent_rate=10, burst_timer=600,
                             sonde_type="RS41")
    assert alt == pytest.approx(26000, abs=100)


def test_estimate_burst_alt_ignores_kill_timer():
    # burst_timer is frequently the burst-KILL countdown (e.g. 30600 s), not
    # time-to-burst; taken literally it would put burst at >170 km. Implausible
    # estimates fall through to the sonde-type default.
    alt = estimate_burst_alt(current_alt=20000, ascent_rate=5.0, burst_timer=30600,
                             sonde_type="RS41")
    assert alt == 35000.0


def test_estimate_burst_alt_ignores_nonpositive_timer():
    alt = estimate_burst_alt(current_alt=20000, ascent_rate=5.0, burst_timer=-1,
                             sonde_type="M20")
    assert alt == 30000.0


def test_estimate_burst_alt_type_default():
    alt = estimate_burst_alt(current_alt=15000, ascent_rate=None, burst_timer=None,
                             sonde_type="RS41")
    assert alt == 35000.0


def test_estimate_burst_alt_generic_default():
    alt = estimate_burst_alt(current_alt=15000, ascent_rate=None, burst_timer=None,
                             sonde_type="UNKNOWNX")
    assert alt == 30000.0


def test_estimate_burst_never_below_current():
    alt = estimate_burst_alt(current_alt=36000, ascent_rate=None, burst_timer=None,
                             sonde_type="RS41")
    assert alt > 36000


def test_estimate_burst_alt_site_climatology_beats_type_table():
    # site history says this launch site bursts at ~33.2 km → beats RS41 default
    alt = estimate_burst_alt(current_alt=15000, ascent_rate=None, burst_timer=None,
                             sonde_type="RS41", site_burst_alt=33200.0)
    assert alt == 33200.0
    # ...but a plausible burst timer still wins
    alt = estimate_burst_alt(current_alt=20000, ascent_rate=10, burst_timer=600,
                             sonde_type="RS41", site_burst_alt=33200.0)
    assert alt == pytest.approx(26000, abs=100)
    # implausible site values (parse glitches) are ignored
    alt = estimate_burst_alt(current_alt=15000, ascent_rate=None, burst_timer=None,
                             sonde_type="RS41", site_burst_alt=90000.0)
    assert alt == 35000.0


def test_parse_grib_path_herbie_layout():
    from windfall.gfs import parse_grib_path
    info = parse_grib_path(
        "/data/gfs/20260609/subset_ab12cd__gfs.t06z.pgrb2.0p25.f003")
    assert info is not None
    assert info.cycle == datetime(2026, 6, 9, 6, tzinfo=timezone.utc)
    assert info.fxx == 3
    assert info.valid_time == datetime(2026, 6, 9, 9, tzinfo=timezone.utc)
    # unparseable names are skipped, not crashed on
    assert parse_grib_path("/data/gfs/README.txt") is None


def test_select_bracketing_picks_newest_cycle_and_valid_pair():
    from windfall.gfs import parse_grib_path, select_bracketing
    files = [
        # old cycle (00z) - must be ignored even though it sorts last by name
        "/d/gfs/20260609/z_old__gfs.t00z.pgrb2.0p25.f006",
        # new cycle (06z), hourly
        "/d/gfs/20260609/a__gfs.t06z.pgrb2.0p25.f000",
        "/d/gfs/20260609/a__gfs.t06z.pgrb2.0p25.f003",
        "/d/gfs/20260609/a__gfs.t06z.pgrb2.0p25.f006",
    ]
    infos = [parse_grib_path(f) for f in files]
    when = datetime(2026, 6, 9, 10, 30, tzinfo=timezone.utc)   # between f003/f006
    lo, hi = select_bracketing(infos, when)
    assert (lo.fxx, hi.fxx) == (3, 6)
    assert lo.cycle.hour == 6 and hi.cycle.hour == 6
    # exactly on a valid hour → both ends are that file
    lo, hi = select_bracketing(infos, datetime(2026, 6, 9, 9, tzinfo=timezone.utc))
    assert lo.fxx == hi.fxx == 3
    # past the last forecast hour → clamp to it
    lo, hi = select_bracketing(infos, datetime(2026, 6, 9, 23, tzinfo=timezone.utc))
    assert lo.fxx == hi.fxx == 6
    # at 01:00 the 06z run did not exist yet - no lookahead: the 00z cycle's
    # only file wins even though the 06z f000 shares its valid time
    lo, hi = select_bracketing(infos, datetime(2026, 6, 9, 1, tzinfo=timezone.utc))
    assert lo.cycle.hour == hi.cycle.hour == 0
    assert lo.fxx == hi.fxx == 6


def test_select_bracketing_never_uses_future_cycle():
    # Backtest honesty: a corpus replay must only see forecast runs that
    # existed at flight time, even when the cache holds newer ones too.
    from windfall.gfs import parse_grib_path, select_bracketing
    files = [
        "/d/gfs/20260608/a__gfs.t18z.pgrb2.0p25.f000",
        "/d/gfs/20260608/a__gfs.t18z.pgrb2.0p25.f006",
        "/d/gfs/20260609/a__gfs.t00z.pgrb2.0p25.f000",
        "/d/gfs/20260609/a__gfs.t00z.pgrb2.0p25.f006",
        "/d/gfs/20260609/a__gfs.t12z.pgrb2.0p25.f000",   # future for this flight
        "/d/gfs/20260609/a__gfs.t12z.pgrb2.0p25.f006",
    ]
    infos = [parse_grib_path(f) for f in files]
    when = datetime(2026, 6, 9, 3, 0, tzinfo=timezone.utc)
    lo, hi = select_bracketing(infos, when)
    assert lo.cycle == hi.cycle == datetime(2026, 6, 9, 0, tzinfo=timezone.utc)
    assert (lo.fxx, hi.fxx) == (0, 6)
    # nothing at-or-before `when` → refuse (None), never a future cycle: a
    # silently-lookahead bracket would bias the very score it feeds
    assert select_bracketing(infos, datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)) is None


def _cube(valid_time, u_val, *, u_east=None):
    """A small synthetic WindCube: 3 levels x 3 lats x 3 lons."""
    import numpy as np
    from windfall.gfs import WindCube
    lats = np.array([46.0, 45.0, 44.0])          # descending, as GRIB stores it
    lons = np.array([6.0, 7.0, 8.0])
    heights = np.zeros((3, 3, 3))
    for k, h in enumerate([10000.0, 5000.0, 1000.0]):   # pressure order (descending h)
        heights[k] = h
    u = np.zeros((3, 3, 3))
    for k, val in enumerate(u_val):
        u[k] = val
        if u_east is not None:
            u[k, :, 2] = u_east[k]               # different wind on the east edge
    v = np.ones((3, 3, 3))
    return WindCube.from_grid(valid_time=valid_time, lats=lats, lons=lons,
                              heights=heights, u=u, v=v)


def test_wind_cube_interpolates_altitude_and_space():
    cube = _cube(T0, [30.0, 20.0, 10.0])         # 10 m/s @1km ... 30 m/s @10km
    # vertical interpolation between levels
    u, v = cube.wind_at(45.0, 7.0, 3000.0)
    assert u == pytest.approx(15.0, abs=0.01)    # halfway 1km→5km
    assert v == pytest.approx(1.0)
    # clamped outside the level range
    assert cube.wind_at(45.0, 7.0, 500.0)[0] == pytest.approx(10.0)
    assert cube.wind_at(45.0, 7.0, 20000.0)[0] == pytest.approx(30.0)
    # horizontal bilinear: east edge blows differently → midpoint blends
    cube2 = _cube(T0, [30.0, 20.0, 10.0], u_east=[30.0, 20.0, 20.0])
    at_centre = cube2.wind_at(45.0, 7.0, 1000.0)[0]
    at_east = cube2.wind_at(45.0, 8.0, 1000.0)[0]
    halfway = cube2.wind_at(45.0, 7.5, 1000.0)[0]
    assert at_centre == pytest.approx(10.0)
    assert at_east == pytest.approx(20.0)
    assert halfway == pytest.approx(15.0, abs=0.01)


def test_cube_pair_interpolates_time():
    from datetime import timedelta
    from windfall.gfs import CubePairWind
    a = _cube(T0, [30.0, 20.0, 10.0])
    b = _cube(T0 + timedelta(hours=3), [30.0, 20.0, 16.0])
    pair = CubePairWind(a, b, t0=T0)
    # at anchor time → cube a; +1.5h → halfway; beyond b → clamp
    assert pair(45.0, 7.0, 1000.0, 0.0)[0] == pytest.approx(10.0)
    assert pair(45.0, 7.0, 1000.0, 5400.0)[0] == pytest.approx(13.0, abs=0.01)
    assert pair(45.0, 7.0, 1000.0, 999999.0)[0] == pytest.approx(16.0)
    # the column profile lerps in time too
    levels = pair.column_levels(45.0, 7.0, 5400.0)
    low = min(levels, key=lambda l: l.height_m)
    assert low.u == pytest.approx(13.0, abs=0.01)


def test_cube_pair_column_cache_bounds_corner_work(monkeypatch):
    # Integrator-style usage: thousands of nearby per-step queries must coalesce
    # into a handful of column builds, not 16 interpolations per step.
    from datetime import timedelta
    from windfall.gfs import CubePairWind, WindCube
    calls = {"n": 0}
    real = WindCube.column_arrays

    def counting(self, lat, lon):
        calls["n"] += 1
        return real(self, lat, lon)

    monkeypatch.setattr(WindCube, "column_arrays", counting)
    a = _cube(T0, [30.0, 20.0, 10.0])
    b = _cube(T0 + timedelta(hours=3), [30.0, 20.0, 16.0])
    pair = CubePairWind(a, b, t0=T0)
    # a slow drift across ~0.1° over 2000 one-second steps
    for step in range(2000):
        u, v = pair(45.0 + step * 5e-5, 7.0, 8000.0 - step * 3.5, float(step))
    # ≤ (3 lat buckets x ~7 time buckets) x 2 cubes, far below 2000 calls
    assert calls["n"] <= 60
    # cached values are still the right physics (vertical interpolation intact)
    assert pair(45.0, 7.0, 3000.0, 0.0)[0] == pytest.approx(15.0, abs=0.01)


def _var_grid(levels, value_per_level, lats=None, lons=None):
    import numpy as np
    from windfall.gfs import VarGrid
    lats = np.array([46.0, 45.0, 44.0]) if lats is None else np.asarray(lats)
    lons = np.array([6.0, 7.0, 8.0]) if lons is None else np.asarray(lons)
    vals = np.empty((len(levels), lats.size, lons.size))
    for k, val in enumerate(value_per_level):
        vals[k] = val
    return VarGrid(lats=lats, lons=lons,
                   levels_hpa=np.asarray(levels, dtype=float), vals=vals)


def test_assemble_cube_joins_mismatched_level_sets():
    # The production failure mode: GFS pgrb2 carries HGT/TMP on MORE isobaric
    # levels than UGRD/VGRD (cfgrib drops the winds when opened together).
    # Assembly must join on the wind levels and pull gh/t where they exist.
    from windfall.gfs import assemble_cube
    u = _var_grid([850, 500, 250], [5.0, 15.0, 30.0])
    v = _var_grid([850, 500, 250], [1.0, 2.0, 3.0])
    gh = _var_grid([1000, 850, 500, 250, 10], [100.0, 1500.0, 5500.0, 10500.0, 31000.0])
    t = _var_grid([1000, 850, 500, 250, 10], [288.0, 280.0, 252.0, 220.0, 230.0])
    cube = assemble_cube(T0, u, v, gh, t)
    assert cube is not None
    assert cube.heights.shape[0] == 3            # joined on the wind levels
    u_at, v_at = cube.wind_at(45.0, 7.0, 5500.0)
    assert u_at == pytest.approx(15.0)
    assert v_at == pytest.approx(2.0)
    # temp present on every wind level → real density via the level column
    col = cube.column(45.0, 7.0)
    mid = next(l for l in col if l.height_m == pytest.approx(5500.0))
    assert mid.temp_k == pytest.approx(252.0)
    assert mid.pressure_pa == pytest.approx(50000.0)


def test_assemble_cube_isa_heights_and_partial_temp():
    from windfall.gfs import assemble_cube, pressure_to_altitude
    u = _var_grid([850, 500], [5.0, 15.0])
    v = _var_grid([850, 500], [1.0, 2.0])
    t = _var_grid([850], [280.0])                # temp missing at 500 → dropped
    cube = assemble_cube(T0, u, v, gh=None, t=t)
    assert cube is not None
    assert cube.temp_k is None                   # partial temp would bias density
    col = sorted(cube.column(45.0, 7.0), key=lambda l: l.height_m)
    assert col[0].height_m == pytest.approx(pressure_to_altitude(85000.0), abs=1.0)
    # winds and missing components degrade to None, not a crash
    assert assemble_cube(T0, u, None) is None
    assert assemble_cube(T0, None, v) is None
    # disjoint wind levels → nothing to join
    v2 = _var_grid([300], [1.0])
    assert assemble_cube(T0, u, v2) is None


def test_herbie_inventory_scans_filenames(tmp_path):
    from windfall.gfs import HerbieGFSSource
    cfg = Config()
    cfg.gfs.path = str(tmp_path)
    day = tmp_path / "gfs" / "20260609"
    day.mkdir(parents=True)
    (day / "subset_x__gfs.t06z.pgrb2.0p25.f000").write_bytes(b"")
    (day / "subset_x__gfs.t06z.pgrb2.0p25.f003").write_bytes(b"")
    (day / "subset_x__gfs.t06z.pgrb2.0p25.f003.idx").write_bytes(b"")  # skipped
    (day / "notes.txt").write_bytes(b"")                               # skipped
    src = HerbieGFSSource(cfg)
    inv = src._inventory()
    assert sorted(i.fxx for i in inv) == [0, 3]
    assert all(i.cycle.hour == 6 for i in inv)


def test_make_gfs_source_disabled():
    cfg = Config()
    cfg.gfs.enabled = False
    assert make_gfs_source(cfg) is None


def test_download_gfs_cycle_no_box():
    from windfall.gfs import download_gfs_cycle
    assert download_gfs_cycle(Config(), None) == []


def test_download_gfs_cycle_without_herbie(tmp_path, monkeypatch):
    # herbie unavailable → graceful empty result, no raise. Simulated by
    # poisoning sys.modules so the test holds whether or not herbie is
    # actually installed in the environment.
    import sys

    from windfall.geo import BBox
    from windfall.gfs import download_gfs_cycle
    monkeypatch.setitem(sys.modules, "herbie", None)
    cfg = Config()
    cfg.gfs.path = str(tmp_path / "gfs")
    assert download_gfs_cycle(cfg, BBox(44, 6, 46, 8)) == []


def _gfs_profile():
    # constant 15 m/s eastward across the column
    levels = [GFSLevel(height_m=a, u=15.0, v=0.0) for a in range(0, 31000, 1000)]
    return build_profile_from_levels(levels)


def test_predictor_uses_gfs_when_no_ascent_profile():
    # Flight joined mid-descent: empty ascent profile, but has descent samples.
    cfg = Config()
    gfs = StaticGFSSource(_gfs_profile())
    predictor = Predictor(cfg, gfs_source=gfs)

    flight = Flight(serial="MIDD", launch_day=DAY, type="RS41", state=FlightState.DESCENT)
    flight.last_lat, flight.last_lon, flight.last_alt = 45.0, 7.0, 8000.0
    flight.last_seen = T0
    # build a few descent samples so the descent model exists
    alt = 9000.0
    t = 0.0
    while alt > 6000:
        rho = isa_density(alt)
        v = 5.5 * rho ** -0.5
        flight.descent_samples.append(DescentSample(t=t, alt=alt, v_obs=v, rho=rho))
        alt -= v * 2
        t += 2
    pred = predictor.predict(flight)
    assert pred is not None
    assert pred.source == PredictionSource.GFS
    assert pred.land_lon > 7.0   # drifted east under the GFS wind
    # GFS source → larger uncertainty (extrapolated bucket)
    assert pred.uncertainty_radius_km > 0
    # already descending: its burst is observed, not predicted - no burst point
    assert pred.burst_lat is None and pred.burst_alt is None


def _mid_descent_flight(serial="MIDD"):
    flight = Flight(serial=serial, launch_day=DAY, type="RS41", state=FlightState.DESCENT)
    flight.last_lat, flight.last_lon, flight.last_alt = 45.0, 7.0, 8000.0
    flight.last_seen = T0
    flight.last_t = T0.timestamp()
    alt = 9000.0
    t = 0.0
    while alt > 6000:
        rho = isa_density(alt)
        v = 5.5 * rho ** -0.5
        flight.descent_samples.append(DescentSample(t=t, alt=alt, v_obs=v, rho=rho))
        alt -= v * 2
        t += 2
    return flight


def test_integrator_samples_wind_field_along_trajectory():
    # The 4-D field path: wind depends on *where the payload currently is*, not
    # on a frozen column at the release point. East of 7.05° the wind triples;
    # a moving-column integrator must land further east than a frozen one.
    from windfall.predictor import GFSWindSource as Base

    class ShearSource(Base):
        def __init__(self, fast_east):
            self.fast = fast_east

        def profile_at(self, lat, lon, when):
            return _gfs_profile()   # density/profile fallback - constant 15 east

        def wind_field(self, lat, lon, when):
            def field(la, lo, alt, sim_t):
                return (30.0, 0.0) if (self.fast and lo > 7.05) else (10.0, 0.0)
            return field

    cfg = Config()
    cfg.ensemble.enabled = False
    moving = Predictor(cfg, gfs_source=ShearSource(fast_east=True))
    frozen = Predictor(cfg, gfs_source=ShearSource(fast_east=False))
    p_moving = moving.predict(_mid_descent_flight("MV"))
    p_frozen = frozen.predict(_mid_descent_flight("FZ"))
    assert p_moving is not None and p_frozen is not None
    assert p_frozen.land_lon > 7.0
    # the shear kicks in mid-flight → strictly further east
    assert p_moving.land_lon > p_frozen.land_lon + 0.05


def test_ensemble_refresh_throttled_per_flight(monkeypatch):
    import windfall.predictor as predmod

    calls = {"n": 0}
    real_scalar = predmod.ensemble_descent
    real_vec = predmod.ensemble_descent_vec

    def counting_scalar(**kw):
        calls["n"] += 1
        return real_scalar(**kw)

    def counting_vec(**kw):
        calls["n"] += 1
        return real_vec(**kw)

    # count whichever implementation ensemble.vectorized selects
    monkeypatch.setattr(predmod, "ensemble_descent", counting_scalar)
    monkeypatch.setattr(predmod, "ensemble_descent_vec", counting_vec)
    cfg = Config()
    cfg.ensemble.n_members = 8
    predictor = Predictor(cfg, gfs_source=StaticGFSSource(_gfs_profile()))

    flight = _mid_descent_flight("THR")
    p1 = predictor.predict(flight)
    p2 = predictor.predict(flight)
    assert p1 is not None and p2 is not None
    assert calls["n"] == 1                       # second call rode the cache
    assert p1.uncertainty_radius_km == p2.uncertainty_radius_km
    flight.last_t += cfg.ensemble.min_interval_seconds + 1
    predictor.predict(flight)
    assert calls["n"] == 2                       # refreshed after the interval


def _ascending_flight(serial, alt, profile_top, vrate=5.0):
    from windfall.profile import FlightProfile
    flight = Flight(serial=serial, launch_day=DAY, type="RS41", state=FlightState.ASCENT)
    flight.last_lat, flight.last_lon, flight.last_alt = 45.0, 7.0, alt
    flight.last_seen = T0
    flight.last_t = T0.timestamp()
    flight.last_vrate = vrate
    p = FlightProfile()
    for a in range(0, int(profile_top) + 1, 300):
        p.add_sample(a, 10.0, 0.0)
    flight.profile = p
    return flight


def test_preburst_gated_when_upper_winds_unknown():
    # Early ascent, no GFS anywhere: the column above 5 km is a clamp of the
    # 5 km wind - refuse to publish (this is the 100+ km junk-path regime).
    cfg = Config()
    cfg.ensemble.n_members_preburst = 8
    predictor = Predictor(cfg)              # no GFS source at all
    early = _ascending_flight("GATE1", alt=5000.0, profile_top=5000.0)
    assert predictor.predict_preburst(early) is None
    # Late ascent: the measured column covers most of the flight (30/35 km of
    # the RS41 default burst) → allowed even without GFS.
    late = _ascending_flight("GATE2", alt=30000.0, profile_top=30000.0)
    assert predictor.predict_preburst(late) is not None


def test_preburst_allowed_when_gfs_field_available():
    cfg = Config()
    cfg.ensemble.n_members_preburst = 8
    predictor = Predictor(cfg, gfs_source=StaticGFSSource(_gfs_profile()))
    early = _ascending_flight("GATE3", alt=5000.0, profile_top=5000.0)
    pred = predictor.predict_preburst(early)
    assert pred is not None
    assert pred.land_lon > 7.0   # advected by real (GFS) winds, not the clamp


def test_preburst_prediction_drifts_downwind():
    cfg = Config()
    gfs = StaticGFSSource(_gfs_profile())
    predictor = Predictor(cfg, gfs_source=gfs)
    flight = Flight(serial="PRE", launch_day=DAY, type="RS41", state=FlightState.ASCENT)
    flight.last_lat, flight.last_lon, flight.last_alt = 45.0, 7.0, 12000.0
    flight.last_seen = T0
    flight.last_vrate = 5.0
    pred = predictor.predict_preburst(flight)
    assert pred is not None
    assert pred.source == PredictionSource.GFS
    # eastward wind over a long ascent+descent → lands well east
    assert pred.land_lon > 7.0
    assert pred.uncertainty_radius_km > 1.0   # pre-burst is uncertain
    # the predicted burst point rides along (for the map's burst marker):
    # above the sonde, east of it, and short of the landing - both legs drift
    # east under the same wind
    assert pred.burst_alt is not None and pred.burst_alt > 12000.0
    assert 7.0 < pred.burst_lon < pred.land_lon
    assert pred.burst_lat == pytest.approx(45.0, abs=0.2)   # no meridional wind


def test_select_bracketing_respects_publication_latency():
    """A 12Z GFS cycle isn't published until ~15:45Z: a backtest predicting at
    13Z must use the 06Z cycle even though 12Z's run time is in the past."""
    from datetime import datetime, timedelta, timezone

    from windfall.gfs import GribFileInfo, select_bracketing

    def mk(hour, fxx):
        return GribFileInfo(path=f"gfs.t{hour:02d}z.f{fxx:03d}",
                            cycle=datetime(2026, 6, 8, hour, tzinfo=timezone.utc),
                            fxx=fxx)

    inv = [mk(6, f) for f in range(10)] + [mk(12, f) for f in range(10)]
    when = datetime(2026, 6, 8, 13, 0, tzinfo=timezone.utc)
    lo, hi = select_bracketing(inv, when, min_age_hours=3.8)
    assert lo.cycle.hour == 6 and hi.cycle.hour == 6
    assert lo.fxx == 7 and hi.fxx == 7        # valid 13Z exactly
    # without latency modelling the (unpublished) 12Z run would win
    lo0, _ = select_bracketing(inv, when, min_age_hours=0.0)
    assert lo0.cycle.hour == 12
    # once it is actually published, prefer it
    later = datetime(2026, 6, 8, 16, 0, tzinfo=timezone.utc)
    lo2, _ = select_bracketing(inv, later, min_age_hours=3.8)
    assert lo2.cycle.hour == 12


def test_window_indices_interior():
    import numpy as np

    from windfall.gfs import window_indices

    lats = np.arange(90.0, -90.25, -0.25)        # GFS native: descending
    lons = np.arange(0.0, 360.0, 0.25)           # 0..359.75
    lat_idx, lon_idx, lons_out = window_indices(lats, lons, 40.7, -112.0, 5.0)
    assert np.all(np.abs(lats[lat_idx] - 40.7) <= 5.0)
    assert lons_out[0] >= 243.0 - 0.25 and lons_out[-1] <= 253.0
    assert np.all(np.diff(lons_out) > 0)          # ascending for WindCube
    # window indices select the matching native columns
    assert np.allclose(lons[lon_idx], lons_out % 360.0)


def test_window_indices_crosses_greenwich_seam():
    import numpy as np

    from windfall.gfs import window_indices

    lats = np.arange(90.0, -90.25, -0.25)
    lons = np.arange(0.0, 360.0, 0.25)
    # window around 2°E spans the 0/360 seam of the native axis
    _, lon_idx, lons_out = window_indices(lats, lons, 47.0, 2.0, 5.0)
    assert np.all(np.diff(lons_out) > 0)
    assert lons_out[0] <= -2.5 and lons_out[-1] >= 6.5   # unwrapped around 2°E
    assert lons_out[-1] <= 180.0     # WindCube._qlon resolves via normalize_lon
    # just east of the seam: axis re-anchored below 360 so queries resolve
    _, _, lons_e = window_indices(lats, lons, 47.0, 358.0, 5.0)
    assert lons_e[-1] < 360.0 and np.all(np.diff(lons_e) > 0)


def test_window_indices_misses_regional_grid():
    import numpy as np

    from windfall.gfs import window_indices

    lats = np.arange(50.0, 40.0, -0.25)          # regional subset
    lons = np.arange(10.0, 20.0, 0.25)
    assert window_indices(lats, lons, -30.0, 150.0, 5.0) is None


def test_prune_grib_cache(tmp_path):
    from windfall.gfs import prune_grib_cache

    day_old = tmp_path / "gfs" / "20260607"
    fresh = tmp_path / "gfs" / "20260610"
    day_old.mkdir(parents=True)
    fresh.mkdir(parents=True)
    old_grib = day_old / "subset_a__gfs.t06z.pgrb2.0p25.f003"
    old_grib.touch()
    (day_old / "subset_a__gfs.t06z.pgrb2.0p25.f003.9a1b.idx").touch()
    new_grib = fresh / "subset_b__gfs.t18z.pgrb2.0p25.f001"
    new_grib.touch()

    now = datetime(2026, 6, 10, 20, 0, tzinfo=timezone.utc)
    assert prune_grib_cache(tmp_path, keep_hours=48.0, now=now) == 1
    assert not old_grib.exists()                      # cycle 06-07 06Z: 86 h old
    assert not list(day_old.glob("*.idx"))            # sidecar removed too
    assert new_grib.exists()                          # 06-10 18Z: 2 h old, kept

    assert prune_grib_cache(tmp_path, keep_hours=0.0, now=now) == 0   # disabled
    assert new_grib.exists()


def test_cube_pair_batch_matches_call():
    """CubePairWind.batch equals per-member __call__ over the same points
    (the batched GFS wind eval must be bit-for-bit the scalar path)."""
    import numpy as np
    from datetime import timedelta
    from windfall.gfs import CubePairWind
    a = _cube(T0, [30.0, 20.0, 10.0], u_east=[30.0, 20.0, 20.0])
    b = _cube(T0 + timedelta(hours=3), [30.0, 20.0, 16.0], u_east=[32.0, 22.0, 26.0])
    pair = CubePairWind(a, b, t0=T0)
    rng = np.random.default_rng(0)
    lats = 44.0 + rng.random(50) * 2.0
    lons = 6.0 + rng.random(50) * 2.0
    alts = 500.0 + rng.random(50) * 10000.0
    for sim_t in (0.0, 5400.0):
        bu, bv = pair.batch(lats, lons, alts, sim_t)
        for i in range(len(lats)):
            su, sv = pair(float(lats[i]), float(lons[i]), float(alts[i]), sim_t)
            assert bu[i] == pytest.approx(su, abs=1e-9)
            assert bv[i] == pytest.approx(sv, abs=1e-9)


def test_cube_pair_batch_shared_column():
    """shared=True samples ONE column at the member centroid and
    hands it to every member - each member must get exactly the centroid
    column's wind at its own altitude, and co-located members must match the
    exact per-bucket path bit-for-bit."""
    import numpy as np
    from datetime import timedelta
    from windfall.gfs import CubePairWind
    a = _cube(T0, [30.0, 20.0, 10.0], u_east=[30.0, 20.0, 20.0])
    b = _cube(T0 + timedelta(hours=3), [30.0, 20.0, 16.0], u_east=[32.0, 22.0, 26.0])
    pair = CubePairWind(a, b, t0=T0)
    rng = np.random.default_rng(2)
    lats = 44.95 + rng.random(40) * 0.1          # ~10 km member cloud
    lons = 6.95 + rng.random(40) * 0.1
    alts = 500.0 + rng.random(40) * 10000.0
    # the centroid exactly as batch_c computes it (wrap-aware lon mean)
    lat0 = float(np.add.reduce(lats)) / lats.size
    d = (lons - lons[0] + 180.0) % 360.0 - 180.0
    lon0 = lons[0] + float(np.add.reduce(d)) / d.size
    for sim_t in (0.0, 5400.0):
        su, sv = pair.batch(lats, lons, alts, sim_t, shared=True)
        for i in range(len(alts)):
            cu, cv = pair(lat0, lon0, float(alts[i]), sim_t)
            assert su[i] == pytest.approx(cu, abs=1e-9)
            assert sv[i] == pytest.approx(cv, abs=1e-9)
    # co-located members: shared and exact per-bucket sampling coincide
    same_lat = np.full(20, 45.3)
    same_lon = np.full(20, 7.2)
    aa = 500.0 + rng.random(20) * 10000.0
    eu, ev = pair.batch(same_lat, same_lon, aa, 0.0)
    su, sv = pair.batch(same_lat, same_lon, aa, 0.0, shared=True)
    assert np.allclose(eu, su, atol=1e-12)
    assert np.allclose(ev, sv, atol=1e-12)
