"""Tests for the Euler descent integrator."""

import math
from datetime import datetime, timezone

import pytest

from windfall.config import IntegratorConfig
from windfall.descent import DescentModel
from windfall.integrator import integrate_descent
from windfall.profile import FlightProfile

T0 = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)


def _flat(elev=0.0):
    return lambda lat, lon: elev


def test_no_wind_lands_directly_below():
    p = FlightProfile()
    p.add_sample(0, 0, 0)
    p.add_sample(10000, 0, 0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    land = integrate_descent(lat=45.0, lon=7.0, alt=5000, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(0), cfg=IntegratorConfig())
    assert land.ok
    assert land.lat == pytest.approx(45.0, abs=1e-6)
    assert land.lon == pytest.approx(7.0, abs=1e-6)
    assert land.eta > T0


def test_constant_east_wind_drifts_east():
    p = FlightProfile()
    # 10 m/s eastward at all sampled alts
    for a in range(0, 11000, 1000):
        p.add_sample(a, 10.0, 0.0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    land = integrate_descent(lat=45.0, lon=7.0, alt=5000, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(0), cfg=IntegratorConfig())
    assert land.ok
    assert land.lon > 7.0     # drifted east
    assert land.lat == pytest.approx(45.0, abs=1e-4)
    # sanity: drift distance ~ wind * time
    from windfall.geo import haversine_km
    drift = haversine_km(45.0, 7.0, land.lat, land.lon)
    assert drift == pytest.approx(10.0 * land.sim_seconds / 1000.0, rel=0.05)


def test_initial_ground_lookup_uses_given_coordinates():
    # regression: the pre-loop ground check passed math.degrees(lon) on a value
    # already in degrees, so the first DEM lookup hit a bogus longitude
    p = FlightProfile()
    p.add_sample(0, 0, 0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    calls = []

    def ground(lat, lon):
        calls.append((lat, lon))
        return 0.0

    integrate_descent(lat=45.0, lon=120.0, alt=100, t0=T0, profile=p,
                      descent=d, ground_fn=ground, cfg=IntegratorConfig())
    assert calls[0][0] == pytest.approx(45.0)
    assert calls[0][1] == pytest.approx(120.0)


def test_terrain_terminates_higher():
    p = FlightProfile()
    p.add_sample(0, 0, 0)
    p.add_sample(10000, 0, 0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    sea = integrate_descent(lat=45.0, lon=7.0, alt=5000, t0=T0, profile=p,
                            descent=d, ground_fn=_flat(0), cfg=IntegratorConfig())
    mountain = integrate_descent(lat=45.0, lon=7.0, alt=5000, t0=T0, profile=p,
                                 descent=d, ground_fn=_flat(2000), cfg=IntegratorConfig())
    # terminating at 2000 m ground happens sooner → less sim time
    assert mountain.sim_seconds < sea.sim_seconds


def test_already_below_ground():
    p = FlightProfile()
    p.add_sample(0, 0, 0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    land = integrate_descent(lat=45.0, lon=7.0, alt=100, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(500), cfg=IntegratorConfig())
    assert land.ok
    assert land.sim_seconds == 0.0


def test_runaway_guard():
    p = FlightProfile()
    p.add_sample(0, 0, 0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    cfg = IntegratorConfig(max_iterations=5)   # too few to reach ground
    land = integrate_descent(lat=45.0, lon=7.0, alt=30000, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(0), cfg=cfg)
    assert not land.ok
    assert "max iterations" in land.reason


def test_path_capture_off_by_default():
    p = FlightProfile()
    p.add_sample(0, 0, 0)
    p.add_sample(10000, 10.0, 0.0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    land = integrate_descent(lat=45.0, lon=7.0, alt=5000, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(0), cfg=IntegratorConfig())
    assert land.path is None


def test_path_capture_traces_to_landing():
    p = FlightProfile()
    for a in range(0, 11000, 1000):
        p.add_sample(a, 10.0, 0.0)   # constant east wind
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    land = integrate_descent(lat=45.0, lon=7.0, alt=8000, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(0), cfg=IntegratorConfig(),
                             capture_path=True, path_max_points=32)
    assert land.ok and land.path is not None
    # bounded by the decimation (< 2 * max_points), and starts at the release
    assert 2 <= len(land.path) <= 64
    assert land.path[0] == pytest.approx((45.0, 7.0, 8000.0))
    # final point is the landing, monotonically descending and drifting east
    assert land.path[-1][0] == pytest.approx(land.lat) and land.path[-1][1] == pytest.approx(land.lon)
    assert land.path[-1][1] > land.path[0][1]            # drifted east
    alts = [pt[2] for pt in land.path]
    assert alts == sorted(alts, reverse=True)            # descending
