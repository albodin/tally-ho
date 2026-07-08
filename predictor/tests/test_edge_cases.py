"""Robustness / edge-case tests."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from windfall.config import Config
from windfall.descent import DescentModel
from windfall.geo import normalize_lon
from windfall.integrator import integrate_descent
from windfall.models import FlightState
from windfall.predictor import Predictor
from windfall.profile import FlightProfile
from windfall.tracker import DescentSample, Flight

T0 = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)


def _flat(e=0.0):
    return lambda lat, lon: e


def test_integrator_crosses_antimeridian():
    p = FlightProfile()
    for a in range(0, 11000, 1000):
        p.add_sample(a, 60.0, 0.0)   # strong eastward wind
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    land = integrate_descent(lat=0.0, lon=179.9, alt=8000, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(0), cfg=Config().integrator)
    assert land.ok
    assert -180.0 <= land.lon <= 180.0          # normalised, no wrap blow-up
    assert land.lon < 0                          # crossed +180 into negative lon


def test_integrator_near_pole_no_crash():
    p = FlightProfile()
    for a in range(0, 11000, 1000):
        p.add_sample(a, 50.0, 50.0)
    d = DescentModel(b=5.5, residual_mps=0, n_points=10)
    land = integrate_descent(lat=89.5, lon=10.0, alt=8000, t0=T0, profile=p,
                             descent=d, ground_fn=_flat(0), cfg=Config().integrator)
    assert land.ok
    assert -90.0 <= land.lat <= 90.0


def test_predictor_none_without_profile_or_gfs():
    cfg = Config()
    predictor = Predictor(cfg)   # no GFS
    flight = Flight(serial="NOPROF", launch_day=T0.date(), state=FlightState.DESCENT)
    flight.last_lat, flight.last_lon, flight.last_alt = 45.0, 7.0, 8000.0
    flight.last_seen = T0
    # has descent samples (so a descent model exists) but empty ascent profile
    flight.descent_samples.append(DescentSample(t=0, alt=8000, v_obs=7.0, rho=0.5))
    flight.descent_samples.append(DescentSample(t=2, alt=7986, v_obs=7.1, rho=0.51))
    assert predictor.predict(flight) is None    # no wind source → degrade to none


def test_predictor_keeps_predicting_after_gap():
    # Telemetry gap mid-descent: predict still works from last good state.
    cfg = Config()
    predictor = Predictor(cfg)
    flight = Flight(serial="GAP", launch_day=T0.date(), type="RS41",
                    state=FlightState.DESCENT)
    p = FlightProfile()
    for a in range(0, 21000, 500):
        p.add_sample(a, 10.0, 0.0)
    flight.profile = p
    flight.last_lat, flight.last_lon, flight.last_alt = 45.0, 7.0, 6000.0
    # last_seen is 20 minutes ago (a long gap) but we still predict from it
    flight.last_seen = T0 - timedelta(minutes=20)
    from windfall.atmosphere import isa_density
    alt, t = 9000.0, 0.0
    while alt > 5500:
        rho = isa_density(alt)
        v = 5.5 * rho ** -0.5
        flight.descent_samples.append(DescentSample(t=t, alt=alt, v_obs=v, rho=rho))
        alt -= v * 2
        t += 2
    pred = predictor.predict(flight, now=T0)
    assert pred is not None
    assert pred.predicted_at == T0


def test_profile_interior_gap_interpolates():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 5.0, 0.0)
    p.add_sample(5000, 9.0, 0.0)   # big interior gap between 1000 and 5000
    # queries in the gap interpolate between the sampled bins, leaning toward
    # the nearer one - no step discontinuity mid-gap
    u_low, _ = p.wind(1500)
    u_high, _ = p.wind(4800)
    assert 5.0 < u_low < 7.0      # near the 1000 m bin
    assert 7.0 < u_high < 9.0     # near the 5000 m bin
    assert u_low == pytest.approx(5.0 + 4.0 * (1500 - 975) / (5025 - 975))


def test_normalize_lon_extremes():
    assert normalize_lon(370) == pytest.approx(10)
    assert normalize_lon(540) == pytest.approx(-180)   # 540 ≡ 180 ≡ -180 in [-180,180)
    assert -180 <= normalize_lon(123456.7) < 180
