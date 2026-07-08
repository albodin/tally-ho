"""Tests for the Monte Carlo landing ensemble."""

import random
from datetime import datetime, timezone

import pytest

from windfall.config import Config
from windfall.descent import DescentModel
from windfall.ensemble import ensemble_descent, ensemble_preburst
from windfall.geo import haversine_km
from windfall.integrator import integrate_descent
from windfall.profile import FlightProfile

T0 = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)


def _profile():
    p = FlightProfile()
    for a in range(0, 11000, 500):
        p.add_sample(a, 10.0, 2.0)
    return p


def _flat(elev=0.0):
    return lambda lat, lon: elev


def _cfg(n=24):
    cfg = Config()
    cfg.ensemble.n_members = n
    cfg.ensemble.n_members_preburst = n
    return cfg


def test_zero_noise_members_collapse_to_deterministic():
    cfg = _cfg()
    cfg.ensemble.wind_sigma_measured_mps = 0.0
    cfg.ensemble.wind_sigma_extrapolated_mps = 0.0
    cfg.ensemble.wind_bias_sigma_mps = 0.0
    cfg.ensemble.b_sigma_rel_fit = 0.0
    p = _profile()
    d = DescentModel(b=5.5, residual_mps=0.0, n_points=10)
    det = integrate_descent(lat=45.0, lon=7.0, alt=8000, t0=T0, profile=p,
                            descent=d, ground_fn=_flat(), cfg=cfg.integrator)
    ens = ensemble_descent(lat=45.0, lon=7.0, alt=8000, t0=T0, profile=p,
                           descent=d, ground_fn=_flat(), cfg=cfg,
                           rng=random.Random(1))
    assert ens is not None and ens.n_members == 24
    assert ens.radius_km == pytest.approx(0.0, abs=1e-6)
    # member dt (2 s) vs deterministic dt (1 s) → tiny discretisation gap only
    assert haversine_km(ens.lat, ens.lon, det.lat, det.lon) < 0.3


def test_noisy_ensemble_spreads_and_is_seed_deterministic():
    cfg = _cfg()
    p = _profile()
    d = DescentModel(b=5.5, residual_mps=0.8, n_points=10)
    kw = dict(lat=45.0, lon=7.0, alt=8000, t0=T0, profile=p, descent=d,
              ground_fn=_flat(), cfg=cfg)
    a = ensemble_descent(rng=random.Random(42), **kw)
    b = ensemble_descent(rng=random.Random(42), **kw)
    c = ensemble_descent(rng=random.Random(43), **kw)
    assert a is not None and a.radius_km > 0.05
    # same seed → identical; different seed → (almost surely) different
    assert (a.lat, a.lon, a.radius_km) == (b.lat, b.lon, b.radius_km)
    assert (a.lat, a.lon) != (c.lat, c.lon)
    # the mean stays in the same neighbourhood as any single member draw
    assert haversine_km(a.lat, a.lon, c.lat, c.lon) < 2.0


def test_ensemble_preburst_runs_and_spreads():
    cfg = _cfg()
    p = _profile()
    ens = ensemble_preburst(lat=45.0, lon=7.0, alt=12000.0, t0=T0,
                            burst_alt=30000.0, ascent_rate=5.0, default_b=5.5,
                            profile=p, ground_fn=_flat(), cfg=cfg,
                            rng=random.Random(7))
    assert ens is not None
    # eastward column → lands east; long modelled flight → wide radius
    assert ens.lon > 7.0
    assert ens.radius_km > 1.0
    assert ens.eta > T0


def test_ensemble_returns_none_when_members_fail():
    cfg = _cfg(n=8)
    p = FlightProfile()
    p.add_sample(1000, 0, 0)
    d = DescentModel(b=5.5, residual_mps=0.0, n_points=10)
    cfg.integrator.max_iterations = 3   # every member hits the runaway guard
    ens = ensemble_descent(lat=45.0, lon=7.0, alt=30000, t0=T0, profile=p,
                           descent=d, ground_fn=_flat(), cfg=cfg,
                           rng=random.Random(1))
    assert ens is None


def test_wind_bias_spreads_members():
    """The per-member constant wind bias is the systematic-model-error term:
    with every other noise source off it alone must spread the members -
    AR(1) noise averages out over the column and under-disperses the radius
    (the 52%-coverage failure mode of the 2026-06-10 backtest)."""
    cfg = _cfg()
    cfg.ensemble.wind_sigma_measured_mps = 0.0
    cfg.ensemble.wind_sigma_extrapolated_mps = 0.0
    cfg.ensemble.wind_bias_sigma_mps = 0.0
    cfg.ensemble.b_sigma_rel_fit = 0.0
    p = _profile()
    d = DescentModel(b=5.5, residual_mps=0.0, n_points=10)
    kw = dict(lat=45.0, lon=7.0, alt=8000, t0=T0, profile=p, descent=d,
              ground_fn=_flat(), cfg=cfg)
    unbiased = ensemble_descent(rng=random.Random(5), **kw)
    cfg.ensemble.wind_bias_sigma_mps = 2.0
    biased = ensemble_descent(rng=random.Random(5), **kw)
    assert unbiased.radius_km == pytest.approx(0.0, abs=1e-6)
    assert biased.radius_km > 0.3
