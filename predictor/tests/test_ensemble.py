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


def test_vectorised_ensemble_matches_scalar_statistically():
    """The vectorised (batch-across-members) descent ensemble draws its RNG
    in a different order, so it is not bit-identical - but the landing
    distribution must match the scalar reference within Monte-Carlo noise."""
    from windfall.ensemble_vec import ensemble_descent_vec

    p = FlightProfile()
    for a in range(0, 20000, 250):
        p.add_sample(a, 12.0, 3.0)          # constant wind; ISA density
    cfg = _cfg(200)
    descent = DescentModel(b=6.0, residual_mps=0.4, n_points=30)
    kw = dict(lat=45.0, lon=7.0, alt=19000.0, t0=T0, profile=p, descent=descent,
              ground_fn=_flat(), cfg=cfg, wind_fn=None, measured_range=p.alt_range())
    dists, ratios = [], []
    for s in range(4):
        a = ensemble_descent(rng=random.Random(s), **kw)
        b = ensemble_descent_vec(rng=random.Random(s), **kw)
        assert a is not None and b is not None and b.n_members > 0
        dists.append(haversine_km(a.lat, a.lon, b.lat, b.lon))
        ratios.append(b.radius_km / a.radius_km)
    # MC noise on the mean ~ radius/sqrt(N); generous but catches physics bugs
    assert sum(dists) / len(dists) < 2.0                 # km
    assert 0.6 < sum(ratios) / len(ratios) < 1.5         # radii agree within ~40%


def test_vectorised_zero_noise_collapses_to_deterministic():
    """With every perturbation off, the vectorised members are identical and land
    exactly on the deterministic integrator - a hard correctness check."""
    from windfall.ensemble_vec import ensemble_descent_vec

    cfg = _cfg(16)
    cfg.ensemble.wind_sigma_measured_mps = 0.0
    cfg.ensemble.wind_sigma_extrapolated_mps = 0.0
    cfg.ensemble.wind_bias_sigma_mps = 0.0
    descent = DescentModel(b=6.0, residual_mps=0.0, n_points=0)  # rel spread → preburst? no: n_points 0
    cfg.ensemble.b_sigma_rel_preburst = 0.0
    p = _profile()
    det = integrate_descent(lat=45.0, lon=7.0, alt=10000.0, t0=T0, profile=p,
                            descent=descent, ground_fn=_flat(), cfg=cfg.integrator)
    ens = ensemble_descent_vec(lat=45.0, lon=7.0, alt=10000.0, t0=T0, profile=p,
                               descent=descent, ground_fn=_flat(), cfg=cfg,
                               wind_fn=None, rng=random.Random(0),
                               measured_range=p.alt_range())
    assert ens is not None
    assert haversine_km(ens.lat, ens.lon, det.lat, det.lon) < 0.05   # ~coincident
    assert ens.radius_km < 0.05


def test_vectorised_preburst_matches_scalar_statistically():
    """The vectorised pre-burst ensemble mixes rising and falling members in one
    lockstep loop; the scalar reference integrates each member's two legs in
    sequence. Same per-member physics, different RNG draw order - the landing
    distributions must agree within Monte-Carlo noise."""
    from windfall.ensemble_vec import ensemble_preburst_vec

    p = FlightProfile()
    for a in range(0, 11000, 250):
        p.add_sample(a, 12.0, 3.0)
    cfg = _cfg(150)
    kw = dict(lat=45.0, lon=7.0, alt=2000.0, t0=T0, burst_alt=9000.0,
              ascent_rate=5.0, default_b=5.5, profile=p, ground_fn=_flat(),
              cfg=cfg, measured_range=p.alt_range())
    dists, ratios = [], []
    for s in range(4):
        a = ensemble_preburst(rng=random.Random(s), **kw)
        b = ensemble_preburst_vec(rng=random.Random(s), **kw)
        assert a is not None and b is not None and b.n_members > 0
        dists.append(haversine_km(a.lat, a.lon, b.lat, b.lon))
        ratios.append(b.radius_km / a.radius_km)
        # median member eta: agrees within the burst-altitude spread's effect
        assert abs((b.eta - a.eta).total_seconds()) < 300.0
    # MC noise on the mean ~ radius/sqrt(N) (radius ~11 km at n=150 → ~1 km
    # floor); generous but catches physics bugs
    assert sum(dists) / len(dists) < 2.5                 # km
    assert 0.6 < sum(ratios) / len(ratios) < 1.5         # radii agree within ~40%


def test_vectorised_preburst_blend_matches_scalar():
    """Same differential through the GFS-blend batch path (shared centroid
    column + grounds= AGL floor), which the pre-burst loop gates differently
    from the descent loop: members *start* low while ascending, so ground
    sampling must engage for the AGL floor without tripping a landing."""
    from windfall.ensemble_vec import ensemble_preburst_vec
    from windfall.gfs import WindCube, CubePairWind
    from windfall.profile import blended_wind_fn
    import numpy as np

    p = FlightProfile()
    for a in range(0, 11000, 250):
        p.add_sample(a, 12.0, 3.0, lat=45.0, lon=7.0, t=1000.0 + a / 5.0)

    def cube(vt, shift):
        lats = np.array([47.0, 45.0, 43.0])
        lons = np.array([5.0, 7.0, 9.0])
        heights = np.zeros((3, 3, 3))
        u = np.zeros((3, 3, 3))
        for k, h in enumerate([12000.0, 6000.0, 500.0]):
            heights[k] = h
            u[k] = 8.0 + h / 1000.0 + shift
        v = np.full((3, 3, 3), 2.0)
        return WindCube.from_grid(valid_time=vt, lats=lats, lons=lons,
                                  heights=heights, u=u, v=v)

    from datetime import timedelta
    pair = CubePairWind(cube(T0, 0.0), cube(T0 + timedelta(hours=1), 2.0), t0=T0)
    cfg = _cfg(150)
    blend = blended_wind_fn(p, pair, cfg.profile, t0_epoch=T0.timestamp(),
                            ground_fn=_flat())
    kw = dict(lat=45.0, lon=7.0, alt=2000.0, t0=T0, burst_alt=9000.0,
              ascent_rate=5.0, default_b=5.5, profile=p, ground_fn=_flat(),
              cfg=cfg, wind_fn=blend, measured_range=p.alt_range())
    dists, ratios = [], []
    for s in range(4):
        a = ensemble_preburst(rng=random.Random(s), **kw)
        b = ensemble_preburst_vec(rng=random.Random(s), **kw)
        assert a is not None and b is not None and b.n_members > 0
        dists.append(haversine_km(a.lat, a.lon, b.lat, b.lon))
        ratios.append(b.radius_km / a.radius_km)
    # radius ~13 km at n=150 → MC floor on the mean ~1.3 km; the same-seed
    # shared-vs-exact column difference is only ~80 m here
    assert sum(dists) / len(dists) < 3.0
    assert 0.6 < sum(ratios) / len(ratios) < 1.5


def test_vectorised_preburst_zero_noise_collapses_to_deterministic():
    """With every perturbation off, each vectorised pre-burst member is the
    deterministic integrate_ascent + integrate_descent chain (at the member dt),
    including the burst-overshoot clamp and the continued field time."""
    from datetime import timedelta

    from windfall.config import IntegratorConfig
    from windfall.ensemble_vec import ensemble_preburst_vec
    from windfall.integrator import integrate_ascent

    cfg = _cfg(16)
    cfg.ensemble.wind_sigma_measured_mps = 0.0
    cfg.ensemble.wind_sigma_extrapolated_mps = 0.0
    cfg.ensemble.wind_bias_sigma_mps = 0.0
    cfg.ensemble.b_sigma_rel_preburst = 0.0
    cfg.ensemble.burst_alt_sigma_m = 0.0
    cfg.ensemble.ascent_rate_sigma_rel = 0.0
    p = _profile()
    mem_cfg = IntegratorConfig(dt_seconds=cfg.ensemble.dt_seconds,
                               max_iterations=cfg.integrator.max_iterations,
                               max_sim_seconds=cfg.integrator.max_sim_seconds)
    b_lat, b_lon, t_asc = integrate_ascent(
        lat=45.0, lon=7.0, alt=12000.0, burst_alt=30000.0, ascent_rate=5.0,
        profile=p, cfg=mem_cfg)
    det = integrate_descent(
        lat=b_lat, lon=b_lon, alt=30000.0, t0=T0 + timedelta(seconds=t_asc),
        profile=p, descent=DescentModel(b=5.5, residual_mps=0.0, n_points=0),
        ground_fn=_flat(), cfg=mem_cfg, t_offset_s=t_asc)
    ens = ensemble_preburst_vec(lat=45.0, lon=7.0, alt=12000.0, t0=T0,
                                burst_alt=30000.0, ascent_rate=5.0,
                                default_b=5.5, profile=p, ground_fn=_flat(),
                                cfg=cfg, rng=random.Random(0),
                                measured_range=p.alt_range())
    assert ens is not None and ens.n_members == 16
    assert haversine_km(ens.lat, ens.lon, det.lat, det.lon) < 0.05
    assert ens.radius_km < 0.05
    assert abs((ens.eta - det.eta).total_seconds()) < 5.0
