"""Tests for wind/density profile construction & sampling."""

import math
import random

import pytest

from windfall.config import ProfileConfig
from windfall.profile import FlightProfile, build_ascent_profile
from windfall.telemetry import parse_frame
from windfall.kinematics import segment


def test_vector_average_in_bin():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(100, 10, 0)
    p.add_sample(120, 20, 0)   # same bin (0..150)
    u, v = p.wind(110)
    assert u == pytest.approx(15.0)   # averaged


def test_clamp_below_and_above_range():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 5, 1)
    p.add_sample(2000, 9, 3)
    # below lowest sampled bin → clamp to lowest
    assert p.wind(0)[0] == pytest.approx(5.0)
    # above highest → clamp to highest
    assert p.wind(9000)[0] == pytest.approx(9.0)


def test_wind_interpolates_between_bins():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 5, 1)   # bin centre 975
    p.add_sample(2000, 9, 3)   # bin centre 2025
    # halfway between the two centres → halfway between the two winds
    u, v = p.wind(1500)
    assert u == pytest.approx(7.0)
    assert v == pytest.approx(2.0)
    # at the centres, the exact bin values
    assert p.wind(975) == pytest.approx((5.0, 1.0))
    assert p.wind(2025) == pytest.approx((9.0, 3.0))


def test_gfs_fill_outside_range_and_wide_gaps():
    p = FlightProfile(bin_size_m=150.0, gap_fill_m=600.0)
    p.add_sample(1000, 5, 1)
    p.add_sample(1300, 7, 2)
    p.add_sample(5000, 9, 3)
    p.gfs_fill = lambda alt: (99.0, 99.0)
    # inside a tightly-sampled span, GFS not used
    assert p.wind(1200)[0] != 99.0
    # a wide interior hole (1300..5000 - e.g. a reception dropout) fills from
    # GFS instead of lerping across kilometres
    assert p.wind(3000) == (99.0, 99.0)
    # outside the range, GFS used
    assert p.wind(8000) == (99.0, 99.0)
    # without GFS, the wide gap falls back to the lerp
    p.gfs_fill = None
    u, _ = p.wind(3000)
    assert 7.0 < u < 9.0


def test_density_falls_back_to_isa():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 5, 1)   # no rho
    rho = p.density(1000)
    assert rho == pytest.approx(1.112, abs=0.02)  # ISA at 1 km


def test_density_uses_measured_bin():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 5, 1, rho=0.5)   # bin centre 975
    # exact at the bin centre; continuous ISA-shape scaling around it
    assert p.density(975) == pytest.approx(0.5)
    assert p.density(1010) == pytest.approx(0.5, abs=0.005)


def test_density_interpolates_log_between_bins():
    from windfall.atmosphere import isa_density
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 0, 0, rho=1.0)
    p.add_sample(4000, 0, 0, rho=0.7)
    lo, hi = p.bin_near(1000).alt, p.bin_near(4000).alt
    mid = p.density((lo + hi) / 2)
    assert mid == pytest.approx((1.0 * 0.7) ** 0.5, rel=1e-6)  # geometric mean
    # outside the measured range: ISA scaled to match the edge, not raw ISA
    edge_ratio = 1.0 / isa_density(lo)
    assert p.density(500) == pytest.approx(isa_density(500) * edge_ratio, rel=1e-6)


def test_measured_fraction():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(0, 1, 0)
    p.add_sample(10_000, 1, 0)   # sampled range ~0..10150
    # descending from 20 km to ground: only lower half measured
    frac = p.measured_fraction(20_000, 0)
    assert 0.4 < frac < 0.6
    # fully inside sampled range
    assert p.measured_fraction(9000, 1000) == pytest.approx(1.0)


def test_json_roundtrip():
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 5, 1, rho=0.9)
    p.add_sample(2000, 9, 3)
    d = p.to_json()
    p2 = FlightProfile.from_json(d)
    assert p2.wind(1000) == pytest.approx(p.wind(1000))
    assert p2.density(1000) == pytest.approx(p.density(1000))


def test_blended_wind_decays_from_measured_to_gfs():
    from windfall.profile import blended_wind_fn
    cfg = ProfileConfig()
    p = FlightProfile(bin_size_m=150.0)
    # measured column sampled at (45, 7) at t=1000: 10 m/s east at all alts
    for a in range(0, 10000, 300):
        p.add_sample(a, 10.0, 0.0, lat=45.0, lon=7.0, t=1000.0)
    gfs = lambda lat, lon, alt, sim_t: (20.0, 4.0)
    wind = blended_wind_fn(p, gfs, cfg, t0_epoch=1000.0)
    # right on the ascent column, fresh → essentially pure measured
    u, v = wind(45.0, 7.0, 5000.0, 0.0)
    assert u == pytest.approx(10.0, abs=0.2)
    # far away (several e-folding distances) → essentially pure GFS
    u, v = wind(45.0, 11.0, 5000.0, 0.0)
    assert u == pytest.approx(20.0, abs=0.5)
    # ...and so does a very stale sample
    u, v = wind(45.0, 7.0, 5000.0, 10 * cfg.gfs_blend_age_s)
    assert u == pytest.approx(20.0, abs=0.5)
    # outside the measured range entirely → GFS
    assert wind(45.0, 7.0, 20000.0, 0.0) == (20.0, 4.0)


def test_blended_wind_uses_gfs_in_interior_holes():
    from windfall.profile import blended_wind_fn
    cfg = ProfileConfig()
    p = FlightProfile(bin_size_m=150.0)
    p.add_sample(1000, 10.0, 0.0, lat=45.0, lon=7.0, t=0.0)
    p.add_sample(8000, 12.0, 0.0, lat=45.0, lon=7.0, t=0.0)
    wind = blended_wind_fn(p, lambda *a: (20.0, 4.0), cfg, t0_epoch=0.0)
    # the 7 km unsampled hole between the bins → pure GFS
    assert wind(45.0, 7.0, 4000.0, 0.0) == (20.0, 4.0)


def test_build_ascent_profile_recovers_wind(flight):
    # Take ascent frames (up to burst) and check recovered wind ~ truth.
    from tests.conftest import wind_at
    frames = [parse_frame(m) for m in flight.frames]
    asc = [f for f in frames if True]
    # restrict to ascent: altitudes strictly increasing until burst
    ascent = []
    for f in frames:
        ascent.append(f)
        if f.alt >= flight.burst_alt - 1:
            break
    p = build_ascent_profile(ascent, ProfileConfig())
    for alt in (3000, 8000, 15000, 22000):
        u, v = p.wind(alt)
        tu, tv = wind_at(alt)
        assert u == pytest.approx(tu, abs=1.5)
        assert v == pytest.approx(tv, abs=1.5)


def test_wind_residual_stats_recovers_injected_error():
    """The ascent measured-minus-model residual should recover a known
    injected bias, de-biased scatter, and a plausible correlation length."""
    from windfall.profile import wind_residual_stats

    p = FlightProfile(bin_size_m=150.0)
    rng = random.Random(0)
    for a in range(0, 30000, 150):
        # measured = model (5, 2) + known bias (3, -2) + small correlated noise
        p.add_sample(a + 75, 5.0 + 3.0 + rng.gauss(0, 1.0),
                     2.0 - 2.0 + rng.gauss(0, 1.0),
                     lat=45.0, lon=7.0, t=1000.0 + a)
    stats = wind_residual_stats(p, lambda la, lo, alt, t: (5.0, 2.0),
                                t0_epoch=1000.0, min_bins=8)
    assert stats is not None
    assert stats.bias_u == pytest.approx(3.0, abs=0.4)
    assert stats.bias_v == pytest.approx(-2.0, abs=0.4)
    assert stats.sigma_mps == pytest.approx(1.0, abs=0.4)
    assert stats.bias_pc_mps == pytest.approx(math.hypot(3.0, 2.0) / math.sqrt(2), abs=0.4)
    assert 200.0 <= stats.corr_len_m <= 6000.0
    assert stats.n_bins >= 8


def test_wind_residual_stats_none_paths():
    """Too few bins, no model sample, or bins without position → None (caller
    keeps the global ensemble constants)."""
    from windfall.profile import wind_residual_stats

    p = FlightProfile(bin_size_m=150.0)
    for a in range(0, 600, 150):   # only 4 bins, below min_bins=8
        p.add_sample(a + 75, 5.0, 2.0, lat=45.0, lon=7.0, t=1000.0 + a)
    assert wind_residual_stats(p, lambda *a: (5.0, 2.0), min_bins=8) is None
    # model returns None everywhere → no residual samples
    p2 = FlightProfile(bin_size_m=150.0)
    for a in range(0, 3000, 150):
        p2.add_sample(a + 75, 5.0, 2.0, lat=45.0, lon=7.0, t=1000.0 + a)
    assert wind_residual_stats(p2, lambda *a: None, min_bins=8) is None


def test_outlier_segment_rejected():
    cfg = ProfileConfig(max_horizontal_mps=200)
    p = FlightProfile(bin_size_m=150.0)
    from windfall.profile import update_profile_from_pair
    from windfall.models import Frame
    from datetime import datetime, timezone, timedelta
    base = datetime(2026, 6, 7, tzinfo=timezone.utc)
    a = Frame("S", 45.0, 7.0, 1000, base.timestamp(), base)
    # 1 second later, jumped 5 degrees lon (~390 km) → ~390km/s, glitch
    b = Frame("S", 45.0, 12.0, 1005, (base + timedelta(seconds=1)).timestamp(),
              base + timedelta(seconds=1))
    added = update_profile_from_pair(p, a, b, cfg)
    assert added is False
    assert p.is_empty()


def test_blended_wind_fn_batch_matches_scalar():
    """The batched blend (.batch) equals the per-member blend across the
    range / AGL-floor / gap / distance-age-weight branches."""
    import numpy as np
    from windfall.config import ProfileConfig
    from windfall.profile import blended_wind_fn

    p = FlightProfile(150.0)
    for a in range(0, 20000, 150):
        p.add_sample(a + 75, 10.0 + a * 0.001, 3.0, lat=45.0, lon=7.0, t=1000.0 + a)

    class MockField:
        def __call__(self, lat, lon, alt, t):
            return (20.0, -5.0)

        def batch(self, lats, lons, alts, t):
            return np.full(len(alts), 20.0), np.full(len(alts), -5.0)

    cfg = ProfileConfig()
    wind = blended_wind_fn(p, MockField(), cfg, t0_epoch=1000.0,
                           ground_fn=lambda lat, lon: 0.0)
    rng = np.random.default_rng(1)
    lats = 44.5 + rng.random(60)
    lons = 6.5 + rng.random(60)
    alts = rng.random(60) * 22000.0
    for sim_t in (0.0, 600.0):
        bu, bv = wind.batch(lats, lons, alts, sim_t)
        for i in range(len(lats)):
            su, sv = wind(float(lats[i]), float(lons[i]), float(alts[i]), sim_t)
            assert bu[i] == pytest.approx(su, abs=1e-3)
            assert bv[i] == pytest.approx(sv, abs=1e-3)


def test_blended_wind_fn_batch_shared_close_to_scalar():
    """shared=True takes the fine-grid fast path (centroid GFS
    column, 10 m row snapping, equirectangular distance). Away from range/hole
    boundaries it must track the scalar blend to well under the ensemble's
    wind-noise floor (sigma >= 0.5 m/s)."""
    import numpy as np
    from windfall.config import ProfileConfig
    from windfall.profile import blended_wind_fn

    p = FlightProfile(150.0)
    for a in range(0, 20000, 150):
        p.add_sample(a + 75, 10.0 + a * 0.001, 3.0, lat=45.0, lon=7.0, t=1000.0 + a)

    class MockField:
        # no .batch: the blend samples the field per member either way, so any
        # shared-vs-scalar difference is the fast weight path alone
        def __call__(self, lat, lon, alt, t):
            return (20.0, -5.0)

    cfg = ProfileConfig()
    wind = blended_wind_fn(p, MockField(), cfg, t0_epoch=1000.0,
                           ground_fn=lambda lat, lon: 0.0)
    rng = np.random.default_rng(3)
    lats = 44.5 + rng.random(60)
    lons = 6.5 + rng.random(60)
    alts = rng.random(60) * 22000.0
    lo, hi = p.alt_range()
    # the fast path quantises altitude to 10 m rows; stay clear of the range
    # edges where that legitimately flips the blend on/off
    away = (np.abs(alts - lo) > 15.0) & (np.abs(alts - hi) > 15.0)
    for sim_t in (0.0, 600.0):
        bu, bv = wind.batch(lats, lons, alts, sim_t, shared=True)
        for i in np.nonzero(away)[0]:
            su, sv = wind(float(lats[i]), float(lons[i]), float(alts[i]), sim_t)
            assert bu[i] == pytest.approx(su, abs=0.1)
            assert bv[i] == pytest.approx(sv, abs=0.1)
    # caller-supplied ground elevations replace the per-member ground_fn loop
    # without changing the result
    bu1, bv1 = wind.batch(lats, lons, alts, 0.0, shared=True)
    bu2, bv2 = wind.batch(lats, lons, alts, 0.0, shared=True, grounds=np.zeros(60))
    assert np.allclose(bu1, bu2, atol=1e-12)
    assert np.allclose(bv1, bv2, atol=1e-12)
