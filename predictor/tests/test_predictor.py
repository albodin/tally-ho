"""End-to-end predictor + replay accuracy tests.

The synthetic flight is generated with the same physics the predictor assumes,
so replaying it through the production predictor must recover the true landing
to within integrator discretisation - this is the core accuracy validation."""

import pytest

from windfall.config import Config
from windfall.geo import haversine_km
from windfall.replay import aggregate, replay_messages
from windfall.uncertainty import uncertainty_radius_km
from windfall.config import UncertaintyConfig
from tests.conftest import simulate_flight


@pytest.fixture(scope="module")
def replayed():
    """One full replay shared by the read-only accuracy assertions below -
    replaying ~2000 descent frames through the ensemble predictor is the
    expensive part, and these tests only inspect its records."""
    f = simulate_flight()
    return f, replay_messages(f.frames)


def test_replay_recovers_landing(replayed):
    flight, res = replayed
    assert res.n_predictions > 20
    # final prediction (near ground) is highly accurate
    assert res.final_error_km < 1.0
    # predicted landing close to true landing
    err = haversine_km(res.truth_lat, res.truth_lon, flight.land_lat, flight.land_lon)
    assert err < 1.0


def test_replay_handles_stringified_wire(flight_stringified):
    res = replay_messages(flight_stringified.frames)
    assert res.n_predictions > 20
    assert res.final_error_km < 1.5


def test_error_converges_during_descent(replayed):
    _, res = replayed
    # error from the highest-altitude prediction vs the lowest
    high = [r for r in res.records if r.alt_at_pred > 15000]
    low = [r for r in res.records if r.alt_at_pred < 3000]
    assert high and low
    mean_high = sum(r.error_km for r in high) / len(high)
    mean_low = sum(r.error_km for r in low) / len(low)
    assert mean_low <= mean_high + 0.5   # tightens (or holds) as it descends


def test_uncertainty_shrinks_as_it_descends(replayed):
    _, res = replayed
    radii = [r.uncertainty_km for r in res.records]
    # earliest (high) prediction has a larger radius than the last (near ground)
    assert radii[0] > radii[-1]
    assert radii[-1] >= 0.0


def test_uncertainty_model_monotonic():
    cfg = UncertaintyConfig()
    far = uncertainty_radius_km(sim_seconds=3600, measured_fraction=0.5,
                                fit_residual_mps=1.0, cfg=cfg)
    near = uncertainty_radius_km(sim_seconds=120, measured_fraction=1.0,
                                 fit_residual_mps=0.1, cfg=cfg)
    assert far > near
    # more unmeasured column → larger radius
    measured = uncertainty_radius_km(sim_seconds=1800, measured_fraction=1.0,
                                     fit_residual_mps=0, cfg=cfg)
    extrap = uncertainty_radius_km(sim_seconds=1800, measured_fraction=0.0,
                                   fit_residual_mps=0, cfg=cfg)
    assert extrap > measured


def test_calibration_on_synthetic_fleet():
    # A small fleet of varied flights; on clean synthetic data the reported
    # radius should comfortably contain the (small) true error.
    results = []
    for i, (blat, blon, burst) in enumerate([
        (45.0, 7.0, 30000), (46.0, 8.0, 28000), (44.0, 6.5, 32000),
        (47.0, 9.0, 26000), (45.5, 7.5, 31000),
    ]):
        f = simulate_flight(serial=f"FLEET{i}", launch_lat=blat, launch_lon=blon,
                            burst_alt=burst)
        results.append(replay_messages(f.frames))
    metrics = aggregate(results)
    assert metrics.n_flights == 5
    assert metrics.mean_final_error_km < 1.5
    # radius should contain the error for the vast majority of predictions
    assert metrics.calibration_rate > 0.9


def test_radius_scale_multiplies_published_radius():
    """`uncertainty.radius_scale` is the knob the backtest's measured r-scale
    feeds (e.g. 1.39 on the 2026-06-10 corpus); it must scale the published
    radius linearly."""
    cfg = UncertaintyConfig()
    base = uncertainty_radius_km(sim_seconds=1800, measured_fraction=0.5,
                                 fit_residual_mps=1.0, cfg=cfg)
    cfg.radius_scale = 1.39
    scaled = uncertainty_radius_km(sim_seconds=1800, measured_fraction=0.5,
                                   fit_residual_mps=1.0, cfg=cfg)
    assert scaled == pytest.approx(base * 1.39, abs=0.01)


def test_shrink_b_toward_climatology_prior():
    """The fitted B shrinks toward the per-family climatology prior as
    pseudo-counts - prior-dominated when few points, fit-dominated when many,
    and a no-op without a prior / with prior_strength 0."""
    from datetime import date

    from windfall.descent import DescentModel
    from windfall.predictor import Predictor
    from windfall.tracker import Flight

    class StubClim:
        def __init__(self, b):
            self._b = b

        def descent_b(self, sonde_type):
            return self._b

    flight = Flight(serial="S1", launch_day=date(2026, 7, 14), type="RS41")
    cfg = Config()
    cfg.descent.prior_strength = 6.0
    p = Predictor(cfg, climatology=StubClim(8.0))

    # few points → prior dominates: (6*8 + 4*5)/(6+4) = 6.8
    few = p._shrink_b(DescentModel(b=5.0, residual_mps=0.3, n_points=4), flight)
    assert few.b == pytest.approx(6.8)
    # many points → fit dominates
    many = p._shrink_b(DescentModel(b=5.0, residual_mps=0.3, n_points=100), flight)
    assert many.b == pytest.approx((6 * 8 + 100 * 5) / 106)
    # no climatology / prior_strength 0 / no prior for family → unchanged
    m = DescentModel(b=5.0, residual_mps=0.3, n_points=4)
    assert Predictor(cfg)._shrink_b(m, flight).b == 5.0
    cfg0 = Config()
    cfg0.descent.prior_strength = 0.0
    assert Predictor(cfg0, climatology=StubClim(8.0))._shrink_b(m, flight).b == 5.0
    assert Predictor(cfg, climatology=StubClim(None))._shrink_b(m, flight).b == 5.0
