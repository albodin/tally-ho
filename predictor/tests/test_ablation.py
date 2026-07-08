"""Tests for the plan-Phase-2 live descent refresh, the wind-assembly ablation
switches, and the ablation runner itself (plan Phase 5)."""

import json

import pytest

from windfall.config import Config, ProfileConfig
from windfall.gfs import StaticGFSSource
from windfall.profile import FlightProfile, blended_wind_fn
from windfall.replay import ablation_report, replay_messages, run_ablation
from windfall.telemetry import parse_frame
from windfall.tracker import FlightTracker
from tests.conftest import fast_ensemble_cfg, simulate_flight, wind_at


def _truth_gfs_source(top_alt: float = 32_000.0) -> StaticGFSSource:
    """A model wind source carrying the simulator's own wind field, so model
    winds are as correct as measured ones and predictions stay near truth."""
    prof = FlightProfile(bin_size_m=250.0)
    alt = 100.0
    while alt < top_alt:
        u, v = wind_at(alt)
        prof.add_sample(alt, u, v)
        alt += 250.0
    return StaticGFSSource(prof)


# ---- profile weight cap -----------------------------------------------------

def test_add_sample_weight_cap_lets_fresh_samples_dominate():
    capped = FlightProfile()
    uncapped = FlightProfile()
    for _ in range(30):
        capped.add_sample(1000.0, 5.0, 0.0)
        uncapped.add_sample(1000.0, 5.0, 0.0)
    for _ in range(8):
        capped.add_sample(1000.0, 15.0, 0.0, weight_cap=8)
        uncapped.add_sample(1000.0, 15.0, 0.0)
    # Uncapped, 8 fresh samples drown in the 30 stale ones (~7.1 m/s); capped,
    # the stale average counts as at most 8 samples and the wind converges on
    # the fresh measurement.
    assert uncapped.wind(1000.0)[0] == pytest.approx(7.1, abs=0.2)
    assert capped.wind(1000.0)[0] > 10.0
    # the true sample count is still tracked
    assert capped.bin_near(1000.0).n == 38


# ---- live descent refresh (tracker) ----------------------------------------

def _profile_after(cfg: Config):
    f = simulate_flight(serial="REFR1", burst_alt=20_000)
    tracker = FlightTracker(cfg)
    for m in f.frames:
        flight, _ = tracker.update(parse_frame(m))
    return flight.profile


def test_descent_refresh_updates_low_bins_with_descent_times():
    on = Config()
    off = Config()
    off.profile.descent_refresh_enabled = False
    bin_on = _profile_after(on).bin_near(2000.0)
    bin_off = _profile_after(off).bin_near(2000.0)
    # With refresh, the 2 km bin's sample-time metadata reflects the descent
    # pass (hours after the ascent crossed it) - that recency is what raises
    # the bin's trust in the measured→model blend near the landing zone.
    assert bin_on.t > bin_off.t + 1000.0
    assert bin_on.n > bin_off.n


# ---- use_measured_winds (model-only ablation switch) ------------------------

def test_use_measured_winds_off_predicts_from_model_only():
    f = simulate_flight(serial="MDL1", burst_alt=24_000)
    gfs = _truth_gfs_source()

    measured = replay_messages(f.frames, cfg=fast_ensemble_cfg(), gfs_source=gfs)
    assert {r.source for r in measured.records} == {"measured"}

    cfg = fast_ensemble_cfg()
    cfg.predict.use_measured_winds = False
    model_only = replay_messages(f.frames, cfg=cfg, gfs_source=gfs)
    assert model_only.n_predictions > 0
    assert {r.source for r in model_only.records} == {"gfs"}
    # the model source carries the true winds, so accuracy holds
    assert model_only.final_error_km < 5.0


# ---- AGL floor in the measured→model blend ----------------------------------

def test_blend_forces_model_winds_below_agl_floor():
    prof = FlightProfile()
    for alt in (1500.0, 5000.0, 10000.0):
        prof.add_sample(alt, 10.0, 0.0, lat=45.0, lon=7.0, t=0.0)
    cfg = ProfileConfig()

    def model(la, lo, alt, sim_t):
        return (0.0, 0.0)

    with_ground = blended_wind_fn(prof, model, cfg, t0_epoch=0.0,
                                  ground_fn=lambda la, lo: 0.0)
    # below 3 km AGL: terrain-local boundary layer → model outright
    assert with_ground(45.0, 7.0, 1500.0, 0.0)[0] == 0.0
    # high up, sampled here and now → measured dominates
    assert with_ground(45.0, 7.0, 10000.0, 0.0)[0] == pytest.approx(10.0, abs=0.5)

    without_ground = blended_wind_fn(prof, model, cfg, t0_epoch=0.0)
    assert without_ground(45.0, 7.0, 1500.0, 0.0)[0] == pytest.approx(10.0, abs=0.5)


# ---- bias-correction wind mode (plan Phase 2 formulation) --------------------

def test_bias_corrected_wind_fn_shifts_model_by_measured_delta():
    from windfall.profile import bias_corrected_wind_fn

    prof = FlightProfile()
    for alt in (5_000.0, 10_000.0):
        prof.add_sample(alt, 10.0, 0.0, lat=45.0, lon=7.0, t=0.0)
    cfg = ProfileConfig()

    def model(la, lo, alt, sim_t=0.0):
        return (6.0, 2.0)

    fn = bias_corrected_wind_fn(prof, model, cfg, t0_epoch=0.0)
    # at the measurement location/time: model + full bias = the measured wind
    u, v = fn(45.0, 7.0, 5_000.0, 0.0)
    assert u == pytest.approx(10.0) and v == pytest.approx(0.0)
    # far downwind the correction decays toward the raw model
    u_far, _ = fn(45.0, 9.5, 5_000.0, 0.0)   # ~200 km east
    assert 6.0 < u_far < 8.0
    # outside the measured range: pure model
    assert fn(45.0, 7.0, 20_000.0, 0.0) == (6.0, 2.0)
    # AGL floor forces pure model down low
    fn_g = bias_corrected_wind_fn(prof, model, cfg, t0_epoch=0.0,
                                  ground_fn=lambda la, lo: 4_000.0)
    assert fn_g(45.0, 7.0, 5_000.0, 0.0) == (6.0, 2.0)


def test_bias_mode_end_to_end_replay():
    f = simulate_flight(serial="BIAS1", burst_alt=24_000)
    cfg = fast_ensemble_cfg()
    cfg.profile.correction_mode = "bias"
    res = replay_messages(f.frames, cfg=cfg, gfs_source=_truth_gfs_source())
    assert res.n_predictions > 0
    assert res.final_error_km < 5.0


# ---- ablation runner ---------------------------------------------------------

def test_run_ablation_compares_wind_assembly_variants(tmp_path):
    for serial in ("AB1", "AB2"):
        f = simulate_flight(serial=serial, burst_alt=24_000)
        (tmp_path / f"{serial}.json").write_text(json.dumps({
            "recovery": {"serial": serial, "lat": f.land_lat, "lon": f.land_lon,
                         "recovered": True},
            "frames": f.frames,
        }))
    gfs = _truth_gfs_source()
    outcomes = run_ablation(tmp_path, cfg=fast_ensemble_cfg(),
                            gfs_factory=lambda cfg: gfs)
    assert list(outcomes) == ["measured-only", "model-only", "blend",
                              "blend+refresh", "bias+refresh", "gfs-only"]
    assert not outcomes["measured-only"].gfs_available
    assert outcomes["model-only"].gfs_available
    for name, o in outcomes.items():
        assert o.metrics.n_flights == 2, name
        assert o.metrics.n_predictions > 0, name
        # the model source carries the true winds → every variant stays sane
        assert o.metrics.mean_final_error_km < 5.0, name
    assert {r.source for r in outcomes["model-only"].results[0].records} == {"gfs"}
    assert {r.source for r in outcomes["measured-only"].results[0].records} == {"measured"}

    report = ablation_report(outcomes)
    assert "model-only" in report and "blend+refresh" in report
    assert "AB1" in report and "AB2" in report


def test_run_ablation_mode_subset_and_errors(tmp_path):
    with pytest.raises(ValueError):
        run_ablation(tmp_path, cfg=Config(), modes=["nope"])
    # empty corpus → None (caller prints the fetch-corpus hint)
    assert run_ablation(tmp_path, cfg=Config(),
                        gfs_factory=lambda cfg: None) is None
