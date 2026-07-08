"""Tests for the offline accuracy harness. The live
store-backed scoring (accuracy_from_store + SQLite) is tested in the
tally-ho app suite."""

from windfall.replay import aggregate, replay_messages
from tests.conftest import fast_ensemble_cfg, simulate_flight


def test_offline_replay_recovers_true_landing():
    # The synthetic flight integrates the same physics the predictor assumes, so
    # replaying it should recover the landing to within integrator discretisation.
    f = simulate_flight(serial="R1", burst_alt=28000)
    result = replay_messages(f.frames)
    assert result.n_predictions > 0
    assert result.final_error_km is not None and result.final_error_km < 2.0


def test_radius_scale_for_target_coverage():
    import pytest
    from windfall.replay import PredRecord, ReplayResult, TARGET_COVERAGE

    # 100 predictions whose error/radius ratio runs 0.01..1.00 → the 68% quantile
    # of the ratios is ~0.68: radii could shrink ~32% and still hit the target.
    res = ReplayResult(serial="X", truth_lat=45.0, truth_lon=7.0)
    for i in range(1, 101):
        err = i / 100.0
        res.records.append(PredRecord(
            alt_at_pred=1000.0, error_km=err, uncertainty_km=1.0,
            sim_seconds=600.0, source="measured", inside_radius=err <= 1.0))
    m = aggregate([res])
    assert m.radius_scale_for_target == pytest.approx(TARGET_COVERAGE, abs=0.02)
    assert f"{TARGET_COVERAGE * 100:.0f}%" in m.report()


def test_replay_truth_override_scores_against_recovered_position():
    f = simulate_flight(serial="T1", burst_alt=24000)
    cfg = fast_ensemble_cfg()
    # truth at the true landing vs. truth shifted ~10 km east: the same replay
    # must score differently, proving the override is what's being compared.
    near = replay_messages(f.frames, cfg=cfg, truth=(f.land_lat, f.land_lon))
    far = replay_messages(f.frames, cfg=cfg, truth=(f.land_lat, f.land_lon + 0.13))
    assert near.truth_source == far.truth_source == "recovered"
    assert near.n_predictions > 0
    assert far.final_error_km > near.final_error_km + 5.0


def test_replay_messages_skips_unparseable_frames():
    f = simulate_flight(serial="T2", burst_alt=24000)
    raw = [{"serial": "T2"}] + f.frames + [{"garbage": True}]
    result = replay_messages(raw, cfg=fast_ensemble_cfg())
    assert result is not None and result.n_predictions > 0
    assert replay_messages([{"nope": 1}]) is None


def test_backtest_corpus(tmp_path):
    import json

    from windfall.replay import backtest_corpus

    for serial in ("BT1", "BT2"):
        f = simulate_flight(serial=serial, burst_alt=24000)
        (tmp_path / f"{serial}.json").write_text(json.dumps({
            "recovery": {"serial": serial, "lat": f.land_lat, "lon": f.land_lon,
                         "recovered": True},
            "frames": f.frames,
        }))
    results = backtest_corpus(tmp_path, cfg=fast_ensemble_cfg())
    assert [r.serial for r in results] == ["BT1", "BT2"]
    for r in results:
        assert r.truth_source == "recovered"
        assert r.final_error_km is not None and r.final_error_km < 5.0
    assert len(backtest_corpus(tmp_path, cfg=fast_ensemble_cfg(), limit=1)) == 1
    metrics = aggregate(results)
    assert metrics.n_flights == 2 and metrics.n_predictions > 0


def _sim_corpus(tmp_path, serials):
    import json

    for serial in serials:
        f = simulate_flight(serial=serial, burst_alt=24000)
        (tmp_path / f"{serial}.json").write_text(json.dumps({
            "recovery": {"serial": serial, "lat": f.land_lat, "lon": f.land_lon,
                         "recovered": True},
            "frames": f.frames,
        }))


def test_backtest_corpus_parallel_matches_serial(tmp_path):
    from windfall.replay import backtest_corpus, backtest_corpus_parallel

    _sim_corpus(tmp_path, ("PA1", "PA2", "PA3"))
    cfg = fast_ensemble_cfg()
    serial = backtest_corpus(tmp_path, cfg=cfg)
    par = backtest_corpus_parallel(tmp_path, cfg=cfg, with_gfs=False, jobs=2)

    # same flights, byte-identical scores: the ensemble seed is stable per
    # serial, so execution order/process must not change the numbers
    assert sorted(r.serial for r in par) == ["PA1", "PA2", "PA3"]
    by_serial = {r.serial: r for r in par}
    for r in serial:
        assert by_serial[r.serial].final_error_km == r.final_error_km
        assert by_serial[r.serial].n_predictions == r.n_predictions

    assert len(backtest_corpus_parallel(tmp_path, cfg=cfg, with_gfs=False,
                                        jobs=2, limit=1)) == 1
    assert backtest_corpus_parallel(tmp_path / "empty", cfg=cfg, jobs=2) == []


def test_parallel_worker_functions_inprocess(tmp_path):
    # The worker init/replay pair runs in subprocesses in production; call them
    # directly to pin behaviour (and so coverage sees them).
    from windfall.replay import _PAR, _par_init, _par_replay

    _sim_corpus(tmp_path, ("PW1",))
    cfg = fast_ensemble_cfg()
    cfg.dem.path = str(tmp_path / "no-dem")     # missing -> flat-ground fallback
    _par_init(cfg, with_dem=True, with_gfs=True)
    assert callable(_PAR["ground_fn"])
    assert _PAR["gfs"] is None                  # gfs.enabled is False by default

    res = _par_replay(str(tmp_path / "PW1.json"))
    assert res is not None and res.serial == "PW1"
    (tmp_path / "junk.json").write_text("{not json")
    assert _par_replay(str(tmp_path / "junk.json")) is None
    _par_init(cfg, with_dem=False, with_gfs=False)   # reset module state


def test_run_ablation_parallel(tmp_path):
    import pytest

    from windfall.replay import run_ablation

    _sim_corpus(tmp_path, ("AB1",))
    cfg = fast_ensemble_cfg()
    outcomes = run_ablation(tmp_path, cfg=cfg, modes=["measured-only"], jobs=2)
    assert outcomes["measured-only"].metrics.n_flights == 1

    with pytest.raises(ValueError, match="jobs > 1"):
        run_ablation(tmp_path, cfg=cfg, modes=["measured-only"], jobs=2,
                     gfs_factory=lambda c: None)
