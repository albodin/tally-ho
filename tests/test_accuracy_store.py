"""Live store-backed accuracy scoring: each recorded landing is
scored against the predictions the daemon saved - the engine's replay metrics
driven by the app's SQLite store."""

from tallyho.app import App
from tallyho.models import Subscriber
from tallyho.notify import FakeNtfySink
from tallyho.store import Store
from windfall.replay import accuracy_from_store, aggregate
from windfall.telemetry import parse_frame
from tests.conftest import fast_ensemble_cfg, simulate_flight


def test_accuracy_from_store_scores_live_landing():
    store = Store(":memory:")
    try:
        store.add_subscriber(Subscriber(
            name="a", lat=45.0, lon=7.0, radius_km=40,
            ntfy_server="https://ntfy.sh", ntfy_topic="a"))
        app = App(fast_ensemble_cfg(), store=store, sink=FakeNtfySink())
        f = simulate_flight(serial="LIVE1", burst_alt=28000)
        app.on_frames([parse_frame(m) for m in f.frames])

        results = accuracy_from_store(store)
        assert len(results) == 1
        r = results[0]
        assert r.serial == "LIVE1"
        # store-scored results carry the flight key for API consumers (offline
        # replay_flight results leave launch_day None)
        assert r.launch_day == store.recent_landings()[0]["launch_day"]
        assert r.n_predictions > 0
        # the live predictor's final error against the recorded landing is small
        assert r.final_error_km is not None and r.final_error_km < 2.0

        metrics = aggregate(results)
        assert metrics.n_flights == 1
        assert 0.0 <= metrics.calibration_rate <= 1.0
        assert "mean final error" in metrics.report()
    finally:
        store.close()


def test_accuracy_from_store_empty():
    store = Store(":memory:")
    try:
        assert accuracy_from_store(store) == []
    finally:
        store.close()


def test_accuracy_skips_truth_older_than_predictions():
    # A truth row hours older than the flight's last prediction is
    # self-contradictory - predictions are only made mid-air, so the flight
    # demonstrably kept flying after the recorded "landing" (a bogus
    # pre-launch row minted at the pad scored 45 good descent predictions as
    # 75-96 km misses). Such flights must not poison the metrics.
    from datetime import date, datetime, timedelta, timezone

    from windfall.models import Prediction, PredictionSource

    store = Store(":memory:")
    try:
        day = date(2026, 7, 12)
        landed_at = datetime(2026, 7, 12, 23, 27, tzinfo=timezone.utc)
        store.record_landing("SUS1", day, land_lat=39.12, land_lon=-108.52,
                             land_alt=1460.0, landed_at=landed_at,
                             detected_by="telemetry")
        for i in range(3):
            store.save_prediction(Prediction(
                serial="SUS1", launch_day=day,
                predicted_at=landed_at + timedelta(hours=2, seconds=10 * i),
                land_lat=39.4, land_lon=-109.3,
                land_eta=landed_at + timedelta(hours=2, minutes=40),
                source=PredictionSource.MEASURED,
                uncertainty_radius_km=9.0, alt_at_pred=25000.0))
        assert accuracy_from_store(store) == []
    finally:
        store.close()
