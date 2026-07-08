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
