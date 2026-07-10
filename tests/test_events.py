"""EventBus + Store change-doorbell + data_version watcher.

Dependency-free (asyncio + threading + sqlite3 only), so this runs in the base
venv without the `api`/`dev` extras. The SSE *endpoint* wiring is tested in
test_web.py behind the httpx2 skip.
"""

import asyncio
import threading
from datetime import date, datetime, timezone

import pytest

from tallyho.events import EventBus
from tallyho.models import AlertType, Prediction, PredictionSource, Subscriber
from tallyho.store import Store
from tallyho.web import _data_version_watcher

DAY = date(2026, 6, 7)
DT = datetime(2026, 6, 7, 0, 11, tzinfo=timezone.utc)


# ---- EventBus ------------------------------------------------------------
def test_publish_from_worker_thread_delivers_to_client():
    """The only cross-thread hop: publish() on a worker thread reaches the loop
    and lands as a dirty-set + wake on the registered client."""
    async def main():
        bus = EventBus(debounce=0.01)
        bus.attach_loop(asyncio.get_running_loop())
        client = bus.register()
        # publish from a genuinely different thread
        await asyncio.get_running_loop().run_in_executor(None, bus.publish, "flights")
        await asyncio.wait_for(client.event.wait(), timeout=2)
        assert client.dirty == {"flights"}

    asyncio.run(main(), debug=True)  # debug mode flags any wrong-thread asyncio touch


def test_burst_coalesces_to_a_single_flush():
    """A burst of publishes within the debounce window is one flush carrying the
    coalesced name set; this is what tames the ~1 Hz append_track_point stream."""
    async def main():
        bus = EventBus(debounce=0.05)
        bus.attach_loop(asyncio.get_running_loop())
        client = bus.register()
        flushes = []
        real_flush = bus._flush
        bus._flush = lambda: (flushes.append(1), real_flush())

        def burst():
            for i in range(100):
                bus.publish("flights" if i % 2 == 0 else "alerts")

        await asyncio.get_running_loop().run_in_executor(None, burst)
        await asyncio.wait_for(client.event.wait(), timeout=2)
        assert client.dirty == {"flights", "alerts"}  # both names, coalesced
        assert sum(flushes) == 1                       # exactly one flush

    asyncio.run(main(), debug=True)


def test_close_wakes_and_ends_a_waiting_consumer():
    async def main():
        bus = EventBus(debounce=0.01)
        bus.attach_loop(asyncio.get_running_loop())
        client = bus.register()

        async def consume():
            await client.event.wait()
            return bus.closed

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.01)
        bus.close()
        assert await asyncio.wait_for(task, timeout=2) is True
        assert bus.closed is True

    asyncio.run(main(), debug=True)


def test_publish_without_a_loop_is_a_silent_noop():
    """Web disabled / not started yet: publish() must not raise, just accumulate
    (a client resyncs on connect anyway)."""
    bus = EventBus()
    bus.publish("flights")          # no loop attached
    bus.publish("flights")          # de-duped in the pending set
    assert bus._pending == {"flights"}
    assert bus._loop is None


# ---- Store change events (the drift test) --------------------------------
def test_store_change_events_match_the_mapping_table():
    """Pins the Store-method → event mapping. If a future write method is added
    without classifying it here, this test is the guard that catches it."""
    store = Store(":memory:")
    events: list[str] = []
    store.on_change = events.append

    def fired(fn):
        events.clear()
        fn()
        return list(events)

    def pred(path=None):
        return Prediction(
            serial="S1", launch_day=DAY, predicted_at=DT,
            land_lat=45.5, land_lon=7.6, land_eta=DT,
            source=PredictionSource.MEASURED, uncertainty_radius_km=2.5, path=path)

    sub = Subscriber(name="a", lat=45.0, lon=7.0, radius_km=10.0,
                     ntfy_server="https://ntfy.sh", ntfy_topic="t")

    # flights
    assert fired(lambda: store.upsert_flight(
        {"serial": "S1", "launch_day": "2026-06-07", "state": "ASCENT"})) == ["flights"]
    assert fired(lambda: store.append_track_point("S1", DAY, 0.0, 45.0, 7.0, 100.0)) == ["flights"]
    assert fired(lambda: store.save_prediction(pred())) == ["flights"]  # no path → one event
    # a path-bearing save fires flights twice (own funnel + save_prediction_path);
    # harmless; the debounce coalesces it in flight.
    assert fired(lambda: store.save_prediction(
        pred(path=[(45.1, 7.2, 8000.0), (45.5, 7.6, 200.0)]))) == ["flights", "flights"]
    assert fired(lambda: store.save_prediction_path(
        pred(path=[(45.1, 7.2, 8000.0), (45.5, 7.6, 200.0)]))) == ["flights"]

    # accuracy
    assert fired(lambda: store.record_landing(
        "S1", DAY, 45.0, 7.0, 100.0, DT, "telemetry")) == ["accuracy"]
    assert fired(store.clear_accuracy) == ["accuracy"]

    # subscribers
    sid = store.add_subscriber(sub)          # (also fired "subscribers"; not asserted here)
    assert fired(lambda: store.set_subscriber_active(sid, False)) == ["subscribers"]
    assert fired(lambda: store.update_subscriber(Subscriber(
        id=sid, name="a2", lat=45.0, lon=7.0, radius_km=11.0,
        ntfy_server="https://ntfy.sh", ntfy_topic="t"))) == ["subscribers"]
    assert fired(lambda: store.delete_subscriber(sid)) == ["subscribers"]
    # add_subscriber itself fires exactly one "subscribers"
    assert fired(lambda: store.add_subscriber(sub)) == ["subscribers"]

    # alerts
    assert fired(lambda: store.record_alert(
        1, "S1", DAY, AlertType.INBOUND, 1.0, 45.0, 7.0, DT)) == ["alerts"]
    assert fired(lambda: store.upsert_alert(
        1, "S1", DAY, AlertType.UPDATE, 1.0, 45.0, 7.0, DT)) == ["alerts"]
    assert fired(lambda: store.update_alert_time(
        1, "S1", DAY, AlertType.INBOUND, DT)) == ["alerts"]
    assert fired(store.clear_alerts) == ["alerts"]

    # tokens
    assert fired(lambda: store.set_ntfy_token("n", "tok")) == ["tokens"]
    assert fired(lambda: store.delete_ntfy_token("n")) == ["tokens"]

    # deliberately unmapped (no UI surface): must fire nothing
    assert fired(lambda: store.save_profile("S1", DAY, {})) == []
    assert fired(lambda: store.save_descent_samples("S1", DAY, [])) == []
    assert fired(lambda: store.add_user("u", "h")) == []
    assert fired(lambda: store.set_kv("k", "v")) == []

    store.close()


def test_changed_is_a_noop_when_no_callback_is_wired():
    """With the web UI disabled, on_change stays None and writes publish nothing
    (no exception)."""
    store = Store(":memory:")
    assert store.on_change is None
    store.upsert_flight({"serial": "S1", "launch_day": "2026-06-07", "state": "ASCENT"})
    store.close()


# ---- data_version watcher (cross-process doorbell) -----------------------
def test_data_version_bumps_only_on_cross_connection_commits(tmp_path):
    """PRAGMA data_version is the watcher's whole basis: a commit on *this*
    connection doesn't bump it (so `tallyho run` never double-fires), a commit
    from another connection does (so standalone `tallyho web` still sees CLI
    writes). Needs a real file - :memory: DBs aren't shared across connections."""
    path = tmp_path / "t.db"
    a, b = Store(path), Store(path)
    try:
        v = a.data_version()
        a.set_kv("x", "1")                 # same connection
        assert a.data_version() == v       # → not bumped
        b.set_kv("y", "2")                 # a different connection
        assert a.data_version() != v       # → bumped
    finally:
        a.close()
        b.close()


def test_data_version_watcher_publishes_changed_on_out_of_connection_write(tmp_path):
    path = tmp_path / "t.db"
    a, b = Store(path), Store(path)

    async def main():
        bus = EventBus(debounce=0.01)
        bus.attach_loop(asyncio.get_running_loop())
        client = bus.register()
        task = asyncio.create_task(_data_version_watcher(a, bus, interval=0.02))
        await asyncio.sleep(0.05)                       # let it read the baseline
        b.set_kv("y", "2")                              # cross-connection write
        await asyncio.wait_for(client.event.wait(), timeout=2)
        task.cancel()
        assert "changed" in client.dirty

    try:
        asyncio.run(main())
    finally:
        a.close()
        b.close()
