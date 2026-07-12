"""Tests for SQLite persistence."""

import dataclasses
from datetime import date, datetime, timezone

import pytest

import tallyho.store
from tallyho.models import AlertType, Prediction, PredictionSource, Subscriber
from tallyho.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_subscriber_crud(store):
    sid = store.add_subscriber(Subscriber(
        name="alice", lat=45.0, lon=7.0, radius_km=30,
        ntfy_server="https://ntfy.sh", ntfy_topic="alice-sondes",
        ntfy_token_ref="NTFY_ALICE",
    ))
    subs = store.list_subscribers()
    assert len(subs) == 1
    assert subs[0].name == "alice"
    assert subs[0].id == sid
    assert subs[0].ntfy_token_ref == "NTFY_ALICE"  # reference, not a token
    assert subs[0].units == "metric"               # default display units
    store.set_subscriber_active(sid, False)
    assert store.list_subscribers(active_only=True) == []
    assert len(store.list_subscribers(active_only=False)) == 1


def test_subscriber_update_get_delete(store):
    sid = store.add_subscriber(Subscriber(
        name="alice", lat=45.0, lon=7.0, radius_km=30,
        ntfy_server="https://ntfy.sh", ntfy_topic="alice-sondes",
    ))
    created = store.get_subscriber(sid)
    assert created is not None and created.created_at is not None

    ok = store.update_subscriber(Subscriber(
        id=sid, name="alice2", lat=46.0, lon=8.0, radius_km=40,
        ntfy_server="https://ntfy.example", ntfy_topic="alice-2",
        ntfy_token_ref="NTFY_ALICE", units="imperial", active=False,
    ))
    assert ok is True
    got = store.get_subscriber(sid)
    assert got.name == "alice2" and got.radius_km == 40 and got.active is False
    assert got.ntfy_token_ref == "NTFY_ALICE"
    assert got.units == "imperial"
    assert got.created_at == created.created_at   # update leaves created_at intact

    # updating / deleting a missing id reports False
    assert store.update_subscriber(dataclasses.replace(got, id=9999)) is False
    assert store.delete_subscriber(9999) is False
    assert store.set_subscriber_active(9999, True) is False

    assert store.delete_subscriber(sid) is True
    assert store.get_subscriber(sid) is None
    assert store.list_subscribers(active_only=False) == []


def test_recent_flights_and_alerts(store):
    store.add_subscriber(Subscriber(
        name="bob", lat=45.0, lon=7.0, radius_km=20,
        ntfy_server="https://ntfy.sh", ntfy_topic="bob"))
    store.upsert_flight({
        "serial": "S1", "launch_day": "2026-06-07", "type": "RS41", "state": "DESCENT",
        "first_seen": "2026-06-07T00:00:00+00:00", "last_seen": "2026-06-07T00:10:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": 30000.0, "max_alt": 30000.0,
        "last_lat": 45.1, "last_lon": 7.1, "last_alt": 8000.0,
    })
    store.upsert_flight({
        "serial": "S0", "launch_day": "2026-06-06", "type": "RS41", "state": "LANDED",
        "first_seen": "2026-06-06T00:00:00+00:00", "last_seen": "2026-06-06T00:30:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": 30000.0, "max_alt": 30000.0,
        "last_lat": 45.2, "last_lon": 7.2, "last_alt": 200.0,
    })
    assert len(store.recent_flights(include_landed=True)) == 2
    active = store.recent_flights(include_landed=False)
    assert [f["serial"] for f in active] == ["S1"]

    store.record_alert(subscriber_id=1, serial="S1", launch_day=date(2026, 6, 7),
                       alert_type=AlertType.INBOUND, distance_km=8.0,
                       land_lat=45.1, land_lon=7.1,
                       sent_at=datetime(2026, 6, 7, 0, 11, tzinfo=timezone.utc))
    alerts = store.recent_alerts()
    assert len(alerts) == 1
    assert alerts[0]["subscriber_name"] == "bob"
    assert alerts[0]["flight_type"] == "RS41"


def test_latest_predictions_for_active(store):
    store.upsert_flight({
        "serial": "S1", "launch_day": "2026-06-07", "type": "RS41", "state": "DESCENT",
        "first_seen": None, "last_seen": "2026-06-07T00:10:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": 30000.0, "max_alt": 30000.0,
        "last_lat": 45.1, "last_lon": 7.1, "last_alt": 8000.0,
    })
    store.upsert_flight({
        "serial": "S0", "launch_day": "2026-06-06", "type": "RS41", "state": "LANDED",
        "first_seen": None, "last_seen": "2026-06-06T00:30:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": 30000.0, "max_alt": 30000.0,
        "last_lat": 45.2, "last_lon": 7.2, "last_alt": 200.0,
    })
    for serial, day, when, lat in [
        ("S1", date(2026, 6, 7), 1, 45.5), ("S1", date(2026, 6, 7), 5, 45.6),
        ("S0", date(2026, 6, 6), 1, 44.0),  # landed - must be excluded
    ]:
        store.save_prediction(Prediction(
            serial=serial, launch_day=day,
            predicted_at=datetime(2026, 6, 7, 1, when, tzinfo=timezone.utc),
            land_lat=lat, land_lon=7.5,
            land_eta=datetime(2026, 6, 7, 1, 30, tzinfo=timezone.utc),
            source=PredictionSource.MEASURED, uncertainty_radius_km=2.0))
    rows = store.latest_predictions_for_active()
    assert len(rows) == 1
    assert rows[0]["serial"] == "S1" and rows[0]["land_lat"] == 45.6  # newest only


def test_flight_upsert_and_get(store):
    row = {
        "serial": "S1", "launch_day": "2026-06-07", "type": "RS41",
        "state": "ASCENT", "first_seen": "2026-06-07T00:00:00+00:00",
        "last_seen": "2026-06-07T00:10:00+00:00", "launch_lat": 45.0,
        "launch_lon": 7.0, "burst_alt": None, "max_alt": 10000.0,
        "last_lat": 45.1, "last_lon": 7.1, "last_alt": 10000.0,
    }
    store.upsert_flight(row)
    row["state"] = "DESCENT"
    row["burst_alt"] = 30000.0
    store.upsert_flight(row)
    got = store.get_flight("S1", date(2026, 6, 7))
    assert got["state"] == "DESCENT"
    assert got["burst_alt"] == 30000.0
    assert len(store.active_flights()) == 1


def test_delete_flight_erases_all_rows_for_one_key(store):
    day, other_day = date(2026, 6, 7), date(2026, 6, 8)
    for d in (day, other_day):
        store.upsert_flight({
            "serial": "S1", "launch_day": d.isoformat(), "type": "RS41",
            "state": "ASCENT", "first_seen": None, "last_seen": None,
            "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": None,
            "max_alt": 5000.0, "last_lat": 45.1, "last_lon": 7.1, "last_alt": 5000.0,
        })
        store.save_profile("S1", d, {"bins": []})
        store.save_descent_samples("S1", d, [[0.0, 5000.0, 10.0, 0.7]])
        store.append_track_point("S1", d, 0.0, 45.0, 7.0, 5000.0)
        store.save_prediction(Prediction(
            serial="S1", launch_day=d,
            predicted_at=datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc),
            land_lat=45.5, land_lon=7.5,
            land_eta=datetime(2026, 6, 7, 1, 30, tzinfo=timezone.utc),
            source=PredictionSource.MEASURED, uncertainty_radius_km=2.0,
        ))

    store.delete_flight("S1", day)
    assert store.get_flight("S1", day) is None
    assert store.load_profile("S1", day) is None
    assert store.load_descent_samples("S1", day) is None
    assert store.track_for("S1", day) == []
    assert store.latest_prediction("S1", day) is None
    # the other launch_day of the same serial is untouched
    assert store.get_flight("S1", other_day) is not None
    assert store.load_profile("S1", other_day) is not None
    assert store.latest_prediction("S1", other_day) is not None


def test_batch_coalesces_commits_and_change_pings(store):
    pings = []
    store.on_change = pings.append
    day = date(2026, 6, 7)
    row = {
        "serial": "B1", "launch_day": day.isoformat(), "type": "RS41",
        "state": "ASCENT", "first_seen": None, "last_seen": None,
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": None,
        "max_alt": 5000.0, "last_lat": 45.1, "last_lon": 7.1, "last_alt": 5000.0,
    }
    with store.batch():
        for i in range(50):
            store.upsert_flight(row)
            store.append_track_point("B1", day, float(i), 45.0, 7.0, 5000.0)
        store.record_landing("B1", day, 45.2, 7.2, 200.0,
                             datetime(2026, 6, 7, 2, 0, tzinfo=timezone.utc), "telemetry")
        assert pings == []                     # everything deferred...
    assert sorted(pings) == ["accuracy", "flights"]   # ...then once per name
    assert store.get_flight("B1", day) is not None
    assert len(store.track_for("B1", day)) == 50

    # a raising body still commits what it wrote (rows match in-memory state)
    pings.clear()
    with pytest.raises(RuntimeError):
        with store.batch():
            store.upsert_flight(dict(row, state="DESCENT"))
            raise RuntimeError("mid-batch failure")
    assert store.get_flight("B1", day)["state"] == "DESCENT"
    assert pings == ["flights"]

    # nested batches defer to the outermost exit
    pings.clear()
    with store.batch():
        with store.batch():
            store.upsert_flight(row)
        assert pings == []
    assert pings == ["flights"]


def test_profile_roundtrip(store):
    prof = {"bin_size_m": 150.0, "bins": [{"alt": 1000, "u": 5, "v": 1, "rho": 0.9, "n": 3}]}
    store.save_profile("S1", date(2026, 6, 7), prof)
    loaded = store.load_profile("S1", date(2026, 6, 7))
    assert loaded == prof
    assert store.load_profile("missing", date(2026, 6, 7)) is None


def test_prediction_save_latest(store):
    p = Prediction(
        serial="S1", launch_day=date(2026, 6, 7),
        predicted_at=datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc),
        land_lat=45.5, land_lon=7.5,
        land_eta=datetime(2026, 6, 7, 1, 30, tzinfo=timezone.utc),
        source=PredictionSource.MEASURED, uncertainty_radius_km=2.0,
    )
    store.save_prediction(p)
    p2 = dataclasses.replace(
        p, predicted_at=datetime(2026, 6, 7, 1, 5, tzinfo=timezone.utc), land_lat=45.6
    )
    store.save_prediction(p2)
    latest = store.latest_prediction("S1", date(2026, 6, 7))
    assert latest["land_lat"] == 45.6


def test_fresh_db_stamped_with_current_schema_version(tmp_path):
    s = Store(tmp_path / "new.db")
    try:
        v = s._conn.execute("PRAGMA user_version").fetchone()[0]
        assert v == tallyho.store._SCHEMA_VERSION
    finally:
        s.close()


def test_pre_versioning_db_adopted_as_baseline(tmp_path):
    # A DB shaped like the current schema but never stamped (user_version 0)
    # must be adopted in place, keeping its data.
    db = tmp_path / "legacy.db"
    s = Store(db)
    s.add_subscriber(Subscriber(
        name="old", lat=45.0, lon=7.0, radius_km=30,
        ntfy_server="https://ntfy.sh", ntfy_topic="old-sondes"))
    s._conn.execute("PRAGMA user_version = 0")
    s._conn.commit()
    s.close()

    s = Store(db)
    try:
        assert s._conn.execute("PRAGMA user_version").fetchone()[0] \
            == tallyho.store._SCHEMA_VERSION
        assert s.list_subscribers()[0].name == "old"
    finally:
        s.close()


def test_pending_migrations_replay_in_order(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    Store(db).close()  # created at the current version

    # Pretend the app has since moved two versions ahead: one SQL migration,
    # one Python callable.
    base = tallyho.store._SCHEMA_VERSION

    def add_mig_b(conn):
        conn.execute("CREATE TABLE mig_b (x)")

    monkeypatch.setattr(tallyho.store, "_SCHEMA_VERSION", base + 2)
    monkeypatch.setattr(tallyho.store, "_MIGRATIONS", {
        base + 1: "CREATE TABLE mig_a (x)",
        base + 2: add_mig_b,
    })
    s = Store(db)
    try:
        tables = {r["name"] for r in s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"mig_a", "mig_b"} <= tables
        assert s._conn.execute("PRAGMA user_version").fetchone()[0] == base + 2
    finally:
        s.close()


def test_db_from_newer_build_refused(tmp_path):
    import sqlite3
    db = tmp_path / "t.db"
    Store(db).close()
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 9999")
    conn.close()
    with pytest.raises(RuntimeError, match="schema version 9999"):
        Store(db)


def test_v2_migration_adds_path_burst_columns(tmp_path):
    """A database whose prediction_paths still has the v1 shape must gain the
    burst-point columns on open - whether it is stamped v1 or predates
    versioning entirely (user_version 0: adopted at the v1 baseline, then the
    migration replays)."""
    for stamp in (1, 0):
        db = tmp_path / f"stamp{stamp}.db"
        s = Store(db)
        s._conn.executescript(
            "DROP TABLE prediction_paths;"
            "CREATE TABLE prediction_paths ("
            "  serial TEXT NOT NULL, launch_day TEXT NOT NULL,"
            "  predicted_at TEXT NOT NULL, source TEXT NOT NULL,"
            "  land_eta TEXT, path_json TEXT NOT NULL,"
            "  PRIMARY KEY (serial, launch_day))")
        s._conn.execute(f"PRAGMA user_version = {stamp}")
        s._conn.commit()
        s.close()

        s = Store(db)
        try:
            cols = {r[1] for r in s._conn.execute(
                "PRAGMA table_info(prediction_paths)")}
            assert {"burst_lat", "burst_lon", "burst_alt"} <= cols
            assert s._conn.execute("PRAGMA user_version").fetchone()[0] \
                == tallyho.store._SCHEMA_VERSION
        finally:
            s.close()


def test_prediction_path_saved_and_excludes_landed(store):
    store.upsert_flight({
        "serial": "P1", "launch_day": "2026-06-07", "type": "RS41", "state": "DESCENT",
        "first_seen": None, "last_seen": "2026-06-07T00:10:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": None, "max_alt": 30000.0,
        "last_lat": 45.1, "last_lon": 7.2, "last_alt": 8000.0,
    })
    store.upsert_flight({
        "serial": "P0", "launch_day": "2026-06-06", "type": "RS41", "state": "LANDED",
        "first_seen": None, "last_seen": "2026-06-06T00:30:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": None, "max_alt": 30000.0,
        "last_lat": 45.2, "last_lon": 7.2, "last_alt": 200.0,
    })
    store.save_prediction(Prediction(
        serial="P1", launch_day=date(2026, 6, 7),
        predicted_at=datetime(2026, 6, 7, 0, 11, tzinfo=timezone.utc),
        land_lat=45.5, land_lon=7.6,
        land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
        source=PredictionSource.MEASURED, uncertainty_radius_km=2.0,
        burst_lat=45.2, burst_lon=7.3, burst_alt=31000.0,
        path=[(45.1, 7.2, 8000.0), (45.3, 7.4, 4000.0), (45.5, 7.6, 200.0)]))
    # landed flight has a path too, but it must be excluded from the active map
    store.save_prediction(Prediction(
        serial="P0", launch_day=date(2026, 6, 6),
        predicted_at=datetime(2026, 6, 6, 0, 20, tzinfo=timezone.utc),
        land_lat=45.2, land_lon=7.2,
        land_eta=datetime(2026, 6, 6, 0, 30, tzinfo=timezone.utc),
        source=PredictionSource.GFS, uncertainty_radius_km=3.0,
        path=[(45.0, 7.0, 9000.0), (45.2, 7.2, 200.0)]))
    rows = store.latest_paths_for_active()
    assert len(rows) == 1
    assert rows[0]["serial"] == "P1"
    assert rows[0]["path"] == [[45.1, 7.2, 8000.0], [45.3, 7.4, 4000.0], [45.5, 7.6, 200.0]]
    # the predicted burst point rides along with the path
    assert (rows[0]["burst_lat"], rows[0]["burst_lon"], rows[0]["burst_alt"]) \
        == (45.2, 7.3, 31000.0)


def test_prediction_path_upserts_latest(store):
    store.upsert_flight({
        "serial": "P1", "launch_day": "2026-06-07", "type": "RS41", "state": "DESCENT",
        "first_seen": None, "last_seen": "2026-06-07T00:10:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": None, "max_alt": 30000.0,
        "last_lat": 45.1, "last_lon": 7.2, "last_alt": 8000.0,
    })
    base = dict(serial="P1", launch_day=date(2026, 6, 7),
                land_lat=45.5, land_lon=7.6,
                land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
                source=PredictionSource.MEASURED, uncertainty_radius_km=2.0)
    store.save_prediction(Prediction(
        predicted_at=datetime(2026, 6, 7, 0, 11, tzinfo=timezone.utc),
        burst_lat=45.2, burst_lon=7.3, burst_alt=31000.0,
        path=[(1.0, 1.0, 5000.0), (2.0, 2.0, 0.0)], **base))
    store.save_prediction(Prediction(
        predicted_at=datetime(2026, 6, 7, 0, 12, tzinfo=timezone.utc),
        path=[(9.0, 9.0, 5000.0), (8.0, 8.0, 0.0)], **base))
    rows = store.latest_paths_for_active()
    assert len(rows) == 1                          # one row per flight (upsert)
    assert rows[0]["path"][0] == [9.0, 9.0, 5000.0]  # newest path won
    # the descent path replaced the pre-burst one: its burst point is gone too
    assert rows[0]["burst_lat"] is None and rows[0]["burst_alt"] is None


def test_landing_record_and_accuracy_inputs(store):
    day = date(2026, 6, 7)
    store.record_landing("L1", day, land_lat=45.50, land_lon=7.60, land_alt=210.0,
                         landed_at=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
                         detected_by="telemetry")
    got = store.get_landing("L1", day)
    assert got["land_lat"] == 45.50 and got["detected_by"] == "telemetry"
    # upsert refines an earlier fix
    store.record_landing("L1", day, land_lat=45.51, land_lon=7.61, land_alt=205.0,
                         landed_at=datetime(2026, 6, 7, 0, 41, tzinfo=timezone.utc),
                         detected_by="timeout")
    assert store.get_landing("L1", day)["land_lat"] == 45.51
    assert len(store.recent_landings()) == 1


def test_predictions_for_orders_and_alt_at_pred(store):
    day = date(2026, 6, 7)
    for minute, alt in [(12, 6000.0), (10, 9000.0), (11, 7500.0)]:
        store.save_prediction(Prediction(
            serial="A1", launch_day=day,
            predicted_at=datetime(2026, 6, 7, 0, minute, tzinfo=timezone.utc),
            land_lat=45.5, land_lon=7.5,
            land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
            source=PredictionSource.MEASURED, uncertainty_radius_km=2.0,
            alt_at_pred=alt))
    rows = store.predictions_for("A1", day)
    assert [r["alt_at_pred"] for r in rows] == [9000.0, 7500.0, 6000.0]  # oldest first


def test_alert_dedup(store):
    args = dict(subscriber_id=1, serial="S1", launch_day=date(2026, 6, 7),
                alert_type=AlertType.INBOUND, distance_km=5.0,
                land_lat=45.5, land_lon=7.5,
                sent_at=datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc))
    assert store.record_alert(**args) is True
    assert store.record_alert(**args) is False   # duplicate
    # different alert_type is allowed
    args2 = {**args, "alert_type": AlertType.LANDED}
    assert store.record_alert(**args2) is True
    last = store.last_alert_at(1, "S1", date(2026, 6, 7), AlertType.INBOUND)
    assert last == datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)


def test_clear_alerts_keeps_airborne_dedup_rows(store):
    day = date(2026, 6, 7)
    store.upsert_flight({"serial": "AIR1", "launch_day": day.isoformat(),
                         "state": "DESCENT", "type": "RS41"})
    store.upsert_flight({"serial": "DOWN1", "launch_day": day.isoformat(),
                         "state": "LANDED", "type": "RS41"})
    sent = datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)
    for serial in ("AIR1", "DOWN1"):
        store.record_alert(subscriber_id=1, serial=serial, launch_day=day,
                           alert_type=AlertType.INBOUND, distance_km=5.0,
                           land_lat=45.5, land_lon=7.5, sent_at=sent)
    # an orphan alert whose flight row is gone is also cleared
    store.record_alert(subscriber_id=1, serial="GONE1", launch_day=day,
                       alert_type=AlertType.LANDED, distance_km=5.0,
                       land_lat=45.5, land_lon=7.5, sent_at=sent)

    assert store.clear_alerts() == 2
    remaining = store.recent_alerts()
    assert [a["serial"] for a in remaining] == ["AIR1"]
    # the airborne flight's de-dup row survived, so its INBOUND won't re-send
    assert store.record_alert(subscriber_id=1, serial="AIR1", launch_day=day,
                              alert_type=AlertType.INBOUND, distance_km=5.0,
                              land_lat=45.5, land_lon=7.5, sent_at=sent) is False


def test_climatology_like_escapes_wildcards(store):
    """Sonde types come from external telemetry, so LIKE metacharacters in them
    must match literally - a spoofed type of '%' or '_' must not broaden the
    climatology scans to other families."""
    day = "2026-06-07"
    for serial, stype, burst, b in [("W1", "RS41", 30000.0, 0.02),
                                    ("W2", "DFM09", 26000.0, 0.05)]:
        store.upsert_flight({
            "serial": serial, "launch_day": day, "type": stype, "state": "LANDED",
            "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": burst,
            "descent_b": b, "max_alt": burst,
            "last_lat": 45.2, "last_lon": 7.2, "last_alt": 200.0,
        })
    # wildcards match nothing (no type literally starts with '%' or '_FM09')
    assert store.site_burst_alts(45.0, 7.0, sonde_type="%") == []
    assert store.type_descent_bs("%") == []
    assert store.type_descent_bs("_FM09") == []
    # legitimate prefixes still match their family
    assert store.site_burst_alts(45.0, 7.0, sonde_type="RS41-SGP") == [30000.0]
    assert store.type_descent_bs("DFM") == [0.05]


def test_flown_track_append_and_query(store):
    day = date(2026, 6, 7)
    pts = [(100.0, 45.0, 7.0, 200.0), (130.0, 45.1, 7.1, 5000.0),
           (160.0, 45.2, 7.2, 9000.0)]
    for t, lat, lon, alt in pts:
        store.append_track_point("T1", day, t, lat, lon, alt)
    # idempotent on (serial, launch_day, t) - a replayed frame doesn't double up
    store.append_track_point("T1", day, 100.0, 45.0, 7.0, 200.0)

    track = store.track_for("T1", day)
    assert [r["t"] for r in track] == [100.0, 130.0, 160.0]  # oldest first
    assert track[0]["lat"] == 45.0 and track[0]["alt"] == 200.0


def test_latest_tracks_for_active_excludes_landed(store):
    day = date(2026, 6, 7)
    for serial, state in [("UP", "DESCENT"), ("DOWN", "LANDED")]:
        store.upsert_flight({
            "serial": serial, "launch_day": day.isoformat(), "type": "RS41",
            "state": state, "first_seen": None, "last_seen": None,
            "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": None, "max_alt": None,
            "last_lat": 45.2, "last_lon": 7.2, "last_alt": 9000.0,
        })
        store.append_track_point(serial, day, 100.0, 45.0, 7.0, 200.0)
        store.append_track_point(serial, day, 160.0, 45.2, 7.2, 9000.0)

    tracks = store.latest_tracks_for_active()
    assert {t["serial"] for t in tracks} == {"UP"}  # the LANDED flight is excluded
    assert tracks[0]["track"] == [[45.0, 7.0, 200.0], [45.2, 7.2, 9000.0]]


# ---- ntfy tokens ------------------------------------------------------------
def test_ntfy_token_roundtrip(store):
    assert store.get_ntfy_token("home") is None
    store.set_ntfy_token("home", "tk_secret_abcd")
    assert store.get_ntfy_token("home") == "tk_secret_abcd"

    toks = store.list_ntfy_tokens()
    assert len(toks) == 1
    assert toks[0]["name"] == "home"
    assert toks[0]["hint"] == "…abcd"        # last 4 only
    assert toks[0]["refs"] == 0
    assert "token" not in toks[0]            # listing never carries the value

    store.set_ntfy_token("home", "tk_rotated_wxyz")   # same name = replace
    assert store.get_ntfy_token("home") == "tk_rotated_wxyz"
    assert len(store.list_ntfy_tokens()) == 1

    assert store.delete_ntfy_token("home") is True
    assert store.delete_ntfy_token("home") is False
    assert store.get_ntfy_token("home") is None


def test_ntfy_token_ref_counting(store):
    store.set_ntfy_token("home", "tk_x")
    assert store.ntfy_token_refs("home") == 0
    store.add_subscriber(Subscriber(
        name="alice", lat=45.0, lon=7.0, radius_km=30,
        ntfy_server="https://ntfy.sh", ntfy_topic="alice-sondes",
        ntfy_token_ref="home"))
    assert store.ntfy_token_refs("home") == 1
    assert store.list_ntfy_tokens()[0]["refs"] == 1
