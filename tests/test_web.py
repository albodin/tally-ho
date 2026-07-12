"""Tests for the optional web UI (onboarding via the browser).

The whole module is skipped unless the `api` extra (FastAPI) is installed, so the
core suite stays dependency-free and offline-capable.
"""

import threading
import time
from datetime import date, datetime, timezone

import pytest
from windfall.geo import haversine_km

fastapi = pytest.importorskip("fastapi")  # skip module without the `api` extra
pytest.importorskip("httpx2")  # FastAPI's TestClient needs httpx2 (the `dev` extra)
from fastapi.testclient import TestClient  # noqa: E402

from tallyho.auth import hash_password  # noqa: E402
from tallyho.config import Config  # noqa: E402
from tallyho.events import EventBus  # noqa: E402
from tallyho.models import AlertType, Prediction, PredictionSource, Subscriber  # noqa: E402
from tallyho.notify import NtfySink  # noqa: E402
from tallyho.store import Store  # noqa: E402
from tallyho.web import create_app  # noqa: E402

# One scrypt hash for the whole module - hashing is deliberately slow.
PASSWORD = "correct-horse-battery"
PASSWORD_HASH = hash_password(PASSWORD)


def login(client, store):
    store.add_user("admin", PASSWORD_HASH)
    r = client.post("/api/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 200


@pytest.fixture
def client():
    store = Store(":memory:")
    app = create_app(Config(), store)
    c = TestClient(app)
    c.store = store  # expose for direct DB assertions
    login(c, store)
    yield c
    store.close()


class RecordingSink(NtfySink):
    """Captures the NtfyMessage instead of hitting the network; ``ok`` controls
    the simulated server verdict."""

    def __init__(self, ok=True):
        self.ok = ok
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)
        return self.ok


@pytest.fixture
def sink_client():
    """Factory: build a client whose /api/test-ntfy uses an injected sink."""
    stores = []

    def make(sink):
        store = Store(":memory:")
        stores.append(store)
        c = TestClient(create_app(Config(), store, ntfy_sink=sink))
        c.store = store
        login(c, store)
        return c

    yield make
    for s in stores:
        s.close()


def _payload(**over):
    body = {
        "name": "home", "lat": 45.07, "lon": 7.69, "radius_km": 30.0,
        "ntfy_server": "https://ntfy.sh", "ntfy_topic": "my-sondes-7a3f",
        "ntfy_token_ref": "NTFY_HOME", "active": True,
    }
    body.update(over)
    return body


# ---- index + health ------------------------------------------------------
def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "leaflet" in r.text.lower()


def test_html_pages_are_no_store(client):
    """The HTML shells must not be browser-cached: "/" swaps from the setup
    wizard to the dashboard after first run, and a heuristically cached page
    survives that switch until a forced reload."""
    for page in ("/", "/login", "/settings"):
        assert client.get(page).headers["cache-control"] == "no-store", page


def test_pages_reference_resolvable_assets(client):
    """Every /static href/src the pages emit must actually be served -
    guards against a renamed/moved CSS or JS file breaking a page."""
    import re

    for page in ("/", "/login", "/settings"):
        html = client.get(page).text
        assets = re.findall(r'(?:href|src)="(/static/[^"]+)"', html)
        assert assets, f"{page} references no static assets?"
        for path in assets:
            assert client.get(path).status_code == 200, f"{page} -> {path}"


def test_static_assets_are_public():
    """CSS/JS (and vendored Leaflet) must load without a session - the login
    page needs its stylesheet before any cookie exists."""
    store = Store(":memory:")
    try:
        c = TestClient(create_app(Config(), store))  # deliberately no login()
        for path, ctype in [("/static/theme.css", "text/css"),
                            ("/static/js/dashboard.js", "javascript"),
                            ("/static/vendor/leaflet/leaflet.js", "javascript")]:
            r = c.get(path, follow_redirects=False)
            assert r.status_code == 200, path
            assert ctype in r.headers["content-type"], path
    finally:
        store.close()


def test_health_stale_without_heartbeat(tmp_path):
    """/api/health is the container healthcheck: no (or stale) heartbeat = 503,
    and it must answer without a login."""
    store = Store(":memory:")
    try:
        c = TestClient(create_app(Config(health_file=str(tmp_path / "hb")), store))
        r = c.get("/api/health")   # deliberately unauthenticated
        assert r.status_code == 503
        assert r.json() == {"status": "stale", "last_frame_age_s": None}
    finally:
        store.close()


def test_health_ok_with_fresh_heartbeat(tmp_path):
    from datetime import datetime, timezone

    hb = tmp_path / "hb"
    hb.write_text(datetime.now(timezone.utc).isoformat())
    store = Store(":memory:")
    try:
        c = TestClient(create_app(Config(health_file=str(hb)), store))
        r = c.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["last_frame_age_s"] < 5
    finally:
        store.close()


def test_stats(client):
    body = client.get("/api/stats").json()
    assert body["subscribers"] == 0 and body["active_flights"] == 0


def test_client_config_default_utc(client, monkeypatch):
    monkeypatch.delenv("TZ", raising=False)  # default must be UTC regardless of host
    assert client.get("/api/config").json() == {"tz": "UTC"}


def test_client_config_reports_display_tz():
    store = Store(":memory:")
    try:
        c = TestClient(create_app(Config(display_tz="America/New_York"), store))
        login(c, store)
        assert c.get("/api/config").json() == {"tz": "America/New_York"}
    finally:
        store.close()


# ---- subscriber CRUD round-trip -----------------------------------------
def test_subscriber_crud_roundtrip(client):
    r = client.post("/api/subscribers", json=_payload())
    assert r.status_code == 201
    sid = r.json()["id"]
    assert r.json()["ntfy_token_ref"] == "NTFY_HOME"  # a token name, not a token
    assert r.json()["units"] == "metric"              # default when omitted

    assert client.get(f"/api/subscribers/{sid}").json()["name"] == "home"
    assert len(client.get("/api/subscribers").json()) == 1

    r = client.put(f"/api/subscribers/{sid}",
                   json=_payload(name="home2", radius_km=42.0, units="imperial"))
    assert r.status_code == 200 and r.json()["radius_km"] == 42.0
    assert r.json()["units"] == "imperial"

    r = client.post(f"/api/subscribers/{sid}/active", json={"active": False})
    assert r.status_code == 200 and r.json()["active"] is False
    assert client.get("/api/subscribers", params={"active_only": True}).json() == []

    assert client.delete(f"/api/subscribers/{sid}").json() == {"deleted": sid}
    assert client.get(f"/api/subscribers/{sid}").status_code == 404


def test_missing_id_is_404(client):
    assert client.get("/api/subscribers/9999").status_code == 404
    assert client.put("/api/subscribers/9999", json=_payload()).status_code == 404
    assert client.post("/api/subscribers/9999/active", json={"active": True}).status_code == 404
    assert client.delete("/api/subscribers/9999").status_code == 404


# ---- the key security property: no raw token ever reaches the DB ----------
def test_raw_token_field_is_rejected_and_never_persisted(client):
    secret = "tk_supersecret123"
    r = client.post("/api/subscribers", json=_payload(ntfy_token="tk_supersecret123"))
    assert r.status_code == 422  # extra="forbid" rejects an actual-token field

    # nothing was written, and the secret appears in no column of the DB
    rows = client.store._conn.execute("SELECT * FROM subscribers").fetchall()
    assert rows == []
    dump = client.store._conn.execute(
        "SELECT group_concat(quote(name)||quote(ntfy_server)||quote(ntfy_topic)"
        "||quote(coalesce(ntfy_token_ref,''))) FROM subscribers").fetchone()[0]
    assert dump is None or secret not in dump


def test_token_ref_is_stored_but_no_token_field_leaks(client):
    client.post("/api/subscribers", json=_payload(ntfy_token_ref="NTFY_HOME"))
    s = client.get("/api/subscribers").json()[0]
    assert s["ntfy_token_ref"] == "NTFY_HOME"
    assert "ntfy_token" not in s  # only the reference is ever serialized


# ---- validation ----------------------------------------------------------
@pytest.mark.parametrize("over", [
    {"lat": 200}, {"lon": -999}, {"radius_km": -1}, {"radius_km": 0},
    {"name": ""}, {"units": "furlongs"},
])
def test_validation_rejects_bad_input(client, over):
    assert client.post("/api/subscribers", json=_payload(**over)).status_code == 422


def test_blank_topic_is_watch_only(client):
    """A blank ntfy_topic is accepted - the location is tracked/shown but never
    notified (so you can run without ntfy)."""
    r = client.post("/api/subscribers", json=_payload(ntfy_topic=""))
    assert r.status_code == 201
    assert r.json()["notify"] is False
    # a topic'd location reports notify=True
    r2 = client.post("/api/subscribers", json=_payload(name="alerted"))
    assert r2.json()["notify"] is True


# ---- test-ntfy: confirm a setup before saving a watched location ----------
def test_test_ntfy_sends_to_the_given_server_and_topic(sink_client):
    sink = RecordingSink(ok=True)
    c = sink_client(sink)
    r = c.post("/api/test-ntfy", json={
        "ntfy_server": "https://ntfy.example", "ntfy_topic": "my-sondes-7a3f"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "note": None}
    # the message went to exactly the server/topic from the request, not a default
    assert len(sink.sent) == 1
    msg = sink.sent[0]
    assert msg.server == "https://ntfy.example"
    assert msg.topic == "my-sondes-7a3f"
    assert msg.token_ref is None  # no token given → unauthenticated


def test_test_ntfy_failure_is_502(sink_client):
    c = sink_client(RecordingSink(ok=False))
    r = c.post("/api/test-ntfy", json={
        "ntfy_server": "https://ntfy.sh", "ntfy_topic": "nope"})
    assert r.status_code == 502
    assert "didn't accept" in r.json()["detail"]


@pytest.mark.parametrize("body", [
    {"ntfy_topic": "t"},                       # missing server
    {"ntfy_server": "", "ntfy_topic": "t"},    # blank server
    {"ntfy_server": "https://ntfy.sh"},        # missing topic
    {"ntfy_server": "https://ntfy.sh", "ntfy_topic": ""},  # blank topic
])
def test_test_ntfy_requires_explicit_server_and_topic(sink_client, body):
    sink = RecordingSink(ok=True)
    c = sink_client(sink)
    assert c.post("/api/test-ntfy", json=body).status_code == 422
    assert sink.sent == []  # validation fails before any send is attempted


def test_test_ntfy_raw_token_field_is_rejected(sink_client):
    """A raw token in the body is a 422 (extra='forbid') - only a ref is allowed,
    mirroring the subscriber endpoint's no-secret rule."""
    sink = RecordingSink(ok=True)
    c = sink_client(sink)
    r = c.post("/api/test-ntfy", json={
        "ntfy_server": "https://ntfy.sh", "ntfy_topic": "t",
        "ntfy_token": "tk_supersecret"})
    assert r.status_code == 422
    assert sink.sent == []


def test_test_ntfy_saved_token_ref_is_used(sink_client):
    sink = RecordingSink(ok=True)
    c = sink_client(sink)
    c.put("/api/tokens/test-tok", json={"token": "tk_value"})
    r = c.post("/api/test-ntfy", json={
        "ntfy_server": "https://ntfy.sh", "ntfy_topic": "t",
        "ntfy_token_ref": "test-tok"})
    assert r.status_code == 200
    assert r.json()["note"] is None  # token saved → no warning
    assert sink.sent[0].token_ref == "test-tok"  # ref passed through for auth


def test_test_ntfy_unsaved_token_ref_sends_unauth_and_warns(sink_client):
    sink = RecordingSink(ok=True)
    c = sink_client(sink)
    r = c.post("/api/test-ntfy", json={
        "ntfy_server": "https://ntfy.sh", "ntfy_topic": "t",
        "ntfy_token_ref": "absent-tok"})
    assert r.status_code == 200
    assert "absent-tok" in r.json()["note"]  # warns no such token is saved
    assert sink.sent[0].token_ref is None  # sent without auth (don't fake a token)


# ---- ntfy tokens: write-only lifecycle -------------------------------------
def test_token_save_list_delete_never_echoes_value(client):
    secret = "tk_supersecret999"
    r = client.put("/api/tokens/home", json={"token": secret})
    assert r.status_code == 200
    assert r.json()["name"] == "home"
    assert r.json()["hint"] == "…t999"       # last 4 only
    assert secret not in r.text              # the value is never echoed back

    listing = client.get("/api/tokens")
    assert [t["name"] for t in listing.json()] == ["home"]
    assert secret not in listing.text
    assert "token" not in listing.json()[0]

    # but the send path can read it (this is what the daemon's sink calls)
    assert client.store.get_ntfy_token("home") == secret

    # same name = replace (rotation)
    client.put("/api/tokens/home", json={"token": "tk_rotated_abcd"})
    assert client.get("/api/tokens").json()[0]["hint"] == "…abcd"
    assert client.store.get_ntfy_token("home") == "tk_rotated_abcd"

    assert client.delete("/api/tokens/home").json() == {"deleted": "home"}
    assert client.delete("/api/tokens/home").status_code == 404
    assert client.get("/api/tokens").json() == []


def test_token_delete_refused_while_referenced(client):
    client.put("/api/tokens/home", json={"token": "tk_x"})
    r = client.post("/api/subscribers", json=_payload(ntfy_token_ref="home"))
    sid = r.json()["id"]

    r = client.delete("/api/tokens/home")
    assert r.status_code == 409
    assert "watched location" in r.json()["detail"]

    client.delete(f"/api/subscribers/{sid}")
    assert client.delete("/api/tokens/home").status_code == 200


@pytest.mark.parametrize("bad", [
    {"token": ""},                             # blank value
    {"token": "x", "name": "smuggled"},        # extra field
    {},                                        # missing value
])
def test_token_validation_rejects_bad_body(client, bad):
    assert client.put("/api/tokens/home", json=bad).status_code == 422


def test_token_name_charset_is_validated(client):
    # %20 = space: path-reachable but rejected by the name rule
    assert client.put("/api/tokens/bad%20name",
                      json={"token": "tk_x"}).status_code == 422
    assert client.put("/api/tokens/" + "a" * 65,
                      json={"token": "tk_x"}).status_code == 422


def test_tokens_require_a_session():
    store = Store(":memory:")
    try:
        c = TestClient(create_app(Config(), store))  # deliberately no login()
        assert c.get("/api/tokens").status_code == 401
        assert c.put("/api/tokens/home",
                     json={"token": "tk_x"}).status_code == 401
    finally:
        store.close()


# ---- dashboard reads -----------------------------------------------------
def _seed_flight_pred_alert(store):
    store.add_subscriber(Subscriber(
        name="bob", lat=45.0, lon=7.0, radius_km=20,
        ntfy_server="https://ntfy.sh", ntfy_topic="bob"))
    store.upsert_flight({
        "serial": "S1", "launch_day": "2026-06-07", "type": "RS41", "state": "DESCENT",
        "first_seen": "2026-06-07T00:00:00+00:00", "last_seen": "2026-06-07T00:10:00+00:00",
        "launch_lat": 45.0, "launch_lon": 7.0, "burst_alt": 30000.0, "max_alt": 30000.0,
        "last_lat": 45.1, "last_lon": 7.2, "last_alt": 8000.0,
    })
    store.save_prediction(Prediction(
        serial="S1", launch_day=date(2026, 6, 7),
        predicted_at=datetime(2026, 6, 7, 0, 11, tzinfo=timezone.utc),
        land_lat=45.5, land_lon=7.6,
        land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
        source=PredictionSource.MEASURED, uncertainty_radius_km=2.5))
    store.record_alert(subscriber_id=1, serial="S1", launch_day=date(2026, 6, 7),
                       alert_type=AlertType.INBOUND, distance_km=8.0,
                       land_lat=45.5, land_lon=7.6,
                       sent_at=datetime(2026, 6, 7, 0, 11, tzinfo=timezone.utc))


def test_flights_endpoint_nests_prediction(client):
    _seed_flight_pred_alert(client.store)
    rows = client.get("/api/flights").json()
    assert len(rows) == 1
    assert rows[0]["serial"] == "S1"
    assert rows[0]["prediction"]["land_lat"] == 45.5


def test_alerts_endpoint_joins_names(client):
    _seed_flight_pred_alert(client.store)
    body = client.get("/api/alerts").json()
    assert body["total"] == 1
    rows = body["items"]
    assert rows[0]["subscriber_name"] == "bob"
    assert rows[0]["flight_type"] == "RS41"
    assert rows[0]["alert_type"] == "INBOUND"


def test_alerts_clear_endpoint(client):
    store = client.store
    _seed_flight_pred_alert(store)  # S1 is in DESCENT - its alert row stays
    store.upsert_flight({"serial": "S0", "launch_day": "2026-06-06",
                         "state": "LANDED", "type": "RS41"})
    store.record_alert(subscriber_id=1, serial="S0", launch_day=date(2026, 6, 6),
                       alert_type=AlertType.LANDED, distance_km=3.0,
                       land_lat=45.2, land_lon=7.2,
                       sent_at=datetime(2026, 6, 6, 0, 30, tzinfo=timezone.utc))
    assert client.get("/api/alerts").json()["total"] == 2

    r = client.delete("/api/alerts")
    assert r.status_code == 200
    assert r.json() == {"deleted": 1}
    # the airborne flight's row is kept (it doubles as the de-dup record)
    assert [a["serial"] for a in client.get("/api/alerts").json()["items"]] == ["S1"]


def test_alerts_pagination(client):
    store = client.store
    day = date(2026, 6, 7)
    # 12 alerts, each a minute apart so the newest-first order is deterministic
    for i in range(12):
        store.record_alert(
            subscriber_id=1, serial=f"S{i:02d}", launch_day=day,
            alert_type=AlertType.INBOUND, distance_km=float(i),
            land_lat=45.0, land_lon=7.0,
            sent_at=datetime(2026, 6, 7, 0, i, tzinfo=timezone.utc))

    # Default page: 10 newest (S11 down to S02), total reports the full history.
    page1 = client.get("/api/alerts").json()
    assert page1["total"] == 12 and page1["limit"] == 10 and page1["offset"] == 0
    assert [a["serial"] for a in page1["items"]] == [f"S{i:02d}" for i in range(11, 1, -1)]

    # Second page: the 2 oldest, in the same newest-first order.
    page2 = client.get("/api/alerts?limit=10&offset=10").json()
    assert page2["offset"] == 10
    assert [a["serial"] for a in page2["items"]] == ["S01", "S00"]


def test_map_feature_collection(client):
    _seed_flight_pred_alert(client.store)
    fc = client.get("/api/map").json()
    assert fc["type"] == "FeatureCollection"
    by_kind = {}
    for f in fc["features"]:
        by_kind.setdefault(f["properties"]["kind"], []).append(f)
    assert set(by_kind) == {"launch", "flight", "prediction", "subscriber"}

    # GeoJSON coordinates are [lon, lat] - assert the order is right
    flight = by_kind["flight"][0]
    assert flight["geometry"]["coordinates"] == [7.2, 45.1]  # [lon, lat]
    launch = by_kind["launch"][0]  # where the flight started (from launch_lat/lon)
    assert launch["geometry"]["coordinates"] == [7.0, 45.0]
    assert launch["properties"]["serial"] == "S1"
    pred = by_kind["prediction"][0]
    assert pred["geometry"]["coordinates"] == [7.6, 45.5]
    assert pred["properties"]["uncertainty_radius_km"] == 2.5
    sub = by_kind["subscriber"][0]
    assert sub["geometry"]["coordinates"] == [7.0, 45.0]
    assert sub["properties"]["radius_km"] == 20.0


def test_map_includes_flown_track(client):
    store = client.store
    _seed_flight_pred_alert(store)  # active flight S1 (DESCENT) on 2026-06-07
    day = date(2026, 6, 7)
    for t, lat, lon, alt in [(0.0, 45.0, 7.0, 200.0), (60.0, 45.05, 7.1, 9000.0),
                             (120.0, 45.1, 7.2, 8000.0)]:
        store.append_track_point("S1", day, t, lat, lon, alt)

    fc = client.get("/api/map").json()
    track = [f for f in fc["features"] if f["properties"]["kind"] == "track"]
    assert len(track) == 1
    assert track[0]["geometry"]["type"] == "LineString"
    # store [lat, lon, alt] → GeoJSON [lon, lat], launch point first
    assert track[0]["geometry"]["coordinates"][0] == [7.0, 45.0]
    assert track[0]["properties"]["serial"] == "S1"


def test_map_includes_predicted_path_and_landing(client):
    store = client.store
    _seed_flight_pred_alert(store)
    # a predicted path for the active flight (stored as [lat, lon, alt])
    store.save_prediction(Prediction(
        serial="S1", launch_day=date(2026, 6, 7),
        predicted_at=datetime(2026, 6, 7, 0, 12, tzinfo=timezone.utc),
        land_lat=45.5, land_lon=7.6,
        land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
        source=PredictionSource.MEASURED, uncertainty_radius_km=2.5,
        path=[(45.1, 7.2, 8000.0), (45.3, 7.4, 4000.0), (45.5, 7.6, 200.0)]))
    # an actual recorded landing (a different, landed flight)
    store.record_landing("S2", date(2026, 6, 6), land_lat=46.0, land_lon=8.0,
                         land_alt=210.0, landed_at=datetime(2026, 6, 6, 1, 0, tzinfo=timezone.utc),
                         detected_by="telemetry")

    fc = client.get("/api/map").json()
    by_kind = {}
    for f in fc["features"]:
        by_kind.setdefault(f["properties"]["kind"], []).append(f)
    assert {"path", "landing"} <= set(by_kind)

    path = by_kind["path"][0]
    assert path["geometry"]["type"] == "LineString"
    # [lat, lon, alt] in store → [lon, lat] in GeoJSON
    assert path["geometry"]["coordinates"][0] == [7.2, 45.1]
    assert path["properties"]["serial"] == "S1"

    landing = by_kind["landing"][0]
    assert landing["geometry"]["coordinates"] == [8.0, 46.0]
    assert landing["properties"]["serial"] == "S2"
    assert landing["properties"]["detected_by"] == "telemetry"


def test_accuracy_endpoint(client):
    store = client.store
    store.record_landing("ACC1", date(2026, 6, 7), land_lat=45.50, land_lon=7.60,
                         land_alt=210.0, landed_at=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
                         detected_by="telemetry")
    # two predictions: a far early one, a near final one
    for minute, lat, alt in [(10, 45.9, 9000.0), (39, 45.51, 500.0)]:
        store.save_prediction(Prediction(
            serial="ACC1", launch_day=date(2026, 6, 7),
            predicted_at=datetime(2026, 6, 7, 0, minute, tzinfo=timezone.utc),
            land_lat=lat, land_lon=7.60,
            land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
            source=PredictionSource.MEASURED, uncertainty_radius_km=5.0,
            alt_at_pred=alt))

    body = client.get("/api/accuracy").json()
    assert body["summary"]["n_flights"] == 1
    assert body["summary"]["n_predictions"] == 2
    assert body["summary"]["mean_final_error_km"] is not None
    # error is bucketed by altitude-at-prediction (the convergence view the UI shows):
    # one prediction at 9000 m → 5-10km bucket, one at 500 m → 0-2km bucket
    assert body["summary"]["bucket_counts"]["5-10km"] == 1
    assert body["summary"]["bucket_counts"]["0-2km"] == 1
    assert body["summary"]["bucket_mean_error_km"]["0-2km"] is not None
    flights = body["flights"]
    assert len(flights) == 1 and flights[0]["serial"] == "ACC1"
    # final (newest) prediction was the close one → small final error
    assert flights[0]["final_error_km"] < 2.0
    # launch_day rides along so the dashboard can link to the history view
    assert flights[0]["launch_day"] == "2026-06-07"


def test_accuracy_endpoint_empty(client):
    body = client.get("/api/accuracy").json()
    assert body["summary"] is None and body["flights"] == []


def test_accuracy_pagination(client):
    store = client.store
    day = date(2026, 6, 7)
    # 3 scored flights, landing times a minute apart (newest-first ordering)
    for i in range(3):
        serial = f"ACC{i}"
        store.record_landing(serial, day, land_lat=45.5, land_lon=7.6, land_alt=210.0,
                             landed_at=datetime(2026, 6, 7, 0, i, tzinfo=timezone.utc),
                             detected_by="telemetry")
        store.save_prediction(Prediction(
            serial=serial, launch_day=day,
            predicted_at=datetime(2026, 6, 7, 0, i, tzinfo=timezone.utc),
            land_lat=45.6, land_lon=7.6,
            land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
            source=PredictionSource.MEASURED, uncertainty_radius_km=5.0, alt_at_pred=500.0))

    page1 = client.get("/api/accuracy?limit=2&offset=0").json()
    # summary spans all 3 flights even though the page shows 2; total is the full count
    assert page1["summary"]["n_flights"] == 3
    assert page1["total"] == 3 and page1["offset"] == 0
    assert [f["serial"] for f in page1["flights"]] == ["ACC2", "ACC1"]

    page2 = client.get("/api/accuracy?limit=2&offset=2").json()
    assert page2["summary"]["n_flights"] == 3  # unchanged by paging
    assert [f["serial"] for f in page2["flights"]] == ["ACC0"]


def test_accuracy_clear_endpoint(client):
    store = client.store
    day = date(2026, 6, 7)

    def _pred(serial):
        return Prediction(
            serial=serial, launch_day=day,
            predicted_at=datetime(2026, 6, 7, 0, 10, tzinfo=timezone.utc),
            land_lat=45.5, land_lon=7.6,
            land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
            source=PredictionSource.MEASURED, uncertainty_radius_km=5.0,
            alt_at_pred=5000.0, path=[(45.6, 7.5, 5000.0), (45.5, 7.6, 200.0)])

    store.upsert_flight({"serial": "ACC1", "launch_day": day.isoformat(),
                         "state": "LANDED", "type": "RS41"})
    store.upsert_flight({"serial": "ACT1", "launch_day": day.isoformat(),
                         "state": "DESCENT", "type": "RS41"})
    store.save_prediction(_pred("ACC1"))
    store.save_prediction(_pred("ACT1"))
    store.record_landing("ACC1", day, land_lat=45.5, land_lon=7.6, land_alt=210.0,
                         landed_at=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
                         detected_by="telemetry")
    assert client.get("/api/accuracy").json()["summary"] is not None

    r = client.delete("/api/accuracy")
    assert r.status_code == 200
    assert r.json()["deleted"] == {"landings": 1, "predictions": 1}

    body = client.get("/api/accuracy").json()
    assert body["summary"] is None and body["flights"] == []
    # only the finished flight's records were wiped; the active flight keeps
    # its current prediction + predicted path for the live map
    assert store.latest_prediction("ACC1", day) is None
    assert store.latest_prediction("ACT1", day) is not None
    assert [p["serial"] for p in store.latest_paths_for_active()] == ["ACT1"]


# ---- per-flight prediction history ----------------------------------------
def _save_pred(store, serial, day, minute, lat, lon, alt=None):
    store.save_prediction(Prediction(
        serial=serial, launch_day=day,
        predicted_at=datetime(2026, 6, 7, 0, minute, tzinfo=timezone.utc),
        land_lat=lat, land_lon=lon,
        land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
        source=PredictionSource.MEASURED, uncertainty_radius_km=2.5,
        alt_at_pred=alt))


def test_history_unknown_flight_404_bad_date_422(client):
    assert client.get("/api/flights/NOPE/2026-06-07/history").status_code == 404
    assert client.get("/api/flights/S1/not-a-date/history").status_code == 422


def test_history_flying_drifts_vs_latest_prediction(client):
    store = client.store
    _seed_flight_pred_alert(store)  # DESCENT flight S1, first pred at (45.5, 7.6)
    day = date(2026, 6, 7)
    _save_pred(store, "S1", day, 20, 45.52, 7.62, alt=5000.0)
    _save_pred(store, "S1", day, 30, 45.55, 7.65, alt=2000.0)

    body = client.get("/api/flights/S1/2026-06-07/history").json()
    assert body["distance_reference"] == "latest_prediction"
    assert body["landing"] is None
    assert body["track"] == []  # active flight: its track is already on the map
    preds = body["predictions"]
    assert len(preds) == 3
    assert [p["predicted_at"] for p in preds] == sorted(p["predicted_at"] for p in preds)
    # latest prediction measured against itself → 0; earlier ones drift from it
    assert preds[-1]["distance_km"] == pytest.approx(0.0)
    assert preds[0]["distance_km"] == pytest.approx(
        haversine_km(45.5, 7.6, 45.55, 7.65))


def test_history_landed_scores_vs_actual_landing(client):
    store = client.store
    day = date(2026, 6, 7)
    store.upsert_flight({"serial": "L1", "launch_day": day.isoformat(),
                         "state": "LANDED", "type": "RS41"})
    _save_pred(store, "L1", day, 10, 45.9, 7.6, alt=9000.0)
    _save_pred(store, "L1", day, 39, 45.51, 7.6, alt=500.0)
    store.record_landing("L1", day, land_lat=45.5, land_lon=7.6, land_alt=210.0,
                         landed_at=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
                         detected_by="telemetry")
    for t, lat, lon, alt in [(0.0, 45.0, 7.0, 200.0), (60.0, 45.3, 7.3, 9000.0),
                             (120.0, 45.5, 7.6, 300.0)]:
        store.append_track_point("L1", day, t, lat, lon, alt)

    body = client.get("/api/flights/L1/2026-06-07/history").json()
    assert body["distance_reference"] == "landing"
    assert body["landing"]["land_lat"] == 45.5
    # landed flight ships its flown track (nothing else draws it on the map)
    assert body["track"][0] == [45.0, 7.0, 200.0]
    preds = body["predictions"]
    assert preds[0]["distance_km"] == pytest.approx(haversine_km(45.9, 7.6, 45.5, 7.6))
    assert preds[-1]["distance_km"] == pytest.approx(haversine_km(45.51, 7.6, 45.5, 7.6))


def test_history_flight_without_predictions(client):
    client.store.upsert_flight({"serial": "N1", "launch_day": "2026-06-07",
                                "state": "ASCENT", "type": "RS41"})
    body = client.get("/api/flights/N1/2026-06-07/history").json()
    assert body["predictions"] == []
    assert body["distance_reference"] is None
    assert body["landing"] is None


def test_history_etag_lets_unchanged_polls_revalidate(client):
    """The dashboard re-polls an open history panel every 15 s; unchanged data
    must come back as a bodyless 304, and any change must invalidate the ETag."""
    store = client.store
    day = date(2026, 6, 7)
    store.upsert_flight({"serial": "E1", "launch_day": day.isoformat(),
                         "state": "DESCENT", "type": "RS41"})
    _save_pred(store, "E1", day, 10, 45.9, 7.6, alt=9000.0)

    r1 = client.get("/api/flights/E1/2026-06-07/history")
    assert r1.status_code == 200
    etag = r1.headers["etag"]
    # no-cache = cache-but-revalidate, so the browser polls with If-None-Match
    assert r1.headers["cache-control"] == "private, no-cache"

    r2 = client.get("/api/flights/E1/2026-06-07/history",
                    headers={"If-None-Match": etag})
    assert r2.status_code == 304
    assert r2.content == b""

    # a new prediction changes the payload → full 200 under a new ETag
    _save_pred(store, "E1", day, 20, 45.5, 7.7, alt=5000.0)
    r3 = client.get("/api/flights/E1/2026-06-07/history",
                    headers={"If-None-Match": etag})
    assert r3.status_code == 200
    assert r3.headers["etag"] != etag
    assert len(r3.json()["predictions"]) == 2


# ---- SSE doorbell endpoint (/api/events) ----------------------------------
# This Starlette's TestClient buffers the *whole* response before returning
# (it runs the ASGI app to completion), so an infinite SSE stream would hang a
# plain client.get("/api/events"). We therefore drive the GET on a worker thread
# and let the generator terminate by closing the bus, then read the buffered
# body. (The live-socket streaming path is what the design doc verified against
# real uvicorn; this asserts the wire format + wiring without a real socket.)
def _drain_events(client, bus, publishes, settle=0.2):
    box = {}

    def run():
        box["r"] = client.get("/api/events")

    t = threading.Thread(target=run)
    t.start()
    deadline = time.time() + 5
    while not bus._clients and time.time() < deadline:  # wait until the generator registers
        time.sleep(0.005)
    publishes()
    time.sleep(settle)          # let the debounce flush + delivery happen
    bus.close()                 # wake the generator so it exits and the body completes
    t.join(timeout=5)
    assert not t.is_alive(), "SSE generator did not terminate on bus.close()"
    return box["r"].text


def _sse_app():
    store = Store(":memory:")
    bus = EventBus(debounce=0.01)   # tighter than prod so tests don't wait a second
    app = create_app(Config(), store, bus=bus)
    return app, store, bus


def test_events_requires_a_session():
    """/api/events lives under /api/*, so the RequireSession guard 401s an
    unauthenticated open before the generator ever starts (a fast, non-streaming
    response - no hang)."""
    store = Store(":memory:")
    try:
        with TestClient(create_app(Config(), store)) as c:  # deliberately no login()
            assert c.get("/api/events").status_code == 401
    finally:
        store.close()


def test_events_delivers_doorbell_with_a_data_line():
    app, store, bus = _sse_app()
    try:
        with TestClient(app) as c:
            login(c, store)
            body = _drain_events(c, bus, lambda: bus.publish("flights"))
    finally:
        store.close()
    assert "event: stats" in body     # liveness frame emitted on connect
    assert "event: flights" in body   # the published doorbell arrived
    assert "data: {}" in body         # every doorbell carries data={} (else it never fires)


def test_events_unregisters_the_client_on_disconnect():
    app, store, bus = _sse_app()
    try:
        with TestClient(app) as c:
            login(c, store)
            _drain_events(c, bus, lambda: bus.publish("flights"))
            assert bus._clients == set()   # the generator's finally ran unregister
    finally:
        store.close()


def test_events_app_keepalive_ticks_stats(monkeypatch):
    """With nothing changing, the endpoint still emits a periodic `event: stats`
    every STATS_TICK so the dashboard's "last frame age" line advances."""
    monkeypatch.setattr("tallyho.web.STATS_TICK", 0.1)
    app, store, bus = _sse_app()
    try:
        with TestClient(app) as c:
            login(c, store)
            body = _drain_events(c, bus, lambda: None, settle=0.45)
    finally:
        store.close()
    assert body.count("event: stats") >= 3   # connect frame + several timeout ticks


def test_events_fastapi_ping_keepalive(monkeypatch):
    """FastAPI's own `: ping` keepalive fires through our endpoint. The ping loop
    reads fastapi.routing._PING_INTERVAL (routing imports the name at module
    load), NOT fastapi.sse._PING_INTERVAL - patching the latter is a no-op."""
    monkeypatch.setattr("fastapi.routing._PING_INTERVAL", 0.1)
    monkeypatch.setattr("tallyho.web.STATS_TICK", 100.0)  # keep our stream silent so pings show
    app, store, bus = _sse_app()
    try:
        with TestClient(app) as c:
            login(c, store)
            body = _drain_events(c, bus, lambda: None, settle=0.35)
    finally:
        store.close()
    assert ": ping" in body
