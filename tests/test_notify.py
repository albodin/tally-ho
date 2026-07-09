"""Tests for ntfy alert lifecycle + de-dup."""

from datetime import date, datetime, timedelta, timezone

import pytest

from tallyho.config import Config
from tallyho.models import (
    AlertType,
    FlightState,
    Prediction,
    PredictionSource,
    Subscriber,
)
from tallyho.notify import AlertManager, FakeNtfySink, NtfyMessage
from tallyho.store import Store
from windfall.tracker import Flight

DAY = date(2026, 6, 7)
T0 = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def _flight(state=FlightState.DESCENT, lat=45.04, lon=7.0, alt=3000.0):
    f = Flight(serial="S1", launch_day=DAY, type="RS41", state=state)
    f.last_lat, f.last_lon, f.last_alt = lat, lon, alt
    f.last_seen = T0
    return f


def _pred(lat=45.04, lon=7.0, radius=2.0, source=PredictionSource.MEASURED):
    return Prediction(serial="S1", launch_day=DAY, predicted_at=T0,
                      land_lat=lat, land_lon=lon,
                      land_eta=T0 + timedelta(minutes=20), source=source,
                      uncertainty_radius_km=radius)


def _sub(store, name="alice", lat=45.0, lon=7.0, r=30.0):
    sid = store.add_subscriber(Subscriber(
        name=name, lat=lat, lon=lon, radius_km=r,
        ntfy_server="https://ntfy.sh", ntfy_topic=f"{name}-sondes",
        ntfy_token_ref="NTFY_TOKEN"))
    return store.list_subscribers()[0] if name == "alice" else \
        [s for s in store.list_subscribers() if s.id == sid][0]


def test_inbound_sent_once(store):
    subs = [_sub(store)]
    mgr = AlertManager(Config(), store, FakeNtfySink())
    flight = _flight()
    sent1 = mgr.handle_prediction(flight, _pred(), subs, now=T0)
    sent2 = mgr.handle_prediction(flight, _pred(), subs, now=T0 + timedelta(seconds=10))
    assert len(sent1) == 1
    assert sent1[0].title.startswith("Tally-ho: RS41 inbound")
    assert sent1[0].token_ref == "NTFY_TOKEN"
    assert sent1[0].click  # map link present
    assert sent2 == []     # de-duped


def test_eta_rendered_in_display_tz(store, monkeypatch):
    # land_eta is T0+20min = 12:20 UTC; the body's ETA should follow display_tz.
    monkeypatch.delenv("TZ", raising=False)  # don't let a host TZ skew the default
    subs = [_sub(store)]
    flight = _flight()
    default = AlertManager(Config(), store, FakeNtfySink())
    assert "ETA 12:20:00 UTC" in default.handle_prediction(flight, _pred(), subs, now=T0)[0].body

    ny = AlertManager(Config(display_tz="America/New_York"), store, FakeNtfySink())
    # fresh flight/sub so de-dup (keyed in the DB) doesn't swallow the second send
    store.clear_alerts()
    subs2 = [_sub(store, name="bob")]
    body = ny.handle_prediction(_flight(), _pred(), subs2, now=T0)[0].body
    assert "ETA 08:20:00 EDT" in body  # 12:20 UTC -> 08:20 EDT


def test_units_formatting(store):
    """Per-subscriber units: metric renders km/m, imperial renders mi/ft -
    same alerts, same internal km, only the strings differ."""
    metric = _sub(store)  # alice, metric default
    sid = store.add_subscriber(Subscriber(
        name="carol", lat=45.0, lon=7.0, radius_km=30.0,
        ntfy_server="https://ntfy.sh", ntfy_topic="carol-sondes", units="imperial"))
    subs = [metric, store.get_subscriber(sid)]
    mgr = AlertManager(Config(), store, FakeNtfySink())

    # INBOUND: landing 45.04,7.0 is ~4.4 km / 2.8 mi due N; uncertainty 2.0 km
    sent = {s.topic: s for s in mgr.handle_prediction(_flight(), _pred(radius=2.0),
                                                      subs, now=T0)}
    km_msg, mi_msg = sent["alice-sondes"], sent["carol-sondes"]
    assert "4.4 km N" in km_msg.title and "±2.0 km" in km_msg.body
    assert "2.8 mi N" in mi_msg.title and "±1.2 mi" in mi_msg.body
    assert "km" not in mi_msg.title and "km" not in mi_msg.body

    # UPDATE: landing moves ~7.8 km / 4.8 mi north
    upd = {s.topic: s for s in mgr.handle_prediction(
        _flight(), _pred(lat=45.11, radius=2.0), subs, now=T0 + timedelta(minutes=11))}
    assert "Moved 7.8 km since last alert." in upd["alice-sondes"].body
    assert "Moved 4.8 mi since last alert." in upd["carol-sondes"].body

    # LANDED: ~2.2 km / 1.4 mi away, altitude 210 m / 689 ft
    flight = _flight(state=FlightState.LANDED, lat=45.02, lon=7.0, alt=210.0)
    landed = {s.topic: s for s in mgr.handle_landed(flight, subs, now=T0)}
    assert "LANDED 2.2 km away" in landed["alice-sondes"].title
    assert "alt 210 m" in landed["alice-sondes"].body
    assert "LANDED 1.4 mi away" in landed["carol-sondes"].title
    assert "alt 689 ft" in landed["carol-sondes"].body


def test_inbound_gated_on_descent(store):
    subs = [_sub(store)]
    mgr = AlertManager(Config(), store, FakeNtfySink())
    # ASCENT prediction must not fire an INBOUND
    flight = _flight(state=FlightState.ASCENT)
    assert mgr.handle_prediction(flight, _pred(), subs, now=T0) == []


def test_no_alert_outside_radius(store):
    subs = [_sub(store, r=1.0)]
    mgr = AlertManager(Config(), store, FakeNtfySink())
    flight = _flight()
    # landing ~30 km away, radius 1 km
    assert mgr.handle_prediction(flight, _pred(lat=45.3), subs, now=T0) == []


def test_watch_only_subscriber_never_notified(store):
    """A blank-topic (watch-only) location is matched for the map but gets no
    push - neither INBOUND on descent nor LANDED."""
    sid = store.add_subscriber(Subscriber(
        name="watcher", lat=45.0, lon=7.0, radius_km=30.0,
        ntfy_server="https://ntfy.sh", ntfy_topic=""))
    sub = store.get_subscriber(sid)
    assert sub.notify_enabled is False
    sink = FakeNtfySink()
    mgr = AlertManager(Config(), store, sink)
    flight = _flight()
    assert mgr.handle_prediction(flight, _pred(), [sub], now=T0) == []
    assert mgr.handle_landed(flight, [sub], now=T0) == []
    assert sink.sent == []  # nothing ever transmitted


def test_update_on_move_and_throttle(store):
    subs = [_sub(store)]
    cfg = Config()
    cfg.notify.update_move_km = 5.0
    cfg.notify.update_throttle_seconds = 600
    mgr = AlertManager(cfg, store, FakeNtfySink())
    flight = _flight()
    mgr.handle_prediction(flight, _pred(lat=45.04, lon=7.0), subs, now=T0)  # INBOUND
    # landing moves ~8 km → UPDATE
    upd = mgr.handle_prediction(flight, _pred(lat=45.11, lon=7.0), subs,
                                now=T0 + timedelta(minutes=11))
    assert len(upd) == 1
    assert "update" in upd[0].title.lower()
    # another move within throttle window → suppressed
    upd2 = mgr.handle_prediction(flight, _pred(lat=45.18, lon=7.0), subs,
                                 now=T0 + timedelta(minutes=12))
    assert upd2 == []


def test_small_move_no_update(store):
    subs = [_sub(store)]
    cfg = Config()
    cfg.notify.update_move_km = 5.0
    mgr = AlertManager(cfg, store, FakeNtfySink())
    flight = _flight()
    mgr.handle_prediction(flight, _pred(lat=45.04), subs, now=T0)
    # moves only ~1 km → no UPDATE
    out = mgr.handle_prediction(flight, _pred(lat=45.05), subs,
                                now=T0 + timedelta(minutes=20))
    assert out == []


def test_landed_alert(store):
    subs = [_sub(store)]
    mgr = AlertManager(Config(), store, FakeNtfySink())
    flight = _flight(state=FlightState.LANDED, lat=45.02, lon=7.0, alt=210.0)
    sent = mgr.handle_landed(flight, subs, now=T0)
    assert len(sent) == 1
    assert "LANDED" in sent[0].title
    assert sent[0].priority == 5   # highest priority
    # sent once
    assert mgr.handle_landed(flight, subs, now=T0 + timedelta(minutes=1)) == []


def test_landed_corrects_wandered_inbound(store):
    # INBOUND fired, but actual landing ends up outside the subscriber radius:
    # LANDED still goes out (correction), keyed on actual position.
    subs = [_sub(store, r=3.0)]
    mgr = AlertManager(Config(), store, FakeNtfySink())
    flight = _flight(state=FlightState.DESCENT, lat=45.02, lon=7.0)
    mgr.handle_prediction(flight, _pred(lat=45.02, radius=2.0), subs, now=T0)  # INBOUND
    # now it actually lands 50 km away (outside radius)
    flight.state = FlightState.LANDED
    flight.last_lat, flight.last_lon = 45.5, 7.0
    sent = mgr.handle_landed(flight, subs, now=T0 + timedelta(minutes=30))
    assert len(sent) == 1   # correction delivered despite being outside radius


def test_dedup_persists_in_store(store):
    subs = [_sub(store)]
    mgr = AlertManager(Config(), store, FakeNtfySink())
    flight = _flight()
    mgr.handle_prediction(flight, _pred(), subs, now=T0)
    row = store.get_alert(subs[0].id, "S1", DAY, AlertType.INBOUND)
    assert row is not None
    assert row["distance_km"] is not None


# ---- HttpNtfySink: token resolution + header assembly (offline) ------------
class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture_urlopen(monkeypatch):
    import urllib.request

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"], captured["timeout"] = req, timeout
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def test_http_sink_resolves_token_ref_via_lookup(monkeypatch):
    from tallyho.notify import HttpNtfySink

    captured = _capture_urlopen(monkeypatch)
    sink = HttpNtfySink(timeout=5.0, token_lookup={"home": "tk_value"}.get)
    ok = sink.send(NtfyMessage(server="https://ntfy.sh/", topic="t",
                               title="T", body="b", token_ref="home"))
    assert ok
    req = captured["req"]
    assert req.full_url == "https://ntfy.sh/t"   # trailing slash normalized
    assert req.get_header("Authorization") == "Bearer tk_value"
    assert req.data == b"b"
    assert captured["timeout"] == 5.0


def test_http_sink_unknown_or_absent_ref_sends_unauthenticated(monkeypatch):
    from tallyho.notify import HttpNtfySink

    captured = _capture_urlopen(monkeypatch)
    sink = HttpNtfySink(token_lookup={}.get)
    # unknown name and no name at all both mean: no Authorization header
    for ref in ("nope", None):
        assert sink.send(NtfyMessage(server="https://ntfy.sh", topic="t",
                                     title="T", body="b", token_ref=ref))
        assert captured["req"].get_header("Authorization") is None
