"""ntfy notifications + alert lifecycle.

* :class:`NtfyMessage` - the publish contract (title/priority/tags/click/actions).
* :class:`NtfySink` - pluggable transport; :class:`HttpNtfySink` posts to a real
  ntfy server, :class:`FakeNtfySink` records messages for tests.
* :class:`AlertManager` - INBOUND (descent-gated) / UPDATE (throttled) / LANDED
  lifecycle with persistent de-dup.

Secrets: a subscriber stores only a *token reference* (the name of a token
saved in the DB); :class:`HttpNtfySink` resolves it to the bearer token at
send time via its injected ``token_lookup`` (normally ``Store.get_ntfy_token``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import Config, display_tzinfo
from windfall.geo import haversine_km
from .geofence import Match, geofence_matches
from .models import AlertType, FlightState, Prediction, Subscriber
from .store import Store

log = logging.getLogger(__name__)


@dataclass(slots=True)
class NtfyMessage:
    server: str
    topic: str
    title: str
    body: str
    priority: int = 3
    tags: list[str] = field(default_factory=list)
    click: str | None = None
    actions: list[str] = field(default_factory=list)
    token_ref: str | None = None
    markdown: bool = False


class NtfySink:
    def send(self, msg: NtfyMessage) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class FakeNtfySink(NtfySink):
    """Records messages instead of sending - the offline test sink."""

    def __init__(self):
        self.sent: list[NtfyMessage] = []

    def send(self, msg: NtfyMessage) -> bool:
        self.sent.append(msg)
        return True


class HttpNtfySink(NtfySink):
    """POSTs to a real ntfy server via stdlib urllib (no extra dependency).

    ``token_lookup`` maps a token name (``Subscriber.ntfy_token_ref``) to its
    bearer token, normally ``Store.get_ntfy_token`` - resolved per send so a
    token saved or rotated in the web UI takes effect immediately."""

    def __init__(self, timeout: float = 10.0, token_lookup=None):
        self.timeout = timeout
        self.token_lookup = token_lookup

    def send(self, msg: NtfyMessage) -> bool:
        import urllib.request

        url = f"{msg.server.rstrip('/')}/{msg.topic}"
        headers = {
            "Title": msg.title,
            "Priority": str(msg.priority),
        }
        if msg.tags:
            headers["Tags"] = ",".join(msg.tags)
        if msg.click:
            headers["Click"] = msg.click
        if msg.actions:
            headers["Actions"] = "; ".join(msg.actions)
        if msg.markdown:
            headers["Markdown"] = "yes"
        token = (self.token_lookup(msg.token_ref)
                 if msg.token_ref and self.token_lookup else None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            url, data=msg.body.encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            log.exception("ntfy send failed for topic %s", msg.topic)
            return False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AlertManager:
    """Lifecycle-aware alerting with persistent de-dup."""

    def __init__(self, cfg: Config, store: Store, sink: NtfySink):
        self.cfg = cfg
        self.store = store
        self.sink = sink
        self._tz = display_tzinfo(cfg)   # human-facing ETA timezone (see config)

    # ---- INBOUND / UPDATE (called per descent prediction) ----------------
    def handle_prediction(
        self,
        flight,
        prediction: Prediction,
        subscribers: list[Subscriber],
        now: datetime | None = None,
    ) -> list[NtfyMessage]:
        """Process a fresh prediction. INBOUND only fires from DESCENT."""
        if flight.state != FlightState.DESCENT:
            return []
        now = now or _utcnow()
        sent: list[NtfyMessage] = []
        for m in geofence_matches(prediction.land_lat, prediction.land_lon, subscribers):
            sub = m.subscriber
            if not sub.notify_enabled:
                continue  # watch-only location: matched for the map, never pushed
            inbound = self.store.get_alert(sub.id, flight.serial, flight.launch_day, AlertType.INBOUND)
            if inbound is None:
                msg = self._inbound_msg(flight, prediction, m)
                if self.store.record_alert(sub.id, flight.serial, flight.launch_day,
                                           AlertType.INBOUND, m.distance_km,
                                           prediction.land_lat, prediction.land_lon, now):
                    if self.sink.send(msg):
                        sent.append(msg)
            else:
                upd = self._maybe_update(flight, prediction, m, inbound, now)
                if upd is not None:
                    sent.append(upd)
        return sent

    def _maybe_update(self, flight, prediction, m: Match, inbound: dict,
                      now: datetime) -> NtfyMessage | None:
        if self.cfg.notify.update_move_km <= 0:
            return None
        sub = m.subscriber
        prev = self.store.get_alert(sub.id, flight.serial, flight.launch_day, AlertType.UPDATE)
        ref = prev or inbound
        ref_lat, ref_lon = ref.get("land_lat"), ref.get("land_lon")
        if ref_lat is None or ref_lon is None:
            return None
        moved = haversine_km(prediction.land_lat, prediction.land_lon, ref_lat, ref_lon)
        if moved < self.cfg.notify.update_move_km:
            return None
        last_sent = self.store.last_alert_at(sub.id, flight.serial, flight.launch_day,
                                             AlertType.UPDATE)
        if last_sent is not None:
            if (now - last_sent).total_seconds() < self.cfg.notify.update_throttle_seconds:
                return None
        msg = self._update_msg(flight, prediction, m, moved)
        self.store.upsert_alert(sub.id, flight.serial, flight.launch_day, AlertType.UPDATE,
                                m.distance_km, prediction.land_lat, prediction.land_lon, now)
        return msg if self.sink.send(msg) else None

    # ---- LANDED (called on the LANDED transition) ------------------------
    def handle_landed(
        self, flight, subscribers: list[Subscriber], now: datetime | None = None
    ) -> list[NtfyMessage]:
        """Final 'go recover it' alert keyed on the actual last-known position.

        Sent to subscribers within radius of the actual landing *and* to any who
        received an INBOUND (so a wandered INBOUND gets corrected)."""
        now = now or _utcnow()
        if flight.last_lat is None or flight.last_lon is None:
            return []
        sent: list[NtfyMessage] = []
        for sub in subscribers:
            if not sub.active or not sub.notify_enabled:
                continue
            d = haversine_km(sub.lat, sub.lon, flight.last_lat, flight.last_lon)
            inside = d <= sub.radius_km
            had_inbound = self.store.get_alert(
                sub.id, flight.serial, flight.launch_day, AlertType.INBOUND) is not None
            if not (inside or had_inbound):
                continue
            if self.store.record_alert(sub.id, flight.serial, flight.launch_day,
                                       AlertType.LANDED, d, flight.last_lat,
                                       flight.last_lon, now):
                msg = self._landed_msg(flight, sub, d)
                if self.sink.send(msg):
                    sent.append(msg)
        return sent

    # ---- message formatting ----------------------------------
    def _inbound_msg(self, flight, prediction: Prediction, m: Match) -> NtfyMessage:
        sub = m.subscriber
        title = f"Tally-ho: {flight.type or 'sonde'} inbound {m.distance_km:.1f} km {m.compass}"
        body = self._body(flight, prediction, m)
        return NtfyMessage(
            server=sub.ntfy_server, topic=sub.ntfy_topic, title=title, body=body,
            priority=4, tags=["balloon", "round_pushpin"],
            click=self._map_url(prediction.land_lat, prediction.land_lon),
            actions=self._track_action(flight.serial),
            token_ref=sub.ntfy_token_ref, markdown=True,
        )

    def _update_msg(self, flight, prediction: Prediction, m: Match, moved: float) -> NtfyMessage:
        sub = m.subscriber
        title = f"Tally-ho: {flight.type or 'sonde'} update {m.distance_km:.1f} km {m.compass}"
        body = self._body(flight, prediction, m) + f"\nMoved {moved:.1f} km since last alert."
        return NtfyMessage(
            server=sub.ntfy_server, topic=sub.ntfy_topic, title=title, body=body,
            priority=3, tags=["balloon", "arrows_counterclockwise"],
            click=self._map_url(prediction.land_lat, prediction.land_lon),
            actions=self._track_action(flight.serial),
            token_ref=sub.ntfy_token_ref, markdown=True,
        )

    def _landed_msg(self, flight, sub: Subscriber, distance_km: float) -> NtfyMessage:
        title = f"Tally-ho: {flight.type or 'sonde'} LANDED {distance_km:.1f} km away"
        body = (
            f"**{flight.serial}** ({flight.type or '?'}) last seen at "
            f"{flight.last_lat:.5f}, {flight.last_lon:.5f}, alt {flight.last_alt:.0f} m.\n"
            f"Distance from you: {distance_km:.1f} km."
        )
        return NtfyMessage(
            server=sub.ntfy_server, topic=sub.ntfy_topic, title=title, body=body,
            priority=5, tags=["balloon", "white_check_mark"],
            click=self._map_url(flight.last_lat, flight.last_lon),
            actions=self._track_action(flight.serial),
            token_ref=sub.ntfy_token_ref, markdown=True,
        )

    def _body(self, flight, prediction: Prediction, m: Match) -> str:
        eta = prediction.land_eta.astimezone(self._tz).strftime("%H:%M:%S %Z")
        return (
            f"**{flight.serial}** ({flight.type or '?'}) predicted landing\n"
            f"{prediction.land_lat:.5f}, {prediction.land_lon:.5f}\n"
            f"ETA {eta} · {m.distance_km:.1f} km {m.compass} of you\n"
            f"uncertainty ±{prediction.uncertainty_radius_km:.1f} km · source {prediction.source.value}"
        )

    def _map_url(self, lat: float, lon: float) -> str:
        return self.cfg.notify.map_url_template.format(lat=lat, lon=lon)

    def _track_action(self, serial: str) -> list[str]:
        url = self.cfg.notify.track_url_template.format(serial=serial)
        return [f"view, Open track, {url}"]
