"""Local dashboard + self-service onboarding web UI (optional `api` extra).

A small FastAPI app that (1) shows live flights, landing predictions and sent
alerts on a Leaflet map + tables, and (2) lets you add/edit/remove *watched
locations* (subscribers) from the browser. It writes only the ``subscribers``
table; the running ``tallyho run`` daemon hot-reloads subscribers from the DB
every ``subscriber_reload_seconds``, so changes take effect with no restart.

Design notes:
* ``fastapi``/``pydantic``/``uvicorn`` are imported lazily (inside the functions
  below) so the core engine still imports and tests offline without the `api`
  extra - mirroring how ``sondehub``/``rasterio``/``herbie`` are isolated.
* This module deliberately does NOT use ``from __future__ import annotations``:
  the request models and route handlers are defined *inside* ``create_app``, and
  FastAPI resolves handler type hints via ``get_type_hints`` - stringized
  annotations would fail to find those locally-scoped classes.
* ntfy tokens are write-only: ``PUT /api/tokens/{name}`` is the single place a
  raw token is accepted, and no response ever carries a value (listings show a
  last-4 hint). A subscriber holds only the token's *name*; its request model
  has no token field and forbids extras, so a token can't land in the wrong
  table.
* Everything but ``/api/health``, ``/login`` and ``POST /api/login`` requires a
  session cookie (see :mod:`tallyho.auth`); the account is created by the
  first-run setup wizard (:mod:`tallyho.setup`).
"""

import hashlib
import json
import logging
import math
import re
import secrets
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from windfall.geo import haversine_km

from .config import Config, display_tz_name
from .models import Subscriber
from .store import Store

log = logging.getLogger(__name__)

_STATIC = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC / "index.html"
_LOGIN_HTML = _STATIC / "login.html"


def heartbeat_age(cfg: Config) -> float | None:
    """Seconds since the daemon last saw a frame (via the heartbeat file), or
    None when no valid heartbeat exists. Shared by ``/api/health`` and the
    ``tallyho health`` CLI fallback."""
    try:
        ts = Path(cfg.health_file).read_text().strip()
        last = datetime.fromisoformat(ts)
    except (OSError, ValueError):
        return None
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds()


def _clean(x):
    """NaN/Inf -> None so the result is strict, parseable JSON (empty altitude
    buckets in the accuracy metrics report as NaN means)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _metrics_json(m) -> dict:
    """Serialize :class:`windfall.replay.Metrics` to a JSON-safe dict."""
    return {
        "n_flights": m.n_flights,
        "n_predictions": m.n_predictions,
        "mean_final_error_km": _clean(m.mean_final_error_km),
        "calibration_rate": _clean(m.calibration_rate),
        "radius_scale_for_target": _clean(m.radius_scale_for_target),
        "bucket_mean_error_km": {k: _clean(v) for k, v in m.bucket_mean_error_km.items()},
        "bucket_counts": m.bucket_counts,
    }


def _serialize_subscriber(s: Subscriber) -> dict:
    """Subscriber -> JSON dict. Note: emits ``ntfy_token_ref`` (a saved token's
    name), never an actual token - there is no token field to leak."""
    return {
        "id": s.id,
        "name": s.name,
        "lat": s.lat,
        "lon": s.lon,
        "radius_km": s.radius_km,
        "ntfy_server": s.ntfy_server,
        "ntfy_topic": s.ntfy_topic,
        "ntfy_token_ref": s.ntfy_token_ref,
        "units": s.units,
        "notify": s.notify_enabled,
        "active": s.active,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }


def create_app(cfg: Config, store: Store, ntfy_sink=None):
    """Build the FastAPI app. ``store`` is injected so tests can pass an
    in-memory Store and the CLI can pass ``Store(cfg.db_path)``. ``ntfy_sink``
    (an :class:`~tallyho.notify.NtfySink`) is likewise injectable so the
    ``/api/test-ntfy`` route can be exercised without real network I/O; when
    ``None`` it lazily builds an :class:`~tallyho.notify.HttpNtfySink`."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Response
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, ConfigDict, Field
        from starlette.middleware.sessions import SessionMiddleware
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the web UI needs FastAPI; install it with: pip install '.[api]'"
        ) from exc

    from .auth import (SESSION_MAX_AGE, LoginLimiter, RequireSession,
                       hash_password, session_secret, verify_password)

    class SubscriberIn(BaseModel):
        # Reject unknown fields so a stray `ntfy_token` (a secret) is a 422, not
        # a silent write - tokens go only through PUT /api/tokens/{name}.
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        name: str = Field(min_length=1)
        lat: float = Field(ge=-90, le=90)
        lon: float = Field(ge=-180, le=180)
        radius_km: float = Field(gt=0)
        ntfy_server: str = "https://ntfy.sh"
        # Blank topic == watch-only: the location is tracked and shown on the map
        # but never sends an ntfy alert (run without ntfy configured).
        ntfy_topic: str = ""
        # NAME of a saved ntfy token (see /api/tokens), never the token itself
        ntfy_token_ref: str | None = None
        units: Literal["metric", "imperial"] = "metric"
        active: bool = True

    class ActiveIn(BaseModel):
        model_config = ConfigDict(extra="forbid")
        active: bool

    class TestNtfyIn(BaseModel):
        # Same no-secret-in-body rule as SubscriberIn: only a token *reference*
        # (a saved token's name) is accepted, never a raw token; extras are a 422.
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        # Both required and non-blank: a test must name an explicit destination,
        # so an omitted/blank server is a 422 rather than a silent ntfy.sh fallback.
        ntfy_server: str = Field(min_length=1)
        ntfy_topic: str = Field(min_length=1)  # a blank topic can't be tested
        ntfy_token_ref: str | None = None

    def _to_subscriber(p: SubscriberIn, sub_id: int | None = None) -> Subscriber:
        return Subscriber(
            id=sub_id, name=p.name, lat=p.lat, lon=p.lon, radius_km=p.radius_km,
            ntfy_server=p.ntfy_server, ntfy_topic=p.ntfy_topic,
            ntfy_token_ref=p.ntfy_token_ref or None, units=p.units, active=p.active,
        )

    app = FastAPI(title="tally-ho", docs_url="/api/docs", openapi_url="/api/openapi.json")
    # Session cookie first (outermost), then the guard that needs it. The
    # signing secret persists in the DB, so sessions survive restarts.
    app.add_middleware(RequireSession)
    app.add_middleware(SessionMiddleware, secret_key=session_secret(store),
                       max_age=SESSION_MAX_AGE, same_site="lax")
    # CSS/JS + vendored Leaflet; auth-exempt (see PUBLIC_PREFIX) so the login
    # page can load its stylesheet.
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    limiter = LoginLimiter()
    # Verifying against this when the username is unknown keeps the response
    # time indistinguishable from a wrong password.
    dummy_hash = hash_password(secrets.token_urlsafe(16))

    class LoginIn(BaseModel):
        model_config = ConfigDict(extra="forbid")
        username: str = Field(min_length=1)
        password: str

    # ---- pages + auth ------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(_INDEX_HTML), media_type="text/html")

    @app.get("/login", include_in_schema=False)
    def login_page():
        return FileResponse(str(_LOGIN_HTML), media_type="text/html")

    @app.post("/api/login")
    def login(payload: LoginIn, request: Request):
        key = request.client.host if request.client else "?"
        wait = limiter.retry_after(key)
        if wait > 0:
            raise HTTPException(status_code=429,
                                detail=f"Too many failed logins; retry in {wait:.0f}s")
        stored = store.get_user_hash(payload.username)
        if not (verify_password(payload.password, stored or dummy_hash)
                and stored is not None):
            limiter.record_failure(key)
            raise HTTPException(status_code=401, detail="Invalid username or password")
        limiter.reset(key)
        request.session["user"] = payload.username
        return {"ok": True, "user": payload.username}

    @app.post("/api/logout")
    def logout(request: Request):
        request.session.clear()
        return {"ok": True}

    @app.get("/api/config")
    def client_config():
        """Front-end bootstrap config. ``tz`` is the IANA timezone the dashboard
        renders all times in (from ``TZ``/``TALLYHO_DISPLAY_TZ``); the browser
        formats with it so the clock matches the server's configured zone rather
        than each viewer's local one."""
        return {"tz": display_tz_name(cfg)}

    @app.get("/api/health")
    def health():
        """Unauthenticated container healthcheck: 200 while the pipeline is
        fresh, 503 once the heartbeat goes stale. Minimal body - the dashboard's
        richer numbers live on the authenticated /api/stats."""
        age = heartbeat_age(cfg)
        status = "ok" if age is not None and age < cfg.health_stale_seconds else "stale"
        body = {"status": status,
                "last_frame_age_s": None if age is None else round(age, 1)}
        return body if status == "ok" else JSONResponse(body, status_code=503)

    @app.get("/api/stats")
    def stats():
        age = heartbeat_age(cfg)
        return {
            "db_path": store.path,
            "active_flights": len(store.active_flights()),
            "subscribers": len(store.list_subscribers(active_only=False)),
            "last_frame_age_s": None if age is None else round(age, 1),
        }

    # ---- dashboard reads -------------------------------------------------
    @app.get("/api/flights")
    def flights(include_landed: bool = False, limit: int = 200):
        out = []
        for r in store.recent_flights(limit=limit, include_landed=include_landed):
            row = dict(r)
            row["prediction"] = store.latest_prediction(
                r["serial"], date.fromisoformat(r["launch_day"]))
            out.append(row)
        return out

    @app.get("/api/flights/{serial}/{launch_day}/history")
    def flight_history(serial: str, launch_day: date, request: Request):
        """One flight's full prediction time-series, each scored against a
        reference point: the recorded actual landing once the flight is down,
        else the *latest* prediction (drift - how far the predicted landing has
        moved). The dashboard's per-sonde history panel reads this.

        ETag'd: this is the page's heaviest response (a LANDED flight ships its
        whole flown track) and the dashboard re-polls it every 15 s while the
        panel is open, so unchanged data revalidates as a bodyless 304 instead
        of re-downloading."""
        flight = store.get_flight(serial, launch_day)
        if flight is None:
            raise HTTPException(status_code=404, detail="flight not found")
        preds = store.predictions_for(serial, launch_day)  # oldest first
        landing = store.get_landing(serial, launch_day)
        if landing is not None:
            reference = "landing"
            ref_lat, ref_lon = landing["land_lat"], landing["land_lon"]
        elif preds:
            reference = "latest_prediction"
            ref_lat, ref_lon = preds[-1]["land_lat"], preds[-1]["land_lon"]
        else:
            reference = None
        out_preds = [{
            "predicted_at": p["predicted_at"],
            "land_lat": p["land_lat"],
            "land_lon": p["land_lon"],
            "land_eta": p["land_eta"],
            "source": p["source"],
            "uncertainty_radius_km": _clean(p["uncertainty_radius_km"]),
            "alt_at_pred": _clean(p["alt_at_pred"]),
            "distance_km": _clean(
                haversine_km(p["land_lat"], p["land_lon"], ref_lat, ref_lon)),
        } for p in preds]
        # Active flights' flown tracks are already on the map (refreshMap); ship
        # the track only for LANDED flights, whose track nothing else draws.
        track = ([[t["lat"], t["lon"], t["alt"]]
                  for t in store.track_for(serial, launch_day)]
                 if flight["state"] == "LANDED" else [])
        payload = {
            "serial": serial,
            "launch_day": launch_day.isoformat(),
            "flight": flight,
            "landing": landing,
            "distance_reference": reference,
            "track": track,
            "predictions": out_preds,
        }
        # Serialize once so the ETag is a hash of the exact bytes served
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"))
        etag = f'"{hashlib.sha256(body.encode()).hexdigest()[:32]}"'
        headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(content=body, media_type="application/json",
                        headers=headers)

    @app.get("/api/alerts")
    def alerts(limit: int = 100):
        return store.recent_alerts(limit=limit)

    @app.delete("/api/alerts")
    def clear_alerts():
        """Wipe the recent-alerts history. Alerts for flights still in the air
        are kept - their rows are the de-dup record, so deleting them would
        re-send the notifications."""
        deleted = store.clear_alerts()
        log.info("cleared alert history: %d row(s)", deleted)
        return {"deleted": deleted}

    @app.get("/api/map")
    def map_data(landings_limit: int = 50):
        """One GeoJSON FeatureCollection for the map. NOTE: GeoJSON coordinates
        are [longitude, latitude] - the reverse of Leaflet's (lat, lon).

        Feature kinds: ``launch`` (where the flight started), ``track`` (the
        actual flown path LineString, launch → current position), ``flight``
        (current position), ``path`` (the predicted trajectory LineString,
        current position → landing), ``prediction`` (the predicted landing point
        + uncertainty), ``landing`` (actual recorded landing), and ``subscriber``
        (watched location)."""
        features = []
        for f in store.active_flights():
            if f["launch_lat"] is not None and f["launch_lon"] is not None:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [f["launch_lon"], f["launch_lat"]]},
                    "properties": {
                        "kind": "launch", "serial": f["serial"], "ftype": f["type"],
                        "first_seen": f["first_seen"],
                    },
                })
            if f["last_lat"] is None or f["last_lon"] is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [f["last_lon"], f["last_lat"]]},
                "properties": {
                    "kind": "flight", "serial": f["serial"], "ftype": f["type"],
                    "state": f["state"], "alt": f["last_alt"],
                },
            })
        for tr in store.latest_tracks_for_active():
            # track rows store [lat, lon, alt]; GeoJSON wants [lon, lat]
            coords = [[pt[1], pt[0]] for pt in tr["track"]]
            if len(coords) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"kind": "track", "serial": tr["serial"]},
            })
        for pp in store.latest_paths_for_active():
            # path rows store [lat, lon, alt]; GeoJSON wants [lon, lat]
            coords = [[pt[1], pt[0]] for pt in pp["path"]]
            if len(coords) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "kind": "path", "serial": pp["serial"], "source": pp["source"],
                    "eta": pp["land_eta"],
                },
            })
        for p in store.latest_predictions_for_active():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p["land_lon"], p["land_lat"]]},
                "properties": {
                    "kind": "prediction", "serial": p["serial"], "source": p["source"],
                    "eta": p["land_eta"], "uncertainty_radius_km": p["uncertainty_radius_km"],
                },
            })
        for lnd in store.recent_landings(limit=landings_limit):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lnd["land_lon"], lnd["land_lat"]]},
                "properties": {
                    "kind": "landing", "serial": lnd["serial"], "ftype": lnd["flight_type"],
                    "alt": lnd["land_alt"], "landed_at": lnd["landed_at"],
                    "detected_by": lnd["detected_by"],
                },
            })
        for s in store.list_subscribers(active_only=False):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [s.lon, s.lat]},
                "properties": {
                    "kind": "subscriber", "id": s.id, "name": s.name,
                    "radius_km": s.radius_km, "active": s.active,
                    "notify": s.notify_enabled,
                },
            })
        return {"type": "FeatureCollection", "features": features}

    @app.get("/api/accuracy")
    def accuracy(limit: int = 200):
        """Score saved predictions against actual recorded landings.
        Returns aggregate metrics plus a per-flight final-error list."""
        from windfall.replay import accuracy_from_store, aggregate

        results = accuracy_from_store(store, limit=limit)
        flights = [{
            "serial": r.serial, "launch_day": r.launch_day,
            "truth_lat": r.truth_lat, "truth_lon": r.truth_lon,
            "final_error_km": r.final_error_km, "n_predictions": r.n_predictions,
        } for r in results]
        return {"summary": _metrics_json(aggregate(results)) if results else None,
                "flights": flights}

    @app.delete("/api/accuracy")
    def clear_accuracy():
        """Wipe the accuracy history (recorded landings + finished flights'
        prediction series). Active flights' current predictions are kept."""
        deleted = store.clear_accuracy()
        log.info("cleared accuracy history: %s", deleted)
        return {"deleted": deleted}

    # ---- watched-location (subscriber) CRUD ------------------------------
    @app.get("/api/subscribers")
    def list_subs(active_only: bool = False):
        return [_serialize_subscriber(s) for s in store.list_subscribers(active_only=active_only)]

    @app.get("/api/subscribers/{sub_id}")
    def get_sub(sub_id: int):
        s = store.get_subscriber(sub_id)
        if s is None:
            raise HTTPException(status_code=404, detail="subscriber not found")
        return _serialize_subscriber(s)

    @app.post("/api/subscribers", status_code=201)
    def create_sub(payload: SubscriberIn):
        sid = store.add_subscriber(_to_subscriber(payload))
        return _serialize_subscriber(store.get_subscriber(sid))

    @app.put("/api/subscribers/{sub_id}")
    def update_sub(sub_id: int, payload: SubscriberIn):
        if not store.update_subscriber(_to_subscriber(payload, sub_id=sub_id)):
            raise HTTPException(status_code=404, detail="subscriber not found")
        return _serialize_subscriber(store.get_subscriber(sub_id))

    @app.post("/api/subscribers/{sub_id}/active")
    def set_active(sub_id: int, payload: ActiveIn):
        if not store.set_subscriber_active(sub_id, payload.active):
            raise HTTPException(status_code=404, detail="subscriber not found")
        return _serialize_subscriber(store.get_subscriber(sub_id))

    @app.delete("/api/subscribers/{sub_id}")
    def delete_sub(sub_id: int):
        if not store.delete_subscriber(sub_id):
            raise HTTPException(status_code=404, detail="subscriber not found")
        return {"deleted": sub_id}

    # ---- ntfy tokens (write-only: a saved value never leaves the server) --
    class TokenIn(BaseModel):
        # The one place a raw secret is accepted, behind the session cookie.
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        token: str = Field(min_length=1)

    def _valid_token_name(name: str) -> str:
        # Path-segment-safe and dropdown-friendly; same charset the UI enforces.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
            raise HTTPException(
                status_code=422,
                detail="token name must be 1-64 chars of letters, digits, _ . -")
        return name

    @app.get("/api/tokens")
    def list_tokens():
        """Saved-token metadata (name, last-4 hint, reference count) - values
        are never returned by any route."""
        return store.list_ntfy_tokens()

    @app.put("/api/tokens/{name}")
    def put_token(name: str, payload: TokenIn):
        """Save or replace a token. The daemon resolves tokens per send, so a
        new/rotated value takes effect without a restart."""
        store.set_ntfy_token(_valid_token_name(name), payload.token)
        return next(t for t in store.list_ntfy_tokens() if t["name"] == name)

    @app.delete("/api/tokens/{name}")
    def delete_token(name: str):
        refs = store.ntfy_token_refs(name)
        if refs:
            raise HTTPException(
                status_code=409,
                detail=f"token {name!r} is used by {refs} watched location(s) - "
                       "point them at another token first")
        if not store.delete_ntfy_token(name):
            raise HTTPException(status_code=404, detail="token not found")
        return {"deleted": name}

    @app.post("/api/test-ntfy")
    def test_ntfy(payload: TestNtfyIn):
        """Send a one-off test notification so a user can confirm an ntfy setup
        before saving a watched location. A token is used only by *reference*
        (a saved token's name), resolved at send time exactly like a live alert
        - no raw token is ever accepted in the request body."""
        from .notify import HttpNtfySink, NtfyMessage

        token_ref = payload.ntfy_token_ref or None
        # If no token is saved under that name, send unauthenticated (works for
        # public topics) and warn - rather than pretend a private topic is
        # reachable when live alerts would also lack the token.
        token_missing = bool(token_ref) and store.get_ntfy_token(token_ref) is None
        msg = NtfyMessage(
            server=payload.ntfy_server, topic=payload.ntfy_topic,
            title="Tally-ho test", priority=3, tags=["balloon", "white_check_mark"],
            body="✅ Test notification from your tally-ho dashboard - "
                 "ntfy is wired up correctly.",
            token_ref=None if token_missing else token_ref,
        )
        sink = (ntfy_sink if ntfy_sink is not None
                else HttpNtfySink(token_lookup=store.get_ntfy_token))
        if not sink.send(msg):
            detail = ("ntfy server didn't accept the message - check the "
                      "server URL and topic")
            if token_ref and not token_missing:
                detail += (f", and that the token {token_ref!r} is "
                           "valid for this topic")
            raise HTTPException(status_code=502, detail=detail + ".")
        note = None
        if token_missing:
            note = (f"sent without auth - no token named {token_ref!r} is saved, "
                    "so a private topic would be rejected. Save it under ntfy "
                    "tokens before relying on alerts.")
        return {"ok": True, "note": note}

    return app


def build_server(cfg: Config, store: Store, host: str = "127.0.0.1", port: int = 8080):
    """Build a (not-yet-running) ``uvicorn.Server`` for the dashboard, sharing the
    given ``store``. Call ``.run()`` on it to serve - in the main thread for the
    standalone ``tallyho web``, or on a daemon thread inside ``tallyho run`` (the
    Store is thread-safe, so both the daemon and the UI can share one)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "the web UI needs uvicorn; install it with: pip install '.[api]'"
        ) from exc
    app = create_app(cfg, store)
    config = uvicorn.Config(app, host=host, port=port, log_level=cfg.log_level.lower())
    return uvicorn.Server(config)


def run_web(cfg: Config, host: str = "127.0.0.1", port: int = 8080) -> int:  # pragma: no cover - needs network
    """Run the dashboard with uvicorn as a standalone process (own DB handle)."""
    store = Store(cfg.db_path)
    if store.count_users() == 0:
        log.error("no account yet - run 'tallyho run' once to complete first-run setup")
        store.close()
        return 2
    server = build_server(cfg, store, host=host, port=port)
    log.info("serving tally-ho web UI on http://%s:%d (login required)", host, port)
    server.run()  # blocks until interrupted
    return 0
