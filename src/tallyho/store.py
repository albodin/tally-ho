"""SQLite persistence.

A single thread-safe store backing flights, wind profiles, subscribers,
predictions, and the alert de-dup table. Ingest can run on Paho's callback
thread, so every connection use is guarded by a lock and the connection is
opened with ``check_same_thread=False``.

ntfy bearer tokens live in their own table, keyed by the name subscribers
reference (``ntfy_token_ref``). They are write-only above this module: the web
API accepts a value but never returns one, and only the send path reads it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from .models import AlertType, Prediction, Subscriber

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subscribers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    radius_km     REAL NOT NULL,
    ntfy_server   TEXT NOT NULL,
    ntfy_topic    TEXT NOT NULL,
    ntfy_token_ref TEXT,
    units         TEXT NOT NULL DEFAULT 'metric',
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);

-- ntfy bearer tokens, keyed by the name subscribers reference (ntfy_token_ref).
-- Write-only outside this module: no API response ever carries `token`.
CREATE TABLE IF NOT EXISTS ntfy_tokens (
    name       TEXT PRIMARY KEY,
    token      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flights (
    serial      TEXT NOT NULL,
    launch_day  TEXT NOT NULL,
    type        TEXT,
    state       TEXT NOT NULL,
    first_seen  TEXT,
    last_seen   TEXT,
    launch_lat  REAL,
    launch_lon  REAL,
    burst_alt   REAL,
    burst_t     REAL,
    descent_b   REAL,
    max_alt     REAL,
    last_lat    REAL,
    last_lon    REAL,
    last_alt    REAL,
    PRIMARY KEY (serial, launch_day)
);

CREATE TABLE IF NOT EXISTS wind_profiles (
    serial       TEXT NOT NULL,
    launch_day   TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    PRIMARY KEY (serial, launch_day)
);

-- Observed descent samples ([t, alt, v_obs, rho] each), checkpointed during
-- descent so a daemon restart resumes the ballistic fit mid-flight instead of
-- degrading to the single-point shortcut.
CREATE TABLE IF NOT EXISTS descent_samples (
    serial       TEXT NOT NULL,
    launch_day   TEXT NOT NULL,
    samples_json TEXT NOT NULL,
    PRIMARY KEY (serial, launch_day)
);

CREATE TABLE IF NOT EXISTS predictions (
    serial                TEXT NOT NULL,
    launch_day            TEXT NOT NULL,
    predicted_at          TEXT NOT NULL,
    land_lat              REAL NOT NULL,
    land_lon              REAL NOT NULL,
    land_eta              TEXT NOT NULL,
    source                TEXT NOT NULL,
    uncertainty_radius_km REAL NOT NULL,
    alt_at_pred           REAL
);
CREATE INDEX IF NOT EXISTS idx_pred_flight
    ON predictions (serial, launch_day, predicted_at);

-- Latest predicted trajectory per flight, for the map (one row per flight, the
-- newest path wins). Kept out of the predictions time-series to keep that table
-- lean for the accuracy harness.
CREATE TABLE IF NOT EXISTS prediction_paths (
    serial       TEXT NOT NULL,
    launch_day   TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    source       TEXT NOT NULL,
    land_eta     TEXT,
    path_json    TEXT NOT NULL,
    PRIMARY KEY (serial, launch_day)
);

-- The actual flown track (downsampled breadcrumb, launch → latest fix) so the
-- map can draw where each sonde has really been, not just its current point.
-- Dense telemetry is thinned at write time (see tracker), so rows stay modest.
CREATE TABLE IF NOT EXISTS flight_track (
    serial      TEXT NOT NULL,
    launch_day  TEXT NOT NULL,
    t           REAL NOT NULL,           -- sonde time, epoch seconds (UTC)
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    alt         REAL,
    PRIMARY KEY (serial, launch_day, t)
);

-- Actual landing position (ground truth) once a flight is LANDED, so live
-- accuracy can be measured against real outcomes.
CREATE TABLE IF NOT EXISTS landings (
    serial      TEXT NOT NULL,
    launch_day  TEXT NOT NULL,
    land_lat    REAL NOT NULL,
    land_lon    REAL NOT NULL,
    land_alt    REAL,
    landed_at   TEXT,
    detected_by TEXT,            -- 'telemetry' (saw it near ground) | 'timeout'
    PRIMARY KEY (serial, launch_day)
);

CREATE TABLE IF NOT EXISTS alerts (
    subscriber_id INTEGER NOT NULL,
    serial        TEXT NOT NULL,
    launch_day    TEXT NOT NULL,
    alert_type    TEXT NOT NULL,
    distance_km   REAL,
    land_lat      REAL,
    land_lon      REAL,
    sent_at       TEXT NOT NULL,
    UNIQUE (subscriber_id, serial, launch_day, alert_type)
);

-- Web-UI accounts (the dashboard requires a login; created by the first-run
-- setup wizard).
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- Small app-owned key/value state (e.g. the session-cookie signing secret).
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# The schema version of a database file lives in ``PRAGMA user_version``.
# To change the schema: bump _SCHEMA_VERSION, edit _SCHEMA (fresh databases
# get the latest shape directly), and register the same change in _MIGRATIONS
# (existing databases replay it). A migration is the SQL upgrading from the
# previous version, or a callable taking the connection when it needs Python
# (data backfills, table rebuilds).
_SCHEMA_VERSION = 1
_MIGRATIONS: dict[int, str | Callable[[sqlite3.Connection], None]] = {
    # 2: "ALTER TABLE flights ADD COLUMN example REAL",
}


def _like_prefix(prefix: str) -> str:
    """``prefix%`` with LIKE metacharacters escaped (pair with ``ESCAPE '\\\\'``).
    Sonde types arrive from external telemetry, so a stray ``%``/``_`` must
    match literally rather than broaden the climatology scans."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class Store:
    def __init__(self, path: str | Path = "data/tallyho.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # WAL allows one writer + many readers across processes (the daemon writes
        # flights/predictions/alerts while `tallyho web` writes subscribers). Retry
        # briefly on a transient writer lock instead of raising "database is locked".
        self._conn.execute("PRAGMA busy_timeout=5000")
        # Set by the web lifespan to EventBus.publish so writes ring the SSE
        # doorbell; None (the default) means writes publish nothing.
        self.on_change: Callable[[str], None] | None = None
        with self._lock:
            self._apply_schema()

    def _changed(self, name: str) -> None:
        cb = self.on_change            # read once; may be reassigned from another thread
        if cb is not None:
            cb(name)                   # must be thread-safe; EventBus.publish is

    def data_version(self) -> int:
        """PRAGMA data_version: unchanged by commits on THIS connection, bumped by
        commits from another connection/process (incl. WAL). Drives the standalone
        ``tallyho web`` cross-process watcher."""
        with self._lock:
            return self._conn.execute("PRAGMA data_version").fetchone()[0]

    def _apply_schema(self) -> None:
        """Create or upgrade the schema, one version at a time. Version 0 means
        a fresh file - or one from before versioning, whose shape already
        matches the v1 baseline - so it gets _SCHEMA (all IF NOT EXISTS) and is
        stamped current. The stamp advances per migration, so a failure partway
        resumes from the last completed step on the next open."""
        v = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if v > _SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} has schema version {v}, newer than this build's "
                f"{_SCHEMA_VERSION} - upgrade tally-ho or restore the matching database")
        if v == 0:
            self._conn.executescript(_SCHEMA)
            v = _SCHEMA_VERSION
        for n in range(v + 1, _SCHEMA_VERSION + 1):
            step = _MIGRATIONS[n]
            if callable(step):
                step(self._conn)
            else:
                self._conn.executescript(step)
            self._conn.execute(f"PRAGMA user_version = {n}")
        self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- flights ----------------------------------------------------------
    def upsert_flight(self, row: dict) -> None:
        cols = (
            "serial", "launch_day", "type", "state", "first_seen", "last_seen",
            "launch_lat", "launch_lon", "burst_alt", "burst_t", "descent_b",
            "max_alt", "last_lat", "last_lon", "last_alt",
        )
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("serial", "launch_day"))
        sql = (
            f"INSERT INTO flights ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(serial, launch_day) DO UPDATE SET {updates}"
        )
        with self._lock:
            self._conn.execute(sql, tuple(row.get(c) for c in cols))
            self._conn.commit()
        self._changed("flights")

    def get_flight(self, serial: str, launch_day: date) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM flights WHERE serial=? AND launch_day=?",
                (serial, launch_day.isoformat()),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def active_flights(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM flights WHERE state != 'LANDED'")
            return [dict(r) for r in cur.fetchall()]

    def recent_flights(self, limit: int = 200, include_landed: bool = True) -> list[dict]:
        """Flights ordered by last activity (for the dashboard tables)."""
        sql = "SELECT * FROM flights"
        if not include_landed:
            sql += " WHERE state != 'LANDED'"
        sql += " ORDER BY last_seen DESC LIMIT ?"
        with self._lock:
            cur = self._conn.execute(sql, (int(limit),))
            return [dict(r) for r in cur.fetchall()]

    # ---- climatology (learned priors) -------------------------------------
    def site_burst_alts(
        self, lat: float, lon: float, box_deg: float = 0.5,
        sonde_type: str | None = None, limit: int = 60,
    ) -> list[float]:
        """Observed burst altitudes of recent flights launched near (lat, lon),
        newest first - the raw material for the per-site burst prior."""
        sql = (
            "SELECT burst_alt FROM flights WHERE burst_alt IS NOT NULL "
            "AND launch_lat BETWEEN ? AND ? AND launch_lon BETWEEN ? AND ?"
        )
        args: list = [lat - box_deg, lat + box_deg, lon - box_deg, lon + box_deg]
        if sonde_type:
            sql += " AND type LIKE ? ESCAPE '\\'"
            args.append(_like_prefix(sonde_type.split("-")[0]))
        sql += " ORDER BY launch_day DESC, last_seen DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            cur = self._conn.execute(sql, args)
            return [r["burst_alt"] for r in cur.fetchall()]

    def type_descent_bs(self, family: str, limit: int = 60) -> list[float]:
        """Fitted ballistic constants of recent flights of a sonde family."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT descent_b FROM flights WHERE descent_b IS NOT NULL "
                "AND type LIKE ? ESCAPE '\\' "
                "ORDER BY launch_day DESC, last_seen DESC LIMIT ?",
                (_like_prefix(family), int(limit)),
            )
            return [r["descent_b"] for r in cur.fetchall()]

    # ---- wind profiles ----------------------------------------------------
    def save_profile(self, serial: str, launch_day: date, profile_json: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO wind_profiles (serial, launch_day, profile_json) "
                "VALUES (?, ?, ?) ON CONFLICT(serial, launch_day) "
                "DO UPDATE SET profile_json=excluded.profile_json",
                (serial, launch_day.isoformat(), json.dumps(profile_json)),
            )
            self._conn.commit()

    def load_profile(self, serial: str, launch_day: date) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT profile_json FROM wind_profiles WHERE serial=? AND launch_day=?",
                (serial, launch_day.isoformat()),
            )
            row = cur.fetchone()
        return json.loads(row["profile_json"]) if row else None

    # ---- descent samples (restart-safe ballistic fit) ----------------------
    def save_descent_samples(self, serial: str, launch_day: date, samples: list) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO descent_samples (serial, launch_day, samples_json) "
                "VALUES (?, ?, ?) ON CONFLICT(serial, launch_day) "
                "DO UPDATE SET samples_json=excluded.samples_json",
                (serial, launch_day.isoformat(), json.dumps(samples)),
            )
            self._conn.commit()

    def load_descent_samples(self, serial: str, launch_day: date) -> list | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT samples_json FROM descent_samples WHERE serial=? AND launch_day=?",
                (serial, launch_day.isoformat()),
            )
            row = cur.fetchone()
        return json.loads(row["samples_json"]) if row else None

    # ---- predictions ------------------------------------------------------
    def save_prediction(self, pred: Prediction) -> None:
        r = pred.to_row()
        with self._lock:
            self._conn.execute(
                "INSERT INTO predictions (serial, launch_day, predicted_at, land_lat, "
                "land_lon, land_eta, source, uncertainty_radius_km, alt_at_pred) "
                "VALUES (:serial, :launch_day, :predicted_at, :land_lat, :land_lon, "
                ":land_eta, :source, :uncertainty_radius_km, :alt_at_pred)",
                r,
            )
            self._conn.commit()
        self._changed("flights")
        # The predicted trajectory rides along in a separate latest-only table.
        if pred.path:
            self.save_prediction_path(pred)

    def save_prediction_path(self, pred: Prediction) -> None:
        """Upsert the newest predicted trajectory for a flight (for the map)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO prediction_paths (serial, launch_day, predicted_at, "
                "source, land_eta, path_json) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(serial, launch_day) DO UPDATE SET "
                "predicted_at=excluded.predicted_at, source=excluded.source, "
                "land_eta=excluded.land_eta, path_json=excluded.path_json",
                (pred.serial, pred.launch_day.isoformat(), pred.predicted_at.isoformat(),
                 pred.source.value, _iso(pred.land_eta),
                 json.dumps([list(p) for p in pred.path])),
            )
            self._conn.commit()
        self._changed("flights")

    def latest_prediction(self, serial: str, launch_day: date) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM predictions WHERE serial=? AND launch_day=? "
                "ORDER BY predicted_at DESC LIMIT 1",
                (serial, launch_day.isoformat()),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def predictions_for(self, serial: str, launch_day: date) -> list[dict]:
        """All predictions for a flight, oldest first (accuracy time-series)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM predictions WHERE serial=? AND launch_day=? "
                "ORDER BY predicted_at ASC",
                (serial, launch_day.isoformat()),
            )
            return [dict(r) for r in cur.fetchall()]

    def latest_predictions_for_active(self) -> list[dict]:
        """Newest prediction for each non-LANDED flight, in one query (for the map)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT p.* FROM predictions p "
                "JOIN flights f ON f.serial=p.serial AND f.launch_day=p.launch_day "
                "WHERE f.state != 'LANDED' AND p.predicted_at = ("
                "    SELECT MAX(p2.predicted_at) FROM predictions p2 "
                "    WHERE p2.serial=p.serial AND p2.launch_day=p.launch_day)"
            )
            return [dict(r) for r in cur.fetchall()]

    def latest_paths_for_active(self) -> list[dict]:
        """Latest predicted trajectory for each non-LANDED flight (for the map).
        Each row's ``path`` is decoded to a list of ``[lat, lon, alt]``."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT pp.* FROM prediction_paths pp "
                "JOIN flights f ON f.serial=pp.serial AND f.launch_day=pp.launch_day "
                "WHERE f.state != 'LANDED'"
            )
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["path"] = json.loads(d.pop("path_json"))
            out.append(d)
        return out

    # ---- flown track (actual breadcrumb, for the map) --------------------
    def append_track_point(
        self, serial: str, launch_day: date, t: float,
        lat: float, lon: float, alt: float | None,
    ) -> None:
        """Append one (downsampled) point to a flight's actual flown track.
        Idempotent on ``(serial, launch_day, t)`` so frame replays don't double up."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO flight_track (serial, launch_day, t, lat, lon, alt) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (serial, launch_day.isoformat(), t, lat, lon, alt),
            )
            self._conn.commit()
        self._changed("flights")

    def track_for(self, serial: str, launch_day: date) -> list[dict]:
        """The full flown track for one flight, oldest point first."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT t, lat, lon, alt FROM flight_track "
                "WHERE serial=? AND launch_day=? ORDER BY t ASC",
                (serial, launch_day.isoformat()),
            )
            return [dict(r) for r in cur.fetchall()]

    def latest_tracks_for_active(self) -> list[dict]:
        """Flown track for each non-LANDED flight (for the map). Each row's
        ``track`` is a list of ``[lat, lon, alt]`` ordered launch → latest."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT tr.serial, tr.launch_day, tr.lat, tr.lon, tr.alt "
                "FROM flight_track tr "
                "JOIN flights f ON f.serial=tr.serial AND f.launch_day=tr.launch_day "
                "WHERE f.state != 'LANDED' ORDER BY tr.serial, tr.launch_day, tr.t ASC"
            )
            rows = cur.fetchall()
        grouped: dict[tuple[str, str], dict] = {}
        for r in rows:
            key = (r["serial"], r["launch_day"])
            entry = grouped.setdefault(
                key, {"serial": r["serial"], "launch_day": r["launch_day"], "track": []})
            entry["track"].append([r["lat"], r["lon"], r["alt"]])
        return list(grouped.values())

    # ---- landings (ground truth) ------------------------------------------
    def record_landing(
        self, serial: str, launch_day: date, land_lat: float, land_lon: float,
        land_alt: float | None, landed_at: datetime | None, detected_by: str,
    ) -> None:
        """Persist the actual landing position. Upserts so a later, lower fix
        refines an earlier one for the same flight."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO landings (serial, launch_day, land_lat, land_lon, "
                "land_alt, landed_at, detected_by) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(serial, launch_day) DO UPDATE SET "
                "land_lat=excluded.land_lat, land_lon=excluded.land_lon, "
                "land_alt=excluded.land_alt, landed_at=excluded.landed_at, "
                "detected_by=excluded.detected_by",
                (serial, launch_day.isoformat(), land_lat, land_lon, land_alt,
                 _iso(landed_at), detected_by),
            )
            self._conn.commit()
        self._changed("accuracy")

    def get_landing(self, serial: str, launch_day: date) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM landings WHERE serial=? AND launch_day=?",
                (serial, launch_day.isoformat()),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def clear_accuracy(self) -> dict:
        """Wipe the accuracy history: every recorded landing plus the prediction
        time-series/paths of finished (LANDED) flights. Predictions of flights
        still in the air are kept so the live map keeps its current predicted
        landings. Returns the deleted row counts."""
        finished = "(SELECT serial, launch_day FROM flights WHERE state='LANDED')"
        with self._lock:
            n_landings = self._conn.execute("DELETE FROM landings").rowcount
            n_preds = self._conn.execute(
                f"DELETE FROM predictions WHERE (serial, launch_day) IN {finished}"
            ).rowcount
            self._conn.execute(
                f"DELETE FROM prediction_paths WHERE (serial, launch_day) IN {finished}")
            self._conn.commit()
        self._changed("accuracy")
        return {"landings": n_landings, "predictions": n_preds}

    def recent_landings(self, limit: int = 100) -> list[dict]:
        """Recently-recorded landings, newest first (for accuracy + the map)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT l.*, f.type AS flight_type FROM landings l "
                "LEFT JOIN flights f ON f.serial=l.serial AND f.launch_day=l.launch_day "
                "ORDER BY l.landed_at DESC LIMIT ?",
                (int(limit),),
            )
            return [dict(r) for r in cur.fetchall()]

    # ---- subscribers ------------------------------------------------------
    def add_subscriber(self, sub: Subscriber) -> int:
        created = sub.created_at or datetime.now(timezone.utc)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO subscribers (name, lat, lon, radius_km, ntfy_server, "
                "ntfy_topic, ntfy_token_ref, units, active, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sub.name, sub.lat, sub.lon, sub.radius_km, sub.ntfy_server,
                 sub.ntfy_topic, sub.ntfy_token_ref, sub.units, int(sub.active),
                 created.isoformat()),
            )
            self._conn.commit()
            sid = int(cur.lastrowid)
        self._changed("subscribers")
        return sid

    def list_subscribers(self, active_only: bool = True) -> list[Subscriber]:
        sql = "SELECT * FROM subscribers"
        if active_only:
            sql += " WHERE active=1"
        with self._lock:
            cur = self._conn.execute(sql)
            rows = cur.fetchall()
        return [self._sub_from_row(r) for r in rows]

    def set_subscriber_active(self, sub_id: int, active: bool) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE subscribers SET active=? WHERE id=?", (int(active), sub_id)
            )
            self._conn.commit()
            changed = cur.rowcount > 0
        self._changed("subscribers")
        return changed

    def get_subscriber(self, sub_id: int) -> Subscriber | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM subscribers WHERE id=?", (sub_id,))
            row = cur.fetchone()
        return self._sub_from_row(row) if row else None

    def update_subscriber(self, sub: Subscriber) -> bool:
        """Update all editable fields by id (created_at is left untouched).
        Returns False if no row with that id exists."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE subscribers SET name=?, lat=?, lon=?, radius_km=?, "
                "ntfy_server=?, ntfy_topic=?, ntfy_token_ref=?, units=?, active=? WHERE id=?",
                (sub.name, sub.lat, sub.lon, sub.radius_km, sub.ntfy_server,
                 sub.ntfy_topic, sub.ntfy_token_ref, sub.units, int(sub.active), sub.id),
            )
            self._conn.commit()
            changed = cur.rowcount > 0
        self._changed("subscribers")
        return changed

    def delete_subscriber(self, sub_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM subscribers WHERE id=?", (sub_id,))
            self._conn.commit()
            changed = cur.rowcount > 0
        self._changed("subscribers")
        return changed

    @staticmethod
    def _sub_from_row(r: sqlite3.Row) -> Subscriber:
        return Subscriber(
            id=r["id"], name=r["name"], lat=r["lat"], lon=r["lon"],
            radius_km=r["radius_km"], ntfy_server=r["ntfy_server"],
            ntfy_topic=r["ntfy_topic"], ntfy_token_ref=r["ntfy_token_ref"],
            units=r["units"], active=bool(r["active"]), created_at=_dt(r["created_at"]),
        )

    # ---- alert de-dup ----------------------------------------
    def record_alert(
        self,
        subscriber_id: int,
        serial: str,
        launch_day: date,
        alert_type: AlertType,
        distance_km: float | None,
        land_lat: float | None,
        land_lon: float | None,
        sent_at: datetime,
    ) -> bool:
        """Insert an alert. Returns True if newly recorded, False if it was a
        duplicate (same subscriber/serial/launch_day/type already sent)."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO alerts (subscriber_id, serial, launch_day, "
                "alert_type, distance_km, land_lat, land_lon, sent_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (subscriber_id, serial, launch_day.isoformat(), alert_type.value,
                 distance_km, land_lat, land_lon, _iso(sent_at)),
            )
            self._conn.commit()
            changed = cur.rowcount > 0
        self._changed("alerts")
        return changed

    def get_alert(
        self, subscriber_id: int, serial: str, launch_day: date, alert_type: AlertType
    ) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM alerts WHERE subscriber_id=? AND serial=? "
                "AND launch_day=? AND alert_type=?",
                (subscriber_id, serial, launch_day.isoformat(), alert_type.value),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def upsert_alert(
        self,
        subscriber_id: int,
        serial: str,
        launch_day: date,
        alert_type: AlertType,
        distance_km: float | None,
        land_lat: float | None,
        land_lon: float | None,
        sent_at: datetime,
    ) -> None:
        """Insert or refresh an alert row (used for repeatable UPDATE alerts)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO alerts (subscriber_id, serial, launch_day, alert_type, "
                "distance_km, land_lat, land_lon, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(subscriber_id, serial, launch_day, alert_type) DO UPDATE SET "
                "distance_km=excluded.distance_km, land_lat=excluded.land_lat, "
                "land_lon=excluded.land_lon, sent_at=excluded.sent_at",
                (subscriber_id, serial, launch_day.isoformat(), alert_type.value,
                 distance_km, land_lat, land_lon, _iso(sent_at)),
            )
            self._conn.commit()
        self._changed("alerts")

    def last_alert_at(
        self, subscriber_id: int, serial: str, launch_day: date, alert_type: AlertType
    ) -> datetime | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT sent_at FROM alerts WHERE subscriber_id=? AND serial=? "
                "AND launch_day=? AND alert_type=?",
                (subscriber_id, serial, launch_day.isoformat(), alert_type.value),
            )
            row = cur.fetchone()
        return _dt(row["sent_at"]) if row else None

    def update_alert_time(
        self, subscriber_id: int, serial: str, launch_day: date,
        alert_type: AlertType, sent_at: datetime,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE alerts SET sent_at=? WHERE subscriber_id=? AND serial=? "
                "AND launch_day=? AND alert_type=?",
                (_iso(sent_at), subscriber_id, serial, launch_day.isoformat(),
                 alert_type.value),
            )
            self._conn.commit()
        self._changed("alerts")

    def clear_alerts(self) -> int:
        """Wipe the recent-alerts history. Rows for flights still in the air are
        kept: the alerts table is also the de-dup record, so deleting an active
        flight's rows would re-send its INBOUND/UPDATE pushes. Returns the
        number of deleted rows."""
        with self._lock:
            n = self._conn.execute(
                "DELETE FROM alerts WHERE (serial, launch_day) NOT IN "
                "(SELECT serial, launch_day FROM flights WHERE state != 'LANDED')"
            ).rowcount
            self._conn.commit()
        self._changed("alerts")
        return n

    def recent_alerts(self, limit: int = 100) -> list[dict]:
        """Recently-sent alerts joined to subscriber name + flight type (dashboard)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT a.*, s.name AS subscriber_name, f.type AS flight_type "
                "FROM alerts a "
                "LEFT JOIN subscribers s ON s.id = a.subscriber_id "
                "LEFT JOIN flights f ON f.serial = a.serial AND f.launch_day = a.launch_day "
                "ORDER BY a.sent_at DESC LIMIT ?",
                (int(limit),),
            )
            return [dict(r) for r in cur.fetchall()]

    # ---- web-UI accounts + app state ---------------------------------------
    def add_user(self, username: str, password_hash: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, _iso(datetime.now(timezone.utc))),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_user_hash(self, username: str) -> str | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT password_hash FROM users WHERE username=?", (username,))
            row = cur.fetchone()
        return row["password_hash"] if row else None

    def count_users(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    # ---- ntfy tokens --------------------------------------------------------
    def set_ntfy_token(self, name: str, token: str) -> None:
        """Save or replace the bearer token for ``name``."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO ntfy_tokens (name, token, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET token=excluded.token, "
                "updated_at=excluded.updated_at",
                (name, token, _iso(datetime.now(timezone.utc))),
            )
            self._conn.commit()
        self._changed("tokens")

    def get_ntfy_token(self, name: str) -> str | None:
        """The token value - for the send path only, never for API responses."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT token FROM ntfy_tokens WHERE name=?", (name,))
            row = cur.fetchone()
        return row["token"] if row else None

    def list_ntfy_tokens(self) -> list[dict]:
        """Token metadata for the UI: name, a last-4 hint, when it was last set,
        and how many subscribers reference it - never the value itself."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.name, t.token, t.updated_at, "
                "(SELECT COUNT(*) FROM subscribers s WHERE s.ntfy_token_ref = t.name) "
                "AS refs FROM ntfy_tokens t ORDER BY t.name").fetchall()
        return [{"name": r["name"], "hint": "…" + r["token"][-4:],
                 "updated_at": r["updated_at"], "refs": r["refs"]} for r in rows]

    def delete_ntfy_token(self, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM ntfy_tokens WHERE name=?", (name,))
            self._conn.commit()
            changed = cur.rowcount > 0
        self._changed("tokens")
        return changed

    def ntfy_token_refs(self, name: str) -> int:
        """How many subscribers reference this token name (delete protection)."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM subscribers WHERE ntfy_token_ref=?",
                (name,)).fetchone()[0]

    def get_kv(self, key: str) -> str | None:
        with self._lock:
            cur = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,))
            row = cur.fetchone()
        return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()
