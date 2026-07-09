"""Configuration loading.

The prediction-engine knobs (tracker/profile/descent/integrator/ensemble/
uncertainty/predict/dem/gfs) live in :mod:`windfall.config`; tally-ho's
:class:`Config` subclasses it with the service-level sections (ingest, alerts,
web, storage), so one flat TOML file still configures everything and the
engine sees identical tuning in the live app and in offline backtests.

Config comes from a TOML file plus environment-variable overrides
(``TALLYHO_<SECTION>_<KEY>``). Secrets (ntfy tokens) are NEVER stored here -
they are saved via the web UI (or ``tallyho token set``) into the DB's
write-only token table, and subscribers reference them by name.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import windfall.config as engine_config

log = logging.getLogger(__name__)
from windfall.config import (  # noqa: F401  (re-exported: app code + tests use these)
    DEMConfig,
    DescentConfig,
    EnsembleConfig,
    GFSConfig,
    IntegratorConfig,
    PredictConfig,
    ProfileConfig,
    TrackerConfig,
    UncertaintyConfig,
)


@dataclass(slots=True)
class IngestConfig:
    # Subscribe to all radiosondes; filter client-side.
    reorder_hold_seconds: float = 3.0      # reorder window
    dedup_ttl_seconds: float = 60.0        # how long to remember dedup keys
    reconnect_base_seconds: float = 1.0    # exponential backoff base
    reconnect_max_seconds: float = 120.0   # backoff ceiling
    reconnect_jitter: float = 0.3          # +/- fraction jitter


@dataclass(slots=True)
class ROIConfig:
    capture_margin_km: float = 300.0      # capture ROI = subscriber circles + this


@dataclass(slots=True)
class NotifyConfig:
    map_url_template: str = "https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=13/{lat}/{lon}"
    track_url_template: str = "https://sondehub.org/{serial}"
    update_move_km: float = 5.0           # UPDATE alert if landing moves > this
    update_throttle_seconds: float = 600.0  # >=1 per 10 min per flight/sub
    request_timeout_seconds: float = 10.0


@dataclass(slots=True)
class WebConfig:
    # Local dashboard / onboarding UI (optional `api` extra). Login required
    # (account created by the first-run setup wizard), so LAN binds are fine -
    # the Docker image binds 0.0.0.0 inside the container. Sessions travel
    # plain HTTP; put a TLS reverse proxy in front for anything beyond a LAN.
    enabled: bool = True   # serve the dashboard in-process from `tallyho run`
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass(slots=True)
class Config(engine_config.Config):
    """Engine config + the live service's sections."""

    db_path: str = "data/tallyho.db"
    health_file: str = "data/heartbeat"   # last-frame timestamp for the healthcheck
    health_stale_seconds: float = 120.0   # healthcheck: last frame must be newer
    tick_seconds: float = 15.0            # maintenance cadence (timeouts, heartbeat)
    subscriber_reload_seconds: float = 300.0  # re-read subscribers/ROI cadence
    # IANA timezone for human-facing times (dashboard clock + ntfy ETAs). Empty
    # means "auto": honor the standard ``TZ`` env var (set via docker-compose),
    # else UTC. See :func:`display_tz_name`.
    display_tz: str = ""
    ingest: IngestConfig = field(default_factory=IngestConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    web: WebConfig = field(default_factory=WebConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a TOML file (if given/present), then env overrides.

    Env override format: ``TALLYHO_<SECTION>_<KEY>`` (e.g.
    ``TALLYHO_INTEGRATOR_DT_SECONDS=2``). Top-level keys use
    ``TALLYHO_<KEY>`` (e.g. ``TALLYHO_DB_PATH``). Engine sections are
    addressed exactly like the app's own.
    """
    return engine_config.load_config(path, cls=Config, env_prefix="TALLYHO_")


def display_tz_name(cfg: Config | None = None) -> str:
    """Validated IANA timezone name for human-facing times.

    Resolution order: an explicit ``cfg.display_tz`` (``TALLYHO_DISPLAY_TZ``),
    then the standard ``TZ`` env var (how docker-compose passes it), then UTC.
    An unset or unrecognized name (e.g. missing tzdata) falls back to ``"UTC"``
    with a warning, so a typo degrades gracefully instead of crashing the app.
    """
    name = ((cfg.display_tz if cfg else "") or os.environ.get("TZ") or "").strip()
    # Strip a leading ':' - `TZ=:/etc/localtime` and `TZ=:America/New_York` are
    # both valid for glibc, but ZoneInfo wants the bare name.
    name = name.lstrip(":")
    if not name or name.upper() == "UTC":
        return "UTC"
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning(
            "unknown timezone %r - set TZ to an IANA name like 'America/New_York'; "
            "showing times in UTC", name)
        return "UTC"
    return name


def display_tzinfo(cfg: Config | None = None) -> tzinfo:
    """:class:`~datetime.tzinfo` for :func:`display_tz_name`. UTC uses the stdlib
    fixed offset so the common case needs no tzdata on the host/image."""
    name = display_tz_name(cfg)
    return timezone.utc if name == "UTC" else ZoneInfo(name)
