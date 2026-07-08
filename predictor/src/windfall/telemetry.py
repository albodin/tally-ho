"""Tolerant telemetry parsing.

The wire format is inconsistent: numeric fields arrive as JSON strings *or*
numbers depending on the uploader, and the ``-9999`` family means "absent".
This module coerces every numeric field through a tolerant funnel and rejects
frames missing the mandatory fields.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .geo import normalize_lon
from .models import Frame

# Any value at or below this is treated as the "-9999 family" absent sentinel
# (seen in the wild as "-9999.0"). No legitimate radiosonde field - altitude,
# temperature, velocity, pressure, lat/lon - reaches this magnitude, so the
# single threshold is safe across all fields.
SENTINEL_THRESHOLD = -9990.0


class FrameError(ValueError):
    """Raised when a frame cannot be parsed into a usable :class:`Frame`."""


def coerce_float(value: object) -> float | None:
    """float() funnel for both ``"44.2"`` and ``44.2``; sentinel/unparseable → None."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; never a telemetry number
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if f <= SENTINEL_THRESHOLD:
        return None
    return f


def coerce_int(value: object) -> int | None:
    """Coerce to int via the float funnel (handles ``"123"``, ``123.0``)."""
    f = coerce_float(value)
    if f is None:
        return None
    return int(f)


def parse_datetime(value: object) -> datetime | None:
    """Parse the sonde ``datetime`` (UTC, ISO-8601, possibly with ``Z`` and
    microseconds). Returns a tz-aware UTC datetime or None."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_frame(msg: dict) -> Frame:
    """Parse one raw telemetry dict into a validated :class:`Frame`.

    Raises :class:`FrameError` if any mandatory field is missing/invalid
    (the spec's "reliably present" set). Optional fields become None on absence or
    sentinel.
    """
    serial = msg.get("serial")
    if not isinstance(serial, str) or not serial.strip():
        raise FrameError("missing/invalid serial")
    serial = serial.strip()

    lat = coerce_float(msg.get("lat"))
    lon = coerce_float(msg.get("lon"))
    alt = coerce_float(msg.get("alt"))
    if lat is None or lon is None or alt is None:
        raise FrameError(f"missing lat/lon/alt for {serial}")
    if not (-90.0 <= lat <= 90.0):
        raise FrameError(f"lat out of range: {lat}")
    lon = normalize_lon(lon)

    dt = parse_datetime(msg.get("datetime"))
    if dt is None:
        raise FrameError(f"missing/invalid datetime for {serial}")

    return Frame(
        serial=serial,
        lat=lat,
        lon=lon,
        alt=alt,
        t=dt.timestamp(),
        dt=dt,
        frame=coerce_int(msg.get("frame")),
        type=_str(msg.get("type")),
        subtype=_str(msg.get("subtype")),
        manufacturer=_str(msg.get("manufacturer")),
        software_name=_str(msg.get("software_name")),
        uploader_callsign=_str(msg.get("uploader_callsign")),
        vel_v=coerce_float(msg.get("vel_v")),
        vel_h=coerce_float(msg.get("vel_h")),
        heading=coerce_float(msg.get("heading")),
        temp=coerce_float(msg.get("temp")),
        humidity=coerce_float(msg.get("humidity")),
        pressure=coerce_float(msg.get("pressure")),
        sats=coerce_float(msg.get("sats")),
        batt=coerce_float(msg.get("batt")),
        burst_timer=coerce_float(msg.get("burst_timer")),
        frequency=coerce_float(msg.get("frequency")),
    )


def try_parse_frame(msg: dict) -> Frame | None:
    """Non-raising variant for stream ingest - returns None on bad frames."""
    try:
        return parse_frame(msg)
    except FrameError:
        return None


def _str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return str(value)
