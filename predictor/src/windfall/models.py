"""Shared data contracts of the prediction engine.

These dataclasses are the canonical types passed between modules. Keeping them
dependency-free (stdlib only) makes the whole core importable without the heavy
I/O extras.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, date, timezone

# Earth radius used throughout (mean radius, metres).
EARTH_RADIUS_M = 6_371_000.0


class FlightState(str, enum.Enum):
    """Per-flight state."""

    ASCENT = "ASCENT"
    FLOAT = "FLOAT"
    DESCENT = "DESCENT"
    LANDED = "LANDED"


class PredictionSource(str, enum.Enum):
    MEASURED = "measured"
    GFS = "gfs"
    EXTRAPOLATION = "extrapolation"


@dataclass(slots=True)
class Frame:
    """One parsed, validated telemetry frame.

    All numeric sentinel (-9999 family) and unparseable values have already been
    converted to ``None`` by :mod:`windfall.telemetry`. ``t`` is sonde time as
    epoch seconds (UTC), derived from ``datetime`` - never ``time_received``.
    """

    serial: str
    lat: float
    lon: float
    alt: float
    t: float                      # epoch seconds, UTC (from sonde `datetime`)
    dt: datetime                  # tz-aware UTC datetime
    frame: int | None = None      # sequence number, may be absent/unreliable
    type: str | None = None
    subtype: str | None = None
    manufacturer: str | None = None
    software_name: str | None = None
    uploader_callsign: str | None = None
    # Optional / per-type kinematics & sensors (may be None).
    vel_v: float | None = None
    vel_h: float | None = None
    heading: float | None = None
    temp: float | None = None     # Celsius
    humidity: float | None = None
    pressure: float | None = None  # hPa
    sats: float | None = None
    batt: float | None = None
    burst_timer: float | None = None
    frequency: float | None = None

    @property
    def dedup_key(self) -> tuple[str, int | str]:
        """Key for de-duplication: (serial, frame) preferred,
        falling back to (serial, datetime-iso) when frame is unreliable."""
        if self.frame is not None:
            return (self.serial, int(self.frame))
        return (self.serial, self.dt.isoformat())


@dataclass(slots=True)
class WindBin:
    """A single altitude bin of the measured profile."""

    alt: float                    # bin centre, metres
    u: float                      # eastward wind, m/s
    v: float                      # northward wind, m/s
    rho: float | None = None      # air density, kg/m^3 (None → use ISA)
    n: int = 0                    # samples averaged into this bin
    # Where/when the samples were taken (means) - drives the measured→GFS blend:
    # trust in this bin decays with distance from here and with age.
    lat: float | None = None
    lon: float | None = None
    t: float | None = None        # epoch seconds

    def to_json(self) -> dict:
        return {"alt": self.alt, "u": self.u, "v": self.v, "rho": self.rho, "n": self.n,
                "lat": self.lat, "lon": self.lon, "t": self.t}

    @classmethod
    def from_json(cls, d: dict) -> "WindBin":
        return cls(alt=d["alt"], u=d["u"], v=d["v"], rho=d.get("rho"), n=d.get("n", 0),
                   lat=d.get("lat"), lon=d.get("lon"), t=d.get("t"))


@dataclass(slots=True)
class DescentSample:
    """One observed descent point feeding the ballistic fit."""

    t: float           # epoch seconds
    alt: float         # m
    v_obs: float       # observed descent speed, m/s (positive down)
    rho: float         # density at this sample, kg/m^3


# A sampled trajectory point: (lat_deg, lon_deg, alt_m). Used for the predicted
# path drawn on the map - the polyline from the sonde's current
# position down to the predicted landing.
TrackPoint = tuple[float, float, float]


@dataclass(slots=True)
class Prediction:
    """Output of the predictor."""

    serial: str
    launch_day: date
    predicted_at: datetime
    land_lat: float
    land_lon: float
    land_eta: datetime
    source: PredictionSource
    uncertainty_radius_km: float
    # Sonde altitude (m) at the moment this prediction was made - lets the
    # accuracy harness bucket error by altitude-at-prediction on live data
    # exactly as the offline replay does. Optional/back-compatible.
    alt_at_pred: float | None = None
    # Predicted burst point (pre-burst predictions only): where the modelled
    # ascent leg tops out before the descent. None once the flight is already
    # descending - its burst is observed, not predicted. Rides with the path
    # into ``prediction_paths`` (not the predictions time-series row).
    burst_lat: float | None = None
    burst_lon: float | None = None
    burst_alt: float | None = None
    # Sampled predicted trajectory (current position → landing) for the map.
    # Not persisted in the predictions row; the latest path is upserted into the
    # separate ``prediction_paths`` table by the store.
    path: list[TrackPoint] | None = None

    def to_row(self) -> dict:
        return {
            "serial": self.serial,
            "launch_day": self.launch_day.isoformat(),
            "predicted_at": self.predicted_at.isoformat(),
            "land_lat": self.land_lat,
            "land_lon": self.land_lon,
            "land_eta": self.land_eta.isoformat(),
            "source": self.source.value,
            "uncertainty_radius_km": self.uncertainty_radius_km,
            "alt_at_pred": self.alt_at_pred,
        }


@dataclass(slots=True)
class Landing:
    """Raw result of the descent integrator."""

    lat: float                    # degrees
    lon: float                    # degrees
    eta: datetime                 # UTC
    steps: int
    sim_seconds: float
    measured_fraction: float      # fraction of remaining column that was measured
    ok: bool = True
    reason: str = ""              # why it failed, if ok is False
    # Optional sampled trajectory (lat, lon, alt) from start to ground crossing,
    # populated only when ``integrate_descent(capture_path=True)``.
    path: list[TrackPoint] | None = None


def launch_day_of(dt: datetime) -> date:
    """Launch-day disambiguation key. Uses UTC date of the frame."""
    return dt.astimezone(timezone.utc).date()
