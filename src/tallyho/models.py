"""Service-side data contracts, plus re-exports of the engine's.

The prediction engine (and its types: Frame, Prediction, FlightState, ...)
lives in the standalone ``windfall`` package; app modules keep importing them
from here so the service code reads the same as before the split. Only the
alerting/subscriber types are tally-ho's own.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from windfall.models import (  # noqa: F401  (re-exported for app modules)
    EARTH_RADIUS_M,
    DescentSample,
    FlightState,
    Frame,
    Landing,
    Prediction,
    PredictionSource,
    TrackPoint,
    WindBin,
    launch_day_of,
)

__all__ = [
    "EARTH_RADIUS_M", "DescentSample", "FlightState", "Frame", "Landing",
    "Prediction", "PredictionSource", "TrackPoint", "WindBin", "launch_day_of",
    "AlertType", "Subscriber",
]


class AlertType(str, enum.Enum):
    """Notification lifecycle."""

    INBOUND = "INBOUND"
    UPDATE = "UPDATE"
    LANDED = "LANDED"


@dataclass(slots=True)
class Subscriber:
    """A friend who wants alerts."""

    name: str
    lat: float
    lon: float
    radius_km: float
    ntfy_server: str
    ntfy_topic: str
    ntfy_token_ref: str | None = None   # name of a saved ntfy token, not the token
    active: bool = True
    id: int | None = None
    created_at: datetime | None = None

    @property
    def notify_enabled(self) -> bool:
        """Whether this watched location pushes ntfy alerts.

        A blank ``ntfy_topic`` means *watch only*: the location still widens the
        capture ROI and shows on the map (so its sondes are tracked/predicted),
        but no ntfy notification is ever sent - letting you run without ntfy
        configured at all."""
        return bool(self.ntfy_topic and self.ntfy_topic.strip())
