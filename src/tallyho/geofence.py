"""Two-tier region-of-interest + geofence matching.

* **Capture ROI (wide)** - union of subscriber circles + a drift margin
  (~300 km). Decides which flights we track/profile at all: ascent happens near
  the launch site, *upwind* of the landing, so we must grab flights that launch
  outside a landing-sized region but drift into it.
* **Alert ROI (narrow)** - the actual subscriber circles, used only for the
  notify decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from windfall.geo import BBox, bearing_deg, circles_bbox, compass_point, haversine_km
from .models import Subscriber


@dataclass(slots=True)
class Match:
    subscriber: Subscriber
    distance_km: float       # from subscriber to predicted landing
    bearing_deg: float       # from subscriber toward the landing
    compass: str


def build_capture_roi(subscribers: list[Subscriber], margin_km: float) -> BBox | None:
    """Capture-ROI bounding box = subscriber circles + drift margin."""
    circles = [(s.lat, s.lon, s.radius_km) for s in subscribers if s.active]
    box = circles_bbox(circles)
    if box is None:
        return None
    return box.expanded_km(margin_km)


def in_capture_roi(box: BBox | None, lat: float, lon: float) -> bool:
    """Whether a position is inside the capture ROI.

    No box means there are no active watched locations, so there is nothing to
    capture *for* - we return False (gate everything out) rather than opening up
    to the entire planet's sonde firehose. Add an active subscriber to populate
    the ROI."""
    if box is None:
        return False
    return box.contains(lat, lon)


def geofence_matches(
    land_lat: float, land_lon: float, subscribers: list[Subscriber]
) -> list[Match]:
    """Subscribers whose alert circle contains the predicted landing."""
    out: list[Match] = []
    for s in subscribers:
        if not s.active:
            continue
        d = haversine_km(s.lat, s.lon, land_lat, land_lon)
        if d <= s.radius_km:
            b = bearing_deg(s.lat, s.lon, land_lat, land_lon)
            out.append(Match(subscriber=s, distance_km=d, bearing_deg=b,
                             compass=compass_point(b)))
    return out
