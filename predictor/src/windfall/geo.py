"""Great-circle geometry helpers.

Pure math, no dependencies. Longitudes are normalised to [-180, 180] and the
antimeridian / polar ``cos(lat) -> 0`` cases are guarded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import EARTH_RADIUS_M

_R_KM = EARTH_RADIUS_M / 1000.0


def normalize_lon(lon: float) -> float:
    """Wrap longitude into [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(_lon_delta(lon1, lon2))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * _R_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlam = math.radians(_lon_delta(lon1, lon2))
    y = math.sin(dlam) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlam)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compass_point(bearing: float) -> str:
    """16-point compass abbreviation for a bearing in degrees."""
    points = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = int((bearing % 360.0) / 22.5 + 0.5) % 16
    return points[idx]


def _lon_delta(lon1: float, lon2: float) -> float:
    """Shortest signed longitude difference, handling antimeridian wrap."""
    d = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    return d


@dataclass(slots=True)
class BBox:
    """Axis-aligned lat/lon bounding box. Does not span the antimeridian
    (acceptable for the capture-ROI use; flights spanning ±180 are not our case)."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon

    def expanded_km(self, km: float) -> "BBox":
        """Grow the box by ``km`` on every side (latitude exact; longitude uses
        the worst-case cos(lat) at the box edge nearest a pole)."""
        dlat = km / _R_KM * 180.0 / math.pi
        worst_lat = max(abs(self.min_lat), abs(self.max_lat))
        cos_lat = max(math.cos(math.radians(min(89.9, worst_lat))), 1e-6)
        dlon = dlat / cos_lat
        return BBox(
            min_lat=max(-90.0, self.min_lat - dlat),
            min_lon=max(-180.0, self.min_lon - dlon),
            max_lat=min(90.0, self.max_lat + dlat),
            max_lon=min(180.0, self.max_lon + dlon),
        )


def circles_bbox(points: list[tuple[float, float, float]]) -> BBox | None:
    """Bounding box covering a set of (lat, lon, radius_km) circles."""
    if not points:
        return None
    box: BBox | None = None
    for lat, lon, radius_km in points:
        dlat = radius_km / _R_KM * 180.0 / math.pi
        cos_lat = max(math.cos(math.radians(min(89.9, abs(lat)))), 1e-6)
        dlon = dlat / cos_lat
        c = BBox(lat - dlat, lon - dlon, lat + dlat, lon + dlon)
        if box is None:
            box = c
        else:
            box = BBox(
                min(box.min_lat, c.min_lat),
                min(box.min_lon, c.min_lon),
                max(box.max_lat, c.max_lat),
                max(box.max_lon, c.max_lon),
            )
    return box
