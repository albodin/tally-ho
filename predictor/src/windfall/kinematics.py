"""Derived kinematics from consecutive frames.

We compute vertical rate, horizontal speed, and the ascent wind vector ourselves
from ``(lat, lon, alt, datetime)`` rather than trusting the reported ``vel_*``
fields, which are frequently absent or sentinel. Reported velocities are used
only as an optional sanity cross-check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import EARTH_RADIUS_M, Frame

R = EARTH_RADIUS_M


@dataclass(slots=True)
class Segment:
    """Kinematics derived from a pair of consecutive frames."""

    dt: float            # seconds (sonde time)
    alt_mid: float       # mean altitude of the pair, m
    vertical_rate: float  # m/s, +up
    wind_u: float        # eastward component of motion, m/s
    wind_v: float        # northward component of motion, m/s
    horizontal_speed: float  # m/s


def segment(a: Frame, b: Frame) -> Segment | None:
    """Kinematics for the ordered pair (a, b). Returns None if dt is non-positive.

    During ascent the balloon's horizontal track *is* the wind, so ``wind_u`` /
    ``wind_v`` here are the ascent wind sample. During descent the
    same components are the payload's ground track.
    """
    dt = b.t - a.t
    if dt <= 0.0:
        return None
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(_lon_delta(a.lon, b.lon))
    lat_mid = math.radians((a.lat + b.lat) / 2.0)
    wind_v = R * dlat / dt
    wind_u = R * math.cos(lat_mid) * dlon / dt
    return Segment(
        dt=dt,
        alt_mid=(a.alt + b.alt) / 2.0,
        vertical_rate=(b.alt - a.alt) / dt,
        wind_u=wind_u,
        wind_v=wind_v,
        horizontal_speed=math.hypot(wind_u, wind_v),
    )


def _lon_delta(lon1: float, lon2: float) -> float:
    return (lon2 - lon1 + 180.0) % 360.0 - 180.0


def windowed_vertical_rate(
    points: list[tuple[float, float]],
    min_points: int = 3,
    min_span_seconds: float = 4.0,
) -> float | None:
    """Vertical rate (m/s, +up) as the least-squares slope of altitude vs time
    over a short window of ``(t, alt)`` points.

    A single 1 Hz frame pair carries 1-3 m/s of GPS-altitude noise - comparable
    to the descent rate near the ground - while the regression over ~25 s of
    points averages it down by an order of magnitude. Returns None when the
    window is too thin to fit; the caller falls back to the frame-pair rate.
    """
    n = len(points)
    if n < min_points:
        return None
    t0 = points[0][0]
    if points[-1][0] - t0 < min_span_seconds:
        return None
    mean_t = sum(p[0] - t0 for p in points) / n
    mean_a = sum(p[1] for p in points) / n
    num = 0.0
    den = 0.0
    for t, a in points:
        dt = (t - t0) - mean_t
        num += dt * (a - mean_a)
        den += dt * dt
    if den <= 0.0:
        return None
    return num / den
