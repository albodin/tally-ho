"""Heun (midpoint-averaged) descent integrator.

The single trajectory engine. It takes a wind+density :class:`FlightProfile`, a
:class:`DescentModel`, and a ground-elevation function, and integrates the
descent forward from the current state to the terrain crossing. The same engine
serves the measured-wind path and the GFS-wind path - only the wind source
differs.

Each step evaluates wind and descent rate at the current state *and* at a
provisional forward-Euler end-of-step state, then advances on their average.
That removes the systematic forward-Euler bias when falling fast (50-70 m/s
near burst) through strong shear layers, at ~2x the cost per step.

When ``wind_fn`` is given, the wind is a full 4-D field queried at the
integrator's *current* position and simulation time -
``wind_fn(lat_deg, lon_deg, alt_m, sim_seconds)`` - instead of the profile's
altitude-only column. ``t_offset_s`` shifts the time argument (the pre-burst
descent leg starts ``t_ascent`` seconds after the field's anchor time).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Callable

from .config import IntegratorConfig
from .descent import DescentModel
from .geo import normalize_lon
from .models import EARTH_RADIUS_M, Landing
from .profile import FlightProfile

R = EARTH_RADIUS_M
_MAX_LAT_RAD = math.radians(89.9)   # guard cos(lat) -> 0 near the poles

# 4-D wind field: (lat_deg, lon_deg, alt_m, sim_seconds) -> (u, v) m/s
WindFn = Callable[[float, float, float, float], "tuple[float, float] | None"]


def _decimate(path: list, max_points: int) -> int:
    """Bound a growing path in place by dropping every other interior point when
    it gets too long (keeping the endpoints). Returns the new sampling stride so
    the caller appends less often as the path lengthens - total memory stays
    O(max_points) regardless of step count."""
    if len(path) < max_points * 2:
        return 1
    del path[1:-1:2]
    return 2


def _clamp_lat(phi: float) -> float:
    return max(-_MAX_LAT_RAD, min(_MAX_LAT_RAD, phi))


def integrate_descent(
    *,
    lat: float,
    lon: float,
    alt: float,
    t0: datetime,
    profile: FlightProfile,
    descent: DescentModel,
    ground_fn,
    cfg: IntegratorConfig,
    capture_path: bool = False,
    path_max_points: int = 64,
    wind_fn: WindFn | None = None,
    t_offset_s: float = 0.0,
) -> Landing:
    """Integrate from (lat, lon, alt) at time ``t0`` down to the ground.

    ``lat``/``lon`` are degrees. ``ground_fn(lat, lon) -> elevation_m``.

    When ``capture_path`` is set, the returned :class:`Landing` carries a
    down-sampled ``path`` of ``(lat, lon, alt)`` points (≲ ``path_max_points``)
    for drawing the predicted descent trajectory on the map.
    """
    phi = math.radians(lat)
    lam = math.radians(lon)
    dt = cfg.dt_seconds
    sim_t = 0.0
    alt_start = alt

    path: list | None = [(lat, lon, alt)] if capture_path else None
    stride = 1

    def wind(p: float, l: float, a: float, ts: float) -> tuple[float, float]:
        if wind_fn is not None:
            w = wind_fn(math.degrees(p), normalize_lon(math.degrees(l)), a,
                        ts + t_offset_s)
            if w is not None:
                return w
        return profile.wind(a)

    ground = ground_fn(lat, lon)
    if alt <= ground:
        return Landing(lat=lat, lon=lon, eta=t0, steps=0, sim_seconds=0.0,
                       measured_fraction=1.0, ok=True, path=path)

    steps = 0
    while steps < cfg.max_iterations and sim_t < cfg.max_sim_seconds:
        steps += 1
        v1 = descent.v_t(profile.density(alt))
        if v1 <= 0.0:
            return Landing(lat=math.degrees(phi), lon=normalize_lon(math.degrees(lam)),
                           eta=t0 + timedelta(seconds=sim_t), steps=steps,
                           sim_seconds=sim_t, measured_fraction=0.0, ok=False,
                           reason="non-positive descent rate")
        wu1, wv1 = wind(phi, lam, alt, sim_t)
        cos_phi = math.cos(_clamp_lat(phi))

        # Heun: provisional forward-Euler end-of-step state, then average the
        # rates evaluated at both ends.
        phi_p = _clamp_lat(phi + (wv1 / R) * dt)
        lam_p = lam + (wu1 / (R * cos_phi)) * dt
        alt_p = alt - v1 * dt
        v2 = descent.v_t(profile.density(alt_p))
        if v2 <= 0.0:
            v2 = v1
        wu2, wv2 = wind(phi_p, lam_p, alt_p, sim_t + dt)

        v_desc = 0.5 * (v1 + v2)
        wu = 0.5 * (wu1 + wu2)
        wv = 0.5 * (wv1 + wv2)

        prev_phi, prev_lam, prev_alt = phi, lam, alt
        phi = _clamp_lat(phi + (wv / R) * dt)
        lam += (wu / (R * cos_phi)) * dt
        alt -= v_desc * dt
        sim_t += dt

        new_lat = math.degrees(phi)
        new_lon = normalize_lon(math.degrees(lam))
        if path is not None and steps % stride == 0:
            path.append((new_lat, new_lon, alt))
            stride *= _decimate(path, path_max_points)
        ground = ground_fn(new_lat, new_lon)
        if alt <= ground:
            # interpolate the final partial step to the ground crossing
            denom = prev_alt - alt
            frac = 1.0 if denom <= 0 else (prev_alt - ground) / denom
            frac = min(1.0, max(0.0, frac))
            f_phi = prev_phi + frac * (phi - prev_phi)
            f_lam = prev_lam + frac * (lam - prev_lam)
            land_lat = math.degrees(f_phi)
            land_lon = normalize_lon(math.degrees(f_lam))
            eta = t0 + timedelta(seconds=sim_t - dt * (1.0 - frac))
            mf = profile.measured_fraction(alt_start, ground)
            if path is not None:
                path.append((land_lat, land_lon, ground))
            return Landing(lat=land_lat, lon=land_lon, eta=eta, steps=steps,
                           sim_seconds=sim_t - dt * (1.0 - frac),
                           measured_fraction=mf, ok=True, path=path)

    # runaway guard
    return Landing(lat=math.degrees(phi), lon=normalize_lon(math.degrees(lam)),
                   eta=t0 + timedelta(seconds=sim_t), steps=steps, sim_seconds=sim_t,
                   measured_fraction=0.0, ok=False, reason="max iterations/time")


def integrate_ascent(
    *,
    lat: float,
    lon: float,
    alt: float,
    burst_alt: float,
    ascent_rate: float,
    profile: FlightProfile,
    cfg: IntegratorConfig,
    path_out: list | None = None,
    path_max_points: int = 32,
    wind_fn: WindFn | None = None,
    t_offset_s: float = 0.0,
) -> tuple[float, float, float]:
    """Integrate the *ascent* up to ``burst_alt`` advected by the wind profile,
    returning the (lat, lon, sim_seconds) at burst. Used for pre-burst landing
    estimates. ``ascent_rate`` is m/s upward. Heun-averaged like the
    descent, with the same optional 4-D ``wind_fn``.

    When ``path_out`` is given, ``(lat, lon, alt)`` samples of the rising leg are
    appended into it (down-sampled to ≲ ``path_max_points``) so the caller can
    prepend the climb to the descent path for the map."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    dt = cfg.dt_seconds
    sim_t = 0.0
    ascent_rate = max(0.1, ascent_rate)
    if path_out is not None:
        path_out.append((lat, lon, alt))
    stride = 1
    steps = 0

    def wind(p: float, l: float, a: float, ts: float) -> tuple[float, float]:
        if wind_fn is not None:
            w = wind_fn(math.degrees(p), normalize_lon(math.degrees(l)), a,
                        ts + t_offset_s)
            if w is not None:
                return w
        return profile.wind(a)

    while alt < burst_alt and steps < cfg.max_iterations and sim_t < cfg.max_sim_seconds:
        steps += 1
        wu1, wv1 = wind(phi, lam, alt, sim_t)
        cos_phi = math.cos(_clamp_lat(phi))
        phi_p = _clamp_lat(phi + (wv1 / R) * dt)
        lam_p = lam + (wu1 / (R * cos_phi)) * dt
        wu2, wv2 = wind(phi_p, lam_p, alt + ascent_rate * dt, sim_t + dt)
        phi = _clamp_lat(phi + (0.5 * (wv1 + wv2) / R) * dt)
        lam += (0.5 * (wu1 + wu2) / (R * cos_phi)) * dt
        alt += ascent_rate * dt
        sim_t += dt
        if path_out is not None and steps % stride == 0:
            path_out.append((math.degrees(phi), normalize_lon(math.degrees(lam)), alt))
            stride *= _decimate(path_out, path_max_points)
    return math.degrees(phi), normalize_lon(math.degrees(lam)), sim_t
