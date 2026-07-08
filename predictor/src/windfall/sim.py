"""Synthetic radiosonde flights with a known landing.

The generator integrates the *same* physics the predictor assumes (ISA density,
ballistic descent ``v_t = B·rho^-1/2``, advection by an altitude-dependent wind
field), so replaying its frames through the production predictor should recover
the true landing to within integrator discretisation - that is the core
accuracy validation. Both test suites (windfall's and tally-ho's)
build their fixtures from this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .atmosphere import R_D, isa_density
from .models import EARTH_RADIUS_M

R = EARTH_RADIUS_M
BASE_TIME = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)


def wind_at(alt: float) -> tuple[float, float]:
    """A smooth synthetic wind field (u east, v north) in m/s.

    Eastward wind grows with altitude (a stylised jet) then eases; a mild,
    steady northward component. Smooth in altitude, matching the plan's
    assumption that the field is spatially smooth."""
    u = 8.0 + 22.0 * math.exp(-((alt - 11_000.0) / 7_000.0) ** 2)
    v = 4.0 + 0.0002 * alt
    return (u, v)


@dataclass
class SimResult:
    frames: list[dict]          # raw telemetry dicts (exercise the parser)
    launch_lat: float
    launch_lon: float
    burst_alt: float
    land_lat: float
    land_lon: float
    land_time: datetime


def simulate_flight(
    *,
    serial: str = "S1234567",
    sonde_type: str = "RS41",
    launch_lat: float = 45.0,
    launch_lon: float = 7.0,
    launch_alt: float = 200.0,
    burst_alt: float = 30_000.0,
    ascent_rate: float = 5.0,       # m/s
    ballistic_b: float = 5.5,       # v_t = B·rho^-1/2
    ground_alt: float = 200.0,      # flat ground for the no-DEM baseline
    frame_dt: float = 1.0,          # seconds between emitted frames
    sim_dt: float = 0.5,            # internal integration step
    include_ptu: bool = True,       # emit pressure/temp (measured density path)
    start: datetime = BASE_TIME,
    stringify: bool = False,        # emit numbers as JSON strings (wire reality)
) -> SimResult:
    """Generate a full ascent+descent flight and its true landing point."""
    frames: list[dict] = []
    lat = math.radians(launch_lat)
    lon = math.radians(launch_lon)
    alt = launch_alt
    t = 0.0
    frame_no = 0
    next_emit = 0.0

    def emit():
        nonlocal frame_no
        # ISA temperature back-out for a plausible PTU reading
        temp_c, press_hpa = _isa_pt(alt)
        lat_d = math.degrees(lat)
        lon_d = math.degrees(lon)
        dt = start + timedelta(seconds=t)
        msg = {
            "serial": serial,
            "type": sonde_type,
            "manufacturer": "Vaisala",
            "software_name": "test",
            "uploader_callsign": "TEST-RX",
            "frame": frame_no,
            "datetime": dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "time_received": dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "lat": lat_d,
            "lon": lon_d,
            "alt": alt,
        }
        if include_ptu:
            msg["pressure"] = press_hpa
            msg["temp"] = temp_c
        if stringify:
            for k in ("lat", "lon", "alt", "pressure", "temp", "frame"):
                if k in msg:
                    msg[k] = str(msg[k])
        frames.append(msg)
        frame_no += 1

    # --- ascent ---
    while alt < burst_alt:
        if t >= next_emit:
            emit()
            next_emit += frame_dt
        u, v = wind_at(alt)
        lat += (v / R) * sim_dt
        lon += (u / (R * math.cos(lat))) * sim_dt
        alt += ascent_rate * sim_dt
        t += sim_dt
    alt = burst_alt
    emit()
    next_emit = t + frame_dt

    # --- descent ---
    while alt > ground_alt:
        u, v = wind_at(alt)
        rho = isa_density(alt)
        v_t = ballistic_b * rho ** -0.5
        lat += (v / R) * sim_dt
        lon += (u / (R * math.cos(lat))) * sim_dt
        alt -= v_t * sim_dt
        t += sim_dt
        if t >= next_emit:
            emit()
            next_emit += frame_dt
    alt = ground_alt
    emit()

    return SimResult(
        frames=frames,
        launch_lat=launch_lat,
        launch_lon=launch_lon,
        burst_alt=burst_alt,
        land_lat=math.degrees(lat),
        land_lon=math.degrees(lon),
        land_time=start + timedelta(seconds=t),
    )


def _isa_pt(alt: float) -> tuple[float, float]:
    """Approximate ISA temperature (C) and pressure (hPa) at altitude for PTU."""
    rho = isa_density(alt)
    # Use the troposphere/stratosphere temperature profile (mirrors atmosphere).
    if alt <= 11_000.0:
        t_k = 288.15 - 0.0065 * alt
    elif alt <= 20_000.0:
        t_k = 216.65
    elif alt <= 32_000.0:
        t_k = 216.65 + 0.001 * (alt - 20_000.0)
    else:
        t_k = 228.65 + 0.0028 * (alt - 32_000.0)
    p_pa = rho * R_D * t_k
    return (t_k - 273.15, p_pa / 100.0)
