"""Air-density models.

Density drives descent rate. Two sources, in priority order:

* **Measured** - for sondes reporting ``pressure`` and ``temp``:
  ``rho = P / (R_d * T)``. Built along ascent, reused for descent (density is
  far more spatially/temporally stable than wind).
* **Modeled** - International Standard Atmosphere when P/T unavailable.

The humidity (virtual-temperature) correction is <1% and deliberately omitted.
"""

from __future__ import annotations

import math

R_D = 287.05            # specific gas constant for dry air, J/(kg·K)
T0_K = 288.15           # ISA sea-level temperature, K
P0_PA = 101_325.0       # ISA sea-level pressure, Pa
G0 = 9.80665            # standard gravity, m/s^2
RHO0 = P0_PA / (R_D * T0_K)  # ~1.225 kg/m^3

# ISA layers as (base geopotential height m, base temp K, lapse rate K/m).
# Covers 0-32 km, which spans the entire radiosonde regime (burst ~25-35 km).
_ISA_LAYERS = [
    (0.0, 288.15, -0.0065),       # troposphere
    (11_000.0, 216.65, 0.0),      # tropopause
    (20_000.0, 216.65, 0.001),    # lower stratosphere
    (32_000.0, 228.65, 0.0028),   # mid stratosphere
]
_ISA_TOP = 47_000.0


def measured_density(pressure_hpa: float, temp_c: float) -> float | None:
    """rho = P / (R_d * T). ``pressure`` is hPa (radiosonde convention), ``temp``
    Celsius. Returns None for non-physical inputs."""
    if pressure_hpa is None or temp_c is None:
        return None
    t_k = temp_c + 273.15
    if t_k <= 0.0 or pressure_hpa <= 0.0:
        return None
    return (pressure_hpa * 100.0) / (R_D * t_k)


def isa_density(alt_m: float) -> float:
    """International Standard Atmosphere density at geometric altitude ``alt_m``.

    Treats geometric ≈ geopotential height (the difference is <0.5% at 30 km,
    well inside the wind-dominated error budget)."""
    h = max(0.0, min(alt_m, _ISA_TOP))
    p = P0_PA
    t = T0_K
    for i, (h_b, t_b, lapse) in enumerate(_ISA_LAYERS):
        h_top = _ISA_LAYERS[i + 1][0] if i + 1 < len(_ISA_LAYERS) else _ISA_TOP
        if h <= h_top:
            dh = h - h_b
            if lapse == 0.0:
                t = t_b
                p = p * math.exp(-G0 * dh / (R_D * t_b))
            else:
                t = t_b + lapse * dh
                p = p * (t / t_b) ** (-G0 / (R_D * lapse))
            return p / (R_D * t)
        # advance base pressure/temp to the top of this layer
        dh = h_top - h_b
        if lapse == 0.0:
            p = p * math.exp(-G0 * dh / (R_D * t_b))
        else:
            t_layer_top = t_b + lapse * dh
            p = p * (t_layer_top / t_b) ** (-G0 / (R_D * lapse))
    # above modelled top: extrapolate isothermally from the last layer
    return p / (R_D * t)
