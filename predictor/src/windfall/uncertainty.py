"""Uncertainty-radius model.

We express confidence as a single actionable number - the radius (km) within
which the true landing is expected to fall - rather than an abstract score, so
it drops straight into the alert text and is directly testable for calibration
against the replay harness.

The radius grows with:

* **time to land** - more flight remaining = more accumulated drift error;
* **the unmeasured fraction** of the remaining wind column (extrapolated/GFS
  air is the dominant residual error after terrain and descent-rate);
* **the descent-fit residual** - a noisy ballistic fit means a noisy ETA/drift.

All constants are calibration knobs (see :class:`tallyho.config.UncertaintyConfig`).
"""

from __future__ import annotations

from .config import UncertaintyConfig


def uncertainty_radius_km(
    *,
    sim_seconds: float,
    measured_fraction: float,
    fit_residual_mps: float,
    cfg: UncertaintyConfig,
) -> float:
    """Compute the predicted-landing uncertainty radius in km."""
    hours = max(0.0, sim_seconds) / 3600.0
    frac = min(1.0, max(0.0, measured_fraction))
    drift_rate = (
        cfg.per_hour_measured_km * frac
        + cfg.per_hour_extrapolated_km * (1.0 - frac)
    )
    radius = cfg.base_km + hours * drift_rate + fit_residual_mps * cfg.fit_residual_km_per_mps
    return round(radius * cfg.radius_scale, 2)
