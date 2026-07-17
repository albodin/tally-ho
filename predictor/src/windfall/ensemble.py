"""Monte Carlo landing ensemble.

The prediction's error sources are explicit, so *sample* them instead of
hand-waving a radius: the ballistic constant (spread set by the fit residual,
wider for the single-point shortcut, widest for the assumed pre-burst chute),
the burst altitude and ascent rate (pre-burst only), and the wind column -
both a per-member **constant bias** (the systematic model error every altitude
shares; AR(1) noise alone averages out over a long descent and under-disperses
the ensemble) and vertically-correlated AR(1) noise, larger in
extrapolated/GFS air than in the measured column. Every member runs the
*production* integrator.

Two things come out, both better than the deterministic single track:

* the **ensemble mean** landing - a better point estimate in terrain, because
  ridges truncate member trajectories asymmetrically in a way one trajectory
  cannot represent;
* an **empirical quantile radius** - calibrated *to the extent the sampled
  error sources span the real ones*; verify against the backtest's coverage
  report and close any residual gap with ``uncertainty.radius_scale``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .atmosphere import RHO0
from .config import Config, IntegratorConfig
from .descent import DescentModel
from .geo import haversine_km, normalize_lon
from .integrator import WindFn, integrate_ascent, integrate_descent
from .profile import FlightProfile, WindResidualStats

GroundFn = "Callable[[float, float], float]"


@dataclass(slots=True)
class EnsembleLanding:
    lat: float
    lon: float
    eta: datetime
    radius_km: float          # quantile of member spread around the mean
    n_members: int            # members that landed successfully


def _member_cfg(cfg: Config) -> IntegratorConfig:
    return IntegratorConfig(
        dt_seconds=cfg.ensemble.dt_seconds,
        max_iterations=cfg.integrator.max_iterations,
        max_sim_seconds=cfg.integrator.max_sim_seconds,
    )


def _ar1_wind(
    base_fn: WindFn | None,
    profile: FlightProfile,
    rng: random.Random,
    cfg: Config,
    measured_range: tuple[float, float] | None,
    stats: "WindResidualStats | None" = None,
) -> WindFn:
    """Wrap a wind source with one member's wind-error draw.

    Two components, sampled per member:

    * a **constant bias** (``wind_bias_sigma_mps``) shared by the whole column -
      the systematic model error mode. Without it every member rides the same
      base field and the spread misses exactly the error that dominates at
      scale (52% measured coverage vs the 68% target, 2026-06-10 backtest);
    * an **AR(1) process** over traversed altitude: e' = a·e + √(1-a²)·σ·N with
      a = exp(-|Δalt|/L). Sigma is smaller inside the *measured* column
      (``measured_range``, None when the whole column is modelled/GFS) than in
      extrapolated air.

    One closure per member; bias and AR(1) state carry across the ascent and
    descent legs so the column error is consistent within a member.
    """
    ecfg = cfg.ensemble
    # When this flight's ascent measured-minus-model residual is available,
    # size the spread from it rather than the global constants (calm day → tight
    # ensemble, high-disagreement day → wide). Perturbations stay zero-mean, so
    # this is a calibration fix (honest radius) and does not move the point.
    if stats is not None:
        floor = ecfg.wind_stats_sigma_floor_mps
        cap = ecfg.wind_stats_sigma_cap_mps
        sigma_meas = min(max(stats.sigma_mps, floor), cap)
        sigma_extrap = min(max(ecfg.wind_stats_extrap_k * stats.sigma_mps, floor), cap)
        bias_sigma = min(max(stats.bias_pc_mps, floor), cap)
        corr_l = max(stats.corr_len_m, 1.0)
    else:
        sigma_meas = ecfg.wind_sigma_measured_mps
        sigma_extrap = ecfg.wind_sigma_extrapolated_mps
        bias_sigma = ecfg.wind_bias_sigma_mps
        corr_l = max(ecfg.wind_corr_length_m, 1.0)
    bias_u = rng.gauss(0.0, bias_sigma)
    bias_v = rng.gauss(0.0, bias_sigma)
    state = {"alt": None, "eu": 0.0, "ev": 0.0}

    def wind(lat: float, lon: float, alt: float, sim_t: float):
        base = None
        if base_fn is not None:
            base = base_fn(lat, lon, alt, sim_t)
        if base is None:
            base = profile.wind(alt)
        sigma = (
            sigma_meas
            if measured_range is not None and measured_range[0] <= alt < measured_range[1]
            else sigma_extrap
        )
        last = state["alt"]
        if last is None:
            state["eu"] = rng.gauss(0.0, sigma)
            state["ev"] = rng.gauss(0.0, sigma)
        else:
            a = math.exp(-abs(alt - last) / corr_l)
            q = sigma * math.sqrt(max(0.0, 1.0 - a * a))
            state["eu"] = a * state["eu"] + rng.gauss(0.0, q)
            state["ev"] = a * state["ev"] + rng.gauss(0.0, q)
        state["alt"] = alt
        return (base[0] + bias_u + state["eu"], base[1] + bias_v + state["ev"])

    return wind


def _b_sigma_rel(descent: DescentModel, cfg: Config) -> float:
    ecfg = cfg.ensemble
    if descent.n_points >= cfg.descent.min_fit_points:
        v_sl = descent.v_t(RHO0)   # sea-level speed - the slowest, so conservative
        rel = (descent.residual_mps / v_sl) if v_sl > 0 else 0.0
        return max(ecfg.b_sigma_rel_fit, rel)
    if descent.n_points >= 1:
        return ecfg.b_sigma_rel_shortcut
    return ecfg.b_sigma_rel_preburst


def _perturbed_model(descent: DescentModel, rel: float, rng: random.Random,
                     cfg: Config) -> DescentModel:
    b = descent.b * math.exp(rng.gauss(0.0, rel))
    b = min(max(b, cfg.descent.b_min), cfg.descent.b_max)
    return DescentModel(b=b, residual_mps=descent.residual_mps,
                        n_points=descent.n_points)


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    mlat = sum(p[0] for p in points) / n
    ref = points[0][1]
    mlon = ref + sum((p[1] - ref + 180.0) % 360.0 - 180.0 for p in points) / n
    return (mlat, normalize_lon(mlon))


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _summarise(
    landings: list[tuple[float, float, datetime]], n_target: int, quantile: float
) -> EnsembleLanding | None:
    if len(landings) < max(3, n_target // 4):
        return None
    mlat, mlon = _mean_point([(l[0], l[1]) for l in landings])
    dists = sorted(haversine_km(mlat, mlon, l[0], l[1]) for l in landings)
    etas = sorted(l[2] for l in landings)
    return EnsembleLanding(
        lat=mlat, lon=mlon, eta=etas[len(etas) // 2],
        radius_km=_quantile(dists, quantile), n_members=len(landings),
    )


def ensemble_descent(
    *,
    lat: float,
    lon: float,
    alt: float,
    t0: datetime,
    profile: FlightProfile,
    descent: DescentModel,
    ground_fn,
    cfg: Config,
    wind_fn: WindFn | None = None,
    rng: random.Random,
    n: int | None = None,
    measured_range: tuple[float, float] | None = None,
    stats: WindResidualStats | None = None,
) -> EnsembleLanding | None:
    """Monte Carlo descent: perturb B and the winds, land every member.

    ``measured_range`` is the altitude span actually measured from this flight
    (None when the column is all GFS/modelled) - wind noise is tighter inside it.
    ``stats`` sizes the wind spread from this flight's ascent residual.
    """
    n = n if n is not None else cfg.ensemble.n_members
    mem_cfg = _member_cfg(cfg)
    rel = _b_sigma_rel(descent, cfg)
    landings: list[tuple[float, float, datetime]] = []
    for _ in range(n):
        member = integrate_descent(
            lat=lat, lon=lon, alt=alt, t0=t0, profile=profile,
            descent=_perturbed_model(descent, rel, rng, cfg),
            ground_fn=ground_fn, cfg=mem_cfg,
            wind_fn=_ar1_wind(wind_fn, profile, rng, cfg, measured_range, stats),
        )
        if member.ok:
            landings.append((member.lat, member.lon, member.eta))
    return _summarise(landings, n, cfg.ensemble.quantile)


def ensemble_preburst(
    *,
    lat: float,
    lon: float,
    alt: float,
    t0: datetime,
    burst_alt: float,
    ascent_rate: float,
    default_b: float,
    profile: FlightProfile,
    ground_fn,
    cfg: Config,
    wind_fn: WindFn | None = None,
    rng: random.Random,
    n: int | None = None,
    measured_range: tuple[float, float] | None = None,
    stats: WindResidualStats | None = None,
) -> EnsembleLanding | None:
    """Monte Carlo pre-burst: perturb burst altitude, ascent rate, the assumed
    chute B, and the winds; fly every member up then down."""
    n = n if n is not None else cfg.ensemble.n_members_preburst
    ecfg = cfg.ensemble
    mem_cfg = _member_cfg(cfg)
    base_model = DescentModel(b=default_b, residual_mps=0.0, n_points=0)
    landings: list[tuple[float, float, datetime]] = []
    for _ in range(n):
        member_wind = _ar1_wind(wind_fn, profile, rng, cfg, measured_range, stats)
        rate_k = max(0.5, ascent_rate * math.exp(rng.gauss(0.0, ecfg.ascent_rate_sigma_rel)))
        burst_k = max(alt + 100.0, burst_alt + rng.gauss(0.0, ecfg.burst_alt_sigma_m))
        b_lat, b_lon, t_asc = integrate_ascent(
            lat=lat, lon=lon, alt=alt, burst_alt=burst_k, ascent_rate=rate_k,
            profile=profile, cfg=mem_cfg, wind_fn=member_wind,
        )
        member = integrate_descent(
            lat=b_lat, lon=b_lon, alt=burst_k,
            t0=t0 + timedelta(seconds=t_asc),
            profile=profile,
            descent=_perturbed_model(base_model, ecfg.b_sigma_rel_preburst, rng, cfg),
            ground_fn=ground_fn, cfg=mem_cfg,
            wind_fn=member_wind, t_offset_s=t_asc,
        )
        if member.ok:
            landings.append((member.lat, member.lon, member.eta))
    return _summarise(landings, n, ecfg.quantile)
