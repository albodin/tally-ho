"""Vectorised Monte-Carlo ensembles.

Advances all N members in lockstep as numpy arrays instead of looping Python over
N independent scalar trajectories. The per-member interpreter, closure, and RNG
overhead that dominates the scalar ensembles collapses into a handful of array
ops per step. :func:`ensemble_descent_vec` twins
:func:`windfall.ensemble.ensemble_descent`; :func:`ensemble_preburst_vec` twins
:func:`windfall.ensemble.ensemble_preburst` - one loop for both legs, each
member flipping from its own perturbed ascent to its own perturbed descent
mid-flight, so at any step the cloud is a mix of rising and falling members.

Physics matches :mod:`windfall.integrator` (Heun) and
:func:`windfall.ensemble._ar1_wind` (per-member B / constant bias / AR(1) wind
perturbation, with the per-flight stats sizing). The RNG draw *order* differs (numpy draws
all members at once; the scalar version draws one member's whole trajectory
before the next), so member-for-member the two are **not** identical - but the
landing distribution is, which the differential tests assert statistically.

Wind: the measured column is altitude-keyed, so it is sampled once onto a fine
altitude grid and batch-interpolated. A 4-D ``wind_fn`` (the GFS blend) is
evaluated through its ``.batch``/``.batch_c`` protocol for the active members
only; with ``ensemble.gfs_shared_column`` (default) the model column is sampled
once per step at the member centroid - the horizontal wind variation across the
member cloud is far smaller than the noise the ensemble adds on purpose. (u, v)
pairs ride one complex array (u + jv) end to end: at ensemble sizes numpy call
dispatch, not flops, is the cost, and complex halves the calls.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from .config import Config
from .descent import DescentModel
from .ensemble import EnsembleLanding, _b_sigma_rel, _summarise
from .geo import normalize_lon
from .integrator import WindFn
from .models import EARTH_RADIUS_M
from .profile import FlightProfile

R = EARTH_RADIUS_M
_MAX_LAT = math.radians(89.9)
# Above this altitude no member can be at/below ground (highest Earth terrain
# ~8849 m); used to skip per-step ground sampling during the high descent.
_GROUND_CHECK_CEIL_M = 9000.0


def _clip_lat(x):
    """np.clip(x, -_MAX_LAT, _MAX_LAT) as two plain ufuncs - np.clip's dispatch
    costs ~2x that, and this runs several times per step."""
    return np.minimum(_MAX_LAT, np.maximum(-_MAX_LAT, x))


def _resolve_sigmas(cfg: Config, stats):
    """Effective (measured, extrapolated, bias) sigmas + corr length - the same
    resolution :func:`windfall.ensemble._ar1_wind` does, batched here once."""
    ec = cfg.ensemble
    if stats is not None:
        floor, cap = ec.wind_stats_sigma_floor_mps, ec.wind_stats_sigma_cap_mps

        def clip(x):
            return min(max(x, floor), cap)

        return (clip(stats.sigma_mps), clip(ec.wind_stats_extrap_k * stats.sigma_mps),
                clip(stats.bias_pc_mps), max(stats.corr_len_m, 1.0))
    return (ec.wind_sigma_measured_mps, ec.wind_sigma_extrapolated_mps,
            ec.wind_bias_sigma_mps, max(ec.wind_corr_length_m, 1.0))


class _MemberWind:
    """One ensemble's batched wind source: base field + per-member constant bias
    + AR(1) noise, returned as one complex u + jv array per call.

    Owns the AR(1) state and mutates it once per :meth:`eval` - exactly like the
    scalar :func:`windfall.ensemble._ar1_wind` closure, which updates on every
    wind() evaluation. Shared by the descent and pre-burst ensembles. The
    constructor draws the per-member bias, so it must be built at the same point
    in the RNG sequence as the scalar draws it (after the B perturbations).
    """

    def __init__(self, *, wind_fn, profile, cfg, nrng, n, measured_range, stats,
                 grid, wc_g):
        self.wind_fn = wind_fn
        self.profile = profile
        self.nrng = nrng
        self.n = n
        self.measured_range = measured_range
        self.grid = grid
        self.wc_g = wc_g            # measured column on `grid` (wind_fn is None)
        sig_meas, sig_extrap, bias_sigma, corr_l = _resolve_sigmas(cfg, stats)
        self.sig_meas = sig_meas
        self.sig_extrap = sig_extrap
        self.corr_l = corr_l
        self.sigma_extrap_arr = np.full(n, sig_extrap)
        self.bias_c = (nrng.normal(0.0, bias_sigma, n)
                       + 1j * nrng.normal(0.0, bias_sigma, n))
        self.e_c = np.zeros(n, dtype=np.complex128)
        self.have_e = False
        self.last_alt = None
        self.gfs_shared = cfg.ensemble.gfs_shared_column and (
            hasattr(wind_fn, "batch") or hasattr(wind_fn, "batch_c"))
        self.batch_c = getattr(wind_fn, "batch_c", None)
        self.batch_fn = getattr(wind_fn, "batch", None)

    def eval(self, phi_a, lam_a, a_a, active, ts, last_ground):
        """base wind + per-member constant bias + AR(1) noise (batched), as one
        complex u + jv array. ``ts`` is the field evaluation time: the Heun
        corrector stage samples at sim_t + dt, matching the scalar integrator.
        ``last_ground`` (optional) is the per-member ground sampled at the end
        of the previous step - the exact positions this stage-1 eval sees - so
        the blend's AGL floor reuses it instead of re-querying ground_fn."""
        n = self.n
        if self.wind_fn is None:
            wc = np.interp(a_a, self.grid, self.wc_g)
        else:
            # landed/frozen members' wind is discarded - evaluate the
            # (expensive) 4-D field only for active ones. Also keeps the shared
            # centroid honest: it must not drag toward long-landed positions.
            idx = np.nonzero(active)[0]
            full = idx.size == n
            lat_d = np.degrees(phi_a if full else phi_a[idx])
            lon_d = (np.degrees(lam_a if full else lam_a[idx]) + 180.0) % 360.0 - 180.0
            a_q = a_a if full else a_a[idx]
            if self.batch_c is not None or self.batch_fn is not None:
                # GFS path: batched 4-D field eval (CubePairWind /
                # blended_wind_fn); shared=True samples the model column once
                # at the member centroid
                if self.gfs_shared:
                    g_q = (None if last_ground is None
                           else (last_ground if full else last_ground[idx]))
                    if self.batch_c is not None:
                        ac = self.batch_c(lat_d, lon_d, a_q, ts, shared=True,
                                          grounds=g_q)
                    else:
                        au, av = self.batch_fn(lat_d, lon_d, a_q, ts, shared=True,
                                               grounds=g_q)
                        ac = au + 1j * av
                elif self.batch_c is not None:
                    ac = self.batch_c(lat_d, lon_d, a_q, ts)
                else:
                    au, av = self.batch_fn(lat_d, lon_d, a_q, ts)
                    ac = au + 1j * av
            else:
                ac = np.empty(idx.size, dtype=np.complex128)
                for j in range(idx.size):
                    w = self.wind_fn(float(lat_d[j]), float(lon_d[j]),
                                     float(a_q[j]), ts)
                    if w is None:
                        w = self.profile.wind(float(a_q[j]))
                    ac[j] = complex(w[0], w[1])
            if full:
                wc = ac
            else:
                wc = np.zeros(n, dtype=np.complex128)
                wc[idx] = ac
        if self.measured_range is not None:
            inside = ((a_a >= self.measured_range[0])
                      & (a_a < self.measured_range[1]))
            sigma = np.where(inside, self.sig_meas, self.sig_extrap)
        else:
            sigma = self.sigma_extrap_arr
        # one draw of 2n reals viewed as n complex: re/im stay iid N(0,1)
        s = self.nrng.standard_normal(2 * n).view(np.complex128)
        if not self.have_e:
            self.e_c = sigma * s
            self.have_e = True
        else:
            a1 = np.exp(-np.abs(a_a - self.last_alt) / self.corr_l)
            q = sigma * np.sqrt(np.maximum(0.0, 1.0 - a1 * a1))
            self.e_c = a1 * self.e_c + q * s
        self.last_alt = a_a
        return wc + self.bias_c + self.e_c


def _ground_at(ground_fn, phi_a, lam_a, mask, n):
    """Ground elevation at the masked members' positions (zeros elsewhere).
    tolist() + zip: plain floats into ground_fn without a numpy-scalar
    __float__ per member - this loop runs every low step."""
    g = np.zeros(n)
    idx = np.nonzero(mask)[0]
    lat_l = np.degrees(phi_a[idx]).tolist()
    lon_l = ((np.degrees(lam_a[idx]) + 180.0) % 360.0 - 180.0).tolist()
    g[idx] = [ground_fn(la, lo) for la, lo in zip(lat_l, lon_l)]
    return g


def ensemble_descent_vec(
    *,
    lat: float,
    lon: float,
    alt: float,
    t0,
    profile: FlightProfile,
    descent: DescentModel,
    ground_fn,
    cfg: Config,
    wind_fn: WindFn | None = None,
    rng,
    n: int | None = None,
    measured_range: tuple[float, float] | None = None,
    stats=None,
) -> EnsembleLanding | None:
    """Vectorised twin of :func:`windfall.ensemble.ensemble_descent`."""
    n = n if n is not None else cfg.ensemble.n_members
    if n < 1:
        return None
    ec = cfg.ensemble
    dt = ec.dt_seconds
    max_steps = cfg.integrator.max_iterations
    max_sim = cfg.integrator.max_sim_seconds
    nrng = np.random.default_rng(rng.getrandbits(64))

    # --- batched perturbations (drawn once) ---
    rel = _b_sigma_rel(descent, cfg)
    B = descent.b * np.exp(nrng.normal(0.0, rel, n))
    B = np.clip(B, cfg.descent.b_min, cfg.descent.b_max)

    # --- measured column on a fine altitude grid (alt-keyed → batchable) ---
    grid = np.arange(0.0, float(alt) + 30.0, 10.0)
    dens_g = np.array([profile.density(z) for z in grid])
    # real profiles are strictly positive → the per-step rho>0 guard (5 array
    # ops per Heun stage) collapses to a plain power
    dens_pos = bool(dens_g.size) and float(dens_g.min()) > 0.0
    wc_g = None
    if wind_fn is None:
        wc_g = np.empty(grid.size, dtype=np.complex128)
        for i, z in enumerate(grid):
            u, v = profile.wind(float(z))
            wc_g[i] = complex(u, v)

    wind = _MemberWind(wind_fn=wind_fn, profile=profile, cfg=cfg, nrng=nrng, n=n,
                       measured_range=measured_range, stats=stats,
                       grid=grid, wc_g=wc_g)

    phi = np.full(n, math.radians(lat))
    lam = np.full(n, math.radians(lon))
    a = np.full(n, float(alt))
    landed = np.zeros(n, dtype=bool)
    llat = np.zeros(n)
    llon = np.zeros(n)
    lsec = np.zeros(n)
    sim_t = 0.0

    # Ground elevations sampled at the end of the previous step - the exact
    # positions the next stage-1 wind eval sees. The blend's AGL floor reuses
    # them instead of re-querying ground_fn per member per call. Sampling has to
    # start min_agl earlier than the termination check needs, so the floor
    # always has elevations by the time a member can trip it.
    last_ground = None
    ground_ceil = _GROUND_CHECK_CEIL_M
    if wind.gfs_shared:
        ground_ceil += max(0.0, cfg.profile.gfs_blend_min_agl_m)

    steps = 0
    while steps < max_steps and sim_t < max_sim:
        active = ~landed
        if not active.any():
            break
        steps += 1
        rho1 = np.interp(a, grid, dens_g)
        if dens_pos:
            v1 = B * rho1 ** -0.5
        else:
            v1 = np.where(rho1 > 0.0, B * np.where(rho1 > 0.0, rho1, 1.0) ** -0.5, 0.0)
        wc1 = wind.eval(phi, lam, a, active, sim_t, last_ground)
        cphi = np.cos(_clip_lat(phi))

        phi_p = _clip_lat(phi + wc1.imag * (dt / R))
        lam_p = lam + wc1.real * (dt / R) / cphi
        a_p = a - v1 * dt
        rho2 = np.interp(a_p, grid, dens_g)
        if dens_pos:
            v2 = B * rho2 ** -0.5
        else:
            v2 = np.where(rho2 > 0.0, B * np.where(rho2 > 0.0, rho2, 1.0) ** -0.5, v1)
        wc2 = wind.eval(phi_p, lam_p, a_p, active, sim_t + dt, last_ground)

        vd = 0.5 * (v1 + v2)
        wc = 0.5 * (wc1 + wc2)

        prev_a = a
        prev_phi = phi
        prev_lam = lam
        # landed members simply keep integrating (their wind is zeroed, their
        # landing already recorded, nothing downstream reads them) - cheaper
        # than masking the update per component
        phi = _clip_lat(phi + wc.imag * (dt / R))
        lam = lam + wc.real * (dt / R) / cphi
        a = a - vd * dt
        sim_t += dt

        # No active member can reach the ground while every one of them is still
        # above the highest terrain on Earth - skip the per-member ground sampling
        # (the dominant per-step cost) until the descent actually approaches it.
        active_alt = a[active]
        if active_alt.size and active_alt.min() > ground_ceil:
            continue
        ground = _ground_at(ground_fn, phi, lam, active, n)
        last_ground = ground
        cross = active & (a <= ground)
        if cross.any():
            denom = prev_a - a
            safe = denom > 0.0
            # divide only where safe (np.where evaluates both branches, so guard
            # the divisor too - frozen members have denom == 0)
            frac = np.where(safe, (prev_a - ground) / np.where(safe, denom, 1.0), 1.0)
            frac = np.clip(frac, 0.0, 1.0)
            fphi = prev_phi + frac * (phi - prev_phi)
            flam = prev_lam + frac * (lam - prev_lam)
            idx = np.nonzero(cross)[0]
            llat[idx] = np.degrees(fphi[idx])
            llon[idx] = np.array([normalize_lon(float(x)) for x in np.degrees(flam[idx])])
            lsec[idx] = sim_t - dt * (1.0 - frac[idx])
            landed[idx] = True
        # members with non-positive descent rate never land → leave active; the
        # step/time guard ends the loop and they are simply dropped (like ok=False)

    landings = [(float(llat[i]), float(llon[i]), t0 + timedelta(seconds=float(lsec[i])))
                for i in range(n) if landed[i]]
    return _summarise(landings, n, ec.quantile)


def ensemble_preburst_vec(
    *,
    lat: float,
    lon: float,
    alt: float,
    t0,
    burst_alt: float,
    ascent_rate: float,
    default_b: float,
    profile: FlightProfile,
    ground_fn,
    cfg: Config,
    wind_fn: WindFn | None = None,
    rng,
    n: int | None = None,
    measured_range: tuple[float, float] | None = None,
    stats=None,
) -> EnsembleLanding | None:
    """Vectorised twin of :func:`windfall.ensemble.ensemble_preburst`.

    One lockstep loop covers both legs: each member climbs at its own perturbed
    rate to its own perturbed burst altitude, flips to the perturbed-B descent
    (the overshoot past burst is clamped back, like the scalar handing
    ``burst_k`` to ``integrate_descent``), and lands - a member's field time
    equals the shared sim clock in both versions, so mixing phases in one loop
    is exact, not an approximation. Budget semantics differ slightly: the
    scalar gives *each leg* the full integrator budget and silently treats a
    ran-out-of-budget ascent as having burst; here one combined budget covers
    the whole flight and members still aloft at the end are dropped like failed
    members. Real flights sit far below either limit.
    """
    n = n if n is not None else cfg.ensemble.n_members_preburst
    if n < 1:
        return None
    ec = cfg.ensemble
    dt = ec.dt_seconds
    max_steps = cfg.integrator.max_iterations
    max_sim = cfg.integrator.max_sim_seconds
    nrng = np.random.default_rng(rng.getrandbits(64))

    # --- batched per-member perturbations (drawn once) ---
    rate = np.maximum(
        0.5, ascent_rate * np.exp(nrng.normal(0.0, ec.ascent_rate_sigma_rel, n)))
    burst = np.maximum(
        alt + 100.0, burst_alt + nrng.normal(0.0, ec.burst_alt_sigma_m, n))
    B = default_b * np.exp(nrng.normal(0.0, ec.b_sigma_rel_preburst, n))
    B = np.clip(B, cfg.descent.b_min, cfg.descent.b_max)

    # --- density/measured wind on a fine grid up to the highest burst ---
    grid = np.arange(0.0, float(burst.max()) + 30.0, 10.0)
    dens_g = np.array([profile.density(z) for z in grid])
    dens_pos = bool(dens_g.size) and float(dens_g.min()) > 0.0
    wc_g = None
    if wind_fn is None:
        wc_g = np.empty(grid.size, dtype=np.complex128)
        for i, z in enumerate(grid):
            u, v = profile.wind(float(z))
            wc_g[i] = complex(u, v)

    wind = _MemberWind(wind_fn=wind_fn, profile=profile, cfg=cfg, nrng=nrng, n=n,
                       measured_range=measured_range, stats=stats,
                       grid=grid, wc_g=wc_g)

    phi = np.full(n, math.radians(lat))
    lam = np.full(n, math.radians(lon))
    a = np.full(n, float(alt))
    ascending = np.ones(n, dtype=bool)
    landed = np.zeros(n, dtype=bool)
    llat = np.zeros(n)
    llon = np.zeros(n)
    lsec = np.zeros(n)
    sim_t = 0.0

    # Unlike the descent ensemble, low altitude does not imply near landing here
    # (members *start* low, ascending), so ground sampling is gated on who needs
    # it: descending members below the terrain ceiling (termination), or - with
    # the shared-column blend - any member low enough for the AGL floor.
    last_ground = None
    agl_ceil = _GROUND_CHECK_CEIL_M + max(0.0, cfg.profile.gfs_blend_min_agl_m)

    steps = 0
    while steps < max_steps and sim_t < max_sim:
        active = ~landed
        if not active.any():
            break
        steps += 1
        rho1 = np.interp(a, grid, dens_g)
        if dens_pos:
            v1 = B * rho1 ** -0.5
        else:
            v1 = np.where(rho1 > 0.0, B * np.where(rho1 > 0.0, rho1, 1.0) ** -0.5, 0.0)
        wc1 = wind.eval(phi, lam, a, active, sim_t, last_ground)
        cphi = np.cos(_clip_lat(phi))

        # signed vertical rate: ascenders climb at their constant perturbed
        # rate (no Heun averaging needed - it is altitude-independent, exactly
        # like integrate_ascent), descenders fall at the density-fed B rate
        w1 = np.where(ascending, rate, -v1)
        phi_p = _clip_lat(phi + wc1.imag * (dt / R))
        lam_p = lam + wc1.real * (dt / R) / cphi
        a_p = a + w1 * dt
        rho2 = np.interp(a_p, grid, dens_g)
        if dens_pos:
            v2 = B * rho2 ** -0.5
        else:
            v2 = np.where(rho2 > 0.0, B * np.where(rho2 > 0.0, rho2, 1.0) ** -0.5, v1)
        wc2 = wind.eval(phi_p, lam_p, a_p, active, sim_t + dt, last_ground)

        w = np.where(ascending, rate, -0.5 * (v1 + v2))
        wc = 0.5 * (wc1 + wc2)

        prev_a = a
        prev_phi = phi
        prev_lam = lam
        phi = _clip_lat(phi + wc.imag * (dt / R))
        lam = lam + wc.real * (dt / R) / cphi
        a = a + w * dt
        sim_t += dt

        # burst: the scalar ascent loop exits once alt >= burst and the descent
        # leg restarts at exactly burst_k - clamp the overshoot the same way
        burst_now = ascending & (a >= burst)
        if burst_now.any():
            a = np.where(burst_now, burst, a)
            ascending = ascending & ~burst_now

        descending = active & ~ascending
        d_alt = a[descending]
        need_land = bool(d_alt.size) and float(d_alt.min()) <= _GROUND_CHECK_CEIL_M
        need_agl = wind.gfs_shared and bool(float(a[active].min()) <= agl_ceil)
        if not (need_land or need_agl):
            last_ground = None      # never reuse elevations across a gap
            continue
        # only members low enough to land or trip the AGL floor get sampled;
        # the zeros left for high members satisfy alt - 0 >= min_agl trivially
        low = active & (a <= agl_ceil)
        ground = _ground_at(ground_fn, phi, lam, low, n)
        last_ground = ground
        cross = descending & (a <= ground)
        if cross.any():
            denom = prev_a - a
            safe = denom > 0.0
            frac = np.where(safe, (prev_a - ground) / np.where(safe, denom, 1.0), 1.0)
            frac = np.clip(frac, 0.0, 1.0)
            fphi = prev_phi + frac * (phi - prev_phi)
            flam = prev_lam + frac * (lam - prev_lam)
            idx = np.nonzero(cross)[0]
            llat[idx] = np.degrees(fphi[idx])
            llon[idx] = np.array([normalize_lon(float(x)) for x in np.degrees(flam[idx])])
            lsec[idx] = sim_t - dt * (1.0 - frac[idx])
            landed[idx] = True

    landings = [(float(llat[i]), float(llon[i]), t0 + timedelta(seconds=float(lsec[i])))
                for i in range(n) if landed[i]]
    return _summarise(landings, n, ec.quantile)
