"""Binned wind + density profile for one flight.

The ascent track *is* the wind: each consecutive-frame segment yields a
``(wind_u, wind_v)`` sample at ``alt_mid``. We vector-average samples into 150 m
altitude bins (locked) and store measured density in the same structure. The
profile is the single ``(u, v)(alt)`` / ``rho(alt)`` input to the descent
integrator - whether it was filled from ascent (primary) or GFS (fallback),
the integrator is identical.

Each bin also remembers *where and when* its samples were taken, so the
measured→GFS blend (:func:`blended_wind_fn`) can decay trust in the measured
column as the descending payload drifts away from it or the sample ages.
"""

from __future__ import annotations

import bisect
import math
from typing import Callable, Iterable

from .atmosphere import isa_density, measured_density
from .config import ProfileConfig
from .geo import haversine_km, normalize_lon
from .kinematics import segment
from .models import Frame, WindBin

# 4-D wind field: (lat_deg, lon_deg, alt_m, sim_seconds) -> (u, v) m/s
WindFieldFn = Callable[[float, float, float, float], "tuple[float, float] | None"]


class FlightProfile:
    """Altitude-binned wind/density profile with edge-fill policy."""

    def __init__(self, bin_size_m: float = 150.0, gap_fill_m: float = 600.0):
        self.bin_size_m = bin_size_m
        # interior altitude gaps wider than this are filled from GFS (when
        # available) instead of linearly interpolating across the hole
        self.gap_fill_m = gap_fill_m
        self._bins: dict[int, WindBin] = {}
        # Optional GFS wind filler, used ABOVE/BELOW the sampled range and for
        # wide interior gaps. Signature: (alt_m) -> (u, v) | None.
        self.gfs_fill: Callable[[float], tuple[float, float] | None] | None = None
        self._sorted_index: list[int] | None = None  # cache of sampled bin idxs
        self._rho_index: list[int] | None = None     # ...of bins carrying rho
        self._rho_alts: list[float] | None = None    # bin centres parallel to it

    # ---- construction -----------------------------------------------------
    def _idx(self, alt: float) -> int:
        return int(alt // self.bin_size_m)

    def add_sample(
        self,
        alt: float,
        u: float,
        v: float,
        rho: float | None = None,
        lat: float | None = None,
        lon: float | None = None,
        t: float | None = None,
        weight_cap: int | None = None,
    ) -> None:
        """Vector-average a wind sample (and optional density / position / time
        metadata) into its bin.

        ``weight_cap`` bounds the *effective* weight of what the bin already
        holds: the existing average counts as at most that many samples, so a
        stream of fresh samples (live descent refresh, plan Phase 2) quickly
        dominates a stale ascent average instead of drowning in its count."""
        idx = self._idx(alt)
        b = self._bins.get(idx)
        if b is None:
            center = (idx + 0.5) * self.bin_size_m
            self._bins[idx] = WindBin(alt=center, u=u, v=v, rho=rho, n=1,
                                      lat=lat, lon=lon, t=t)
            self._sorted_index = None
            self._rho_index = None
            return
        n = b.n if weight_cap is None else min(b.n, weight_cap)
        b.u = (b.u * n + u) / (n + 1)
        b.v = (b.v * n + v) / (n + 1)
        if rho is not None:
            if b.rho is None:
                b.rho = rho
                self._rho_index = None
            else:
                # density bins also scalar-average over contributing samples
                b.rho = (b.rho * n + rho) / (n + 1)
        if lat is not None:
            b.lat = lat if b.lat is None else b.lat + (lat - b.lat) / (n + 1)
        if lon is not None:
            if b.lon is None:
                b.lon = lon
            else:
                delta = (lon - b.lon + 180.0) % 360.0 - 180.0
                b.lon = normalize_lon(b.lon + delta / (n + 1))
        if t is not None:
            b.t = t if b.t is None else b.t + (t - b.t) / (n + 1)
        b.n += 1

    # ---- queries ----------------------------------------------------------
    def is_empty(self) -> bool:
        return not self._bins

    @property
    def n_bins(self) -> int:
        """Number of sampled altitude bins."""
        return len(self._bins)

    def alt_range(self) -> tuple[float, float] | None:
        if not self._bins:
            return None
        idxs = self._index()
        return (idxs[0] * self.bin_size_m, (idxs[-1] + 1) * self.bin_size_m)

    def _index(self) -> list[int]:
        if self._sorted_index is None:
            self._sorted_index = sorted(self._bins)
        return self._sorted_index

    def bin_near(self, alt: float) -> WindBin | None:
        """The sampled bin whose centre is nearest ``alt`` (None when empty)."""
        idxs = self._index()
        if not idxs:
            return None
        pos = bisect.bisect_left(idxs, self._idx(alt))
        best: WindBin | None = None
        for cand in idxs[max(0, pos - 1):pos + 1]:
            b = self._bins[cand]
            if best is None or abs(b.alt - alt) < abs(best.alt - alt):
                best = b
        return best

    def wind(self, alt: float) -> tuple[float, float]:
        """Wind ``(u, v)`` at altitude. Inside the sampled range, linearly
        interpolate between the centres of the surrounding sampled bins (the
        half-bins past the end centres clamp). Interior gaps wider than
        ``gap_fill_m`` and altitudes outside the range fill from GFS if
        configured, else clamp/lerp."""
        if not self._bins:
            if self.gfs_fill is not None:
                filled = self.gfs_fill(alt)
                if filled is not None:
                    return filled
            return (0.0, 0.0)
        idxs = self._index()
        lo, hi = idxs[0], idxs[-1]
        idx = self._idx(alt)
        if idx < lo or idx > hi:
            if self.gfs_fill is not None:
                filled = self.gfs_fill(alt)
                if filled is not None:
                    return filled
            edge = self._bins[lo] if idx < lo else self._bins[hi]
            return (edge.u, edge.v)
        # rightmost sampled bin at-or-below alt's bin, then pick the neighbour
        # on alt's side of that bin centre
        pos = bisect.bisect_right(idxs, idx) - 1
        below = self._bins[idxs[pos]]
        if alt < below.alt:
            if pos == 0:
                return (below.u, below.v)
            return self._lerp_or_fill(self._bins[idxs[pos - 1]], below, alt)
        if pos + 1 >= len(idxs):
            return (below.u, below.v)
        return self._lerp_or_fill(below, self._bins[idxs[pos + 1]], alt)

    def _lerp_or_fill(self, a: WindBin, b: WindBin, alt: float) -> tuple[float, float]:
        if (b.alt - a.alt) > self.gap_fill_m and self.gfs_fill is not None:
            filled = self.gfs_fill(alt)
            if filled is not None:
                return filled
        return self._lerp(a, b, alt)

    @staticmethod
    def _lerp(a: WindBin, b: WindBin, alt: float) -> tuple[float, float]:
        span = b.alt - a.alt
        t = 0.0 if span <= 0 else min(1.0, max(0.0, (alt - a.alt) / span))
        return (a.u + t * (b.u - a.u), a.v + t * (b.v - a.v))

    def _rho_idx(self) -> tuple[list[int], list[float]]:
        if self._rho_index is None:
            self._rho_index = sorted(
                i for i, b in self._bins.items() if b.rho is not None and b.rho > 0.0)
            self._rho_alts = [self._bins[i].alt for i in self._rho_index]
        return self._rho_index, self._rho_alts  # type: ignore[return-value]

    def density(self, alt: float) -> float:
        """Air density at altitude. Inside the measured-rho range, log-linear
        interpolation between bin centres (density is exponential-ish in
        altitude). Outside it, ISA *scaled to match the nearest measured edge*,
        so the handoff is continuous instead of stepping onto raw ISA exactly
        where the descent is fastest."""
        idxs, alts = self._rho_idx()
        if not idxs:
            return isa_density(alt)
        lo_b = self._bins[idxs[0]]
        hi_b = self._bins[idxs[-1]]
        if alt <= lo_b.alt:
            return isa_density(alt) * (lo_b.rho / isa_density(lo_b.alt))
        if alt >= hi_b.alt:
            return isa_density(alt) * (hi_b.rho / isa_density(hi_b.alt))
        pos = bisect.bisect_right(alts, alt)
        a = self._bins[idxs[pos - 1]]
        b = self._bins[idxs[pos]]
        span = b.alt - a.alt
        t = 0.0 if span <= 0 else (alt - a.alt) / span
        return a.rho * (b.rho / a.rho) ** t

    def measured_fraction(self, alt_from: float, alt_to: float) -> float:
        """Fraction of the altitude interval [alt_to, alt_from] (a descent from
        ``alt_from`` down to ``alt_to``) covered by the measured sampled range.
        Drives the uncertainty radius."""
        rng = self.alt_range()
        span = alt_from - alt_to
        if rng is None or span <= 0:
            return 0.0
        lo = max(alt_to, rng[0])
        hi = min(alt_from, rng[1])
        return max(0.0, min(1.0, (hi - lo) / span))

    # ---- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "bin_size_m": self.bin_size_m,
            "bins": [b.to_json() for _, b in sorted(self._bins.items())],
        }

    @classmethod
    def from_json(cls, d: dict) -> "FlightProfile":
        p = cls(bin_size_m=d.get("bin_size_m", 150.0))
        for raw in d.get("bins", []):
            b = WindBin.from_json(raw)
            p._bins[p._idx(b.alt)] = b
        p._sorted_index = None
        p._rho_index = None
        return p


def blended_wind_fn(
    profile: FlightProfile,
    gfs_fn: WindFieldFn,
    cfg: ProfileConfig,
    t0_epoch: float | None = None,
    ground_fn: Callable[[float, float], float] | None = None,
) -> WindFieldFn:
    """Blend the measured ascent column with a 4-D GFS field.

    Inside the measured range the weight on the measured wind decays
    exponentially with (a) the horizontal distance between the integrator's
    current position and where that altitude was actually sampled, and (b) the
    age of the sample at integration time. Altitudes falling in a wide
    unsampled hole, or outside the measured range entirely, use pure GFS.
    Below ``gfs_blend_min_agl_m`` above ground (when a ground model is wired)
    the measured column is ignored outright: boundary-layer winds at the
    landing zone are terrain-local, not what the launch site measured.
    """
    rng = profile.alt_range()
    d0 = max(cfg.gfs_blend_distance_km, 1e-6)
    a0 = max(cfg.gfs_blend_age_s, 1e-6)

    def wind(lat: float, lon: float, alt: float, sim_t: float):
        g = gfs_fn(lat, lon, alt, sim_t)
        if rng is None or not (rng[0] <= alt < rng[1]):
            return g if g is not None else profile.wind(alt)
        mu, mv = profile.wind(alt)
        if g is None:
            return (mu, mv)
        if ground_fn is not None and cfg.gfs_blend_min_agl_m > 0.0 \
                and (alt - ground_fn(lat, lon)) < cfg.gfs_blend_min_agl_m:
            return g
        b = profile.bin_near(alt)
        w = 1.0
        if b is not None:
            if abs(b.alt - alt) > cfg.interior_gap_fill_m:
                return g  # unsampled hole → trust GFS outright
            arg = 0.0
            if b.lat is not None and b.lon is not None:
                arg += haversine_km(lat, lon, b.lat, b.lon) / d0
            if t0_epoch is not None and b.t is not None:
                arg += max(0.0, (t0_epoch + sim_t) - b.t) / a0
            w = math.exp(-arg)
        gu, gv = g
        return (w * mu + (1.0 - w) * gu, w * mv + (1.0 - w) * gv)

    return wind


def plausible_segment(seg, cfg: ProfileConfig) -> bool:
    """Plausibility gate against GPS glitches before binning."""
    if seg is None or seg.dt < cfg.min_dt_seconds:
        return False
    if seg.horizontal_speed > cfg.max_horizontal_mps:
        return False
    if abs(seg.vertical_rate) > cfg.max_alt_step_mps:
        return False
    return True


def update_profile_from_pair(
    profile: FlightProfile, a: Frame, b: Frame, cfg: ProfileConfig,
    weight_cap: int | None = None,
) -> bool:
    """Add the wind/density sample from consecutive ascent frames (a, b) to the
    profile if it passes the plausibility gate. Returns True if added."""
    seg = segment(a, b)
    if not plausible_segment(seg, cfg):
        return False
    # Density from the *upper* frame's P/T (associated with alt_mid is fine -
    # density varies slowly with altitude across one 150 m bin).
    rho = None
    if b.pressure is not None and b.temp is not None:
        rho = measured_density(b.pressure, b.temp)
    lon_mid = normalize_lon(a.lon + ((b.lon - a.lon + 180.0) % 360.0 - 180.0) / 2.0)
    profile.add_sample(seg.alt_mid, seg.wind_u, seg.wind_v, rho,
                       lat=(a.lat + b.lat) / 2.0, lon=lon_mid, t=(a.t + b.t) / 2.0,
                       weight_cap=weight_cap)
    return True


def build_ascent_profile(
    frames: Iterable[Frame], cfg: ProfileConfig | None = None
) -> FlightProfile:
    """Build a profile from an ordered sequence of ascent frames."""
    cfg = cfg or ProfileConfig()
    profile = FlightProfile(bin_size_m=cfg.bin_size_m, gap_fill_m=cfg.interior_gap_fill_m)
    prev: Frame | None = None
    for f in frames:
        if prev is not None:
            update_profile_from_pair(profile, prev, f, cfg)
        prev = f
    return profile


def bias_corrected_wind_fn(
    profile: FlightProfile,
    gfs_fn: WindFieldFn,
    cfg: ProfileConfig,
    t0_epoch: float | None = None,
    ground_fn: Callable[[float, float], float] | None = None,
) -> WindFieldFn:
    """The plan-Phase-2 bias formulation: ``wind = model + w·(measured - model
    at the place/time the layer was sampled)``.

    Unlike :func:`blended_wind_fn` (which *replaces* model wind with the
    measured value as w→1), the bias mode keeps the model field's own spatial
    and temporal structure and only shifts it by the measured-minus-model
    offset Δ(z) observed where the sonde actually flew. Δ uses the nearest
    sampled bin (bias varies far more slowly with altitude than wind itself)
    and is memoised per bin for the lifetime of this callable - one prediction
    cycle. The decay weight and the AGL floor are shared with the blend.
    """
    rng = profile.alt_range()
    d0 = max(cfg.gfs_blend_distance_km, 1e-6)
    a0 = max(cfg.gfs_blend_age_s, 1e-6)
    deltas: dict[int, tuple[float, float] | None] = {}

    def _delta(b: WindBin) -> tuple[float, float] | None:
        key = profile._idx(b.alt)
        if key in deltas:
            return deltas[key]
        # model wind where/when this layer was measured; sim_t is relative to
        # t0 (negative = in the past; the field clamps to its earliest cube)
        sim_t = 0.0
        if t0_epoch is not None and b.t is not None:
            sim_t = b.t - t0_epoch
        model = gfs_fn(b.lat if b.lat is not None else 0.0,
                       b.lon if b.lon is not None else 0.0,
                       b.alt, sim_t) if (b.lat is not None and b.lon is not None) else None
        delta = None if model is None else (b.u - model[0], b.v - model[1])
        deltas[key] = delta
        return delta

    def wind(lat: float, lon: float, alt: float, sim_t: float):
        g = gfs_fn(lat, lon, alt, sim_t)
        if g is None:
            return profile.wind(alt)
        if rng is None or not (rng[0] <= alt < rng[1]):
            return g
        if ground_fn is not None and cfg.gfs_blend_min_agl_m > 0.0 \
                and (alt - ground_fn(lat, lon)) < cfg.gfs_blend_min_agl_m:
            return g
        b = profile.bin_near(alt)
        if b is None or abs(b.alt - alt) > cfg.interior_gap_fill_m:
            return g
        delta = _delta(b)
        if delta is None:
            return g
        arg = 0.0
        if b.lat is not None and b.lon is not None:
            arg += haversine_km(lat, lon, b.lat, b.lon) / d0
        if t0_epoch is not None and b.t is not None:
            arg += max(0.0, (t0_epoch + sim_t) - b.t) / a0
        w = math.exp(-arg)
        return (g[0] + w * delta[0], g[1] + w * delta[1])

    return wind
