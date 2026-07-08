"""GFS fallback wind source.

Used when a flight is first seen mid-descent (no ascent profile) or for a
pre-burst estimate. There is **no external predictor binary** - GFS is just an
alternate way to fill the *same* binned ``(u, v)(alt)`` / ``rho(alt)`` profile
the measured path produces, then the **identical** descent integrator runs.

Beyond the column fallback, the source exposes a full **4-D wind field**
(:meth:`GFSWindSource.wind_field`): bilinear in lat/lon, linear in altitude
between isobaric levels, linear in time between the two forecast valid hours
bracketing the prediction. The integrator queries it at its *current* position
every step, so a trajectory drifting 100 km samples the winds where it actually
is - not a frozen column at the release point.

Layers:
* pure, testable helpers (``parse_grib_path``, ``select_bracketing``,
  :class:`WindCube`, ``build_profile_from_levels``, ``estimate_burst_alt``,
  ``pressure_to_altitude``);
* :class:`StaticGFSSource` - a fixed profile for tests/demos;
* :class:`HerbieGFSSource` - reads downloaded GRIB regions via cfgrib/xarray
  (lazy import). The GRIB is fetched by the ``gfs-downloader`` cron.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .atmosphere import G0, P0_PA, R_D, T0_K, isa_density
from .config import Config
from .geo import BBox, normalize_lon
from .predictor import GFSWindSource
from .profile import FlightProfile

log = logging.getLogger(__name__)

# Typical burst altitudes by sonde type (m), for pre-burst estimates.
_TYPE_BURST_ALT = {
    "RS41": 35_000.0,
    "RS92": 33_000.0,
    "DFM": 33_000.0,
    "M10": 30_000.0,
    "M20": 30_000.0,
    "IMET": 30_000.0,
}
_DEFAULT_BURST_ALT = 30_000.0
# Physical ceiling for a *timer-derived* burst estimate. The telemetry
# ``burst_timer`` is frequently the burst-KILL countdown (e.g. 30600 s, or a
# 65535 sentinel), not time-to-burst - taken at face value it would put burst
# at 100+ km. Estimates above this ceiling mean the timer wasn't time-to-burst;
# fall through to the site/type default instead.
_MAX_BURST_ALT = 38_000.0


@dataclass(slots=True)
class GFSLevel:
    """One GFS isobaric level sample at a point."""

    height_m: float          # geopotential height (≈ geometric) of the level
    u: float                 # eastward wind, m/s
    v: float                 # northward wind, m/s
    pressure_pa: float | None = None
    temp_k: float | None = None


def pressure_to_altitude(pressure_pa: float) -> float:
    """ISA-inverse altitude (m) for a pressure (Pa) - fallback when GFS
    geopotential height is unavailable. Valid through the troposphere/low
    stratosphere covering the sonde regime."""
    lapse = 0.0065
    if pressure_pa >= 22_632.0:   # below ~11 km
        return (T0_K / lapse) * (1.0 - (pressure_pa / P0_PA) ** (R_D * lapse / G0))
    # isothermal layer above the tropopause
    h_trop = 11_000.0
    p_trop = 22_632.0
    t_trop = 216.65
    return h_trop + (R_D * t_trop / G0) * math.log(p_trop / pressure_pa)


def density_at(level: GFSLevel) -> float | None:
    if level.pressure_pa is not None and level.temp_k is not None and level.temp_k > 0:
        return level.pressure_pa / (R_D * level.temp_k)
    return None


def build_profile_from_levels(
    levels: list[GFSLevel], bin_size_m: float = 150.0
) -> FlightProfile:
    """Convert GFS isobaric levels to the binned profile the integrator uses."""
    prof = FlightProfile(bin_size_m=bin_size_m)
    for lv in sorted(levels, key=lambda x: x.height_m):
        rho = density_at(lv)
        if rho is None:
            rho = isa_density(lv.height_m)
        prof.add_sample(lv.height_m, lv.u, lv.v, rho)
    return prof


def estimate_burst_alt(
    *,
    current_alt: float,
    ascent_rate: float | None,
    burst_timer: float | None,
    sonde_type: str | None,
    site_burst_alt: float | None = None,
    cfg: Config | None = None,
) -> float:
    """Estimate burst altitude for a pre-burst prediction.

    Priority: burst_timer + ascent rate → site climatology (the median observed
    burst altitude of past flights launched nearby - same balloon batch, same
    fill, twice daily) → sonde-type default → generic default. Never below the
    current altitude."""
    if burst_timer is not None and burst_timer > 0 and ascent_rate is not None and ascent_rate > 0:
        est = current_alt + ascent_rate * burst_timer
        if est <= _MAX_BURST_ALT:
            return max(est, current_alt + 100.0)
        # implausibly high → the timer was a kill countdown, not time-to-burst
    if site_burst_alt is not None and 5_000.0 < site_burst_alt <= _MAX_BURST_ALT:
        return max(site_burst_alt, current_alt + 100.0)
    if sonde_type:
        for key, alt in _TYPE_BURST_ALT.items():
            if sonde_type.upper().startswith(key):
                return max(alt, current_alt + 100.0)
    return max(_DEFAULT_BURST_ALT, current_alt + 100.0)


# ---- GRIB inventory (pure, filename-based) ---------------------------------

_CYCLE_RE = re.compile(r"\.t(\d{2})z\.")
# GFS subsets carry ``.f003``; HRRR files are ``...wrfprsf03.grib2`` etc.
_FXX_RE = re.compile(r"(?:\.f|wrf(?:prs|sfc|nat)f)(\d{2,3})(?=$|[._])")
_DATE_RE = re.compile(r"(20\d{6})")


@dataclass(slots=True, frozen=True)
class GribFileInfo:
    """Cycle/forecast-hour identity of one downloaded GRIB file."""

    path: str
    cycle: datetime          # run time, tz-aware UTC
    fxx: int                 # forecast hour

    @property
    def valid_time(self) -> datetime:
        return self.cycle + timedelta(hours=self.fxx)


def parse_grib_path(path) -> GribFileInfo | None:
    """Parse cycle date/hour + forecast hour from a Herbie-style GRIB path,
    e.g. ``.../gfs/20260609/subset_ab12__gfs.t06z.pgrb2.0p25.f003``. The date
    usually lives in the parent directory name; the *last* date-looking token in
    the path is the one nearest the file. Returns None if it doesn't parse."""
    s = str(path)
    name = Path(s).name
    m_cyc = _CYCLE_RE.search(name)
    m_fxx = _FXX_RE.search(name)
    dates = _DATE_RE.findall(s)
    if not (m_cyc and m_fxx and dates):
        return None
    try:
        day = datetime.strptime(dates[-1], "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return GribFileInfo(path=s, cycle=day + timedelta(hours=int(m_cyc.group(1))),
                        fxx=int(m_fxx.group(1)))


def select_bracketing(
    infos: list[GribFileInfo], when: datetime,
    min_age_hours: float = 0.0,
) -> tuple[GribFileInfo, GribFileInfo] | None:
    """Pick the two files of the *newest cycle* whose valid times bracket
    ``when`` (clamping to the cycle's ends when ``when`` falls outside). The
    newest run is always the best forecast; mixing cycles would interpolate
    between inconsistent model states.

    Cycles *run after* ``when`` are refused outright - None when nothing else
    exists, never a future cycle. Live this never triggers (runs are always in
    the past), but a backtest cache may hold cycles newer than the flight, and
    serving one silently would be lookahead bias - a forecast that did not
    exist when the sonde flew, biasing the score it was meant to measure.
    ``min_age_hours`` models *publication latency* the same way: a 12Z GFS
    cycle isn't on the open-data bucket until ~15:45Z, so a backtest of a 13Z
    flight must not use it even though the run time itself is in the past.
    Live this is a no-op too - the cache only ever holds cycles that were
    already published."""
    if not infos:
        return None
    cutoff = when - timedelta(hours=min_age_hours)
    past = [i.cycle for i in infos if i.cycle <= cutoff]
    if not past:
        log.warning(
            "no GRIB cycle published by %s (publication latency %.1f h) - "
            "refusing the %d future cycle(s) in the cache rather than look ahead",
            when.isoformat(), min_age_hours, len({i.cycle for i in infos}))
        return None
    cycle = max(past)
    cands = sorted((i for i in infos if i.cycle == cycle), key=lambda i: i.fxx)
    before = [i for i in cands if i.valid_time <= when]
    after = [i for i in cands if i.valid_time >= when]
    lo = before[-1] if before else cands[0]
    hi = after[0] if after else cands[-1]
    return (lo, hi)


# ---- 4-D wind field (pure numpy interpolation, testable offline) -----------

def window_indices(
    lats: np.ndarray, lons: np.ndarray, lat: float, lon: float, half_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Indices of the ``±half_deg`` window around (lat, lon) on a regular
    lat/lon grid, plus the window's longitude axis in a convention
    :meth:`WindCube._qlon` resolves correctly.

    GFS files are global (download subsets *levels*, not area): a full cube is
    ~1 GB of float arrays, the windowed one a few MB. Longitudes are matched on
    the circle, so a window crossing the grid's 0/360 seam comes back as a
    non-contiguous index set ordered west→east with an unwrapped, ascending
    axis. None when the window misses the grid entirely (regional file)."""
    lat_idx = np.where(np.abs(lats - lat) <= half_deg)[0]
    qlon = lon % 360.0
    start = qlon - half_deg
    offset = (lons - start) % 360.0
    lon_idx = np.where(offset <= 2.0 * half_deg)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        return None
    order = np.argsort(offset[lon_idx])
    lon_idx = lon_idx[order]
    lons_out = start + offset[lon_idx]
    if lons_out[-1] >= 360.0:     # keep east-of-Greenwich windows resolvable
        lons_out = lons_out - 360.0
    return lat_idx, lon_idx, lons_out


def _level_first(a) -> np.ndarray:
    """Coerce a (lat, lon) grid to (1, lat, lon) - the level axis leads."""
    a = np.asarray(a, dtype=float)
    return a[None] if a.ndim == 2 else a


def _axis_pos(axis: np.ndarray, x: float) -> tuple[int, int, float]:
    """Bracketing indices + fractional weight along a sorted-ascending axis,
    clamped at the ends."""
    if x <= axis[0] or axis.size == 1:
        return 0, 0, 0.0
    if x >= axis[-1]:
        n = axis.size - 1
        return n, n, 0.0
    j = int(np.searchsorted(axis, x))
    j0 = j - 1
    return j0, j, float((x - axis[j0]) / (axis[j] - axis[j0]))


@dataclass(slots=True)
class WindCube:
    """One forecast hour's (level, lat, lon) grids, ready for interpolation.

    ``wind_at`` is bilinear in the horizontal and linear in altitude along each
    corner column's own geopotential heights - the correct order (interpolate
    vertically per column first, then blend horizontally), since level heights
    vary across the grid."""

    valid_time: datetime
    lats: np.ndarray         # (nlat,) ascending
    lons: np.ndarray         # (nlon,) ascending, in the grid's own convention
    heights: np.ndarray      # (nlev, nlat, nlon) metres, level axis ascending
    u: np.ndarray            # (nlev, nlat, nlon)
    v: np.ndarray
    temp_k: np.ndarray | None = None
    pressure_pa: np.ndarray | None = None    # (nlev,)
    # Projected-grid support (HRRR's Lambert-conformal 3 km grid): when set,
    # ``lats``/``lons`` hold the grid's regular y/x axes in METRES and ``proj``
    # maps (lat_deg, lon_deg) -> (y_m, x_m). All interpolation logic is shared.
    proj: Callable[[float, float], "tuple[float, float]"] | None = None

    @classmethod
    def from_grid(cls, *, valid_time, lats, lons, heights, u, v,
                  temp_k=None, pressure_pa=None, proj=None) -> "WindCube":
        """Normalise raw GRIB arrays: latitudes are usually stored descending,
        levels by descending pressure; sort everything ascending."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        heights = _level_first(heights)
        u = _level_first(u)
        v = _level_first(v)
        if temp_k is not None:
            temp_k = _level_first(temp_k)
        if pressure_pa is not None:
            pressure_pa = np.atleast_1d(np.asarray(pressure_pa, dtype=float))
        if lats.size > 1 and lats[0] > lats[-1]:
            lats = lats[::-1].copy()
            heights = heights[:, ::-1]
            u, v = u[:, ::-1], v[:, ::-1]
            temp_k = temp_k[:, ::-1] if temp_k is not None else None
        if lons.size > 1 and lons[0] > lons[-1]:
            lons = lons[::-1].copy()
            heights = heights[:, :, ::-1]
            u, v = u[:, :, ::-1], v[:, :, ::-1]
            temp_k = temp_k[:, :, ::-1] if temp_k is not None else None
        order = np.argsort(np.nanmean(heights.reshape(heights.shape[0], -1), axis=1))
        heights, u, v = heights[order], u[order], v[order]
        temp_k = temp_k[order] if temp_k is not None else None
        pressure_pa = pressure_pa[order] if pressure_pa is not None else None
        return cls(valid_time=valid_time, lats=lats, lons=lons,
                   heights=np.ascontiguousarray(heights),
                   u=np.ascontiguousarray(u), v=np.ascontiguousarray(v),
                   temp_k=np.ascontiguousarray(temp_k) if temp_k is not None else None,
                   pressure_pa=pressure_pa, proj=proj)

    def _qlon(self, lon: float) -> float:
        # GFS native grids run 0..360; regional subsets keep that convention
        return lon % 360.0 if self.lons[-1] > 180.0 else normalize_lon(lon)

    def _corners(self, lat: float, lon: float):
        if self.proj is not None:
            y, x = self.proj(lat, lon)
            i0, i1, wy = _axis_pos(self.lats, y)
            j0, j1, wx = _axis_pos(self.lons, x)
        else:
            i0, i1, wy = _axis_pos(self.lats, lat)
            j0, j1, wx = _axis_pos(self.lons, self._qlon(lon))
        return (
            (i0, j0, (1.0 - wy) * (1.0 - wx)),
            (i0, j1, (1.0 - wy) * wx),
            (i1, j0, wy * (1.0 - wx)),
            (i1, j1, wy * wx),
        )

    def wind_at(self, lat: float, lon: float, alt: float) -> tuple[float, float]:
        out_u = 0.0
        out_v = 0.0
        for i, j, w in self._corners(lat, lon):
            if w <= 0.0:
                continue
            h = self.heights[:, i, j]
            out_u += w * float(np.interp(alt, h, self.u[:, i, j]))
            out_v += w * float(np.interp(alt, h, self.v[:, i, j]))
        return (out_u, out_v)

    def column_arrays(self, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Horizontally-blended ``(heights, u, v)`` level arrays at a point -
        the fast path behind the column cache (one vector op per corner)."""
        nlev = self.heights.shape[0]
        h = np.zeros(nlev)
        uu = np.zeros(nlev)
        vv = np.zeros(nlev)
        for i, j, w in self._corners(lat, lon):
            if w <= 0.0:
                continue
            h += w * self.heights[:, i, j]
            uu += w * self.u[:, i, j]
            vv += w * self.v[:, i, j]
        return h, uu, vv

    def column(self, lat: float, lon: float) -> list[GFSLevel]:
        """The horizontally-interpolated level column at a point."""
        nlev = self.heights.shape[0]
        h = np.zeros(nlev)
        uu = np.zeros(nlev)
        vv = np.zeros(nlev)
        tt = np.zeros(nlev) if self.temp_k is not None else None
        for i, j, w in self._corners(lat, lon):
            if w <= 0.0:
                continue
            h += w * self.heights[:, i, j]
            uu += w * self.u[:, i, j]
            vv += w * self.v[:, i, j]
            if tt is not None:
                tt += w * self.temp_k[:, i, j]
        return [
            GFSLevel(
                height_m=float(h[k]), u=float(uu[k]), v=float(vv[k]),
                pressure_pa=float(self.pressure_pa[k]) if self.pressure_pa is not None else None,
                temp_k=float(tt[k]) if tt is not None else None,
            )
            for k in range(nlev)
        ]


@dataclass(slots=True)
class VarGrid:
    """One GRIB variable's (level, lat, lon) grids, as read per-shortName."""

    lats: np.ndarray
    lons: np.ndarray
    levels_hpa: np.ndarray   # (nlev,)
    vals: np.ndarray         # (nlev, nlat, nlon)

    def by_level(self) -> dict[float, np.ndarray]:
        # round to kill float jitter between messages of the same file
        return {round(float(l), 3): self.vals[k] for k, l in enumerate(self.levels_hpa)}


def assemble_cube(
    valid_time: datetime,
    u: VarGrid | None,
    v: VarGrid | None,
    gh: VarGrid | None = None,
    t: VarGrid | None = None,
    proj=None,
) -> WindCube | None:
    """Build a :class:`WindCube` from independently-read variable grids.

    GFS pgrb2 carries HGT/TMP on more isobaric levels than UGRD/VGRD, so the
    variables cannot be read through one cfgrib filter (the winds get dropped as
    conflicting). Instead each variable is read alone and joined here on the
    levels that have *both* wind components; heights/temperature come from
    gh/t where those levels exist, ISA-inverse heights otherwise. Temperature is
    attached only when it covers every wind level (partial columns would bias
    the density profile).
    """
    if u is None or v is None:
        return None
    v_by = v.by_level()
    gh_by = gh.by_level() if gh is not None else {}
    t_by = t.by_level() if t is not None else {}
    common = [
        (k, round(float(lvl), 3)) for k, lvl in enumerate(u.levels_hpa)
        if round(float(lvl), 3) in v_by
    ]
    if not common:
        return None
    nlat, nlon = u.vals.shape[1], u.vals.shape[2]
    heights = np.empty((len(common), nlat, nlon), dtype=float)
    uu = np.empty_like(heights)
    vv = np.empty_like(heights)
    temps = np.empty_like(heights)
    have_all_temps = bool(t_by)
    pressures = np.empty(len(common), dtype=float)
    for i, (k, lvl) in enumerate(common):
        pressures[i] = lvl * 100.0
        uu[i] = u.vals[k]
        vv[i] = v_by[lvl]
        if lvl in gh_by:
            heights[i] = gh_by[lvl]
        else:
            heights[i].fill(pressure_to_altitude(lvl * 100.0))
        if lvl in t_by:
            temps[i] = t_by[lvl]
        else:
            have_all_temps = False
    return WindCube.from_grid(
        valid_time=valid_time, lats=u.lats, lons=u.lons,
        heights=heights, u=uu, v=vv,
        temp_k=temps if have_all_temps else None,
        pressure_pa=pressures, proj=proj,
    )


class CubePairWind:
    """Time-interpolating 4-D wind field between two forecast valid times.

    Callable as ``(lat_deg, lon_deg, alt_m, sim_seconds) -> (u, v)`` with the
    simulation time measured from the anchor ``t0`` (the prediction time) - the
    signature the integrator's ``wind_fn`` hook expects.

    Sampling is column-cached: queries are bucketed to ~5 km / 5 min (far finer
    than the 0.25° grid and hourly steps carrying the data), and each bucket
    pre-blends one ``(heights, u, v)`` column. A per-step call is then a dict
    hit + two ``np.interp`` - without this, every integrator step costs 16
    corner interpolations and an ensemble refresh stalls the ingest thread."""

    _Q_INV_DEG = 20.0        # 1/20° ≈ 5 km buckets
    _Q_SECONDS = 300.0
    _CACHE_MAX = 4096

    def __init__(self, a: WindCube, b: WindCube, t0: datetime):
        if b.valid_time < a.valid_time:
            a, b = b, a
        self.a = a
        self.b = b
        self._t0 = t0.timestamp()
        self._ta = a.valid_time.timestamp()
        span = b.valid_time.timestamp() - self._ta
        self._span = span if span > 0 else None
        self._cols: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _w(self, sim_t: float) -> float:
        if self._span is None:
            return 0.0
        return min(1.0, max(0.0, (self._t0 + sim_t - self._ta) / self._span))

    def _column(self, lat: float, lon: float, sim_t: float):
        key = (round(lat * self._Q_INV_DEG), round(lon * self._Q_INV_DEG),
               int((self._t0 + sim_t) // self._Q_SECONDS))
        col = self._cols.get(key)
        if col is None:
            ha, ua, va = self.a.column_arrays(lat, lon)
            w = self._w(sim_t)
            if w <= 0.0:
                col = (ha, ua, va)
            else:
                hb, ub, vb = self.b.column_arrays(lat, lon)
                if hb.shape != ha.shape:
                    col = (hb, ub, vb) if w >= 0.5 else (ha, ua, va)
                else:
                    col = (ha + w * (hb - ha), ua + w * (ub - ua), va + w * (vb - va))
            # FIFO-evict rather than clear(): a full clear mid-integration makes
            # every subsequent step a 16-corner recompute at once
            while len(self._cols) >= self._CACHE_MAX:
                self._cols.pop(next(iter(self._cols)))
            self._cols[key] = col
        return col

    def __call__(self, lat: float, lon: float, alt: float, sim_t: float = 0.0):
        h, u, v = self._column(lat, lon, sim_t)
        return (float(np.interp(alt, h, u)), float(np.interp(alt, h, v)))

    def column_levels(self, lat: float, lon: float, sim_t: float = 0.0) -> list[GFSLevel]:
        cols_a = self.a.column(lat, lon)
        w = self._w(sim_t)
        if w <= 0.0:
            return cols_a
        cols_b = self.b.column(lat, lon)
        if len(cols_a) != len(cols_b):
            return cols_a if w < 0.5 else cols_b
        out = []
        for la, lb in zip(cols_a, cols_b):
            temp = None
            if la.temp_k is not None and lb.temp_k is not None:
                temp = la.temp_k + w * (lb.temp_k - la.temp_k)
            out.append(GFSLevel(
                height_m=la.height_m + w * (lb.height_m - la.height_m),
                u=la.u + w * (lb.u - la.u),
                v=la.v + w * (lb.v - la.v),
                pressure_pa=la.pressure_pa,
                temp_k=temp,
            ))
        return out


class StaticGFSSource(GFSWindSource):
    """A fixed profile returned for any location/time - for tests and demos."""

    def __init__(self, profile: FlightProfile):
        self._profile = profile

    def profile_at(self, lat: float, lon: float, when: datetime) -> FlightProfile | None:
        # return a copy so callers can attach a gfs_fill without mutating ours
        return copy.deepcopy(self._profile)

    def wind_filler(self, lat: float, lon: float, when: datetime):
        prof = self._profile
        return lambda alt: prof.wind(alt)

    def wind_field(self, lat: float, lon: float, when: datetime):
        prof = self._profile
        return lambda la, lo, alt, sim_t: prof.wind(alt)


def scan_inventory(path: Path) -> list[GribFileInfo]:
    """Cycle/forecast-hour inventory of every GRIB under ``path``, parsed
    from filenames alone (shared by the GFS and HRRR sources)."""
    infos: list[GribFileInfo] = []
    try:
        if path.exists():
            for p in path.rglob("*"):
                if not p.is_file() or p.suffix == ".idx":
                    continue
                info = parse_grib_path(p)
                if info is not None:
                    infos.append(info)
    except OSError:
        log.exception("failed to scan GRIB inventory at %s", path)
    return infos


class HerbieGFSSource(GFSWindSource):
    """Read downloaded GFS GRIB and serve the 4-D interpolating wind field.

    The ``gfs-downloader`` cron writes one cycle's files into ``path`` - level-subset but spatially *global* (GRIB byte ranges are
    per-message). The inventory is scanned from *filenames* (cycle + forecast
    hour), the two files bracketing the prediction time are loaded into
    :class:`WindCube`s windowed ``±gfs.window_deg`` around the flight (a global
    cube is ~1 GB; the window a few MB), and all sampling interpolates - never
    nearest-file / nearest-gridpoint. Requires cfgrib/xarray (lazy import);
    degrades to None if unavailable so the predictor falls back to measured
    winds only."""

    _INVENTORY_TTL_S = 60.0
    _CUBE_CACHE_MAX = 6      # windowed cubes are a few MB each

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = Path(cfg.gfs.path)
        self._cubes: dict[tuple, WindCube | None] = {}
        self._inv: list[GribFileInfo] = []
        self._inv_at: float | None = None

    # ---- public API --------------------------------------------------------
    def profile_at(self, lat: float, lon: float, when: datetime) -> FlightProfile | None:  # pragma: no cover - needs GRIB
        pair = self._pair(when, lat, lon)
        if pair is None:
            return None
        levels = pair.column_levels(lat, lon, 0.0)
        if not levels:
            return None
        return build_profile_from_levels(levels, self.cfg.profile.bin_size_m)

    def wind_field(self, lat: float, lon: float, when: datetime):  # pragma: no cover - needs GRIB
        return self._pair(when, lat, lon)

    def wind_filler(self, lat: float, lon: float, when: datetime):  # pragma: no cover - needs GRIB
        # Lazy: edge-fill is rarely hit on the measured path, so don't read GRIB
        # until wind() actually queries outside the sampled range.
        holder: dict[str, CubePairWind | None] = {}

        def fill(alt: float):
            if "p" not in holder:
                holder["p"] = self._pair(when, lat, lon)
            pair = holder["p"]
            if pair is None:
                return None
            return pair(lat, lon, alt, 0.0)

        return fill

    # ---- internals ----------------------------------------------------------
    def _inventory(self) -> list[GribFileInfo]:
        now = time.monotonic()
        if self._inv_at is not None and now - self._inv_at < self._INVENTORY_TTL_S:
            return self._inv
        self._inv = scan_inventory(self.path)
        self._inv_at = now
        return self._inv

    def _pair(self, when: datetime, lat: float, lon: float) -> CubePairWind | None:  # pragma: no cover - needs GRIB
        sel = select_bracketing(self._inventory(), when,
                                self.cfg.gfs.publication_latency_hours)
        if sel is None:
            return None
        a = self._cube(sel[0], lat, lon)
        b = a if sel[1].path == sel[0].path else self._cube(sel[1], lat, lon)
        if a is None or b is None:
            return None
        return CubePairWind(a, b, t0=when)

    def _cube(self, info: GribFileInfo, lat: float, lon: float) -> WindCube | None:  # pragma: no cover - needs GRIB
        # window-bucketed bounded cache, like HerbieHRRRSource: flights within
        # the same ~1° share a cube, and an unbounded full-file cache of global
        # grids grows by the gigabyte per forecast hour crossed
        key = (info.path, round(lat), round(lon))
        if key in self._cubes:
            return self._cubes[key]
        cube = None
        try:
            import xarray as xr
        except ImportError:
            log.warning("xarray/cfgrib not installed; GFS source disabled")
            self._cubes[key] = None
            return None
        try:
            # One open per variable: in GFS pgrb2, HGT/TMP exist on MORE
            # isobaric levels than UGRD/VGRD, and cfgrib silently drops the
            # conflicting wind variables when everything is opened through one
            # typeOfLevel filter. Per-shortName opens can never conflict.
            grids = {
                short: self._read_var(xr, info.path, short, lat, lon,
                                      self.cfg.gfs.window_deg)
                for short in ("u", "v", "gh", "t")
            }
            cube = assemble_cube(info.valid_time, **grids)
            if cube is None:
                log.warning("GFS file %s has no usable wind levels", info.path)
        except Exception:
            log.exception("failed to load GFS cube from %s", info.path)
        while len(self._cubes) >= self._CUBE_CACHE_MAX:
            self._cubes.pop(next(iter(self._cubes)))   # FIFO, keep recent cubes
        self._cubes[key] = cube
        return cube

    @staticmethod
    def _read_var(xr, path: str, short: str, lat: float, lon: float,
                  window_deg: float) -> VarGrid | None:  # pragma: no cover - needs GRIB
        """Read one variable's isobaric (level, lat, lon) grids windowed
        ``±window_deg`` around the flight, or None when the file doesn't carry
        it (e.g. pre-TMP downloads). Falls back to the full grid when the
        window misses it (a regional subset from another deployment)."""
        try:
            # indexpath="": don't read/write cfgrib's on-disk pickle index -
            # parallel backtest workers race on it (truncated index -> EOFError
            # noise); a fresh header scan per open costs ~a second.
            ds = xr.open_dataset(
                path, engine="cfgrib",
                backend_kwargs={"indexpath": "", "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa", "shortName": short}},
            )
        except Exception:
            return None
        try:
            if short not in ds:
                return None
            lats = np.asarray(ds["latitude"].values, dtype=float)
            lons = np.asarray(ds["longitude"].values, dtype=float)
            vals = _level_first(ds[short].values)
            win = window_indices(lats, lons, lat, lon, window_deg)
            if win is not None:
                lat_idx, lon_idx, lons = win
                lats = lats[lat_idx]
                vals = vals[:, lat_idx][:, :, lon_idx]
            return VarGrid(
                lats=lats,
                lons=lons,
                levels_hpa=np.atleast_1d(ds["isobaricInhPa"].values).astype(float),
                vals=vals,
            )
        finally:
            ds.close()


def prune_grib_cache(path: Path, keep_hours: float,
                     now: datetime | None = None) -> int:
    """Delete cached GRIB files (and their ``.idx`` sidecars) whose *cycle* is
    older than ``keep_hours``. Returns the number of GRIB files removed.
    No-op when ``keep_hours <= 0``."""
    if keep_hours <= 0:
        return 0
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=keep_hours)
    n = 0
    for info in scan_inventory(path):
        if info.cycle >= cutoff:
            continue
        p = Path(info.path)
        for victim in (p, *p.parent.glob(p.name + "*.idx")):
            try:
                victim.unlink(missing_ok=True)
            except OSError:
                log.warning("could not prune %s", victim)
        n += 1
    if n:
        log.info("pruned %d GRIB file(s) older than %.0f h from %s",
                 n, keep_hours, path)
    return n


def download_gfs_cycle(
    cfg: Config,
    box: BBox | None,
    fxx: list[int] | None = None,
    levels: str | None = None,
    run_date: datetime | None = None,
) -> list:
    """Download one GFS cycle's wind region into ``cfg.gfs.path`` via Herbie. Shared by the in-app timer thread and the standalone
    ``scripts/gfs_download.py``. Returns the downloaded file paths (empty list if
    there's nothing to do or Herbie is unavailable - caller degrades gracefully).

    ``run_date`` pins a specific (possibly historical) cycle instead of the
    latest - NOAA's AWS open-data archive keeps GFS back to 2021, so backtests
    can use the forecast that was actually available at flight time. Latest-mode
    downloads prune cycles older than ``gfs.keep_hours`` afterwards; pinned
    (historical) downloads never prune, so backtest archives are safe."""
    pinned = run_date is not None
    if box is None:
        log.warning("no capture ROI (no active subscribers); skipping GFS download")
        return []
    fxx = cfg.gfs.download_fxx if fxx is None else fxx
    levels = cfg.gfs.download_levels if levels is None else levels
    try:
        from herbie import Herbie
    except ImportError:
        log.error("herbie not installed; pip install '.[gfs]' to enable GFS download")
        return []
    try:
        # herbie >=~2024 renamed the helper to ``HerbieLatest``; older releases
        # exported it as ``Herbie_latest``. Accept either.
        from herbie import HerbieLatest
    except ImportError:  # pragma: no cover - version-dependent
        from herbie import Herbie_latest as HerbieLatest

    dest = Path(cfg.gfs.path)
    dest.mkdir(parents=True, exist_ok=True)

    # Herbie's first arg is a *date*, not the string "latest". Anchor the most
    # recently published cycle once (probing fxx=0), then pull every forecast
    # hour from that same run so the cycle is consistent across fxx.
    if run_date is None:
        try:  # pragma: no cover - needs network + herbie
            run_date = HerbieLatest(
                model="gfs", product="pgrb2.0p25", fxx=0, save_dir=str(dest)
            ).date
        except Exception:
            log.exception("failed to find latest GFS cycle; skipping GFS download")
            return []
    log.info("downloading GFS for bbox=%s cycle=%s fxx=%s", box, run_date, fxx)

    out = []
    for fh in fxx:  # pragma: no cover - needs network + herbie
        try:
            H = Herbie(run_date, model="gfs", product="pgrb2.0p25", fxx=fh,
                       save_dir=str(dest))
            local = H.download(levels)
            log.info("downloaded GFS fxx=%dh -> %s", fh, local)
            out.append(local)
        except Exception:
            log.exception("failed to download GFS fxx=%dh", fh)
    if out and not pinned:  # pragma: no cover - needs network + herbie
        prune_grib_cache(dest, cfg.gfs.keep_hours)
    return out


def make_gfs_source(cfg: Config) -> GFSWindSource | None:
    """Return a GFS source if enabled and the cache dir exists, else None."""
    if not cfg.gfs.enabled:
        return None
    if not Path(cfg.gfs.path).exists():
        log.warning("GFS enabled but path %s missing; GFS source disabled", cfg.gfs.path)
        return None
    return HerbieGFSSource(cfg)
