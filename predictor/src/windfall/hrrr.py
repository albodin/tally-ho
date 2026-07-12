"""HRRR wind source + HRRR-over-GFS composition (plan Phase 0).

HRRR is NOAA's 3 km CONUS model, cycled hourly - dramatically better
boundary-layer and terrain-flow winds than GFS's 0.25° where it exists. Its
pressure-level files top out near 50 hPa (~20-21 km), so GFS still owns the
column above; :class:`CompositeWindSource` samples HRRR below a configured
ceiling and GFS above, with a linear blend across the seam.

The HRRR grid is **Lambert conformal**, not lat/lon: cfgrib exposes 2-D
latitude/longitude arrays and the projection parameters in the GRIB attributes.
Rather than regridding, :class:`LambertGrid` implements the (spherical) forward
LCC transform, and the shared :class:`~windfall.gfs.WindCube` interpolates on
the regular projected y/x axes - identical bilinear/level/time logic to GFS.

Files are full-CONUS (1799x1059 per level - gigabytes as float arrays), so only
a window around the flight is materialised per load (``hrrr.window_deg``).

Everything heavy stays behind lazy imports; the pure pieces (projection,
composition, factory) are tested offline.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Config
from .gfs import (
    CubePairWind,
    GFSWindSource,
    GribFileInfo,
    VarGrid,
    WindCube,
    assemble_cube,
    build_profile_from_levels,
    make_gfs_source,
    scan_inventory,
    select_bracketing,
)
from .profile import FlightProfile

log = logging.getLogger(__name__)


# ---- Lambert conformal grid (pure math, testable offline) -------------------

@dataclass(slots=True)
class LambertGrid:
    """Forward Lambert-conformal transform + the grid's regular y/x axes.

    Spherical LCC per the GRIB2 template NCEP uses (sphere radius 6 371 229 m).
    ``proj(lat, lon) -> (y_m, x_m)`` feeds :class:`WindCube`'s projected mode;
    axes are anchored so the first grid point sits at (0, 0).
    """

    n: float                 # cone constant
    rf: float                # R * F (projection scale numerator)
    rho0: float              # radial coordinate of the latitude of origin
    lon0_rad: float          # central meridian, radians
    x0: float                # projected coords of the first grid point...
    y0: float                # ...so axes can be anchored at (0, 0)
    dx: float
    dy: float
    nx: int
    ny: int

    R_SPHERE = 6_371_229.0   # GRIB2 sphere for NCEP LCC grids

    # HRRR CONUS defaults, used when a GRIB attribute is missing.
    _DEFAULTS = {
        "GRIB_Latin1InDegrees": 38.5,
        "GRIB_Latin2InDegrees": 38.5,
        "GRIB_LaDInDegrees": 38.5,
        "GRIB_LoVInDegrees": 262.5,
        "GRIB_latitudeOfFirstGridPointInDegrees": 21.138123,
        "GRIB_longitudeOfFirstGridPointInDegrees": 237.280472,
        "GRIB_DxInMetres": 3000.0,
        "GRIB_DyInMetres": 3000.0,
    }

    @classmethod
    def from_attrs(cls, attrs: dict, nx: int, ny: int) -> "LambertGrid":
        def get(key: str) -> float:
            return float(attrs.get(key, cls._DEFAULTS[key]))

        lat1 = math.radians(get("GRIB_Latin1InDegrees"))
        lat2 = math.radians(get("GRIB_Latin2InDegrees"))
        lad = math.radians(get("GRIB_LaDInDegrees"))
        lov = math.radians(get("GRIB_LoVInDegrees"))
        if abs(lat1 - lat2) < 1e-9:
            n = math.sin(lat1)
        else:
            n = (math.log(math.cos(lat1) / math.cos(lat2))
                 / math.log(math.tan(math.pi / 4 + lat2 / 2)
                            / math.tan(math.pi / 4 + lat1 / 2)))
        f = math.cos(lat1) * math.tan(math.pi / 4 + lat1 / 2) ** n / n
        rf = cls.R_SPHERE * f
        rho0 = rf / math.tan(math.pi / 4 + lad / 2) ** n
        grid = cls(n=n, rf=rf, rho0=rho0, lon0_rad=lov, x0=0.0, y0=0.0,
                   dx=get("GRIB_DxInMetres"), dy=get("GRIB_DyInMetres"),
                   nx=nx, ny=ny)
        grid.y0, grid.x0 = grid.proj(
            get("GRIB_latitudeOfFirstGridPointInDegrees"),
            get("GRIB_longitudeOfFirstGridPointInDegrees"))
        return grid

    def proj(self, lat: float, lon: float) -> tuple[float, float]:
        """(lat°, lon°) → (y_m, x_m) relative to the first grid point."""
        rho = self.rf / math.tan(math.pi / 4 + math.radians(lat) / 2) ** self.n
        dlon = math.radians(lon) - self.lon0_rad
        dlon = (dlon + math.pi) % (2 * math.pi) - math.pi
        a = self.n * dlon
        x = rho * math.sin(a)
        y = self.rho0 - rho * math.cos(a)
        return (y - self.y0, x - self.x0)

    def y_axis(self) -> np.ndarray:
        return np.arange(self.ny, dtype=float) * self.dy

    def x_axis(self) -> np.ndarray:
        return np.arange(self.nx, dtype=float) * self.dx

    def window(self, lat: float, lon: float, half_deg: float) -> tuple[slice, slice]:
        """Index slices covering ``±half_deg`` around (lat, lon), clamped to the
        grid. The window's projected extent is taken from its 4 corners plus the
        edge midpoints (the LCC image of a lat/lon box is curved)."""
        pts = [(lat + dy * half_deg, lon + dx * half_deg)
               for dy in (-1.0, 0.0, 1.0) for dx in (-1.0, 0.0, 1.0)]
        ys, xs = zip(*(self.proj(la, lo) for la, lo in pts))
        i0 = max(0, int(min(ys) // self.dy) - 1)
        i1 = min(self.ny, int(max(ys) // self.dy) + 2)
        j0 = max(0, int(min(xs) // self.dx) - 1)
        j1 = min(self.nx, int(max(xs) // self.dx) + 2)
        return (slice(i0, i1), slice(j0, j1))


# ---- HRRR source -------------------------------------------------------------

class HerbieHRRRSource(GFSWindSource):
    """Read downloaded HRRR prs GRIB and serve the 4-D interpolating wind field.

    Same shape as :class:`~windfall.gfs.HerbieGFSSource` - filename inventory,
    newest-cycle valid-time bracketing, :class:`CubePairWind` time-lerp - except
    cubes carry the Lambert projection and only a window around the flight is
    loaded (full-CONUS files would not fit in memory). Requires cfgrib/xarray
    (lazy import); degrades to None so the caller falls back to GFS/measured."""

    _INVENTORY_TTL_S = 60.0
    _CUBE_CACHE_MAX = 6      # windows are ~50-100 MB each

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = Path(cfg.hrrr.path)
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
                                self.cfg.hrrr.publication_latency_hours)
        if sel is None:
            return None
        a = self._cube(sel[0], lat, lon)
        b = a if sel[1].path == sel[0].path else self._cube(sel[1], lat, lon)
        if a is None or b is None:
            return None
        return CubePairWind(a, b, t0=when)

    def _cube(self, info: GribFileInfo, lat: float, lon: float) -> WindCube | None:  # pragma: no cover - needs GRIB
        # window-bucketed cache: flights within the same ~1° share a cube, and
        # the window half-width (≥3°) dwarfs any drift across the bucket
        key = (info.path, round(lat), round(lon))
        if key in self._cubes:
            return self._cubes[key]
        cube = None
        try:
            import xarray as xr
        except ImportError:
            log.warning("xarray/cfgrib not installed; HRRR source disabled")
            self._cubes[key] = None
            return None
        try:
            grids: dict[str, VarGrid | None] = {}
            proj = None
            for short in ("u", "v", "gh", "t"):
                got = self._read_var(xr, info.path, short, lat, lon)
                if got is None:
                    grids[short] = None
                    continue
                grids[short], grid = got
                proj = proj or grid.proj
            cube = assemble_cube(info.valid_time, proj=proj, **grids)
            if cube is None:
                log.warning("HRRR file %s has no usable wind levels", info.path)
        except Exception:
            log.exception("failed to load HRRR cube from %s", info.path)
        while len(self._cubes) >= self._CUBE_CACHE_MAX:
            self._cubes.pop(next(iter(self._cubes)))   # FIFO, keep recent cubes
        self._cubes[key] = cube
        return cube

    def _read_var(self, xr, path: str, short: str, lat: float, lon: float):  # pragma: no cover - needs GRIB
        """One variable's (level, y, x) window around (lat, lon), plus the
        Lambert grid built from that variable's GRIB attributes. Returns None
        when the file doesn't carry the variable."""
        try:
            # indexpath="": no shared on-disk cfgrib index - parallel backtest
            # workers race on it (see HerbieGFSSource._read_var).
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
            da = ds[short]
            ny = da.sizes.get("y", 0)
            nx = da.sizes.get("x", 0)
            if not (ny and nx):
                return None    # not a projected grid - wrong product
            grid = LambertGrid.from_attrs(dict(da.attrs), nx=nx, ny=ny)
            ysl, xsl = grid.window(lat, lon, self.cfg.hrrr.window_deg)
            if ysl.start >= ysl.stop or xsl.start >= xsl.stop:
                log.warning("(%.2f, %.2f) is outside the HRRR grid", lat, lon)
                return None
            vals = da.isel(y=ysl, x=xsl).values
            if vals.ndim == 2:
                vals = vals[None]
            var = VarGrid(
                lats=grid.y_axis()[ysl],
                lons=grid.x_axis()[xsl],
                levels_hpa=np.atleast_1d(ds["isobaricInhPa"].values).astype(float),
                vals=np.asarray(vals, dtype=float),
            )
            return var, grid
        finally:
            ds.close()


# ---- HRRR-over-GFS composition ------------------------------------------------

def blend_fields(primary, fallback, ceiling_m: float, ramp_m: float):
    """A 4-D wind callable sampling ``primary`` below ``ceiling_m - ramp_m``,
    ``fallback`` above ``ceiling_m``, linearly blended in between."""
    lo = ceiling_m - max(ramp_m, 1.0)

    def wind(lat: float, lon: float, alt: float, sim_t: float = 0.0):
        if alt <= lo:
            return primary(lat, lon, alt, sim_t)
        if alt >= ceiling_m:
            return fallback(lat, lon, alt, sim_t)
        w = (ceiling_m - alt) / (ceiling_m - lo)
        pu, pv = primary(lat, lon, alt, sim_t)
        fu, fv = fallback(lat, lon, alt, sim_t)
        return (w * pu + (1.0 - w) * fu, w * pv + (1.0 - w) * fv)

    return wind


class CompositeWindSource(GFSWindSource):
    """HRRR below the ceiling, GFS above - one source, plan Phase 0.

    Degrades per-call: if either side has no data for the prediction time, the
    other serves the whole column (HRRR alone clamps above its top level -
    still better than nothing; GFS alone is the pre-HRRR behaviour)."""

    def __init__(self, primary: GFSWindSource, fallback: GFSWindSource,
                 ceiling_m: float = 20_000.0, ramp_m: float = 1_500.0):
        self.primary = primary
        self.fallback = fallback
        self.ceiling_m = ceiling_m
        self.ramp_m = ramp_m

    def wind_field(self, lat: float, lon: float, when: datetime):
        pf = self.primary.wind_field(lat, lon, when)
        ff = self.fallback.wind_field(lat, lon, when)
        if pf is None:
            return ff
        if ff is None:
            return pf
        return blend_fields(pf, ff, self.ceiling_m, self.ramp_m)

    def wind_filler(self, lat: float, lon: float, when: datetime):
        p_fill = self.primary.wind_filler(lat, lon, when)
        f_fill = self.fallback.wind_filler(lat, lon, when)
        if p_fill is None:
            return f_fill
        if f_fill is None:
            return p_fill

        def fill(alt: float):
            if alt < self.ceiling_m:
                got = p_fill(alt)
                if got is not None:
                    return got
            return f_fill(alt)

        return fill

    def profile_at(self, lat: float, lon: float, when: datetime) -> FlightProfile | None:
        """Merged column: HRRR bins below the ceiling, GFS bins above."""
        base = self.fallback.profile_at(lat, lon, when)
        fine = self.primary.profile_at(lat, lon, when)
        if fine is None:
            return base
        if base is None:
            return fine
        merged = FlightProfile(bin_size_m=fine.bin_size_m, gap_fill_m=fine.gap_fill_m)
        for b in fine._bins.values():
            if b.alt < self.ceiling_m:
                merged.add_sample(b.alt, b.u, b.v, b.rho, lat=b.lat, lon=b.lon, t=b.t)
        for b in base._bins.values():
            if b.alt >= self.ceiling_m:
                merged.add_sample(b.alt, b.u, b.v, b.rho, lat=b.lat, lon=b.lon, t=b.t)
        return merged if not merged.is_empty() else base


# ---- download + factory --------------------------------------------------------

def download_hrrr_cycle(
    cfg: Config,
    fxx: list[int] | None = None,
    levels: str | None = None,
    run_date: datetime | None = None,
) -> list:
    """Download one HRRR cycle's pressure-level winds into ``cfg.hrrr.path`` via
    Herbie. No spatial subsetting exists at download time (byte ranges are
    per-message), so files are full-CONUS but variable-subset; the reader
    windows them at load. ``run_date`` pins a historical cycle (NOAA's AWS
    archive keeps HRRR back to 2014) for lookahead-free backtests. Latest-mode
    downloads prune cycles older than ``hrrr.keep_hours`` afterwards; pinned
    downloads never prune."""
    pinned = run_date is not None
    fxx = cfg.hrrr.download_fxx if fxx is None else fxx
    levels = cfg.hrrr.download_levels if levels is None else levels
    try:
        from herbie import Herbie
    except ImportError:
        log.error("herbie not installed; pip install 'windfall[gfs]' to enable HRRR download")
        return []
    try:
        from herbie import HerbieLatest
    except ImportError:  # pragma: no cover - version-dependent
        from herbie import Herbie_latest as HerbieLatest

    dest = Path(cfg.hrrr.path)
    dest.mkdir(parents=True, exist_ok=True)

    if run_date is None:
        try:  # pragma: no cover - needs network + herbie
            run_date = HerbieLatest(
                model="hrrr", product="prs", fxx=0, save_dir=str(dest)
            ).date
        except Exception:
            log.exception("failed to find latest HRRR cycle; skipping HRRR download")
            return []
    log.info("downloading HRRR cycle=%s fxx=%s", run_date, fxx)

    out = []
    for fh in fxx:  # pragma: no cover - needs network + herbie
        try:
            h = Herbie(run_date, model="hrrr", product="prs", fxx=fh,
                       save_dir=str(dest))
            if h.idx is None:
                # Cycle files publish sequentially; later fxx lag f00 by
                # minutes, and subset downloads need the .idx sidecar.
                log.info("HRRR fxx=%dh not published yet; skipping until next poll", fh)
                continue
            local = h.download(levels)
            log.info("downloaded HRRR fxx=%dh -> %s", fh, local)
            out.append(local)
        except Exception:
            log.exception("failed to download HRRR fxx=%dh", fh)
    if out and not pinned:  # pragma: no cover - needs network + herbie
        from .gfs import prune_grib_cache
        prune_grib_cache(dest, cfg.hrrr.keep_hours)
    return out


def make_hrrr_source(cfg: Config) -> GFSWindSource | None:
    """Return an HRRR source if enabled and the cache dir exists, else None."""
    if not cfg.hrrr.enabled:
        return None
    if not Path(cfg.hrrr.path).exists():
        log.warning("HRRR enabled but path %s missing; HRRR source disabled",
                    cfg.hrrr.path)
        return None
    return HerbieHRRRSource(cfg)


def make_wind_source(cfg: Config) -> GFSWindSource | None:
    """The model wind source the predictor should use: HRRR-over-GFS when both
    are wired, either alone otherwise, None when neither."""
    gfs = make_gfs_source(cfg)
    hrrr = make_hrrr_source(cfg)
    if hrrr is None:
        return gfs
    if gfs is None:
        log.warning("HRRR without GFS: winds above ~%.0f km clamp to the HRRR "
                    "ceiling - enable GFS for the upper column",
                    cfg.hrrr.ceiling_m / 1000.0)
        return hrrr
    return CompositeWindSource(hrrr, gfs, ceiling_m=cfg.hrrr.ceiling_m,
                               ramp_m=cfg.hrrr.blend_ramp_m)
