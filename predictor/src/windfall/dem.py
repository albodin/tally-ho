"""Terrain (DEM) termination.

We terminate the descent at true ground elevation rather than 0 m MSL - the
single biggest improvement over chasemapper-class predictors, decisive in
terrain.

The engine only needs a ``ground_fn(lat, lon) -> elevation_m``. Four
implementations:

* :class:`FlatGround` - constant elevation; the offline/test fallback.
* :class:`CallableGround` - wrap an arbitrary function (synthetic terrain).
* :class:`CopernicusDEM` - bilinear lookups into Copernicus **GLO-30** (30 m,
  global) 1°x1° tiles, resolved by their global naming scheme.
* :class:`IndexedDEM` - bilinear lookups into *any* geographic-CRS GeoTIFF
  tile set, indexed by each file's bounds: drop USGS **3DEP** 1/3-arc-second
  (10 m) tiles in the directory for Wasatch-grade terrain, or mix
  resolutions - the finest tile covering the point wins.

rasterio dataset handles are not thread-safe, so reads are lock-guarded and
opened datasets cached. :func:`make_ground_model` picks per
``dem.source`` (auto/glo30/tiles) when enabled, tiles are present and rasterio
is importable, else degrades to flat ground.

Tiles get on disk one of two ways: :func:`download_dem_tiles` pulls the GLO-30
tiles covering a bbox from the AWS Open Data mirror (the app runs it on an
in-app timer thread when ``dem.download_in_process``), or the user drops any
tile set in ``dem.path`` by hand. :class:`ReloadableGround` is the stable
``ground_fn`` handle the app hands the tracker/predictor - ``reload()`` re-runs
the factory so tiles that arrived after startup are picked up mid-flight.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Callable

from .config import DEMConfig
from .geo import BBox

log = logging.getLogger(__name__)


class FlatGround:
    """Constant ground elevation (offline fallback / sea-level baseline)."""

    def __init__(self, elevation_m: float = 0.0):
        self.elevation_m = elevation_m

    def __call__(self, lat: float, lon: float) -> float:
        return self.elevation_m


class CallableGround:
    """Adapt any ``(lat, lon) -> elevation`` function to the ground interface."""

    def __init__(self, fn: Callable[[float, float], float]):
        self.fn = fn

    def __call__(self, lat: float, lon: float) -> float:
        return self.fn(lat, lon)


def tile_name(lat: float, lon: float) -> str:
    """Copernicus GLO-30 COG tile name covering (lat, lon).

    Tiles key on the integer-degree SW corner, e.g. (45.3, 7.8) -> the
    ``...N45_00_E007_00...`` tile."""
    ns_deg = math.floor(lat)
    ew_deg = math.floor(lon)
    ns = f"N{ns_deg:02d}" if ns_deg >= 0 else f"S{-ns_deg:02d}"
    ew = f"E{ew_deg:03d}" if ew_deg >= 0 else f"W{-ew_deg:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tiles_for_bbox(box: BBox) -> list[str]:
    """Every GLO-30 tile name needed to cover ``box`` (for pre-fetch)."""
    names: list[str] = []
    lat = math.floor(box.min_lat)
    while lat <= math.floor(box.max_lat):
        lon = math.floor(box.min_lon)
        while lon <= math.floor(box.max_lon):
            names.append(tile_name(lat + 0.5, lon + 0.5))
            lon += 1
        lat += 1
    return names


def sample_bilinear(ds, lat: float, lon: float, fallback: float = 0.0) -> float:
    """Bilinear elevation read from an open (geographic-CRS) raster dataset.
    Uses the tuple window form, so test fakes don't need rasterio installed."""
    import numpy as np

    inv = ~ds.transform
    col_f, row_f = inv * (lon, lat)
    c0, r0 = int(math.floor(col_f)), int(math.floor(row_f))
    # clamp to valid 2x2 window
    c0 = min(max(c0, 0), ds.width - 2)
    r0 = min(max(r0, 0), ds.height - 2)
    block = ds.read(1, window=((r0, r0 + 2), (c0, c0 + 2))).astype("float64")
    nodata = ds.nodata
    if nodata is not None:
        block = np.where(block == nodata, np.nan, block)
    if np.all(np.isnan(block)):
        return fallback
    # bilinear weights from the fractional position within the cell
    fc = min(max(col_f - c0, 0.0), 1.0)
    fr = min(max(row_f - r0, 0.0), 1.0)
    w = np.array([[(1 - fr) * (1 - fc), (1 - fr) * fc],
                  [fr * (1 - fc), fr * fc]])
    vals = np.nan_to_num(block, nan=np.nanmean(block))
    return float(np.sum(w * vals))


class CopernicusDEM:
    """Bilinear ground lookups into GLO-30 tiles via rasterio."""

    # memo cache size cap; entries are one float per ~30 m cell, so this stays
    # a few MB while easily covering a whole descent's worth of lookups
    _MEMO_MAX = 200_000

    def __init__(self, path: str, fallback_elevation_m: float = 0.0):
        self.path = Path(path)
        self.fallback = fallback_elevation_m
        self._lock = threading.Lock()
        self._datasets: dict[str, object] = {}   # tile name -> open rasterio dataset
        self._missing: set[str] = set()
        # The integrator queries ground every 1 s sim step (thousands of reads
        # per prediction, re-run per telemetry frame), and consecutive steps
        # often fall in the same DEM cell. Memoise per 1-arcsec cell - GLO-30's
        # native resolution - so repeat lookups skip the raster read entirely.
        self._memo: dict[tuple[int, int], float] = {}

    def __call__(self, lat: float, lon: float) -> float:  # pragma: no cover - needs rasterio+tiles
        cell = (int(lat * 3600.0), int(lon * 3600.0))
        with self._lock:
            hit = self._memo.get(cell)
            if hit is not None:
                return hit
            ds = self._open(tile_name(lat, lon))
            if ds is None:
                return self.fallback
            elev = sample_bilinear(ds, lat, lon)
            while len(self._memo) >= self._MEMO_MAX:
                self._memo.pop(next(iter(self._memo)))   # FIFO, not clear()
            self._memo[cell] = elev
            return elev

    def _open(self, name: str):  # pragma: no cover - needs rasterio+tiles
        if name in self._datasets:
            return self._datasets[name]
        if name in self._missing:
            return None
        try:
            import rasterio  # lazy: only needed when a real DEM is configured
        except ImportError:
            log.warning("rasterio not installed; DEM disabled (pip install '.[dem]')")
            self._missing.add(name)
            return None
        candidates = [
            self.path / f"{name}.tif",
            self.path / name / f"{name}.tif",
        ]
        for cand in candidates:
            if cand.exists():
                ds = rasterio.open(cand)
                self._datasets[name] = ds
                return ds
        self._missing.add(name)
        return None

    def close(self) -> None:  # pragma: no cover
        with self._lock:
            for ds in self._datasets.values():
                try:
                    ds.close()
                except Exception:
                    pass
            self._datasets.clear()
            self._memo.clear()


class IndexedDEM:
    """Bounds-indexed bilinear lookups into an arbitrary GeoTIFF tile set.

    Built for USGS 3DEP 1/3-arc-second (10 m) 1°x1° tiles - a good pick
    for Wasatch terrain - but accepts any geographic-CRS rasters: the directory
    is scanned once, each file's bounds recorded, and a lookup opens (and
    caches) the finest-resolution tile covering the point. Projected-CRS files
    are skipped with a warning (use 3DEP's standard GeoTIFF product, which is
    geographic NAD83 ≈ WGS84 at DEM accuracy)."""

    _MEMO_MAX = 200_000

    def __init__(self, path: str, fallback_elevation_m: float = 0.0, opener=None):
        self.path = Path(path)
        self.fallback = fallback_elevation_m
        self._opener = opener            # injectable for tests; None → rasterio
        self._lock = threading.Lock()
        self._tiles: list[tuple[tuple[float, float, float, float], float, str]] | None = None
        self._datasets: dict[str, object] = {}
        # 3DEP native resolution is 1/3 arcsec; memoise per such cell
        self._memo: dict[tuple[int, int], float] = {}

    def _open_raw(self, p):
        if self._opener is not None:
            return self._opener(p)
        try:
            import rasterio
        except ImportError:
            log.warning("rasterio not installed; DEM disabled (pip install '.[dem]')")
            return None
        try:
            return rasterio.open(p)
        except Exception:
            log.warning("unreadable raster %s; skipping", p)
            return None

    def _index(self) -> list:
        if self._tiles is not None:
            return self._tiles
        tiles = []
        for p in sorted(self.path.rglob("*")):
            if p.suffix.lower() not in (".tif", ".tiff"):
                continue
            ds = self._open_raw(p)
            if ds is None:
                continue
            crs = getattr(ds, "crs", None)
            if crs is not None and not getattr(crs, "is_geographic", True):
                log.warning("DEM tile %s is not in a geographic CRS; skipping "
                            "(use e.g. the 3DEP 1/3-arcsec GeoTIFF product)", p)
                ds.close()
                continue
            b = ds.bounds
            tiles.append(((b.left, b.bottom, b.right, b.top),
                          float(max(ds.res)), str(p)))
            ds.close()
        if not tiles:
            log.warning("no usable rasters under %s; flat fallback", self.path)
        else:
            log.info("indexed %d DEM tile(s) under %s", len(tiles), self.path)
        self._tiles = tiles
        return tiles

    def _tile_for(self, lat: float, lon: float) -> str | None:
        best = None
        for (left, bottom, right, top), res, p in self._index():
            if left <= lon <= right and bottom <= lat <= top:
                if best is None or res < best[0]:
                    best = (res, p)
        return best[1] if best else None

    def __call__(self, lat: float, lon: float) -> float:
        cell = (int(lat * 10800.0), int(lon * 10800.0))
        with self._lock:
            hit = self._memo.get(cell)
            if hit is not None:
                return hit
            p = self._tile_for(lat, lon)
            if p is None:
                return self.fallback
            ds = self._datasets.get(p)
            if ds is None:
                ds = self._open_raw(p)
                if ds is None:
                    return self.fallback
                self._datasets[p] = ds
            elev = sample_bilinear(ds, lat, lon, fallback=self.fallback)
            while len(self._memo) >= self._MEMO_MAX:
                self._memo.pop(next(iter(self._memo)))   # FIFO, not clear()
            self._memo[cell] = elev
            return elev

    def close(self) -> None:
        with self._lock:
            for ds in self._datasets.values():
                try:
                    ds.close()
                except Exception:
                    pass
            self._datasets.clear()
            self._memo.clear()


def detect_source(path: Path) -> str:
    """'glo30' when GLO-30-named tiles are present, else 'tiles'."""
    for p in path.rglob("Copernicus_DSM_*"):
        return "glo30"
    return "tiles"


def make_ground_model(cfg: DEMConfig) -> Callable[[float, float], float]:
    """Return the best available ground model (degrade path)."""
    if not cfg.enabled:
        log.info("DEM disabled in config; using flat sea-level ground")
        return FlatGround(0.0)
    path = Path(cfg.path)
    if not path.exists():
        log.warning("DEM path %s missing; using flat sea-level ground", path)
        return FlatGround(0.0)
    try:
        import rasterio  # noqa: F401
    except ImportError:
        log.warning("rasterio not installed; using flat sea-level ground")
        return FlatGround(0.0)
    source = getattr(cfg, "source", "auto")
    if source == "auto":
        source = detect_source(path)
    if source == "tiles":
        return IndexedDEM(str(path))
    return CopernicusDEM(str(path))


class ReloadableGround:
    """Stable ``ground_fn`` handle whose backing model can be rebuilt at runtime.

    The tracker and predictor capture ``ground_fn`` once at construction, but
    the in-app DEM downloader adds tiles while they run. ``reload()`` re-runs
    :func:`make_ground_model` - re-detecting the source (an empty dir indexes
    as ``tiles``; the first downloaded GLO-30 tile flips it to ``glo30``) and
    dropping the old model's missing-tile/memo caches - so new tiles take
    effect without a restart."""

    def __init__(self, cfg: DEMConfig):
        self.cfg = cfg
        self._model = make_ground_model(cfg)

    def __call__(self, lat: float, lon: float) -> float:
        return self._model(lat, lon)

    def reload(self) -> None:
        old = self._model
        self._model = make_ground_model(self.cfg)
        close = getattr(old, "close", None)
        if close is not None:
            close()


def _fetch_url(url: str, dest: Path) -> bool:
    """Stream ``url`` to ``dest``. True on success, False on 404 (the tile
    definitively doesn't exist - an ocean-only square), raise otherwise."""
    import shutil
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=60.0) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh, 256 * 1024)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


ABSENT_FILE = "absent_tiles.json"


def _load_absent(path: Path) -> set[str]:
    """Tile names recorded absent upstream; empty on any read/parse problem
    (worst case is one extra 404 round, re-recorded on the same pass)."""
    import json

    try:
        return {str(n) for n in json.loads(path.read_text())}
    except (OSError, ValueError, TypeError):
        return set()


def _save_absent(path: Path, absent: set[str]) -> bool:
    import json

    try:
        part = path.with_name(path.name + ".part")
        part.write_text(json.dumps(sorted(absent), indent=0) + "\n")
        part.replace(path)
        return True
    except OSError:
        log.warning("could not persist absent-tile list to %s; ocean squares "
                    "will be re-probed after a restart", path)
        return False


def download_dem_tiles(
    cfg: DEMConfig,
    box: BBox | None,
    *,
    skip: set[str] | None = None,
    fetch: Callable[[str, Path], bool] | None = None,
) -> list[str]:
    """Download the GLO-30 tiles covering ``box`` into ``cfg.path``; returns the
    newly downloaded tile names. Shared by the in-app timer thread and any
    external scheduler. Idempotent: tiles already on disk (either GLO-30
    layout) are skipped, so once the ROI is covered every pass is a free
    existence check.

    ``skip`` is the caller's in-memory set of tiles absent upstream: a 404
    means an ocean-only square with no tile (sea-level fallback is correct
    there). New 404s are recorded in ``absent_tiles.json`` in the DEM dir as
    they are discovered (a restart even mid-pass keeps them) and merged back
    into ``skip`` each pass, so neither later passes nor restarts re-request
    them (delete the file to force a recheck, e.g. should
    Copernicus ever add coverage). Transient errors are logged and *not*
    recorded - the next pass retries. Fetches run on ``cfg.download_workers``
    threads (clamped 1..16). ``fetch`` is injectable for tests (it must be
    thread-safe if workers > 1); the default streams via urllib to a ``.part``
    file renamed into place, so a killed download never leaves a truncated
    .tif."""
    if box is None:
        log.debug("no capture ROI (no active subscribers); skipping DEM download")
        return []
    if fetch is None:
        fetch = _fetch_url
    if skip is None:
        skip = set()
    dest = Path(cfg.path)
    new: list[str] = []
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        log.error("DEM dir %s is not writable by this process (container runs "
                  "as its uid:gid - chown the host dir to match, or set "
                  "PUID/PGID); skipping DEM download pass", dest)
        return new
    absent_file = dest / ABSENT_FILE
    skip |= _load_absent(absent_file)
    absent_save_ok = True   # one warning per pass, not per ocean square

    def on_disk(name: str) -> bool:
        return (dest / f"{name}.tif").exists() or (dest / name / f"{name}.tif").exists()

    names = [n for n in tiles_for_bbox(box) if n not in skip]
    todo = [n for n in names if not on_disk(n)]
    # progress denominator: ROI tiles believed to exist upstream; 404s
    # discovered this pass drop out of it as they are found
    total = len(names)
    have = total - len(todo)
    lock = threading.Lock()
    abort = threading.Event()   # unwritable dir: fails every tile identically

    def fetch_one(name: str) -> None:
        nonlocal total, have, absent_save_ok
        if abort.is_set():
            return
        part = dest / f"{name}.tif.part"
        try:
            ok = fetch(cfg.download_url.format(name=name), part)
        except PermissionError:
            # one actionable line and no further fetches - not a traceback
            # (or a queue drain) per remaining tile
            with lock:
                if not abort.is_set():
                    abort.set()
                    log.error("DEM dir %s is not writable by this process "
                              "(container runs as its uid:gid - chown the host "
                              "dir to match, or set PUID/PGID); skipping DEM "
                              "download pass", dest)
            return
        except Exception:
            log.exception("DEM tile %s download failed; will retry next pass", name)
            part.unlink(missing_ok=True)
            return
        if not ok:
            part.unlink(missing_ok=True)
            with lock:
                skip.add(name)
                total -= 1
                log.info("no GLO-30 tile %s (ocean-only square); sea-level fallback there", name)
                # persist immediately - the first pass over a big ROI can run
                # for an hour, and a restart mid-pass must not re-probe these
                if absent_save_ok:
                    absent_save_ok = _save_absent(absent_file, skip)
            return
        part.replace(dest / f"{name}.tif")
        with lock:
            new.append(name)
            have += 1
            log.info("downloaded DEM tile %s (%d/%d for ROI)", name, have, total)

    if todo:
        from concurrent.futures import ThreadPoolExecutor

        workers = max(1, min(16, cfg.download_workers))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="dem-fetch") as pool:
            list(pool.map(fetch_one, todo))
    if new:
        log.info("downloaded %d DEM tile(s) into %s", len(new), dest)
    return new
