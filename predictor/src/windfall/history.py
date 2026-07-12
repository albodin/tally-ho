"""SondeHub historical data: recovery reports + archived flight telemetry.

Two public, no-auth sources:

* **Recovery reports** - ``https://api.v2.sondehub.org/recovered``: chasers
  report where a sonde was *physically found*. This is the gold ground truth;
  the telemetry archive's last frame is biased (reception is usually lost some
  hundreds of metres AGL, so the true landing sits further downwind).
* **Telemetry archive** - the ``sondehub-history`` S3 bucket holds every frame
  SondeHub ever heard, one gzipped JSON file per serial
  (``serial/<serial>.json.gz``). It lags live flights by hours; for a sonde
  still in the air use :func:`fetch_live_telemetry` (the ES-backed
  ``/sonde/<serial>`` endpoint) instead.

:func:`download_corpus` joins the two into a local directory of flight files
(``<serial>.json`` = ``{"recovery": {...}, "frames": [...]}``) that the
offline backtest (``windfall backtest`` / :func:`windfall.replay.backtest_corpus`)
consumes. Only the two ``fetch_*`` functions touch the network; everything
else takes injectable fetchers, so the corpus logic tests fully offline.

:func:`plan_wind_cycles` + :func:`fetch_corpus_winds` close the loop for
model-wind backtests: they work out which historical GFS/HRRR cycles (and
forecast hours) each corpus flight needs - honouring publication latency, so
no lookahead - and pull them into the shared GRIB caches via the same
download helpers the live daemon uses.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .geo import BBox
from .telemetry import coerce_float, parse_datetime

log = logging.getLogger(__name__)

RECOVERED_URL = "https://api.v2.sondehub.org/recovered"
SERIAL_URL = "https://sondehub-history.s3.amazonaws.com/serial/{serial}.json.gz"
LIVE_SERIAL_URL = "https://api.v2.sondehub.org/sonde/{serial}"


def fetch_recovered(
    duration: str = "7d",
    lat: float | None = None,
    lon: float | None = None,
    distance_km: float | None = None,
    timeout: float = 30.0,
) -> list[dict]:  # pragma: no cover - network
    """Recovery reports from the SondeHub API, optionally geo-filtered.

    ``duration`` is SondeHub syntax (``1d``/``7d``/``3m``...); the API's
    ``distance`` parameter is metres, ours is km.
    """
    params: dict[str, str] = {"duration": duration}
    if lat is not None and lon is not None:
        params["lat"] = str(lat)
        params["lon"] = str(lon)
        if distance_km is not None:
            params["distance"] = str(int(distance_km * 1000))
    url = RECOVERED_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def fetch_telemetry(serial: str, timeout: float = 60.0) -> list[dict] | None:  # pragma: no cover - network
    """Full frame history for one serial from the S3 archive, or None if the
    archive has no file for it (very recent flights land there with a delay)."""
    url = SERIAL_URL.format(serial=urllib.parse.quote(serial))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    return json.loads(gzip.decompress(data))


def fetch_live_telemetry(serial: str, timeout: float = 60.0) -> list[dict] | None:  # pragma: no cover - network
    """Frame history for one serial from the live SondeHub API.

    Unlike the S3 archive (:func:`fetch_telemetry`, which lags by hours), this
    covers flights still in the air - it is what backfills a sonde first heard
    mid-flight. The endpoint 302s to a generated S3 export; urllib follows the
    redirect. Returns the raw frame dicts (MQTT shape), or None if SondeHub
    doesn't know the serial."""
    url = LIVE_SERIAL_URL.format(serial=urllib.parse.quote(serial))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    return json.loads(data)


@dataclass(slots=True)
class CorpusFlight:
    """One cached flight: the recovery report plus its archived frames."""

    serial: str
    recovery: dict
    frames: list[dict]
    path: Path

    @property
    def truth(self) -> tuple[float, float] | None:
        """The recovered (found-on-the-ground) position, if usable."""
        lat, lon = self.recovery.get("lat"), self.recovery.get("lon")
        if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
            return None
        return (float(lat), float(lon))


# Every frame field telemetry.parse_frame reads. Archive frames carry far
# more (per-listener uploader arrays are most of the bytes); stripping to this
# whitelist is lossless to the engine and shrinks files 10-20x - corpus JSON
# parse time and worker memory shrink with them.
FRAME_FIELDS = (
    "serial", "lat", "lon", "alt", "datetime", "frame", "type", "subtype",
    "manufacturer", "software_name", "uploader_callsign", "vel_v", "vel_h",
    "heading", "temp", "humidity", "pressure", "sats", "batt",
    "burst_timer", "frequency",
)


def _slim_frames(frames: list[dict]) -> list[dict]:
    return [{k: m[k] for k in FRAME_FIELDS if k in m} for m in frames]


def download_corpus(
    out_dir: str | Path,
    *,
    duration: str = "7d",
    lat: float | None = None,
    lon: float | None = None,
    distance_km: float | None = None,
    limit: int | None = None,
    only_recovered: bool = True,
    slim: bool = True,
    recovered_fn: Callable[..., list[dict]] | None = None,
    telemetry_fn: Callable[[str], list[dict] | None] | None = None,
) -> list[Path]:
    """Fetch recovery reports, then each flight's archived telemetry, into
    ``out_dir`` (one ``<serial>.json`` per flight). Existing files are kept
    as-is (the corpus is an append-only cache), so re-runs only download new
    flights. Returns the paths now in the corpus for this report set.
    ``slim`` keeps only the frame fields the engine parses (see
    :data:`FRAME_FIELDS`)."""
    if recovered_fn is None:
        recovered_fn = fetch_recovered
    if telemetry_fn is None:
        telemetry_fn = fetch_telemetry
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports = recovered_fn(duration=duration, lat=lat, lon=lon, distance_km=distance_km)
    paths: list[Path] = []
    seen: set[str] = set()
    for rec in reports:
        serial = rec.get("serial")
        if not serial or serial in seen:
            continue
        seen.add(serial)
        if only_recovered and not rec.get("recovered"):
            continue
        path = out / f"{serial}.json"
        if not path.exists():
            frames = telemetry_fn(serial)
            if not frames:
                log.info("no archived telemetry for %s (yet); skipping", serial)
                continue
            if slim:
                frames = _slim_frames(frames)
            path.write_text(json.dumps({"recovery": rec, "frames": frames}))
            log.info("saved %s (%d frames)", path, len(frames))
        paths.append(path)
        if limit is not None and len(paths) >= limit:
            break
    return paths


def slim_corpus(corpus_dir: str | Path) -> tuple[int, int, int]:
    """Rewrite every corpus file with frames stripped to :data:`FRAME_FIELDS`
    (atomic per file). Returns (files rewritten, bytes before, bytes after)."""
    import os

    n = before = after = 0
    for path in corpus_paths(corpus_dir):
        size = path.stat().st_size
        cf = load_corpus_file(path)
        if cf is None:
            continue
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(
            {"recovery": cf.recovery, "frames": _slim_frames(cf.frames)}))
        os.replace(tmp, path)
        n += 1
        before += size
        after += path.stat().st_size
        log.info("slimmed %s: %.0f -> %.0f MB", path.name, size / 1e6,
                 path.stat().st_size / 1e6)
    return n, before, after


def corpus_paths(corpus_dir: str | Path) -> list[Path]:
    """The flight files of a corpus directory (sorted by serial), unparsed -
    corpus files run to ~100 MB of 1 Hz frames, so callers that can spread the
    JSON parsing across workers (or stop at a limit) start from this."""
    root = Path(corpus_dir)
    return sorted(root.glob("*.json")) if root.exists() else []


def load_corpus_file(path: Path) -> CorpusFlight | None:
    """Parse one corpus flight file; None (with a warning) if unreadable."""
    try:
        doc = json.loads(path.read_text())
        frames = doc["frames"]
        recovery = doc.get("recovery") or {}
        if not isinstance(frames, list) or not frames:
            raise ValueError("no frames")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        log.warning("skipping unreadable corpus file %s", path)
        return None
    return CorpusFlight(serial=recovery.get("serial") or path.stem,
                        recovery=recovery, frames=frames, path=path)


def iter_corpus(corpus_dir: str | Path) -> list[CorpusFlight]:
    """Load every flight file in a corpus directory (sorted by serial).
    Unreadable/odd files are skipped with a warning, not fatal."""
    flights = (load_corpus_file(p) for p in corpus_paths(corpus_dir))
    return [cf for cf in flights if cf is not None]


# ---- historical model-wind fetch planning (offline, pure) ------------------

@dataclass(slots=True)
class CyclePlan:
    """One model cycle a corpus backtest needs: which run, which forecast
    hours, the region it must cover, and the flights that want it."""

    cycle: datetime          # run time, tz-aware UTC
    fxx: list[int]           # forecast hours still to download
    box: BBox                # union of the flights' margins (GFS download log)
    serials: list[str]


def flight_extent(flight: CorpusFlight) -> tuple[datetime, datetime, BBox] | None:
    """Time span and position bounding box of one corpus flight, including the
    recovered ground-truth position (it sits downwind of the last-heard frame).
    None when no frame carries a parseable time + position."""
    t_lo = t_hi = None
    lats: list[float] = []
    lons: list[float] = []
    for m in flight.frames:
        dt = parse_datetime(m.get("datetime"))
        lat, lon = coerce_float(m.get("lat")), coerce_float(m.get("lon"))
        if dt is None or lat is None or lon is None:
            continue
        t_lo = dt if t_lo is None or dt < t_lo else t_lo
        t_hi = dt if t_hi is None or dt > t_hi else t_hi
        lats.append(lat)
        lons.append(lon)
    if t_lo is None:
        return None
    if flight.truth is not None:
        lats.append(flight.truth[0])
        lons.append(flight.truth[1])
    return t_lo, t_hi, BBox(min(lats), min(lons), max(lats), max(lons))


def plan_wind_cycles(
    flights: list[CorpusFlight],
    *,
    cycle_hours: float,
    latency_hours: float,
    margin_km: float = 75.0,
    pad_hours: float = 2.0,
    max_fxx: int = 120,
    have: set[tuple[datetime, int]] | None = None,
) -> list[CyclePlan]:
    """Which model cycles (and forecast hours) a backtest of ``flights`` needs.

    Per flight: the newest cycle already *published* at launch - run time on
    the ``cycle_hours`` grid, at least ``latency_hours`` before the first frame
    (publication latency; mirrors :func:`windfall.gfs.select_bracketing`, so
    the backtest will actually pick what this downloads) - with hourly forecast
    steps whose valid times cover first frame → last frame + ``pad_hours``
    (descent simulated past loss of signal). Flights sharing a cycle merge;
    (cycle, fxx) pairs in ``have`` (the existing cache inventory) drop out."""
    merged: dict[datetime, list] = {}     # cycle -> [lo, hi, box, serials]
    for cf in flights:
        ext = flight_extent(cf)
        if ext is None:
            log.warning("no usable frames in %s; cannot plan a wind fetch", cf.serial)
            continue
        start, end, box = ext
        cutoff = start - timedelta(hours=latency_hours)
        day = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(hours=cycle_hours)
        cycle = day + step * int((cutoff - day) / step)
        lo = max(0, int((start - cycle) / timedelta(hours=1)))
        hi = math.ceil((end + timedelta(hours=pad_hours) - cycle) / timedelta(hours=1))
        if hi > max_fxx:
            log.warning("%s needs fxx up to %d; clamping to %d", cf.serial, hi, max_fxx)
            hi = max_fxx
        lo = min(lo, hi)
        box = box.expanded_km(margin_km)
        cur = merged.get(cycle)
        if cur is None:
            merged[cycle] = [lo, hi, box, [cf.serial]]
        else:
            cur[0] = min(cur[0], lo)
            cur[1] = max(cur[1], hi)
            cur[2] = BBox(min(cur[2].min_lat, box.min_lat), min(cur[2].min_lon, box.min_lon),
                          max(cur[2].max_lat, box.max_lat), max(cur[2].max_lon, box.max_lon))
            cur[3].append(cf.serial)

    plans: list[CyclePlan] = []
    for cycle in sorted(merged):
        lo, hi, box, serials = merged[cycle]
        fxx = [f for f in range(lo, hi + 1)
               if have is None or (cycle, f) not in have]
        if fxx:
            plans.append(CyclePlan(cycle=cycle, fxx=fxx, box=box, serials=serials))
    return plans


# Per-model planning constants: cycle grid spacing and how far out the archive
# carries hourly forecast steps (GFS hourly to f120; HRRR prs files to f18).
_MODEL_GRID = {"gfs": (6.0, 120), "hrrr": (1.0, 18)}


def fetch_corpus_winds(
    cfg,
    corpus_dir: str | Path,
    *,
    model: str = "gfs",
    margin_km: float = 75.0,
    pad_hours: float = 2.0,
    dry_run: bool = False,
    printer: Callable[[str], None] = print,
) -> int:
    """Plan and download every historical model cycle a corpus backtest needs
    (the ``windfall fetch-gfs`` command). ``model`` is ``gfs``/``hrrr``/``both``;
    already-cached (cycle, fxx) pairs are skipped. Returns a CLI exit code."""
    flights = iter_corpus(corpus_dir)
    if not flights:
        printer(f"no flights in {corpus_dir} (run 'fetch-corpus' first)")
        return 2

    from .gfs import download_gfs_cycle, scan_inventory

    models = ("gfs", "hrrr") if model == "both" else (model,)
    for name in models:
        cycle_hours, max_fxx = _MODEL_GRID[name]
        mcfg = getattr(cfg, name)
        plans = plan_wind_cycles(
            flights, cycle_hours=cycle_hours, max_fxx=max_fxx,
            latency_hours=mcfg.publication_latency_hours,
            margin_km=margin_km, pad_hours=pad_hours,
            have={(i.cycle, i.fxx) for i in scan_inventory(Path(mcfg.path))})
        if not plans:
            printer(f"{name}: all needed cycles already in {mcfg.path}")
            continue
        printer(f"{name}: {len(plans)} cycle(s) to download into {mcfg.path}")
        for p in plans:
            printer(f"  {p.cycle:%Y-%m-%d %HZ}  fxx {p.fxx[0]}-{p.fxx[-1]}"
                    f" ({len(p.fxx)} file(s))  flights: {', '.join(p.serials)}")
        if dry_run:
            continue
        for p in plans:
            run_date = p.cycle.replace(tzinfo=None)   # Herbie wants naive UTC
            try:
                if name == "hrrr":
                    from .hrrr import download_hrrr_cycle
                    paths = download_hrrr_cycle(cfg, fxx=p.fxx, run_date=run_date)
                else:
                    paths = download_gfs_cycle(cfg, p.box, fxx=p.fxx, run_date=run_date)
            except OSError as e:
                printer(f"  {p.cycle:%Y-%m-%d %HZ}: download failed: {e}")
                return 1
            if not paths:
                printer(f"  {p.cycle:%Y-%m-%d %HZ}: nothing downloaded "
                        "(is herbie installed? pip install 'windfall[gfs]')")
    return 0
