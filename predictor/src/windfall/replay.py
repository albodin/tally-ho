"""Offline replay / accuracy harness.

Replays archived (or synthetic) flights frame-by-frame through the **exact
production predictor** - the same :class:`FlightTracker` + :class:`Predictor`
the live app uses. There is deliberately no offline/online fork, so the
accuracy and calibration numbers measured here transfer directly to production.

Ground truth is the last valid telemetry position before signal loss (≈ landing;
typically some hundreds of m above true ground in terrain/trees) - or,
better, a chaser-reported recovered position (``truth=`` override): reception is
usually lost well above ground, so the true landing sits further downwind, and
the recovery report is where the sonde physically was. :func:`backtest_corpus`
runs a whole directory of archived flights (see :mod:`windfall.history`) against
recovered truth.

Metrics:
* predicted-vs-actual landing error (km), bucketed by altitude-at-prediction;
* convergence - does error shrink as the sonde descends;
* calibration - is the reported ``uncertainty_radius_km`` honest (true error
  inside the radius at the expected rate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Protocol

from .config import Config
from .geo import haversine_km
from .models import Frame, FlightState, Prediction
from .predictor import GFSWindSource, GroundFn, Predictor
from .telemetry import FrameError, parse_frame
from .tracker import FlightTracker

log = logging.getLogger(__name__)


class PredictionStore(Protocol):
    """What :func:`accuracy_from_store` needs from the live app's store."""

    def recent_landings(self, limit: int = 200) -> list[dict]: ...
    def predictions_for(self, serial: str, launch_day: date) -> list[dict]: ...

# Altitude buckets (m) for error reporting.
ALT_BUCKETS = [(0, 2000), (2000, 5000), (5000, 10000), (10000, 20000), (20000, 99000)]


@dataclass(slots=True)
class PredRecord:
    alt_at_pred: float
    error_km: float
    uncertainty_km: float
    sim_seconds: float
    source: str
    inside_radius: bool


@dataclass(slots=True)
class ReplayResult:
    serial: str
    truth_lat: float
    truth_lon: float
    records: list[PredRecord] = field(default_factory=list)
    truth_source: str = "last_frame"     # or "recovered" (chaser ground truth)
    # ISO launch day when scored from a live store (offline replays have none) -
    # lets API consumers link a result back to its (serial, launch_day) flight.
    launch_day: str | None = None

    @property
    def final_error_km(self) -> float | None:
        return self.records[-1].error_km if self.records else None

    @property
    def n_predictions(self) -> int:
        return len(self.records)


def replay_flight(
    frames: list[Frame],
    cfg: Config | None = None,
    ground_fn: GroundFn | None = None,
    gfs_source: GFSWindSource | None = None,
    truth: tuple[float, float] | None = None,
    predict_every_s: float = 0.0,
) -> ReplayResult:
    """Replay one flight's ordered frames and record every descent prediction.

    ``truth`` overrides the landing ground truth (e.g. a chaser-reported
    recovered position); default is the last telemetry position.
    ``predict_every_s`` throttles descent predictions by sonde time, mirroring
    the daemon's ``predict.descent_predict_seconds`` - real archived flights
    are ~1 Hz and per-frame ensembles would make corpus runs take hours.
    0 (default) keeps the historical per-frame behaviour."""
    cfg = cfg or Config()
    tracker = FlightTracker(cfg, store=None, ground_fn=ground_fn)
    predictor = Predictor(cfg, ground_fn=ground_fn, gfs_source=gfs_source)

    if truth is None:
        result = ReplayResult(serial=frames[0].serial,
                              truth_lat=frames[-1].lat, truth_lon=frames[-1].lon)
    else:
        result = ReplayResult(serial=frames[0].serial, truth_lat=truth[0],
                              truth_lon=truth[1], truth_source="recovered")
    last_pred_t: float | None = None
    for f in frames:
        flight, _events = tracker.update(f)
        if flight.state == FlightState.DESCENT:
            if last_pred_t is not None and f.t - last_pred_t < predict_every_s:
                continue
            last_pred_t = f.t
            pred = predictor.predict(flight)
            if pred is not None:
                err = haversine_km(pred.land_lat, pred.land_lon,
                                   result.truth_lat, result.truth_lon)
                result.records.append(PredRecord(
                    alt_at_pred=flight.last_alt,
                    error_km=err,
                    uncertainty_km=pred.uncertainty_radius_km,
                    sim_seconds=(pred.land_eta - pred.predicted_at).total_seconds(),
                    source=pred.source.value,
                    inside_radius=err <= pred.uncertainty_radius_km,
                ))
    return result


def parse_raw_frames(raw: list[dict]) -> list[Frame]:
    """Raw telemetry dicts -> time-sorted Frames, skipping unparseable ones
    (real archive data has a few)."""
    frames = []
    for m in raw:
        try:
            frames.append(parse_frame(m))
        except FrameError:
            continue
    frames.sort(key=lambda f: f.t)
    return frames


def replay_messages(raw: list[dict], **kw) -> ReplayResult | None:
    """Replay from raw telemetry dicts; None if nothing parses."""
    frames = parse_raw_frames(raw)
    if not frames:
        return None
    return replay_flight(frames, **kw)


def backtest_corpus(
    corpus_dir,
    cfg: Config | None = None,
    ground_fn: GroundFn | None = None,
    gfs_source: GFSWindSource | None = None,
    limit: int | None = None,
) -> list[ReplayResult]:
    """Replay every archived flight in a corpus directory (built by
    ``windfall fetch-corpus`` / :func:`windfall.history.download_corpus`) through
    the production predictor, scored against the *recovered* position when the
    report carries one (falling back to last-heard). Pure offline given the
    cached corpus - vary config/code and re-run to compare. Descent predictions
    run at the daemon's ``predict.descent_predict_seconds`` cadence, not per
    frame: archived flights are ~1 Hz and this is what production would do.

    Flights load, replay and free one at a time - a whole corpus of raw frame
    dicts at once is tens of GB."""
    from .history import corpus_paths, load_corpus_file

    cfg = cfg or Config()
    results: list[ReplayResult] = []
    for path in corpus_paths(corpus_dir):
        if limit is not None and len(results) >= limit:
            break
        cf = load_corpus_file(path)
        if cf is None:
            continue
        frames = parse_raw_frames(cf.frames)
        if not frames:
            continue
        truth = cf.truth
        del cf
        results.append(replay_flight(
            frames, cfg=cfg, ground_fn=ground_fn, gfs_source=gfs_source,
            truth=truth, predict_every_s=cfg.predict.descent_predict_seconds))
    return results


# ---- parallel corpus backtest -----------------------------------------------
# Flights are independent, so a corpus replays embarrassingly parallel across
# worker processes. The eager-object API of backtest_corpus can't cross process
# boundaries (ground_fn/gfs_source hold rasterio datasets and GRIB cube caches),
# so each worker rebuilds its sources from the picklable Config - and parses its
# own corpus JSON, which is itself a large share of single-threaded runtime.

_PAR: dict = {}      # per-worker-process state, set once by _par_init


def _par_init(cfg: Config, with_dem: bool, with_gfs: bool) -> None:
    _PAR["cfg"] = cfg
    _PAR["ground_fn"] = None
    _PAR["gfs"] = None
    if with_dem:
        from .dem import make_ground_model
        _PAR["ground_fn"] = make_ground_model(cfg.dem)
    if with_gfs:
        from .hrrr import make_wind_source
        _PAR["gfs"] = make_wind_source(cfg)


def _par_replay(path_str: str) -> ReplayResult | None:
    from .history import load_corpus_file
    from pathlib import Path

    cf = load_corpus_file(Path(path_str))
    if cf is None:
        return None
    cfg = _PAR["cfg"]
    frames = parse_raw_frames(cf.frames)
    if not frames:
        return None
    truth = cf.truth
    # Frames are compact slots objects; the raw JSON tree is 10-20x the file
    # size as dicts. Release it before the replay holds memory for minutes.
    del cf
    return replay_flight(frames, cfg=cfg, ground_fn=_PAR["ground_fn"],
                         gfs_source=_PAR["gfs"], truth=truth,
                         predict_every_s=cfg.predict.descent_predict_seconds)


def backtest_corpus_parallel(
    corpus_dir,
    cfg: Config | None = None,
    *,
    with_dem: bool = False,
    with_gfs: bool = True,
    limit: int | None = None,
    jobs: int = 2,
) -> list[ReplayResult]:
    """:func:`backtest_corpus` across ``jobs`` worker processes.

    Same scoring, same cadence; only the execution differs. ``limit`` bounds
    the *files attempted* (not parseable results - workers discover
    parseability). A flight that errors is logged and skipped, not fatal."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .history import corpus_paths

    cfg = cfg or Config()
    paths = corpus_paths(corpus_dir)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        return []
    results: list[ReplayResult] = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_par_init,
                             initargs=(cfg, with_dem, with_gfs)) as pool:
        futures = {pool.submit(_par_replay, str(p)): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception:
                log.exception("replay failed for %s (%d/%d)", futures[fut], i, len(paths))
                continue
            if res is None:
                log.warning("skipped %s (%d/%d)", futures[fut], i, len(paths))
                continue
            results.append(res)
            log.info("replayed %s: final %s (%d/%d)", res.serial,
                     f"{res.final_error_km:.2f} km" if res.final_error_km is not None
                     else "-", i, len(paths))
    return results


@dataclass
class Metrics:
    n_flights: int
    n_predictions: int
    mean_final_error_km: float
    bucket_mean_error_km: dict[str, float]
    bucket_counts: dict[str, int]
    calibration_rate: float          # fraction of predictions with error <= radius
    # Multiply every published radius by this to make exactly TARGET_COVERAGE of
    # predictions fall inside it - the one-number calibration knob. ~1.0 means
    # the radii are honest; >1 under-confident radii, <1 over-confident.
    radius_scale_for_target: float = float("nan")

    def report(self) -> str:
        lines = [
            f"flights={self.n_flights} predictions={self.n_predictions}",
            f"mean final error = {self.mean_final_error_km:.2f} km",
            f"calibration (error<=radius) = {self.calibration_rate * 100:.0f}% "
            f"(target {TARGET_COVERAGE * 100:.0f}%)",
            f"radius scale for {TARGET_COVERAGE * 100:.0f}% coverage = "
            f"{self.radius_scale_for_target:.2f}",
            "error by altitude-at-prediction:",
        ]
        for key in self.bucket_mean_error_km:
            lines.append(
                f"  {key:>12}: {self.bucket_mean_error_km[key]:.2f} km "
                f"(n={self.bucket_counts[key]})"
            )
        return "\n".join(lines)


# The published radius is meant to be a ~68% (1-sigma) bound.
TARGET_COVERAGE = 0.68


def aggregate(results: list[ReplayResult]) -> Metrics:
    """Aggregate per-flight replays into accuracy/calibration metrics."""
    finals = [r.final_error_km for r in results if r.final_error_km is not None]
    bucket_err: dict[str, list[float]] = {_bucket_key(b): [] for b in ALT_BUCKETS}
    inside = 0
    total = 0
    ratios: list[float] = []
    for r in results:
        for rec in r.records:
            total += 1
            inside += int(rec.inside_radius)
            bucket_err[_bucket_for(rec.alt_at_pred)].append(rec.error_km)
            if rec.uncertainty_km > 0:
                ratios.append(rec.error_km / rec.uncertainty_km)
    return Metrics(
        n_flights=len(results),
        n_predictions=total,
        mean_final_error_km=(sum(finals) / len(finals)) if finals else float("nan"),
        bucket_mean_error_km={
            k: (sum(v) / len(v) if v else float("nan")) for k, v in bucket_err.items()
        },
        bucket_counts={k: len(v) for k, v in bucket_err.items()},
        calibration_rate=(inside / total) if total else float("nan"),
        radius_scale_for_target=_coverage_scale(ratios, TARGET_COVERAGE),
    )


def _coverage_scale(ratios: list[float], target: float) -> float:
    """The factor every radius would need so ``target`` of predictions land
    inside it - i.e. the ``target`` quantile of error/radius ratios."""
    if not ratios:
        return float("nan")
    s = sorted(ratios)
    pos = target * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def accuracy_from_store(store: PredictionStore, limit: int = 200) -> list[ReplayResult]:
    """Build per-flight :class:`ReplayResult`s from *live* captured data: each
    recorded actual landing is the truth, scored against every prediction the
    daemon saved for that flight. Reuses the exact replay metrics, so
    'how correct were we?' on real flights reads the same as the offline harness.
    """
    results: list[ReplayResult] = []
    for lnd in store.recent_landings(limit=limit):
        launch_day = date.fromisoformat(lnd["launch_day"])
        preds = store.predictions_for(lnd["serial"], launch_day)
        if not preds:
            continue
        truth_lat, truth_lon = lnd["land_lat"], lnd["land_lon"]
        res = ReplayResult(serial=lnd["serial"], truth_lat=truth_lat,
                           truth_lon=truth_lon, launch_day=lnd["launch_day"])
        for p in preds:
            err = haversine_km(p["land_lat"], p["land_lon"], truth_lat, truth_lon)
            sim = (datetime.fromisoformat(p["land_eta"])
                   - datetime.fromisoformat(p["predicted_at"])).total_seconds()
            res.records.append(PredRecord(
                alt_at_pred=p["alt_at_pred"] if p["alt_at_pred"] is not None else 0.0,
                error_km=err,
                uncertainty_km=p["uncertainty_radius_km"],
                sim_seconds=sim,
                source=p["source"],
                inside_radius=err <= p["uncertainty_radius_km"],
            ))
        results.append(res)
    return results


def _bucket_key(b: tuple[int, int]) -> str:
    lo, hi = b
    return f"{lo // 1000}-{hi // 1000}km"


def _bucket_for(alt: float) -> str:
    for lo, hi in ALT_BUCKETS:
        if lo <= alt < hi:
            return _bucket_key((lo, hi))
    return _bucket_key(ALT_BUCKETS[-1])


# ---- ablation (plan Phase 5) -----------------------------------------------
# Run the same corpus once per wind-assembly variant so each component's
# contribution is measured, not assumed. If a component buys nothing, drop it.


@dataclass(slots=True)
class AblationMode:
    name: str
    description: str
    wants_gfs: bool                       # mode gets the GFS source (if available)
    configure: Callable[[Config], None]   # in-place tweak of a copied config


def _measured_only(cfg: Config) -> None:
    pass                                   # GFS withheld via wants_gfs=False


def _model_only(cfg: Config) -> None:
    cfg.predict.use_measured_winds = False


def _blend(cfg: Config) -> None:
    cfg.profile.descent_refresh_enabled = False


def _blend_refresh(cfg: Config) -> None:
    pass                                   # production defaults


def _bias_refresh(cfg: Config) -> None:
    cfg.profile.correction_mode = "bias"


def _gfs_only(cfg: Config) -> None:
    # quantifies HRRR-over-GFS: identical to blend+refresh when no HRRR cache
    cfg.hrrr.enabled = False


ABLATION_MODES: list[AblationMode] = [
    AblationMode("measured-only", "ascent column only, no model winds",
                 False, _measured_only),
    AblationMode("model-only", "model (GFS) winds only, measured column ignored",
                 True, _model_only),
    AblationMode("blend", "model + measured ascent column (no live refresh)",
                 True, _blend),
    AblationMode("blend+refresh", "model + measured column, refreshed during descent",
                 True, _blend_refresh),
    AblationMode("bias+refresh", "model shifted by the measured-minus-model bias",
                 True, _bias_refresh),
    AblationMode("gfs-only", "blend+refresh with HRRR disabled (GFS only)",
                 True, _gfs_only),
]


@dataclass(slots=True)
class AblationOutcome:
    mode: AblationMode
    metrics: Metrics
    results: list[ReplayResult]
    gfs_available: bool


def run_ablation(
    corpus_dir,
    cfg: Config | None = None,
    ground_fn: GroundFn | None = None,
    modes: list[str] | None = None,
    limit: int | None = None,
    gfs_factory: Callable[[Config], GFSWindSource | None] | None = None,
    jobs: int = 1,
    with_dem: bool = False,
) -> dict[str, AblationOutcome] | None:
    """Backtest the corpus once per ablation mode (plan Phase 5).

    Each mode runs on a deep copy of ``cfg`` so the variants stay independent.
    ``gfs_factory`` builds the model wind source per mode (default: the GFS
    cache at ``cfg.gfs.path``); modes that want model winds but find none still
    run - their record counts expose the degradation honestly. Returns None if
    the corpus is empty.

    ``jobs > 1`` replays each mode via :func:`backtest_corpus_parallel`; the
    sources are then rebuilt per worker from config, so injected ``ground_fn``/
    ``gfs_factory`` objects can't be honoured - pass ``with_dem`` instead.
    """
    import copy

    cfg = cfg or Config()
    if jobs > 1 and (ground_fn is not None or gfs_factory is not None):
        raise ValueError("jobs > 1 rebuilds sources from config per worker; "
                         "use with_dem= instead of ground_fn/gfs_factory")
    wanted = ABLATION_MODES if modes is None else [
        m for m in ABLATION_MODES if m.name in set(modes)]
    if modes is not None and len(wanted) != len(set(modes)):
        known = {m.name for m in ABLATION_MODES}
        raise ValueError(f"unknown ablation mode(s): {sorted(set(modes) - known)}")

    if gfs_factory is None:
        def gfs_factory(c: Config):
            from .hrrr import make_wind_source
            return make_wind_source(c)

    outcomes: dict[str, AblationOutcome] = {}
    for mode in wanted:
        mode_cfg = copy.deepcopy(cfg)
        mode.configure(mode_cfg)
        gfs = gfs_factory(mode_cfg) if mode.wants_gfs else None
        if jobs > 1:
            log.info("ablation mode %s (%d jobs)", mode.name, jobs)
            results = backtest_corpus_parallel(
                corpus_dir, cfg=mode_cfg, with_dem=with_dem,
                with_gfs=mode.wants_gfs, limit=limit, jobs=jobs)
        else:
            results = backtest_corpus(corpus_dir, cfg=mode_cfg, ground_fn=ground_fn,
                                      gfs_source=gfs, limit=limit)
        if not results:
            return None
        outcomes[mode.name] = AblationOutcome(
            mode=mode, metrics=aggregate(results), results=results,
            gfs_available=gfs is not None)
    return outcomes


def ablation_report(outcomes: dict[str, AblationOutcome]) -> str:
    """Human-readable comparison: headline numbers per mode, then the
    per-flight final-error matrix (modes as columns)."""
    name_w = max(len(n) for n in outcomes) + 2
    lines = ["ablation over the wind-assembly pipeline:", ""]
    header = (f"{'mode':<{name_w}} {'flights':>7} {'preds':>6} "
              f"{'final err':>10} {'calib':>6} {'r-scale':>8}  notes")
    lines.append(header)
    lines.append("-" * len(header))
    for name, o in outcomes.items():
        m = o.metrics
        note = o.mode.description
        if o.mode.wants_gfs and not o.gfs_available:
            note += " [NO GFS DATA - degraded]"
        lines.append(
            f"{name:<{name_w}} {m.n_flights:>7} {m.n_predictions:>6} "
            f"{m.mean_final_error_km:>8.2f}km {m.calibration_rate:>5.0%} "
            f"{m.radius_scale_for_target:>8.2f}  {note}")

    serials: list[str] = []
    for o in outcomes.values():
        for r in o.results:
            if r.serial not in serials:
                serials.append(r.serial)
    mode_names = list(outcomes)
    lines.append("")
    lines.append("per-flight final error (km):")
    lines.append("  " + f"{'serial':>12} " + " ".join(f"{n:>14}" for n in mode_names))
    for s in serials:
        cells = []
        for n in mode_names:
            err = next((r.final_error_km for r in outcomes[n].results
                        if r.serial == s), None)
            cells.append(f"{err:>14.2f}" if err is not None else f"{'-':>14}")
        lines.append("  " + f"{s:>12} " + " ".join(cells))
    return "\n".join(lines)
