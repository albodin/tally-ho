"""Prediction-engine configuration.

Config comes from a TOML file plus environment-variable overrides
(``WINDFALL_<SECTION>_<KEY>`` when running standalone). The tally-ho app embeds
this config by subclassing :class:`Config` with its service-level sections and
its own env prefix, so the engine's knobs are tuned identically in both worlds.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackerConfig:
    burst_drop_m: float = 300.0            # alt below running max to call burst
    burst_consecutive: int = 3             # consecutive negative-rate frames
    # A burst altitude is only *recorded* (and fed to the per-site prior) once the
    # flight has been observed to genuinely climb this far above its first fix. A
    # sonde first heard already descending (launched out of range, drifted into
    # reception on the way down) never gains this, so it is tracked as DESCENT but
    # keeps ``burst_alt = None`` - we never saw its burst, so it must not poison
    # the climatology with a bogus low "burst" at the first-heard altitude.
    ascent_min_gain_m: float = 500.0
    # Undo a called burst if the sonde later climbs back this far above the
    # recorded burst altitude: a strong downdraft (or a GPS spike poisoning
    # max_alt) can briefly fake the drop, but a real burst never re-ascends. The
    # revert clears burst_alt so the *real* burst higher up is the one recorded.
    burst_revert_climb_m: float = 500.0
    float_rate_abs_mps: float = 1.0        # |vert rate| below this = floaty
    float_min_alt_m: float = 12_000.0      # only consider FLOAT above this band
    float_window_seconds: float = 300.0    # sustained window to declare FLOAT
    landed_alt_above_ground_m: float = 300.0  # near-ground band for LANDED
    landed_timeout_seconds: float = 180.0  # telemetry gap while low → LANDED
    # A DESCENT flight that drops below the radio horizon *above* the low band
    # (common in hills, where the last fix is 2-3 km up) is on the ground within
    # minutes; after this gap it is closed out - without a landing-truth record,
    # since the last fix was nowhere near the ground.
    descent_lost_timeout_seconds: float = 600.0
    # An ASCENT flight silent this long is down: the radio horizon only grows on
    # the way up, so losing an ascending sonde for good means the tracker missed
    # the rest of the flight (it was stopped with sondes aloft, or the sonde
    # died). 90 min covers a worst-case remaining climb + descent; without this,
    # such flights sat "mid-air" until stale_flight_seconds. Closed out as
    # EXPIRED - the mid-ascent last fix is nowhere near the landing. FLOAT is
    # exempt (floaters legitimately stay up for hours) and keeps the stale sweep.
    ascent_lost_timeout_seconds: float = 5400.0
    # A flight silent this long is over no matter its altitude (drifted out of
    # receiver range mid-air). It is closed out WITHOUT a landing record and
    # evicted from memory, so it stops showing as active forever.
    stale_flight_seconds: float = 6 * 3600.0
    # Claim a launch site only when the first frame is this close to the ground
    # (AGL when a DEM is wired, MSL over the flat fallback). A sonde first heard
    # already high launched out of range - its launch site is unknown, and
    # pinning the map's 🚀 to the first-heard position would be wrong.
    launch_max_agl_m: float = 1500.0
    new_ascent_climb_m: float = 500.0      # climb above the LANDING alt to declare a fresh flight
    # ...sustained for this many consecutive frames. A landed sonde keeps pinging
    # from the ground for hours with noisy GPS fixes; one (or two) high fixes must
    # not resurrect it into a ghost ASCENT flight - a real relaunch keeps climbing.
    new_ascent_consecutive: int = 3
    # Flown-track breadcrumb capture (the actual path, launch → now, for the map).
    # A frame is appended to the track once it has moved enough since the last
    # kept point - downsamples the dense telemetry so the stored trail stays small.
    track_min_move_m: float = 250.0        # ...this far horizontally, OR
    track_min_alt_m: float = 200.0         # ...this much in altitude, OR
    track_min_interval_s: float = 30.0     # ...at least this long since the last point
    # Vertical rate from a least-squares fit of altitude vs time over a short
    # sliding window rather than a single frame pair - GPS altitude noise on
    # 1 Hz pairs is 1-3 m/s, comparable to the descent rate near the ground.
    # Feeds burst/float/landed detection and the ballistic descent samples.
    vrate_window_seconds: float = 25.0
    vrate_min_points: int = 3              # fall back to the frame pair below this
    vrate_min_span_seconds: float = 4.0    # ...or when the window spans less time
    # Frame-level teleport gate. The wind profile is already
    # protected by the segment plausibility gate, but the state machine was not:
    # one corrupt frame (DFM teleports in the wild) poisons ``max_alt`` forever,
    # degrading burst detection to rate-only and recording a garbage burst_alt
    # into climatology. A frame implying an impossible speed from the last
    # accepted fix is rejected outright; after ``glitch_accept_after``
    # *consecutive* rejections the new position is accepted as real (the sonde
    # genuinely is where it says - we were wrong, stop fighting the data).
    glitch_horizontal_mps: float = 250.0   # max plausible ground speed (jet ~120)
    glitch_vertical_mps: float = 150.0     # max plausible |vrate| (fall peaks ~75)
    glitch_accept_after: int = 5


@dataclass(slots=True)
class ProfileConfig:
    bin_size_m: float = 150.0
    max_horizontal_mps: float = 200.0     # plausibility gate
    # Max |vertical rate| for a valid segment. Must clear real post-burst fall
    # rates (40-70 m/s in thin air) or descent live-refresh silently rejects the
    # very bins it should be updating right after burst; real GPS teleports
    # imply hundreds of m/s, so the glitch margin survives the higher bound.
    max_alt_step_mps: float = 90.0
    min_dt_seconds: float = 0.5           # ignore segments shorter than this
    # Measured→GFS blending. The ascent column is gospel
    # near where/when it was sampled; as the descending payload drifts away or
    # the sample ages, trust shifts toward GFS. Only active when a 4-D GFS wind
    # field is available; weight = exp(-(distance/D + age/A)).
    gfs_blend_enabled: bool = True
    gfs_blend_distance_km: float = 60.0   # e-folding horizontal distance of trust
    gfs_blend_age_s: float = 5400.0       # e-folding age of trust (1.5 h)
    # Below this height above ground the measured column is ignored outright in
    # favour of model winds (plan Phase 2): boundary-layer winds at the landing
    # zone are terrain-local, and the launch-site measurement misleads there.
    # Only active when a ground model and a 4-D model field are both wired.
    gfs_blend_min_agl_m: float = 3000.0
    # An interior altitude gap in the measured profile wider than this (e.g. a
    # reception dropout) is filled from GFS instead of lerping across it.
    interior_gap_fill_m: float = 600.0
    # How the measured column corrects the model field (plan Phase 2):
    #   "blend" - replace: wind = w·measured + (1-w)·model (w decays with
    #             distance/age). Simple, strong where the model is biased AND
    #             the trajectory stays near the ascent column.
    #   "bias"  - correct: wind = model + w·(measured - model at the place/time
    #             the layer was actually sampled). Preserves the model's own
    #             spatial structure; the measurement only shifts it.
    # Both share the decay constants and the AGL floor below; ablate to choose.
    correction_mode: str = "blend"
    # Live refresh (plan Phase 2): descent telemetry is itself a wind
    # measurement - closer to the landing zone and newer than the ascent column.
    # When enabled, descent frame pairs keep updating the profile bins they
    # re-cross, with the old bin contents down-weighted so the fresher, nearer
    # sample dominates quickly.
    descent_refresh_enabled: bool = True
    # Effective prior weight of an already-populated bin when a descent sample
    # refreshes it - the old vector average counts as at most this many samples.
    descent_refresh_n_cap: int = 8


@dataclass(slots=True)
class DescentConfig:
    transient_seconds: float = 45.0       # discard post-burst transient
    transient_drop_m: float = 1000.0      # ...or first ~1 km of descent
    b_min: float = 2.0                    # ballistic constant clamp
    b_max: float = 30.0
    recency_halflife_s: float = 120.0     # weight recent descent points more
    min_fit_points: int = 4               # below this, use single-point shortcut
    default_b: float = 5.5                # ballistic constant for pre-burst/no-data
    # Descent character sometimes changes mid-fall (balloon remnants detach,
    # the chute finally inflates ~5-10 km down where air thickens). When the
    # recent per-point ballistic constants step away from the older ones by
    # more than this relative jump, the fit resets to post-change samples only.
    regime_change_rel: float = 0.25
    regime_recent_points: int = 12        # window defining "recent" for the test


@dataclass(slots=True)
class IntegratorConfig:
    dt_seconds: float = 1.0               # integrator step (1-2 s)
    max_iterations: int = 200_000         # runaway guard
    max_sim_seconds: float = 6 * 3600.0


@dataclass(slots=True)
class EnsembleConfig:
    # Monte Carlo landing ensemble. The error sources are
    # explicit, so sample them: ballistic constant (fit residual), burst altitude
    # and ascent rate (pre-burst), and a vertically-correlated wind error. Each
    # member runs the production integrator; the ensemble mean is the landing
    # estimate and the spread quantile is the uncertainty radius.
    enabled: bool = True
    n_members: int = 100                  # descent ensemble size
    n_members_preburst: int = 60          # pre-burst sweep runs for every sonde aloft
    dt_seconds: float = 2.0               # member step (perturbations dominate dt error)
    # Refresh the ensemble at most this often per flight (sonde time); between
    # refreshes the cached mean-offset/radius rides on the per-frame
    # deterministic prediction. Keeps per-frame descent predictions cheap.
    min_interval_seconds: float = 60.0
    wind_sigma_measured_mps: float = 1.2  # wind error inside the measured column
    wind_sigma_extrapolated_mps: float = 3.0  # ...in extrapolated/GFS air
    wind_corr_length_m: float = 1500.0    # vertical correlation length of wind error
    # Per-member CONSTANT wind offset (m/s per component), drawn once per member.
    # The AR(1) noise above decorrelates over ~wind_corr_length_m and averages
    # out over a 25 km descent, so it cannot represent the *systematic* model
    # bias every member shares - which is exactly the error mode that made the
    # published radii cover only 52% instead of 68% on the full-corpus backtest
    # (2026-06-10). Calibration knob: tune against `windfall backtest`.
    wind_bias_sigma_mps: float = 1.5
    b_sigma_rel_fit: float = 0.05         # relative B spread floor for a real fit
    b_sigma_rel_shortcut: float = 0.18    # ...for the single-point shortcut
    b_sigma_rel_preburst: float = 0.30    # ...for the assumed pre-burst chute
    burst_alt_sigma_m: float = 1500.0     # pre-burst: burst-altitude spread
    ascent_rate_sigma_rel: float = 0.10   # pre-burst: ascent-rate spread
    quantile: float = 0.68                # radius = this quantile of member spread
    seed: int | None = None               # fixed RNG seed (None → stable per flight)


@dataclass(slots=True)
class UncertaintyConfig:
    # Heuristic uncertainty radius. Constants are calibration
    # knobs - tune against the replay harness.
    base_km: float = 0.3                  # irreducible floor near ground
    per_hour_measured_km: float = 1.5     # km of error per hour of measured flight
    per_hour_extrapolated_km: float = 6.0  # ...per hour of extrapolated/GFS column
    fit_residual_km_per_mps: float = 0.15  # descent-fit residual contribution
    # Multiplier on every published radius (ensemble and heuristic alike). The
    # backtest reports `radius scale for 68% coverage` - the measured value
    # drops straight in here, closing the calibration loop the harness was
    # already measuring but nothing consumed.
    radius_scale: float = 1.0


@dataclass(slots=True)
class PredictConfig:
    # Pre-burst landing estimates for sondes still ascending/floating.
    # Informational only - alerts stay gated on DESCENT. Refreshed for
    # every active flight on the maintenance tick (see ``predict_active_seconds``).
    preburst_enabled: bool = True
    predict_active_seconds: float = 30.0  # cadence of the airborne-flight sweep
    path_max_points: int = 64             # max sampled points in a predicted path
    # Re-predict a DESCENT flight at most this often (sonde time). Telemetry is
    # ~1 Hz but the answer barely moves frame-to-frame; predicting every frame
    # just burns the ingest thread and bloats the predictions table. The burst
    # transition itself always predicts immediately.
    descent_predict_seconds: float = 10.0
    # Publish a pre-burst estimate only when the wind column ahead is known:
    # GFS reachable above the measured range, OR the measured column already
    # covers at least this fraction of the estimated-burst column. Without
    # this gate, an early-ascent sonde's whole flight gets advected by its
    # topmost (boundary-layer) wind clamped over 30 km of column - errors of
    # 100+ km published with a straight face. No path beats a junk path.
    preburst_min_coverage: float = 0.7
    # Ablation switch (plan Phase 5): when False the measured ascent column is
    # ignored entirely and predictions run on model (GFS) winds alone - the
    # baseline that tells you what the ascent correction actually buys.
    use_measured_winds: bool = True


@dataclass(slots=True)
class DEMConfig:
    path: str = "dem"                     # directory of DEM tiles
    enabled: bool = True                  # if False or unavailable → flat-ground fallback
    # "glo30": Copernicus GLO-30 tiles, resolved by their global naming scheme.
    # "tiles": any GeoTIFF set, indexed by each file's bounds (drop USGS 3DEP
    #          10 m 1°x1° tiles - or any geographic-CRS rasters - in `path`).
    # "auto":  GLO-30 names if present, else the bounds index.
    source: str = "auto"
    # Auto-download the GLO-30 tiles covering the capture ROI on an in-app timer
    # thread (like GFS/HRRR). Terrain is static: once the ROI's tiles are on
    # disk every pass is a free existence check, and only a subscriber edit that
    # grows the ROI fetches more. Set False to manage tiles yourself (pre-seeded
    # read-only mount, scripts/prefetch_dem.sh, or a 3DEP set).
    download_in_process: bool = True
    download_check_seconds: float = 300.0  # re-check cadence (ROI can grow)
    # Parallel tile fetches per pass (clamped 1..16).
    download_workers: int = 4
    download_url: str = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


@dataclass(slots=True)
class GFSConfig:
    path: str = "gfs"                     # shared volume of downloaded GRIB
    enabled: bool = False                 # off until a downloaded cycle exists
    download_cadence_hours: float = 6.0
    download_in_process: bool = True      # run the downloader as an in-app timer
                                          # thread (no separate container)
    # Hourly forecast steps so any prediction time has a tight bracketing pair
    # (the wind field interpolates between the two nearest valid hours).
    download_fxx: list = field(default_factory=lambda: list(range(13)))
    # Wind + geopotential height + temperature (temperature → real GFS density),
    # isobaric ("mb") levels only - surface/10m/tropopause messages are dead
    # weight. Non-capturing group keeps pandas' str.contains warning quiet.
    download_levels: str = r":(?:UGRD|VGRD|HGT|TMP):\d+(?:\.\d+)? mb:"
    # A cycle only becomes usable this long after its run time - GFS lands on
    # the open-data bucket ~3.5-4 h after the hour. Keeps backtests honest
    # (no forecast that wasn't published yet); a no-op for live caches.
    publication_latency_hours: float = 3.8
    # Half-width (degrees) of the subgrid loaded around the flight: downloaded
    # files are level-subset but spatially GLOBAL (GRIB byte ranges are
    # per-message), and a full 0.25° cube is ~1 GB as float arrays - the
    # window is a few MB. 5° ≈ 550 km, far beyond any drift.
    window_deg: float = 5.0
    # After a live (latest-cycle) download, delete cached cycles older than
    # this (~1.6 GB/cycle, ~6 GB/day unpruned). 0 disables - the default, so
    # a bare CLI run can never eat a manually curated backtest archive;
    # deployments set it (docker-compose does). Pinned-date downloads
    # (fetch-gfs / --date backtest fetches) never prune.
    keep_hours: float = 0.0


@dataclass(slots=True)
class HRRRConfig:
    """HRRR - NOAA's 3 km CONUS model, hourly cycles. Far better boundary-layer
    and terrain-flow winds than GFS where it exists; its pressure-level files
    top out near 50 hPa (~20-21 km), so GFS still covers the column above
    (plan Phase 0). When both sources are enabled the predictor samples HRRR
    below the ceiling and GFS above, with a linear blend across the seam."""

    path: str = "hrrr"                    # cache of downloaded HRRR GRIB
    enabled: bool = False                 # off until a downloaded cycle exists
    ceiling_m: float = 20_000.0           # hand over to GFS above this altitude
    blend_ramp_m: float = 1_500.0         # width of the HRRR→GFS blend zone
    # Half-width (degrees) of the subgrid loaded around the flight: HRRR files
    # are full-CONUS (1799x1059 @ 3 km - gigabytes as float arrays), so only a
    # window around the query is materialised. 3° ≈ 330 km, far beyond any drift.
    window_deg: float = 3.0
    download_cadence_hours: float = 1.0   # HRRR runs hourly
    download_in_process: bool = True
    # Hourly cycles mean short forecast horizons suffice for a live descent;
    # a few hours of headroom covers publication latency (~50 min) and gaps.
    download_fxx: list = field(default_factory=lambda: [0, 1, 2, 3])
    download_levels: str = r":(?:UGRD|VGRD|HGT|TMP):\d+(?:\.\d+)? mb:"
    publication_latency_hours: float = 1.0   # HRRR publishes ~50 min after the hour
    keep_hours: float = 0.0   # live-download cache pruning; 0 = off (see GFSConfig)


@dataclass(slots=True)
class Config:
    """Engine configuration: every knob the predictor + accuracy harness need."""

    log_level: str = "INFO"
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    descent: DescentConfig = field(default_factory=DescentConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    dem: DEMConfig = field(default_factory=DEMConfig)
    gfs: GFSConfig = field(default_factory=GFSConfig)
    hrrr: HRRRConfig = field(default_factory=HRRRConfig)


def _apply(obj: Any, data: dict) -> None:
    """Recursively overlay a dict onto a dataclass instance, in place."""
    for key, value in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key: {key!r}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, *, cls: type = Config,
                env_prefix: str = "WINDFALL_") -> Any:
    """Load config from a TOML file (if given/present), then env overrides.

    Env override format: ``<PREFIX><SECTION>_<KEY>`` (e.g.
    ``WINDFALL_INTEGRATOR_DT_SECONDS=2``). Top-level keys use
    ``<PREFIX><KEY>``. The embedding app passes its own ``cls``/``env_prefix``.
    """
    cfg = cls()
    if path is not None:
        p = Path(path)
        if p.exists():
            with p.open("rb") as fh:
                _apply(cfg, tomllib.load(fh))
    apply_env(cfg, env_prefix)
    return cfg


def apply_env(cfg: Any, prefix: str) -> None:
    section_names = {f.name for f in fields(cfg) if is_dataclass(getattr(cfg, f.name))}
    for env_key, raw in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        rest = env_key[len(prefix):].lower()
        if rest == "config":
            # <PREFIX>CONFIG is the conventional config-file *path* variable,
            # consumed by the CLI before loading - not a key override
            continue
        target = cfg
        attr = rest
        for section in section_names:
            if rest.startswith(section + "_"):
                target = getattr(cfg, section)
                attr = rest[len(section) + 1:]
                break
        if not hasattr(target, attr):
            # TOML loading raises on unknown keys; env overrides can't (the
            # environment may carry unrelated vars), but a typo'd knob silently
            # doing nothing is a tuning-session footgun - make it visible.
            log.warning("ignoring env override %s: no config key %r", env_key, rest)
            continue
        setattr(target, attr, _coerce_like(getattr(target, attr), raw))


def _coerce_like(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw
