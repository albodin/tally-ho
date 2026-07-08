"""The production predictor.

`Predictor.predict` is the single function that turns a flight's current state +
measured profile + descent samples into a landing :class:`Prediction`. The
replay harness drives *this exact function* - there is no offline /
online fork - so tuning numbers transfer directly to production.

Wind source selection:
* measured ascent profile when present → ``source = measured``;
* otherwise GFS, if a GFS wind source is wired (Phase 5) → ``source = gfs``;
* GFS also fills above/below the measured range and wide interior gaps as an
  edge policy;
* when the GFS source can serve a full 4-D field, the integrator samples it at
  its *current position and time* each step, and measured winds are blended
  toward GFS as the trajectory drifts away from the ascent column or the
  samples age.

When the Monte Carlo ensemble is enabled, the landing
point is the ensemble mean and the uncertainty radius is the empirical member
spread; the heuristic linear radius remains as the fallback. Ensembles refresh
at most every ``ensemble.min_interval_seconds`` of sonde time per flight - in
between, the cached mean-offset/radius rides on the per-frame deterministic
prediction, keeping the hot path cheap.
"""

from __future__ import annotations

import logging
import random
import zlib
from datetime import datetime, timedelta
from typing import Callable, Optional

from .climatology import Climatology
from .config import Config
from .descent import DescentModel, fit_descent, shortcut_descent
from .ensemble import ensemble_descent, ensemble_preburst
from .geo import normalize_lon
from .integrator import WindFn, integrate_ascent, integrate_descent
from .models import FlightState, Prediction, PredictionSource
from .profile import FlightProfile, bias_corrected_wind_fn, blended_wind_fn
from .tracker import Flight
from .uncertainty import uncertainty_radius_km

log = logging.getLogger(__name__)

GroundFn = Callable[[float, float], float]


class GFSWindSource:
    """Interface for a GFS wind source (implemented in Phase 5, :mod:`windfall.gfs`)."""

    def profile_at(self, lat: float, lon: float, when: datetime) -> Optional[FlightProfile]:
        raise NotImplementedError

    def wind_filler(self, lat: float, lon: float, when: datetime):
        """Return an ``(alt) -> (u, v) | None`` callable for edge-fill, or None."""
        return None

    def wind_field(self, lat: float, lon: float, when: datetime) -> WindFn | None:
        """Return a 4-D wind field ``(lat_deg, lon_deg, alt_m, sim_seconds) ->
        (u, v)`` anchored at ``when`` (sim time 0), or None when unavailable."""
        return None


class Predictor:
    def __init__(
        self,
        cfg: Config,
        ground_fn: GroundFn | None = None,
        gfs_source: GFSWindSource | None = None,
        climatology: Climatology | None = None,
    ):
        self.cfg = cfg
        self.ground_fn = ground_fn or (lambda lat, lon: 0.0)
        self.gfs_source = gfs_source
        self.climatology = climatology
        # (serial, kind) -> (sonde_t, dlat, dlon, radius_km) of the last ensemble
        self._ens_cache: dict[tuple[str, str], tuple[float, float, float, float]] = {}

    def predict_for_flight(self, flight: Flight, now: datetime | None = None) -> Prediction | None:
        """State-dispatching entry point: a descent prediction once burst has
        been detected, otherwise the informational pre-burst estimate while the
        sonde is still going up. This is what the live app calls so *every* sonde
        aloft gets a current landing prediction + path."""
        if flight.state == FlightState.DESCENT:
            return self.predict(flight, now=now)
        if flight.state in (FlightState.ASCENT, FlightState.FLOAT):
            if not self.cfg.predict.preburst_enabled:
                return None
            return self.predict_preburst(flight, now=now)
        return None

    def predict(
        self, flight: Flight, now: datetime | None = None, capture_path: bool = True
    ) -> Prediction | None:
        """Predict the landing for ``flight`` from its current state, or None if
        there is not enough information this cycle."""
        if flight.last_lat is None or flight.last_lon is None or flight.last_alt is None:
            return None
        when = flight.last_seen or now

        descent = self._descent_model(flight)
        if descent is None:
            return None

        profile, source = self._wind_profile(flight, when)
        if profile is None:
            return None
        wind_fn = self._wind_fn(flight, profile, source, when)

        landing = integrate_descent(
            lat=flight.last_lat,
            lon=flight.last_lon,
            alt=flight.last_alt,
            t0=when,
            profile=profile,
            descent=descent,
            ground_fn=self.ground_fn,
            cfg=self.cfg.integrator,
            capture_path=capture_path,
            path_max_points=self.cfg.predict.path_max_points,
            wind_fn=wind_fn,
        )
        if not landing.ok:
            log.debug("integrator failed for %s: %s", flight.serial, landing.reason)
            return None

        land_lat, land_lon = landing.lat, landing.lon
        radius: float | None = None
        if self._ensemble_on(self.cfg.ensemble.n_members):
            off = self._ensemble_offset(
                "descent", flight, landing.lat, landing.lon,
                lambda rng: ensemble_descent(
                    lat=flight.last_lat, lon=flight.last_lon, alt=flight.last_alt,
                    t0=when, profile=profile, descent=descent,
                    ground_fn=self.ground_fn, cfg=self.cfg,
                    wind_fn=wind_fn, rng=rng,
                    measured_range=(profile.alt_range()
                                    if source == PredictionSource.MEASURED else None),
                ),
            )
            if off is not None:
                dlat, dlon, r = off
                land_lat += dlat
                land_lon = normalize_lon(land_lon + dlon)
                radius = round(max(self.cfg.uncertainty.base_km,
                                   r * self.cfg.uncertainty.radius_scale), 2)
        if radius is None:
            # GFS-sourced winds are not "measured from this flight" - they fall
            # in the extrapolated bucket of the uncertainty model.
            measured_fraction = (
                0.0 if source == PredictionSource.GFS else landing.measured_fraction)
            radius = uncertainty_radius_km(
                sim_seconds=landing.sim_seconds,
                measured_fraction=measured_fraction,
                fit_residual_mps=descent.residual_mps,
                cfg=self.cfg.uncertainty,
            )
        path = landing.path
        if path and (land_lat != landing.lat or land_lon != landing.lon):
            path[-1] = (land_lat, land_lon, path[-1][2])
        return Prediction(
            serial=flight.serial,
            launch_day=flight.launch_day,
            predicted_at=now or when,
            land_lat=land_lat,
            land_lon=land_lon,
            land_eta=landing.eta,
            source=source,
            uncertainty_radius_km=radius,
            alt_at_pred=flight.last_alt,
            path=path,
        )

    def predict_preburst(self, flight: Flight, now: datetime | None = None) -> Prediction | None:
        """Pre-burst landing estimate: estimate the burst altitude
        (site climatology beats the type table), integrate the ascent up to burst
        advected by GFS (or measured) winds, then descend on a learned-or-default
        chute. Informational only - alerts stay gated on DESCENT."""
        if flight.state not in (FlightState.ASCENT, FlightState.FLOAT):
            return None
        if flight.last_lat is None or flight.last_lon is None or flight.last_alt is None:
            return None
        when = flight.last_seen or now
        if when is None:
            return None

        from .gfs import estimate_burst_alt  # local import avoids a cycle

        profile, source = self._wind_profile(flight, when)
        # Pre-burst can also use the measured ascent column (clamped above range).
        if profile is None or profile.is_empty():
            if not self.cfg.predict.use_measured_winds or flight.profile.is_empty():
                return None
            profile, source = flight.profile, PredictionSource.MEASURED
        wind_fn = self._wind_fn(flight, profile, source, when)

        site_burst = None
        default_b = self.cfg.descent.default_b
        if self.climatology is not None:
            site_burst = self.climatology.burst_alt(
                flight.launch_lat if flight.launch_lat is not None else flight.last_lat,
                flight.launch_lon if flight.launch_lon is not None else flight.last_lon,
                flight.type,
            )
            learned_b = self.climatology.descent_b(flight.type)
            if learned_b is not None:
                default_b = learned_b

        burst_alt = estimate_burst_alt(
            current_alt=flight.last_alt,
            ascent_rate=flight.last_vrate,
            burst_timer=getattr(flight.prev_frame, "burst_timer", None),
            sonde_type=flight.type,
            site_burst_alt=site_burst,
            cfg=self.cfg,
        )
        if not self._preburst_winds_known(profile, source, wind_fn, burst_alt):
            return None
        ascent_rate = (
            flight.robust_ascent_rate()
            or (flight.last_vrate if (flight.last_vrate and flight.last_vrate > 0) else None)
            or 5.0
        )
        ascent_path: list = []
        b_lat, b_lon, t_ascent = integrate_ascent(
            lat=flight.last_lat, lon=flight.last_lon, alt=flight.last_alt,
            burst_alt=burst_alt, ascent_rate=ascent_rate, profile=profile,
            cfg=self.cfg.integrator,
            path_out=ascent_path,
            path_max_points=max(8, self.cfg.predict.path_max_points // 2),
            wind_fn=wind_fn,
        )
        descent = DescentModel(b=default_b, residual_mps=0.0, n_points=0)
        landing = integrate_descent(
            lat=b_lat, lon=b_lon, alt=burst_alt,
            t0=when + timedelta(seconds=t_ascent),
            profile=profile, descent=descent, ground_fn=self.ground_fn,
            cfg=self.cfg.integrator,
            capture_path=True,
            path_max_points=self.cfg.predict.path_max_points,
            wind_fn=wind_fn, t_offset_s=t_ascent,
        )
        if not landing.ok:
            return None

        land_lat, land_lon = landing.lat, landing.lon
        radius: float | None = None
        if self._ensemble_on(self.cfg.ensemble.n_members_preburst):
            off = self._ensemble_offset(
                "preburst", flight, landing.lat, landing.lon,
                lambda rng: ensemble_preburst(
                    lat=flight.last_lat, lon=flight.last_lon, alt=flight.last_alt,
                    t0=when, burst_alt=burst_alt, ascent_rate=ascent_rate,
                    default_b=default_b, profile=profile,
                    ground_fn=self.ground_fn, cfg=self.cfg,
                    wind_fn=wind_fn, rng=rng,
                    measured_range=(profile.alt_range()
                                    if source == PredictionSource.MEASURED else None),
                ),
            )
            if off is not None:
                dlat, dlon, r = off
                land_lat += dlat
                land_lon = normalize_lon(land_lon + dlon)
                radius = round(max(self.cfg.uncertainty.base_km,
                                   r * self.cfg.uncertainty.radius_scale), 2)
        if radius is None:
            radius = uncertainty_radius_km(
                sim_seconds=landing.sim_seconds + t_ascent,
                measured_fraction=0.0,   # pre-burst column is essentially all modelled
                fit_residual_mps=2.0,    # no descent observed yet → inflate
                cfg=self.cfg.uncertainty,
            )
        # Full predicted track: the rising leg up to burst, then the descent.
        path = ascent_path + (landing.path or [])
        if path and (land_lat != landing.lat or land_lon != landing.lon):
            path[-1] = (land_lat, land_lon, path[-1][2])
        return Prediction(
            serial=flight.serial, launch_day=flight.launch_day,
            predicted_at=now or when, land_lat=land_lat, land_lon=land_lon,
            land_eta=landing.eta, source=source, uncertainty_radius_km=radius,
            alt_at_pred=flight.last_alt, path=path,
        )

    # ---- internals --------------------------------------------------------
    def _preburst_winds_known(
        self,
        profile: FlightProfile,
        source: PredictionSource,
        wind_fn: WindFn | None,
        burst_alt: float,
    ) -> bool:
        """Whether the wind column the flight will traverse is known well enough
        to publish a pre-burst estimate. With neither a GFS field nor edge-fill,
        winds above the measured range clamp to the topmost sample - a 30 km
        column extrapolated from one boundary-layer wind lands 100+ km wrong, so
        we publish nothing instead."""
        if source == PredictionSource.GFS:
            return True            # the whole column is model winds already
        if wind_fn is not None:
            return True            # 4-D GFS field sampled along the trajectory
        rng = profile.alt_range()
        if rng is None:
            return False
        if profile.gfs_fill is not None:
            # probe just above the measured range: does edge-fill actually have
            # data (a downloaded cycle), or is the filler wired but dry?
            if profile.gfs_fill(rng[1] + profile.bin_size_m) is not None:
                return True
        covered = (rng[1] - rng[0]) / max(burst_alt - rng[0], 1.0)
        return covered >= self.cfg.predict.preburst_min_coverage

    def _descent_model(self, flight: Flight) -> DescentModel | None:
        model = fit_descent(flight.descent_samples, self.cfg.descent,
                            burst_t=flight.burst_t, burst_alt=flight.burst_alt)
        if model is not None:
            return model
        if flight.descent_samples:
            last = flight.descent_samples[-1]
            return shortcut_descent(last.v_obs, last.rho, self.cfg.descent)
        return None

    def _wind_profile(
        self, flight: Flight, when: datetime | None
    ) -> tuple[FlightProfile | None, PredictionSource]:
        if self.cfg.predict.use_measured_winds and not flight.profile.is_empty():
            # Measured profile; wire GFS edge-fill above/below sampled range.
            if self.gfs_source is not None and when is not None:
                flight.profile.gfs_fill = self.gfs_source.wind_filler(
                    flight.last_lat, flight.last_lon, when
                )
            return flight.profile, PredictionSource.MEASURED
        if self.gfs_source is not None and when is not None:
            gfs_profile = self.gfs_source.profile_at(flight.last_lat, flight.last_lon, when)
            if gfs_profile is not None and not gfs_profile.is_empty():
                return gfs_profile, PredictionSource.GFS
        return None, PredictionSource.MEASURED

    def _wind_fn(
        self,
        flight: Flight,
        profile: FlightProfile,
        source: PredictionSource,
        when: datetime | None,
    ) -> WindFn | None:
        """The 4-D wind callable for the integrator, when one can be built."""
        if self.gfs_source is None or when is None:
            return None
        field = self.gfs_source.wind_field(flight.last_lat, flight.last_lon, when)
        if field is None:
            return None
        if source == PredictionSource.GFS:
            return field
        if not self.cfg.profile.gfs_blend_enabled:
            return None
        make = (bias_corrected_wind_fn if self.cfg.profile.correction_mode == "bias"
                else blended_wind_fn)
        return make(profile, field, self.cfg.profile,
                    t0_epoch=flight.last_t, ground_fn=self.ground_fn)

    def _ensemble_on(self, n_members: int) -> bool:
        return self.cfg.ensemble.enabled and n_members >= 2

    def _seed(self, flight: Flight) -> int:
        if self.cfg.ensemble.seed is not None:
            return self.cfg.ensemble.seed
        # stable per flight → successive predictions move smoothly, not jitter
        return zlib.crc32(flight.serial.encode())

    def _ensemble_offset(
        self,
        kind: str,
        flight: Flight,
        det_lat: float,
        det_lon: float,
        run,
    ) -> tuple[float, float, float] | None:
        """Run (or reuse) the ensemble and express it as an offset from the
        deterministic landing + a radius. The offset/radius is cached per flight
        and refreshed at most every ``ensemble.min_interval_seconds`` of sonde
        time, so per-frame predictions stay cheap between refreshes."""
        key = (flight.serial, kind)
        t_now = flight.last_t if flight.last_t is not None else 0.0
        cached = self._ens_cache.get(key)
        if cached is not None and abs(t_now - cached[0]) < self.cfg.ensemble.min_interval_seconds:
            return cached[1], cached[2], cached[3]
        ens = run(random.Random(self._seed(flight)))
        if ens is None:
            # keep riding a stale offset rather than snapping back to nothing
            return (cached[1], cached[2], cached[3]) if cached else None
        dlat = ens.lat - det_lat
        dlon = (ens.lon - det_lon + 180.0) % 360.0 - 180.0
        # evict oldest entries rather than clear(): a full clear would force a
        # synchronized ensemble recompute across every active flight at once
        while len(self._ens_cache) > 512:
            self._ens_cache.pop(next(iter(self._ens_cache)))
        self._ens_cache[key] = (t_now, dlat, dlon, ens.radius_km)
        return dlat, dlon, ens.radius_km
