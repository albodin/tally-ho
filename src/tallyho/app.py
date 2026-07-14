"""The monolith: ingest → tracker → predictor → geofence → notify.

`App` wires the modules together. It is driven by frame batches (from the live
SondeHub stream in production, or from tests/replay offline), so the whole
pipeline is exercisable without a network connection.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from windfall.climatology import Climatology
from .backfill import HistoryFetcher, merge_history
from .config import Config
from windfall.dem import ReloadableGround, download_dem_tiles
from .geofence import build_capture_roi, in_capture_roi
from windfall.geo import haversine_km
from windfall.gfs import download_gfs_cycle
from windfall.hrrr import download_hrrr_cycle, make_wind_source
from .ingest import SondeHubStream, TelemetryProcessor
from .models import Frame, FlightState
from .notify import AlertManager, HttpNtfySink, NtfySink
from windfall.predictor import GFSWindSource, Predictor
from .store import Store
from windfall.tracker import FlightTracker, TrackerEvent

log = logging.getLogger(__name__)

# A pending silence-recovery fetch holds its flight out of the timeout sweep;
# past this age the hold is abandoned (a wedged fetch worker must not keep
# ghost flights mid-air forever - that is the very bug recovery exists to fix).
_RECOVERY_HOLD_MAX_SECONDS = 1800.0

# Above this many queued raw frames the consumer is visibly behind real time:
# shed optional work (pre-burst sweep) and defer the maintenance tick, whose
# silence timeouts would misread queued-but-unprocessed flights as silent.
_INGEST_BACKLOG_SHED = 5_000


class App:
    def __init__(
        self,
        cfg: Config,
        store: Store | None = None,
        sink: NtfySink | None = None,
        gfs_source: GFSWindSource | None = None,
        config_path: str | Path | None = None,
        backfill_fetch_fn=None,
    ):
        self.cfg = cfg
        # the config.toml behind cfg; the in-app web UI's settings editor
        # writes it (None = embedders/tests without one -> editor read-only)
        self.config_path = config_path
        self.store = store or Store(cfg.db_path)
        # Reloadable so tiles the in-app DEM downloader fetches later are picked
        # up by the tracker/predictor (which capture ground_fn here) mid-run.
        self.ground_fn = ReloadableGround(cfg.dem)
        self.tracker = FlightTracker(cfg, store=self.store, ground_fn=self.ground_fn)
        # Resume in-flight sondes across a restart so the next frame doesn't get
        # treated as a fresh launch (which would move the launch marker to the
        # sonde's current position and lose state/profile).
        self.tracker.load_active()
        # Read GFS from the local cache (filled by the in-app downloader thread
        # or a standalone run of scripts/gfs_download.py) unless one is injected.
        self.predictor = Predictor(
            cfg, ground_fn=self.ground_fn,
            gfs_source=gfs_source if gfs_source is not None else make_wind_source(cfg),
            # learned priors (per-site burst altitude, per-type chute B) from
            # this receiver's own flight history
            climatology=Climatology(self.store),
        )
        self.alerts = AlertManager(cfg, self.store, sink or HttpNtfySink(
            timeout=cfg.notify.request_timeout_seconds,
            token_lookup=self.store.get_ntfy_token))
        self.last_frame_at: datetime | None = None
        self._subscribers = []
        self._capture_box = None
        # sonde-time of the last saved prediction per descending flight - the
        # per-frame throttle (see PredictConfig.descent_predict_seconds)
        self._last_pred_t: dict[tuple, float] = {}
        # History backfill: a sonde first heard mid-air - or a tracked flight
        # that goes silent (restart with sondes aloft, stream drop) - gets its
        # missed frames fetched from SondeHub (worker thread) and its flight
        # rebuilt on this thread - see apply_backfills(). One attempt per flight
        # per trigger; live frames heard while a fetch runs are buffered for the
        # rebuild. While a silence-recovery fetch is pending its flight is held
        # out of the timeout sweep (tick), so the rebuild - not a blind expiry -
        # decides its fate.
        self._backfill = HistoryFetcher(cfg.ingest, fetch_fn=backfill_fetch_fn)
        self._backfill_requested: set[tuple] = set()
        self._backfill_buffer: dict[str, list[Frame]] = {}
        self._recovery_requested: set[tuple] = set()
        self._recovery_pending: dict[str, datetime] = {}   # serial → requested at
        self._gfs_stop = threading.Event()
        # ROI-change wake-ups: a new/moved location kicks the ROI-dependent
        # downloaders immediately instead of waiting out their cadences (GFS
        # sleeps 6 h between passes; HRRR is full-CONUS and needs no kick).
        self._gfs_kick = threading.Event()
        self._dem_kick = threading.Event()
        self._gfs_thread: threading.Thread | None = None
        self._hrrr_thread: threading.Thread | None = None
        self._dem_thread: threading.Thread | None = None
        # tiles known absent upstream (ocean squares 404) - asked at most once
        # per process, not once per check pass
        self._dem_absent: set[str] = set()
        self._web_thread: threading.Thread | None = None
        self._web_server = None
        self.reload_subscribers()

    # ---- subscriber / ROI cache ------------------------------------------
    def reload_subscribers(self) -> None:
        old_box = self._capture_box
        self._subscribers = self.store.list_subscribers(active_only=True)
        self._capture_box = build_capture_roi(self._subscribers, self.cfg.roi.capture_margin_km)
        log.info("loaded %d subscribers; capture ROI=%s",
                 len(self._subscribers), self._capture_box)
        if self._capture_box != old_box:
            # The ROI appeared or moved (a location added/edited in the UI):
            # wake the GFS/DEM downloaders now so the first location gets winds
            # and terrain within minutes, not a download cadence later.
            self._gfs_kick.set()
            self._dem_kick.set()
        self._drop_outside_roi()

    def _drop_outside_roi(self) -> None:
        """Stop tracking flights whose last known position is outside the
        capture ROI - the box moved/shrank (subscriber edit) or the sonde flew
        away. Closed out without a landing-truth row or alert; once evicted,
        the per-frame gate keeps their frames out. With no active
        subscribers there is no box and every tracked flight is dropped."""
        for flight in list(self.tracker.flights.values()):
            if flight.state == FlightState.LANDED:
                continue  # already invisible; husk kept for serial-reuse detection
            if flight.last_lat is None or flight.last_lon is None:
                continue
            if in_capture_roi(self._capture_box, flight.last_lat, flight.last_lon):
                continue
            log.info("dropping flight %s: outside capture ROI at %.3f,%.3f",
                     flight.serial, flight.last_lat, flight.last_lon)
            self.tracker.drop(flight)
            self._last_pred_t.pop((flight.serial, flight.launch_day), None)

    # ---- frame processing -------------------------------------------------
    def on_frames(self, frames: list[Frame]) -> None:
        for f in frames:
            self.process_frame(f)

    def process_frame(self, frame: Frame) -> None:
        self.last_frame_at = datetime.now(timezone.utc)
        # Capture-ROI gate: process flights inside the wide capture box, or any
        # flight we are already tracking.
        already = self.tracker.get(frame.serial) is not None
        if not already and not in_capture_roi(self._capture_box, frame.lat, frame.lon):
            return

        flight, events = self.tracker.update(frame)

        if TrackerEvent.NEW_FLIGHT in events:
            self._maybe_backfill(flight, frame)
        elif frame.serial in self._backfill_buffer:
            # heard while this serial's history fetch runs: keep for the rebuild
            buf = self._backfill_buffer[frame.serial]
            if len(buf) < 5000:   # a fetch is seconds; cap a stuck worker's cost
                buf.append(frame)

        if TrackerEvent.RESUMED in events:
            # A silence close-out contradicted by live frames - the sonde is
            # still flying. Retract the provisional landing truth + LANDED
            # alert rows so the real landing re-records and re-alerts at the
            # corrected position (no-op for a resumed EXPIRED flight, which
            # never recorded either).
            self.store.retract_landing(flight.serial, flight.launch_day)

        if TrackerEvent.LANDED in events:
            self._record_landing(flight, now=flight.last_seen, detected_by="telemetry")
            self.alerts.handle_landed(flight, self._subscribers, now=flight.last_seen)
            return

        if flight.state == FlightState.LANDED and flight.landed_by_timeout:
            # Late ground pings on a timeout-landed flight: keep the recorded
            # landing (and the map's marker) on the freshest fix.
            self._refine_timeout_landing(flight)
            return

        if flight.state == FlightState.DESCENT:
            # Predict immediately on the burst transition, then at most every
            # descent_predict_seconds of sonde time - not every 1 Hz frame.
            if TrackerEvent.BURST in events or self._descent_pred_due(flight):
                pred = self.predictor.predict(flight)
                if pred is not None:
                    self._mark_predicted(flight)
                    self.store.save_prediction(pred)
                    self.alerts.handle_prediction(flight, pred, self._subscribers,
                                                  now=flight.last_seen)

    def _descent_pred_due(self, flight) -> bool:
        if flight.last_t is None:
            return True
        last = self._last_pred_t.get((flight.serial, flight.launch_day))
        return last is None or flight.last_t - last >= self.cfg.predict.descent_predict_seconds

    def _mark_predicted(self, flight) -> None:
        if flight.last_t is None:
            return
        if len(self._last_pred_t) > 4096:
            self._last_pred_t.clear()
        self._last_pred_t[(flight.serial, flight.launch_day)] = flight.last_t

    # ---- first-detection history backfill ----------------------------------
    def _maybe_backfill(self, flight, frame: Frame) -> bool:
        """Queue a SondeHub history fetch for a flight first heard mid-air.
        First heard near the ground means we have the whole flight already, so
        there is nothing to fetch (``launch_lat`` is only set on a near-ground
        first frame)."""
        if not self.cfg.ingest.backfill_enabled:
            return False
        if flight.launch_lat is not None:
            return False
        key = (flight.serial, flight.launch_day)
        if key in self._backfill_requested:
            return False
        if len(self._backfill_requested) > 4096:
            self._backfill_requested.clear()
        self._backfill_requested.add(key)
        if not self._backfill.request(flight.serial):
            return False
        log.info("%s first heard mid-air at %.0f m; fetching missed history "
                 "from SondeHub", flight.serial, frame.alt)
        self._backfill_buffer[flight.serial] = [frame]
        return True

    def apply_backfills(self) -> int:
        """Drain completed history fetches and rebuild their flights. Runs on
        the consumer thread (replay shares the tracker/DEM/predictor with frame
        processing). Returns the number of flights rebuilt."""
        rebuilt = 0
        for serial, raw in self._backfill.drain():
            self._recovery_pending.pop(serial, None)
            live = self._backfill_buffer.pop(serial, [])
            if not raw:
                continue
            try:
                if self._rebuild_from_history(serial, raw, live):
                    rebuilt += 1
            except Exception:  # noqa: BLE001 - a bad rebuild must not kill ingest
                log.exception("history rebuild for %s failed; keeping the "
                              "live-only flight", serial)
        return rebuilt

    def _rebuild_from_history(self, serial: str, raw: list[dict],
                              live: list[Frame]) -> bool:
        """Replace a tracked flight with one rebuilt from its full SondeHub
        history: forget the in-memory flight, erase its rows, and replay
        history + buffered live frames through the tracker (launch site, ascent
        wind profile, burst/descent/landed state all come out real). Serves
        both backfill triggers - history reaching *earlier* than we first heard
        (first-detection) or *later* than we last heard (silence recovery). One
        prediction/alert pass runs at the end - replay itself stays silent."""
        flight = self.tracker.get(serial)
        if flight is None or flight.state == FlightState.LANDED:
            return False   # dropped (ROI) or resolved while the fetch ran
        frames = merge_history(raw, live, serial)
        if not frames:
            return False
        starts_earlier = flight.first_seen is None or frames[0].dt < flight.first_seen
        ends_later = flight.last_seen is None or frames[-1].dt > flight.last_seen
        if not (starts_earlier or ends_later):
            return False   # history adds nothing beyond what we heard live
        log.info("rebuilding %s from %d frame(s) of history %s → %s (heard "
                 "live %s → %s)", serial, len(frames), frames[0].dt,
                 frames[-1].dt, flight.first_seen, flight.last_seen)
        self.tracker.forget(serial)
        self._last_pred_t.pop((serial, flight.launch_day), None)

        landed = False
        # One transaction + one SSE ping: the rebuilt flight must appear on the
        # dashboard atomically, not animate through its replayed history (and
        # thousands of per-frame commits would stall this thread for seconds).
        with self.store.batch():
            self.store.delete_flight(serial, flight.launch_day)
            for f in frames:
                fl, events = self.tracker.update(f)
                if TrackerEvent.NEW_FLIGHT in events:
                    landed = False   # replay can span a reused serial's older flight
                if TrackerEvent.LANDED in events:
                    landed = True

        if landed:
            # the history shows it already came down - close the loop now
            self._record_landing(fl, now=fl.last_seen, detected_by="telemetry")
            self.alerts.handle_landed(fl, self._subscribers, now=fl.last_seen)
        elif fl.state == FlightState.DESCENT:
            # falling right now: predict immediately off the rebuilt profile
            pred = self.predictor.predict(fl)
            if pred is not None:
                self._mark_predicted(fl)
                self.store.save_prediction(pred)
                self.alerts.handle_prediction(fl, pred, self._subscribers,
                                              now=fl.last_seen)
        return True

    def _request_recoveries(self, now: datetime) -> int:
        """Queue one history fetch for each tracked flight that has gone
        silent. Frames the daemon missed - it was down while the sondes flew
        on (the restart-with-sondes-aloft case), or the stream dropped - are
        still in SondeHub; replaying them closes the flight out with its real
        fate (a landing during the downtime becomes a real LANDED with ground
        truth) instead of a blind timeout expiry. One attempt per flight."""
        if not self.cfg.ingest.backfill_enabled:
            return 0
        n = 0
        for flight in list(self.tracker.flights.values()):
            if flight.state == FlightState.LANDED or flight.last_seen is None:
                continue
            gap = (now - flight.last_seen).total_seconds()
            if gap < self.cfg.ingest.backfill_silent_seconds:
                continue
            key = (flight.serial, flight.launch_day)
            # skip if already attempted, or a fetch for this serial is running
            if key in self._recovery_requested or flight.serial in self._backfill_buffer:
                continue
            if len(self._recovery_requested) > 4096:
                self._recovery_requested.clear()
            self._recovery_requested.add(key)
            if not self._backfill.request(flight.serial):
                continue
            log.info("%s silent %.0f s at %.0f m; fetching missed frames from "
                     "SondeHub", flight.serial, gap, flight.last_alt or 0.0)
            self._backfill_buffer[flight.serial] = []
            self._recovery_pending[flight.serial] = now
            n += 1
        return n

    def _recovery_hold(self, now: datetime) -> set[str]:
        """Serials whose recovery fetch is still pending: the timeout sweep
        must not close their flights out from under the coming rebuild.
        Overaged entries are dropped, not held (see _RECOVERY_HOLD_MAX_SECONDS)."""
        for serial in [s for s, t in self._recovery_pending.items()
                       if (now - t).total_seconds() > _RECOVERY_HOLD_MAX_SECONDS]:
            log.warning("recovery fetch for %s never completed; releasing its "
                        "flight to the timeout sweep", serial)
            del self._recovery_pending[serial]
        return set(self._recovery_pending)

    def predict_active(self, now: datetime | None = None) -> int:
        """Refresh pre-burst landing predictions + paths for every sonde still
        going up. DESCENT flights are predicted per-frame in
        :meth:`process_frame`, so this only sweeps ASCENT/FLOAT flights - giving
        each airborne sonde a current predicted path on the map even before
        burst. Returns the number of predictions saved. Informational only:
        alerts stay gated on DESCENT."""
        if not self.cfg.predict.preburst_enabled:
            return 0
        saved = 0
        for flight in list(self.tracker.flights.values()):
            if flight.state not in (FlightState.ASCENT, FlightState.FLOAT):
                continue
            pred = self.predictor.predict_for_flight(flight, now=now)
            if pred is not None:
                self.store.save_prediction(pred)
                saved += 1
        return saved

    def tick(self, now: datetime | None = None) -> None:
        """Periodic maintenance: out-of-ROI drop, silence-recovery fetches,
        landed-by-timeout sweep. The ROI sweep runs first so a flight that left
        the box (or was left behind by a subscriber edit) is dropped silently
        instead of alerting on a later timeout; recoveries queue before the
        timeout sweep so a flight already past its timeout at queue time (the
        first tick after a restart) is held for its rebuild, not expired."""
        now = now or datetime.now(timezone.utc)
        self._drop_outside_roi()
        self._request_recoveries(now)
        for flight, event in self.tracker.check_timeouts(now, hold=self._recovery_hold(now)):
            if event == TrackerEvent.LANDED:
                self._record_landing(flight, now=flight.last_seen or now,
                                     detected_by="timeout")
                self.alerts.handle_landed(flight, self._subscribers, now=now)

    def _record_landing(self, flight, now: datetime | None, detected_by: str) -> None:
        """Persist the actual landing position as accuracy ground truth."""
        if flight.last_lat is None or flight.last_lon is None:
            return
        # A "landing" within landing_truth_min_fall_m of the flight's apogee
        # never saw a real descent (e.g. GPS settling on the launch pad walked
        # the state machine to LANDED pre-launch). A poisoned truth row is
        # worse than none: it silently corrupts the accuracy/calibration
        # metrics and, being upserted by (serial, launch_day), outlives
        # rebuilds of the flight itself.
        if (flight.last_alt is not None and flight.max_alt != float("-inf")
                and flight.max_alt - flight.last_alt
                < self.cfg.tracker.landing_truth_min_fall_m):
            log.warning("not recording landing truth for %s: apogee %.0f m is "
                        "only %.0f m above the landing fix - no real descent "
                        "was observed", flight.serial, flight.max_alt,
                        flight.max_alt - flight.last_alt)
            return
        self.store.record_landing(
            serial=flight.serial, launch_day=flight.launch_day,
            land_lat=flight.last_lat, land_lon=flight.last_lon,
            land_alt=flight.last_alt, landed_at=now, detected_by=detected_by,
        )

    def _refine_timeout_landing(self, flight) -> None:
        """A timeout-landed sonde still pinging from the ground is reporting
        better landing truth than the fix recorded at the close-out (the last
        frame heard BEFORE the silence, possibly hundreds of metres short).
        Refresh the truth row once the fix meaningfully improves on it: enough
        horizontal movement, or clearly lower. No re-alert - a refinement is
        the same landing, better located (only a RESUMED flight's re-landing
        re-alerts). Telemetry landings are left alone: their fix is already on
        the ground, and ground noise must not walk a confirmed landing around."""
        if flight.last_lat is None or flight.last_lon is None:
            return
        lnd = self.store.get_landing(flight.serial, flight.launch_day)
        if lnd is None:    # e.g. the min-fall truth guard refused this flight
            return
        tcfg = self.cfg.tracker
        moved_m = haversine_km(lnd["land_lat"], lnd["land_lon"],
                               flight.last_lat, flight.last_lon) * 1000.0
        lower = (lnd["land_alt"] is not None and flight.last_alt is not None
                 and lnd["land_alt"] - flight.last_alt >= tcfg.landing_refine_alt_m)
        if moved_m < tcfg.landing_refine_move_m and not lower:
            return
        # keep the original landing time - the position is refined, but the
        # sonde came down when it came down, not when this ping arrived
        landed_at = (datetime.fromisoformat(lnd["landed_at"])
                     if lnd["landed_at"] else flight.last_seen)
        self.store.record_landing(
            serial=flight.serial, launch_day=flight.launch_day,
            land_lat=flight.last_lat, land_lon=flight.last_lon,
            land_alt=flight.last_alt, landed_at=landed_at, detected_by="timeout",
        )

    # ---- health -----------------------------------------------
    def healthy(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.last_frame_at is None:
            return False
        return (now - self.last_frame_at).total_seconds() < self.cfg.health_stale_seconds

    def write_heartbeat(self) -> None:
        if self.last_frame_at is None:
            return
        path = Path(self.cfg.health_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.last_frame_at.isoformat())

    # ---- in-app GFS downloader --------------------------
    def start_gfs_downloader(self) -> bool:
        """Start the periodic GFS download as a daemon timer thread, so no
        separate container is needed. No-op unless GFS is enabled and configured
        to run in-process. Returns True if a thread was started."""
        if not (self.cfg.gfs.enabled and self.cfg.gfs.download_in_process):
            return False
        if self._gfs_thread is not None and self._gfs_thread.is_alive():
            return False
        self._gfs_stop.clear()
        self._gfs_thread = threading.Thread(
            target=self._gfs_loop, name="gfs-downloader", daemon=True)
        self._gfs_thread.start()
        log.info("started in-app GFS downloader (every %.1fh)",
                 self.cfg.gfs.download_cadence_hours)
        return True

    def stop_gfs_downloader(self) -> None:
        self._gfs_stop.set()
        # wake any loop sleeping on its kick event so it exits promptly
        self._gfs_kick.set()
        self._dem_kick.set()

    def start_hrrr_downloader(self) -> bool:
        """HRRR companion to the GFS timer thread: hourly cycles, own cadence.
        No-op unless HRRR is enabled and configured to run in-process."""
        if not (self.cfg.hrrr.enabled and self.cfg.hrrr.download_in_process):
            return False
        if self._hrrr_thread is not None and self._hrrr_thread.is_alive():
            return False
        self._hrrr_thread = threading.Thread(
            target=self._hrrr_loop, name="hrrr-downloader", daemon=True)
        self._hrrr_thread.start()
        log.info("started in-app HRRR downloader (every %.1fh)",
                 self.cfg.hrrr.download_cadence_hours)
        return True

    def start_dem_downloader(self) -> bool:
        """DEM companion to the GFS/HRRR timer threads: fetch the GLO-30 tiles
        covering the capture ROI, so terrain termination needs no manual
        prefetch. Terrain is static - once the ROI's tiles are on disk every
        pass is a free existence check; a subscriber edit that grows the ROI is
        picked up within a check interval. No-op unless DEM is enabled and
        configured to download in-process."""
        if not (self.cfg.dem.enabled and self.cfg.dem.download_in_process):
            return False
        if self._dem_thread is not None and self._dem_thread.is_alive():
            return False
        self._dem_thread = threading.Thread(
            target=self._dem_loop, name="dem-downloader", daemon=True)
        self._dem_thread.start()
        log.info("started in-app DEM tile downloader (check every %.0fs)",
                 self.cfg.dem.download_check_seconds)
        return True

    def _dem_loop(self) -> None:
        cadence = max(30.0, self.cfg.dem.download_check_seconds)
        while not self._gfs_stop.is_set():
            self._dem_kick.clear()
            try:
                if download_dem_tiles(self.cfg.dem, self._capture_box,
                                      skip=self._dem_absent):
                    # new tiles on disk: rebuild the ground model so the
                    # tracker/predictor see real terrain from the next lookup
                    self.ground_fn.reload()
            except Exception:  # noqa: BLE001 - never let the timer thread die
                log.exception("DEM tile download pass failed")
            self._dem_kick.wait(cadence)   # cadence, or an ROI-change kick

    def _hrrr_loop(self) -> None:
        cadence = max(600.0, self.cfg.hrrr.download_cadence_hours * 3600.0)
        while not self._gfs_stop.is_set():
            try:
                download_hrrr_cycle(self.cfg)
            except Exception:  # noqa: BLE001 - never let the timer thread die
                log.exception("HRRR download cycle failed")
            if self._gfs_stop.wait(cadence):
                break

    def _gfs_loop(self) -> None:
        cadence = max(600.0, self.cfg.gfs.download_cadence_hours * 3600.0)
        while not self._gfs_stop.is_set():
            self._gfs_kick.clear()
            try:
                download_gfs_cycle(self.cfg, self._capture_box)
            except Exception:  # noqa: BLE001 - never let the timer thread die
                log.exception("GFS download cycle failed")
            self._gfs_kick.wait(cadence)   # cadence, or an ROI-change kick

    # ---- in-app web dashboard ---------------------------
    def start_web_server(self) -> bool:
        """Serve the dashboard / onboarding UI in-process on a daemon thread, so
        no separate container is needed. It shares this App's (thread-safe) Store,
        so locations added in the browser are picked up by the next subscriber
        reload - no restart. No-op unless ``cfg.web.enabled``. Returns True if a
        thread was started."""
        if not self.cfg.web.enabled:
            return False
        if self._web_thread is not None and self._web_thread.is_alive():
            return False
        from .web import build_server

        self._web_server = build_server(
            self.cfg, self.store, host=self.cfg.web.host, port=self.cfg.web.port,
            config_path=self.config_path)
        self._web_thread = threading.Thread(
            target=self._web_server.run, name="web-ui", daemon=True)
        self._web_thread.start()
        log.info("started in-app web UI on http://%s:%d",
                 self.cfg.web.host, self.cfg.web.port)
        return True

    def stop_web_server(self) -> None:
        if self._web_server is not None:
            self._web_server.should_exit = True

    # ---- live run ---------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - needs network/broker
        """Single-threaded consumer loop: the MQTT thread only enqueues raw
        frames; this thread does all parsing/tracking/prediction + maintenance,
        so DEM lookups stay on one thread."""
        raw_q: queue.Queue = queue.Queue(maxsize=200_000)
        processor = TelemetryProcessor(self.cfg.ingest, on_frames=self.on_frames)

        dropped = {"n": 0, "t": 0.0}

        def enqueue(msg: dict) -> None:
            try:
                raw_q.put_nowait(msg)
            except queue.Full:
                # rate-limit: one line per window, not one per dropped frame
                dropped["n"] += 1
                now = time.monotonic()
                if now - dropped["t"] >= 10.0:
                    log.warning("ingest queue full; dropped %d frame(s) in the "
                                "last 10s - consumer falling behind", dropped["n"])
                    dropped["n"] = 0
                    dropped["t"] = now

        stream = SondeHubStream(enqueue, self.cfg.ingest)
        t = threading.Thread(target=stream.run_forever, name="sondehub", daemon=True)
        t.start()
        log.info("started SondeHub ingest")
        # always alive (idle when backfill_enabled is off), so the settings
        # editor can hot-toggle backfill without a thread-start path
        self._backfill.start()
        self.start_gfs_downloader()
        self.start_hrrr_downloader()
        self.start_dem_downloader()
        self.start_web_server()

        last_tick = 0.0
        last_predict = 0.0
        last_reload = time.monotonic()
        try:
            while True:
                try:
                    processor.handle_raw(raw_q.get(timeout=1.0))
                    # drain a backlog batch before re-checking the timers, so a
                    # burst of buffered frames clears at full speed
                    for _ in range(500):
                        processor.handle_raw(raw_q.get_nowait())
                except queue.Empty:
                    pass
                self.apply_backfills()
                now_mono = time.monotonic()
                if now_mono - last_tick >= self.cfg.tick_seconds:
                    # The silence sweeps compare wall-clock now to *sonde-time*
                    # last_seen; while a backlog drains, every tracked flight
                    # looks silent even though its frames are sitting in the
                    # queue - a descent got expired mid-air that way
                    # (26004618, 2026-07-13). Defer maintenance until caught up.
                    backlog = raw_q.qsize()
                    if backlog > _INGEST_BACKLOG_SHED:
                        log.warning("ingest backlog %d; deferring maintenance tick",
                                    backlog)
                    else:
                        self.tick()
                    self.write_heartbeat()
                    last_tick = now_mono
                if now_mono - last_predict >= self.cfg.predict.predict_active_seconds:
                    # Shed the optional pre-burst sweep while behind on ingest -
                    # informational map paths must never starve frame processing.
                    backlog = raw_q.qsize()
                    if backlog > _INGEST_BACKLOG_SHED:
                        log.warning("ingest backlog %d; skipping pre-burst sweep", backlog)
                    else:
                        self.predict_active()
                    last_predict = now_mono
                if now_mono - last_reload >= self.cfg.subscriber_reload_seconds:
                    self.reload_subscribers()
                    last_reload = now_mono
        except KeyboardInterrupt:
            log.info("shutting down")
            stream.stop()
            self._backfill.stop()
            self.stop_gfs_downloader()
            self.stop_web_server()
