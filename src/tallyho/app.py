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
from .config import Config
from windfall.dem import ReloadableGround, download_dem_tiles
from .geofence import build_capture_roi, in_capture_roi
from windfall.gfs import download_gfs_cycle
from windfall.hrrr import download_hrrr_cycle, make_wind_source
from .ingest import SondeHubStream, TelemetryProcessor
from .models import Frame, FlightState
from .notify import AlertManager, HttpNtfySink, NtfySink
from windfall.predictor import GFSWindSource, Predictor
from .store import Store
from windfall.tracker import FlightTracker, TrackerEvent

log = logging.getLogger(__name__)


class App:
    def __init__(
        self,
        cfg: Config,
        store: Store | None = None,
        sink: NtfySink | None = None,
        gfs_source: GFSWindSource | None = None,
    ):
        self.cfg = cfg
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
            timeout=cfg.notify.request_timeout_seconds))
        self.last_frame_at: datetime | None = None
        self._subscribers = []
        self._capture_box = None
        # sonde-time of the last saved prediction per descending flight - the
        # per-frame throttle (see PredictConfig.descent_predict_seconds)
        self._last_pred_t: dict[tuple, float] = {}
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
        self.last_frame_at = frame.dt
        # Capture-ROI gate: process flights inside the wide capture box, or any
        # flight we are already tracking.
        already = self.tracker.get(frame.serial) is not None
        if not already and not in_capture_roi(self._capture_box, frame.lat, frame.lon):
            return

        flight, events = self.tracker.update(frame)

        if TrackerEvent.LANDED in events:
            self._record_landing(flight, now=flight.last_seen, detected_by="telemetry")
            self.alerts.handle_landed(flight, self._subscribers, now=flight.last_seen)
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
        """Periodic maintenance: out-of-ROI drop + landed-by-timeout sweep. The ROI sweep runs first so a flight that left the
        box (or was left behind by a subscriber edit) is dropped silently
        instead of alerting on a later timeout."""
        now = now or datetime.now(timezone.utc)
        self._drop_outside_roi()
        for flight, event in self.tracker.check_timeouts(now):
            if event == TrackerEvent.LANDED:
                self._record_landing(flight, now=flight.last_seen or now,
                                     detected_by="timeout")
                self.alerts.handle_landed(flight, self._subscribers, now=now)

    def _record_landing(self, flight, now: datetime | None, detected_by: str) -> None:
        """Persist the actual landing position as accuracy ground truth."""
        if flight.last_lat is None or flight.last_lon is None:
            return
        self.store.record_landing(
            serial=flight.serial, launch_day=flight.launch_day,
            land_lat=flight.last_lat, land_lon=flight.last_lon,
            land_alt=flight.last_alt, landed_at=now, detected_by=detected_by,
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
            self.cfg, self.store, host=self.cfg.web.host, port=self.cfg.web.port)
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
                now_mono = time.monotonic()
                if now_mono - last_tick >= self.cfg.tick_seconds:
                    self.tick()
                    self.write_heartbeat()
                    last_tick = now_mono
                if now_mono - last_predict >= self.cfg.predict.predict_active_seconds:
                    # Shed the optional pre-burst sweep while behind on ingest -
                    # informational map paths must never starve frame processing.
                    backlog = raw_q.qsize()
                    if backlog > 5_000:
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
            self.stop_gfs_downloader()
            self.stop_web_server()
