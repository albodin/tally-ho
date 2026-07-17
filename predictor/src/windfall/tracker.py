"""Flight state machine + burst/float detection.

One :class:`Flight` per ``(serial, launch_day)``. Serials recur across days, so a
fresh ascent for a known serial starts a new flight. The tracker
builds the ascent wind/density profile, detects burst vs float, collects descent
samples for the ballistic fit, and emits transition events the app reacts to.

The tracker does **not** run the predictor - that is the app's job, keeping the
state machine independent of the prediction engine (and of the DEM/GFS).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Protocol

from .atmosphere import measured_density
from .config import Config
from .geo import haversine_km
from .kinematics import segment, windowed_vertical_rate
from .models import DescentSample, Frame, FlightState, launch_day_of
from .profile import FlightProfile, update_profile_from_pair

log = logging.getLogger(__name__)

GroundFn = Callable[[float, float], float]  # (lat, lon) -> ground elevation m

__all__ = ["DescentSample", "Flight", "FlightStore", "FlightTracker",
           "GroundFn", "TrackerEvent"]


class FlightStore(Protocol):
    """What the tracker needs from a persistence layer (duck-typed): the live
    app passes its SQLite store, tests pass an in-memory fake, and ``None``
    disables persistence entirely - the replay harness runs that way."""

    def upsert_flight(self, row: dict) -> None: ...
    def active_flights(self) -> list[dict]: ...
    def save_profile(self, serial: str, launch_day: date, profile: dict) -> None: ...
    def load_profile(self, serial: str, launch_day: date) -> dict | None: ...
    def track_for(self, serial: str, launch_day: date) -> list[dict]: ...
    def append_track_point(self, serial: str, launch_day: date, t: float,
                           lat: float, lon: float, alt: float) -> None: ...
    # Descent samples persist so a daemon restart mid-descent resumes the
    # ballistic fit instead of degrading to the single-point shortcut. Each
    # sample is ``[t, alt, v_obs, rho]``. Optional: the tracker probes with
    # getattr and degrades gracefully when a store predates these methods.
    def save_descent_samples(self, serial: str, launch_day: date,
                             samples: list) -> None: ...
    def load_descent_samples(self, serial: str, launch_day: date) -> list | None: ...


class TrackerEvent(str, enum.Enum):
    NEW_FLIGHT = "NEW_FLIGHT"
    BURST = "BURST"
    FLOAT = "FLOAT"
    LANDED = "LANDED"
    # Signal lost long ago mid-air: the flight is closed out (state LANDED) but
    # its last position is NOT a landing - no ground truth is recorded.
    EXPIRED = "EXPIRED"
    # A silence close-out contradicted by the sonde transmitting again: the
    # flight reopened (its prior state, or DESCENT for a timeout landing) and
    # any provisional landing record should be retracted by the app.
    RESUMED = "RESUMED"


@dataclass
class Flight:
    serial: str
    launch_day: date
    type: str | None = None
    state: FlightState = FlightState.ASCENT
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    launch_lat: float | None = None
    launch_lon: float | None = None
    max_alt: float = float("-inf")
    burst_alt: float | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    last_alt: float | None = None
    last_t: float | None = None
    last_vrate: float | None = None      # most recent derived vertical rate, m/s
    descent_b: float | None = None       # ballistic constant fitted at landing
    burst_t: float | None = None         # sonde time at burst detection (epoch s)
    landed_alt: float | None = None      # altitude when LANDED was declared (stable re-ascent anchor)
    # True when LANDED came from the silence sweep rather than telemetry
    # reaching the ground band: a *provisional* landing. Silence while low is
    # usually a landing, but sometimes just a reception gap - frames heard
    # later, well below the declared landing altitude, reopen the descent
    # (see _is_still_descending).
    landed_by_timeout: bool = False
    # State at a silence close-out (EXPIRED mid-air), None for real landings.
    # If the sonde transmits again the silence was reception loss, not the
    # flight ending - _resolve_flight resumes this state instead of letting
    # the LANDED husk swallow the rest of the flight (frames would update the
    # track but never predict, alert, or record the real landing).
    expired_state: FlightState | None = None
    # transient runtime state
    prev_frame: Frame | None = None
    first_alt: float | None = None       # altitude of the first frame seen this session
    ascended: bool = False               # observed to genuinely climb (gate for recording a burst)
    neg_rate_count: int = 0
    glitch_count: int = 0                # consecutive teleport-gated frames
    reascent_count: int = 0              # consecutive climbing frames since LANDED (serial-reuse gate)
    redescent_count: int = 0             # consecutive frames below a timeout landing (resume gate)
    float_since_t: float | None = None
    last_track: tuple[float, float, float, float] | None = None  # (t, lat, lon, alt) last point kept
    profile: FlightProfile = field(default_factory=FlightProfile)
    descent_samples: list[DescentSample] = field(default_factory=list)
    # (t, alt) sliding window for the regression-based vertical rate
    alt_window: list[tuple[float, float]] = field(default_factory=list)
    # recent positive ascent rates (windowed), for a robust pre-burst estimate
    ascent_rates: list[float] = field(default_factory=list)

    def robust_ascent_rate(self) -> float | None:
        """Median of recent windowed ascent rates - far steadier than the last
        single frame pair, which used to drive the pre-burst estimate."""
        if len(self.ascent_rates) < 5:
            return None
        rates = sorted(self.ascent_rates)
        return rates[len(rates) // 2]

    def to_row(self) -> dict:
        return {
            "serial": self.serial,
            "launch_day": self.launch_day.isoformat(),
            "type": self.type,
            "state": self.state.value,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "launch_lat": self.launch_lat,
            "launch_lon": self.launch_lon,
            "burst_alt": self.burst_alt,
            "burst_t": self.burst_t,
            "descent_b": self.descent_b,
            "max_alt": None if self.max_alt == float("-inf") else self.max_alt,
            "last_lat": self.last_lat,
            "last_lon": self.last_lon,
            "last_alt": self.last_alt,
        }

    @classmethod
    def from_row(cls, row: dict) -> "Flight":
        """Rebuild a flight from a persisted ``flights`` row (inverse of
        :meth:`to_row`) so the tracker can resume it after a restart. The wind
        profile, last-kept track point, and transient runtime fields are restored
        separately by :meth:`FlightTracker.load_active`."""
        def _dt(v):
            return datetime.fromisoformat(v) if v else None
        return cls(
            serial=row["serial"],
            launch_day=date.fromisoformat(row["launch_day"]),
            type=row["type"],
            state=FlightState(row["state"]),
            first_seen=_dt(row["first_seen"]),
            last_seen=_dt(row["last_seen"]),
            launch_lat=row["launch_lat"],
            launch_lon=row["launch_lon"],
            burst_alt=row["burst_alt"],
            burst_t=row.get("burst_t"),
            descent_b=row.get("descent_b"),
            max_alt=float("-inf") if row["max_alt"] is None else row["max_alt"],
            last_lat=row["last_lat"],
            last_lon=row["last_lon"],
            last_alt=row["last_alt"],
        )


class FlightTracker:
    def __init__(
        self,
        cfg: Config,
        store: FlightStore | None = None,
        ground_fn: GroundFn | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.ground_fn = ground_fn or (lambda lat, lon: 0.0)
        self.flights: dict[tuple[str, date], Flight] = {}
        self._active_key: dict[str, tuple[str, date]] = {}

    def load_active(self) -> int:
        """Rehydrate in-flight (non-LANDED) flights from the store on startup so a
        restart *resumes* them instead of treating the next frame as a brand-new
        launch. Without this, a mid-air frame would reset ``first_seen`` and
        clobber the real ``launch_lat/launch_lon`` (and lose state/profile).
        Returns the number of flights restored."""
        if self.store is None:
            return 0
        rows = self.store.active_flights()
        # Assign _active_key from oldest to newest so that if a serial somehow has
        # two active flights, the most recent one wins as the active key.
        rows.sort(key=lambda r: (r["last_seen"] or r["first_seen"] or ""))
        for row in rows:
            flight = Flight.from_row(row)
            # A persisted in-flight sonde has real tracking history: if it's still
            # ASCENT/FLOAT it genuinely climbed (first-heard-descending catches
            # reach DESCENT within minutes), so re-open the burst gate. first_alt
            # isn't persisted, so without this a burst right after a restart -
            # measured against the resume altitude - could be missed.
            flight.ascended = flight.state in (FlightState.ASCENT, FlightState.FLOAT)
            key = (flight.serial, flight.launch_day)
            saved = self.store.load_profile(flight.serial, flight.launch_day)
            if saved is not None:
                flight.profile = FlightProfile.from_json(saved)
            else:
                flight.profile = FlightProfile(bin_size_m=self.cfg.profile.bin_size_m, gap_fill_m=self.cfg.profile.interior_gap_fill_m)
            track = self.store.track_for(flight.serial, flight.launch_day)
            if track:
                last = track[-1]
                flight.last_track = (last["t"], last["lat"], last["lon"], last["alt"])
                flight.last_t = last["t"]
            load_samples = getattr(self.store, "load_descent_samples", None)
            if load_samples is not None and flight.state == FlightState.DESCENT:
                saved_samples = load_samples(flight.serial, flight.launch_day)
                if saved_samples:
                    flight.descent_samples = [
                        DescentSample(t=s[0], alt=s[1], v_obs=s[2], rho=s[3])
                        for s in saved_samples
                    ]
            self.flights[key] = flight
            self._active_key[flight.serial] = key
        if rows:
            log.info("rehydrated %d active flight(s) from store", len(rows))
        return len(rows)

    # ---- main entry -------------------------------------------------------
    def update(self, frame: Frame) -> tuple[Flight, list[TrackerEvent]]:
        # Teleport gate BEFORE flight resolution: a glitch frame must neither
        # poison the active flight's state (max_alt → burst detection/burst_alt)
        # nor fake the altitude jump that spawns a "new ascent" for a LANDED
        # serial. Checked against the last *accepted* fix.
        key = self._active_key.get(frame.serial)
        if key is not None and self._is_glitch(self.flights[key], frame):
            return self.flights[key], []
        flight, events = self._resolve_flight(frame)
        self._apply_frame(flight, frame, events)
        if self.store is not None:
            self.store.upsert_flight(flight.to_row())
        return flight, events

    def get(self, serial: str) -> Flight | None:
        key = self._active_key.get(serial)
        return self.flights.get(key) if key else None

    def check_timeouts(self, now: datetime,
                       hold: "set[str] | None" = None) -> list[tuple[Flight, TrackerEvent]]:
        """Mark flights LANDED when telemetry stops while low, expire flights
        whose telemetry stopped long ago at any altitude, and evict long-silent
        LANDED flights from memory. Serials in ``hold`` are left alone this
        sweep - the caller is resolving their real fate out of band (a history
        fetch is in flight) and a timeout close-out now would preempt it."""
        out = []
        tcfg = self.cfg.tracker
        for key, flight in list(self.flights.items()):
            if flight.last_seen is None:
                continue
            gap = (now - flight.last_seen).total_seconds()
            if flight.state == FlightState.LANDED:
                # keep finished flights around briefly (late ground frames,
                # serial-reuse detection), then free the memory
                if gap > tcfg.stale_flight_seconds:
                    del self.flights[key]
                    if self._active_key.get(flight.serial) == key:
                        del self._active_key[flight.serial]
                continue
            if hold and flight.serial in hold:
                continue
            if flight.last_alt is None:
                continue
            if gap > tcfg.stale_flight_seconds:
                # Drifted out of receiver range mid-air and never came back:
                # close it out so it stops showing as an active flight, but do
                # NOT record its last mid-air position as a landing.
                log.info("expiring stale flight %s (silent %.0f s at %.0f m)",
                         flight.serial, gap, flight.last_alt)
                flight.expired_state = flight.state
                flight.state = FlightState.LANDED
                self._persist(flight, profile=True)
                out.append((flight, TrackerEvent.EXPIRED))
                continue
            if (flight.state == FlightState.ASCENT
                    and gap > tcfg.ascent_lost_timeout_seconds):
                # Silence only ever grows the radio horizon on the way up, so a
                # long-lost ascending sonde burst and landed unheard (daemon
                # downtime, dead sonde). Close it out - EXPIRED, not a landing:
                # the last fix is mid-ascent, nowhere near where it came down.
                log.info("expiring lost ascent flight %s (silent %.0f s at "
                         "%.0f m)", flight.serial, gap, flight.last_alt)
                flight.expired_state = flight.state
                flight.state = FlightState.LANDED
                self._persist(flight, profile=True)
                out.append((flight, TrackerEvent.EXPIRED))
                continue
            if gap <= tcfg.landed_timeout_seconds or flight.state != FlightState.DESCENT:
                continue
            ground = self.ground_fn(flight.last_lat, flight.last_lon)
            low = flight.last_alt <= ground + max(tcfg.landed_alt_above_ground_m, 2000.0)
            if low:
                flight.state = FlightState.LANDED
                flight.landed_alt = flight.last_alt
                # Provisional: the silence is usually touchdown cutting the
                # link, but can be a mere reception gap - frames heard later,
                # clearly below this altitude, reopen the descent.
                flight.landed_by_timeout = True
                self._finalize_descent_b(flight)
                self._persist(flight, profile=True)
                out.append((flight, TrackerEvent.LANDED))
            elif gap > tcfg.descent_lost_timeout_seconds:
                # Dropped below the radio horizon while still above the low
                # band (hills): it is on the ground by now, but the last fix
                # was high - close it out without a landing-truth record.
                log.info("descent flight %s lost above ground band (%.0f m, "
                         "silent %.0f s); closing out", flight.serial,
                         flight.last_alt, gap)
                flight.expired_state = flight.state
                flight.state = FlightState.LANDED
                # no landing truth, but the chute fit is still good climatology
                self._finalize_descent_b(flight)
                self._persist(flight, profile=True)
                out.append((flight, TrackerEvent.EXPIRED))
        return out

    def forget(self, serial: str) -> Flight | None:
        """Remove a serial's active flight from memory without persisting
        anything - unlike :meth:`drop`, which closes the flight out as LANDED.
        For callers about to *rebuild* the flight (e.g. replaying fetched
        history through :meth:`update`): the provisional in-memory state simply
        vanishes, and any rows already persisted are the caller's to clean up.
        Returns the removed flight, or None if the serial wasn't tracked."""
        key = self._active_key.pop(serial, None)
        if key is None:
            return None
        return self.flights.pop(key, None)

    def drop(self, flight: Flight) -> None:
        """Close out a flight immediately and evict it from memory - e.g. it
        left the caller's region of interest (the tracker itself has no
        geography). State goes to LANDED so it stops listing as active, but its
        last position is NOT a landing truth. If the serial transmits again the
        caller's ingest gate decides whether it gets re-tracked (as a new
        flight, since this one is gone)."""
        key = (flight.serial, flight.launch_day)
        if flight.state == FlightState.DESCENT:
            # no landing truth, but the chute fit is still good climatology
            self._finalize_descent_b(flight)
        flight.state = FlightState.LANDED
        self._persist(flight, profile=True)
        self.flights.pop(key, None)
        if self._active_key.get(flight.serial) == key:
            del self._active_key[flight.serial]

    # ---- internals --------------------------------------------------------
    def _is_glitch(self, flight: Flight, frame: Frame) -> bool:
        """Whether ``frame`` is a teleport relative to the flight's last
        accepted fix - an implied speed no real sonde reaches. After
        ``glitch_accept_after`` consecutive rejections the position is accepted
        as real: the sonde is consistently *there*, so the old fix (or our
        thresholds) was the wrong party."""
        tcfg = self.cfg.tracker
        if flight.last_lat is None or flight.last_lon is None or flight.last_t is None:
            return False
        dt = frame.t - flight.last_t
        if dt <= 0.0:
            return False    # duplicate/out-of-order; downstream already ignores it
        h_mps = haversine_km(flight.last_lat, flight.last_lon,
                             frame.lat, frame.lon) * 1000.0 / dt
        v_mps = (abs(frame.alt - flight.last_alt) / dt
                 if flight.last_alt is not None else 0.0)
        if h_mps <= tcfg.glitch_horizontal_mps and v_mps <= tcfg.glitch_vertical_mps:
            flight.glitch_count = 0
            return False
        flight.glitch_count += 1
        if flight.glitch_count > tcfg.glitch_accept_after:
            log.warning("%s: %d consecutive teleport frames - accepting the new "
                        "position as real", flight.serial, flight.glitch_count)
            flight.glitch_count = 0
            return False
        log.info("%s: teleport frame rejected (%.0f m/s horizontal, %.0f m/s "
                 "vertical over %.1f s)", flight.serial, h_mps, v_mps, dt)
        return True

    def _resolve_flight(self, frame: Frame) -> tuple[Flight, list[TrackerEvent]]:
        key = self._active_key.get(frame.serial)
        if key is not None:
            flight = self.flights[key]
            if flight.state == FlightState.LANDED and flight.expired_state is not None:
                # Closed out by a silence timeout mid-air, but the sonde is
                # transmitting again: the silence was reception loss, not the
                # flight ending. Resume the state it was expired from - the
                # ordinary transition machinery re-sorts it (a reopened ASCENT
                # that is actually falling calls its burst within a few
                # frames). Without this the LANDED husk swallowed the rest of
                # the flight: frames kept updating the track but nothing
                # predicted, alerted, or recorded the real landing
                # (26004618, 2026-07-13, expired mid-descent at 22 km).
                log.info("%s transmitting again %.0f s after its timeout "
                         "close-out; resuming %s", flight.serial,
                         frame.t - (flight.last_t or frame.t),
                         flight.expired_state.value)
                flight.state = flight.expired_state
                flight.expired_state = None
                flight.reascent_count = 0
                return flight, [TrackerEvent.RESUMED]
            if (flight.state == FlightState.LANDED and flight.landed_by_timeout
                    and self._is_still_descending(flight, frame)):
                # A timeout landing contradicted by live frames well below the
                # declared landing altitude: the silence was a reception gap
                # and the sonde is still falling. Reopen the descent so it
                # predicts again and the *real* landing supersedes the
                # provisional record (which the app retracts on this event).
                log.info("%s heard %.0f m below its timeout landing at %.0f m;"
                         " resuming DESCENT", flight.serial,
                         (flight.landed_alt or 0.0) - frame.alt,
                         flight.landed_alt or 0.0)
                flight.state = FlightState.DESCENT
                flight.landed_by_timeout = False
                flight.landed_alt = None
                flight.redescent_count = 0
                flight.reascent_count = 0
                # the close-out fit was premature - refit at the real landing
                # with the full descent's samples
                flight.descent_b = None
                return flight, [TrackerEvent.RESUMED]
            if flight.state != FlightState.LANDED or not self._is_new_ascent(flight, frame):
                return flight, []
        # new flight (first sighting, or a fresh ascent for a reused serial)
        return self._new_flight(frame), [TrackerEvent.NEW_FLIGHT]

    def _is_new_ascent(self, landed: Flight, frame: Frame) -> bool:
        """A LANDED flight's serial reappears: is this a genuinely new launch, or
        just the landed sonde still pinging from where it came down?

        A radiosonde keeps transmitting for hours after landing; its ground GPS
        fixes are noisy (spikes of 100s of m) and they cross the UTC day boundary
        while the sonde never moved. Neither a single high fix nor a new calendar
        day is a relaunch - treating them as one spawned ghost flights that sat on
        the ground stuck in ASCENT. Only a *sustained* climb well above the
        landing altitude counts: a real relaunch keeps going up.

        Genuine cross-day serial reuse (a different sonde, the same printed serial
        weeks later) is handled elsewhere - the old flight is long gone from memory
        by then (``stale_flight_seconds``), so its next frame takes the ordinary
        new-flight path rather than this resurrection check."""
        if landed.first_seen is None:
            return True
        tcfg = self.cfg.tracker
        # Anchor to the altitude at landing, not the rolling last fix: successive
        # noisy ground pings must not let the bar creep upward frame by frame.
        anchor = landed.landed_alt if landed.landed_alt is not None else landed.last_alt
        if anchor is None or frame.alt <= anchor + tcfg.new_ascent_climb_m:
            landed.reascent_count = 0
            return False
        landed.reascent_count += 1
        return landed.reascent_count >= tcfg.new_ascent_consecutive

    def _is_still_descending(self, landed: Flight, frame: Frame) -> bool:
        """A timeout-LANDED flight's serial is transmitting again: is the sonde
        still falling (the silence was a reception gap, not touchdown), or is
        this the landed sonde pinging from the ground?

        Mirror image of :meth:`_is_new_ascent`: anchored to the altitude at
        the landing declaration, and requiring a *sustained* run of fixes well
        below it - ground GPS fixes are noisy (spikes of 100s of m), so a
        single low fix must not reopen a real landing."""
        tcfg = self.cfg.tracker
        anchor = landed.landed_alt if landed.landed_alt is not None else landed.last_alt
        if anchor is None or frame.alt >= anchor - tcfg.redescent_drop_m:
            landed.redescent_count = 0
            return False
        landed.redescent_count += 1
        return landed.redescent_count >= tcfg.redescent_consecutive

    def _new_flight(self, frame: Frame) -> Flight:
        lday = launch_day_of(frame.dt)
        flight = Flight(serial=frame.serial, launch_day=lday, type=frame.type)
        flight.profile = FlightProfile(bin_size_m=self.cfg.profile.bin_size_m, gap_fill_m=self.cfg.profile.interior_gap_fill_m)
        self.flights[(frame.serial, lday)] = flight
        self._active_key[frame.serial] = (frame.serial, lday)
        return flight

    def _apply_frame(self, flight: Flight, frame: Frame, events: list[TrackerEvent]) -> None:
        if flight.first_seen is None:
            flight.first_seen = frame.dt
            flight.first_alt = frame.alt
            # Only claim a launch site when we actually heard the sonde near
            # the ground. First-heard-high means it launched out of range -
            # the launch site is unknown, not the mid-air first-fix position.
            ground = self.ground_fn(frame.lat, frame.lon)
            if frame.alt <= ground + self.cfg.tracker.launch_max_agl_m:
                flight.launch_lat = frame.lat
                flight.launch_lon = frame.lon

        vrate = self._vertical_rate(flight, frame)

        flight.max_alt = max(flight.max_alt, frame.alt)
        # A burst may only be recorded once we've seen the sonde truly climb. A
        # sonde first heard already falling never trips this, so its inevitable
        # "drop below max" is tracked as descent without a bogus burst altitude.
        if (not flight.ascended and flight.first_alt is not None
                and flight.max_alt - flight.first_alt >= self.cfg.tracker.ascent_min_gain_m):
            flight.ascended = True

        # Collect ascent-rate history for the robust pre-burst estimate.
        if flight.state == FlightState.ASCENT and vrate is not None and vrate > 0:
            flight.ascent_rates.append(vrate)
            if len(flight.ascent_rates) > 600:
                del flight.ascent_rates[:300]

        # Build the ascent/float wind+density profile. During
        # DESCENT the falling payload is itself a wind measurement - newer and
        # closer to the landing zone than the ascent column - so (when enabled)
        # it keeps refreshing the bins it re-crosses, with the stale contents
        # down-weighted (live refresh).
        pcfg = self.cfg.profile
        descending = flight.state == FlightState.DESCENT
        if flight.prev_frame is not None and (
                flight.state in (FlightState.ASCENT, FlightState.FLOAT)
                or (descending and pcfg.descent_refresh_enabled)):
            added = update_profile_from_pair(
                flight.profile, flight.prev_frame, frame, pcfg,
                weight_cap=pcfg.descent_refresh_n_cap if descending else None)
            if added and flight.profile.n_bins % 25 == 0:
                self._persist(flight, profile=True)

        # --- transitions ---
        if flight.state in (FlightState.ASCENT, FlightState.FLOAT):
            self._maybe_burst(flight, frame, vrate, events)
        if flight.state == FlightState.ASCENT:
            self._maybe_float(flight, frame, vrate, events)

        if flight.state == FlightState.DESCENT:
            if self._maybe_revert_burst(flight, frame):
                pass  # was a transient dip, not a burst - now ASCENT again
            elif vrate is not None and vrate < 0:
                self._collect_descent(flight, frame, vrate)
                self._maybe_landed(flight, frame, events)

        flight.last_lat, flight.last_lon, flight.last_alt = frame.lat, frame.lon, frame.alt
        flight.last_t = frame.t
        flight.last_seen = frame.dt
        if vrate is not None:
            flight.last_vrate = vrate
        flight.prev_frame = frame

        self._maybe_track(flight, frame)

    def _vertical_rate(self, flight: Flight, frame: Frame) -> float | None:
        """Regression-based vertical rate over a short sliding window, falling
        back to the single frame pair while the window is still thin."""
        tcfg = self.cfg.tracker
        flight.alt_window.append((frame.t, frame.alt))
        cutoff = frame.t - tcfg.vrate_window_seconds
        while flight.alt_window and flight.alt_window[0][0] < cutoff:
            flight.alt_window.pop(0)
        vrate = windowed_vertical_rate(
            flight.alt_window, tcfg.vrate_min_points, tcfg.vrate_min_span_seconds)
        if vrate is None and flight.prev_frame is not None:
            seg = segment(flight.prev_frame, frame)
            if seg is not None:
                vrate = seg.vertical_rate
        return vrate

    def _maybe_track(self, flight: Flight, frame: Frame) -> None:
        """Append this frame to the flown-track breadcrumb if it has moved enough
        since the last kept point. The first frame of a flight is its
        launch point and is always kept. Downsampling keeps the stored trail small
        without losing the shape of the path."""
        if self.store is None:
            return
        prev = flight.last_track
        if prev is not None:
            _, plat, plon, palt = prev
            tcfg = self.cfg.tracker
            moved_m = haversine_km(plat, plon, frame.lat, frame.lon) * 1000.0
            d_alt = abs(frame.alt - palt)
            d_t = frame.t - prev[0]
            if (moved_m < tcfg.track_min_move_m and d_alt < tcfg.track_min_alt_m
                    and d_t < tcfg.track_min_interval_s):
                return
        self.store.append_track_point(
            flight.serial, flight.launch_day, frame.t, frame.lat, frame.lon, frame.alt)
        flight.last_track = (frame.t, frame.lat, frame.lon, frame.alt)

    def _maybe_burst(self, flight: Flight, frame: Frame, vrate, events) -> None:
        tcfg = self.cfg.tracker
        dropped = frame.alt < flight.max_alt - tcfg.burst_drop_m
        if dropped and vrate is not None and vrate < 0:
            flight.neg_rate_count += 1
        else:
            flight.neg_rate_count = 0
        if flight.neg_rate_count < tcfg.burst_consecutive:
            return
        # Never meaningfully airborne? A sonde cold-starting on the launch pad
        # can emit a sustained fake "descent" (GPS altitude settling downward),
        # and a flight that never left the ground must not enter DESCENT - it
        # would immediately "land" at the pad and mint a bogus landing-truth
        # row and LANDED alert. Genuine flights, including those first heard
        # already falling, have their apogee far above the ground.
        ground = self.ground_fn(frame.lat, frame.lon)
        if flight.max_alt < ground + tcfg.min_airborne_agl_m:
            return
        # A sustained drop below the running max: the sonde is descending, so
        # start tracking it as such (this is what drives the landing prediction).
        flight.state = FlightState.DESCENT
        flight.float_since_t = None
        # restart the vrate window so descent rates aren't diluted by the
        # tail of ascent points still inside it
        del flight.alt_window[:-1]
        if flight.ascended:
            # We watched it climb to this apogee, so this max IS the burst
            # altitude - record it and let it feed the per-site burst prior.
            flight.burst_alt = flight.max_alt
            flight.burst_t = frame.t
        # else: first heard already falling - genuinely descending, but we never
        # observed its burst. Leave burst_alt None so it cannot poison the prior.
        events.append(TrackerEvent.BURST)
        self._persist(flight, profile=True)

    def _maybe_revert_burst(self, flight: Flight, frame: Frame) -> bool:
        """Undo a called burst when the sonde climbs back above its supposed
        apogee. A strong downdraft - or a GPS spike that poisoned ``max_alt`` -
        can briefly fake the ``burst_drop_m`` drop while the balloon is still
        rising; a real burst never re-ascends. Reverting to ASCENT clears the
        false burst so the *real* burst higher up is the one recorded, and stops
        the spurious descent predictions the false burst was emitting.

        Only meaningful once a burst altitude was recorded (``ascended``); a
        descent caught mid-fall has ``burst_alt = None`` and nothing to revert."""
        if flight.burst_alt is None:
            return False
        if frame.alt <= flight.burst_alt + self.cfg.tracker.burst_revert_climb_m:
            return False
        log.info("%s: re-ascended %.0f m above called burst %.0f m - reverting "
                 "to ASCENT (transient dip, not a burst)", flight.serial,
                 frame.alt - flight.burst_alt, flight.burst_alt)
        flight.state = FlightState.ASCENT
        flight.burst_alt = None
        flight.burst_t = None
        flight.neg_rate_count = 0
        flight.float_since_t = None
        flight.descent_samples.clear()
        self._persist(flight, profile=True)
        return True

    def _maybe_float(self, flight: Flight, frame: Frame, vrate, events) -> None:
        tcfg = self.cfg.tracker
        if frame.alt < tcfg.float_min_alt_m or vrate is None:
            flight.float_since_t = None
            return
        if abs(vrate) < tcfg.float_rate_abs_mps:
            if flight.float_since_t is None:
                flight.float_since_t = frame.t
            elif frame.t - flight.float_since_t >= tcfg.float_window_seconds:
                flight.state = FlightState.FLOAT
                events.append(TrackerEvent.FLOAT)
        else:
            flight.float_since_t = None

    def _collect_descent(self, flight: Flight, frame: Frame, vrate) -> None:
        rho = None
        if frame.pressure is not None and frame.temp is not None:
            rho = measured_density(frame.pressure, frame.temp)
        if rho is None:
            rho = flight.profile.density(frame.alt)
        flight.descent_samples.append(
            DescentSample(t=frame.t, alt=frame.alt, v_obs=-vrate, rho=rho)
        )
        # checkpoint every ~30 samples (~30 s at 1 Hz) so a restart mid-descent
        # resumes the ballistic fit instead of starting over from one point
        if len(flight.descent_samples) % 30 == 0:
            self._save_descent_samples(flight)

    def _maybe_landed(self, flight: Flight, frame: Frame, events) -> None:
        ground = self.ground_fn(frame.lat, frame.lon)
        if frame.alt <= ground + self.cfg.tracker.landed_alt_above_ground_m:
            flight.state = FlightState.LANDED
            flight.landed_alt = frame.alt
            self._finalize_descent_b(flight)
            events.append(TrackerEvent.LANDED)

    def _finalize_descent_b(self, flight: Flight) -> None:
        """Fit and store the flight's ballistic constant once it is finished, so
        future pre-burst predictions can use a learned per-type prior instead of
        the static default. Clamped fits are garbage in →
        garbage out, so only clean fits are kept."""
        if flight.descent_b is not None or not flight.descent_samples:
            return
        from .descent import fit_descent, shortcut_descent  # local: avoids a cycle

        self._save_descent_samples(flight)
        model = fit_descent(flight.descent_samples, self.cfg.descent,
                            burst_t=flight.burst_t, burst_alt=flight.burst_alt)
        if model is None:
            last = flight.descent_samples[-1]
            model = shortcut_descent(last.v_obs, last.rho, self.cfg.descent)
        if model is not None and not model.clamped:
            flight.descent_b = model.b
            self._persist(flight, profile=True)

    def _save_descent_samples(self, flight: Flight) -> None:
        if self.store is None or not flight.descent_samples:
            return
        save = getattr(self.store, "save_descent_samples", None)
        if save is None:    # store predates descent-sample persistence
            return
        save(flight.serial, flight.launch_day,
             [[s.t, s.alt, s.v_obs, s.rho] for s in flight.descent_samples])

    def _persist(self, flight: Flight, profile: bool = False) -> None:
        if self.store is None:
            return
        self.store.upsert_flight(flight.to_row())
        if profile and not flight.profile.is_empty():
            self.store.save_profile(flight.serial, flight.launch_day, flight.profile.to_json())
