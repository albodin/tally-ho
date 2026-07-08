"""Shared engine-test fixtures.

The synthetic-flight generator lives in :mod:`windfall.sim` (it is part of the
package's public accuracy story); re-exported here so tests keep importing it
from ``tests.conftest``. The persistence fakes implement the tracker's
``FlightStore`` protocol in memory - the SQLite implementation belongs to the
tally-ho app and is tested there.
"""

from __future__ import annotations

import pytest

from windfall.sim import (  # noqa: F401  (re-exported for tests)
    BASE_TIME,
    SimResult,
    simulate_flight,
    wind_at,
)


def fast_ensemble_cfg(cfg=None):
    """A Config with a tiny Monte Carlo ensemble, for wiring-level tests: the
    ensemble code path still runs, without the statistical cost.
    Accuracy/calibration tests keep production defaults."""
    from windfall.config import Config
    cfg = cfg or Config()
    cfg.ensemble.n_members = 6
    cfg.ensemble.n_members_preburst = 6
    cfg.ensemble.min_interval_seconds = 300.0
    return cfg


class FakeFlightStore:
    """In-memory implementation of :class:`windfall.tracker.FlightStore`.

    Constructor mirrors ``Store(":memory:")`` so tests read the same either
    way; ``get_flight`` is provided for assertions even though the tracker
    itself never calls it."""

    def __init__(self, _path: str | None = None):
        self.flights: dict[tuple, dict] = {}   # (serial, launch_day_iso) -> row
        self.profiles: dict[tuple, dict] = {}
        self.tracks: dict[tuple, list[dict]] = {}
        self.descent_samples: dict[tuple, list] = {}

    @staticmethod
    def _key(serial, launch_day) -> tuple:
        return (serial, str(launch_day))

    def upsert_flight(self, row: dict) -> None:
        self.flights[(row["serial"], row["launch_day"])] = dict(row)

    def get_flight(self, serial, launch_day) -> dict | None:
        return self.flights.get(self._key(serial, launch_day))

    def active_flights(self) -> list[dict]:
        return [dict(r) for r in self.flights.values() if r["state"] != "LANDED"]

    def save_profile(self, serial, launch_day, profile: dict) -> None:
        self.profiles[self._key(serial, launch_day)] = profile

    def load_profile(self, serial, launch_day) -> dict | None:
        return self.profiles.get(self._key(serial, launch_day))

    def save_descent_samples(self, serial, launch_day, samples: list) -> None:
        self.descent_samples[self._key(serial, launch_day)] = [list(s) for s in samples]

    def load_descent_samples(self, serial, launch_day) -> list | None:
        return self.descent_samples.get(self._key(serial, launch_day))

    def track_for(self, serial, launch_day) -> list[dict]:
        return list(self.tracks.get(self._key(serial, launch_day), []))

    def append_track_point(self, serial, launch_day, t, lat, lon, alt) -> None:
        self.tracks.setdefault(self._key(serial, launch_day), []).append(
            {"t": t, "lat": lat, "lon": lon, "alt": alt})

    def close(self) -> None:
        pass


@pytest.fixture
def flight() -> SimResult:
    return simulate_flight()


@pytest.fixture
def flight_stringified() -> SimResult:
    return simulate_flight(stringify=True)
