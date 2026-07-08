"""Shared app-test fixtures.

Synthetic flights come from the engine package (:mod:`windfall.sim`) - the
same generator the engine's own accuracy tests use, so the service tests
exercise identical physics.
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
    """A Config with a tiny Monte Carlo ensemble, for wiring-level tests
    (alerting, CLI, DEM hookup): the ensemble code path still runs, without the
    statistical cost. Accuracy/calibration tests keep production defaults."""
    from tallyho.config import Config
    cfg = cfg or Config()
    cfg.ensemble.n_members = 6
    cfg.ensemble.n_members_preburst = 6
    cfg.ensemble.min_interval_seconds = 300.0
    return cfg


@pytest.fixture
def flight() -> SimResult:
    return simulate_flight()


@pytest.fixture
def flight_stringified() -> SimResult:
    return simulate_flight(stringify=True)
