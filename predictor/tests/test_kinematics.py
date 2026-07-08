"""Tests for derived kinematics."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from windfall.kinematics import segment
from windfall.models import EARTH_RADIUS_M, Frame

BASE = datetime(2026, 6, 7, tzinfo=timezone.utc)


def mk(lat, lon, alt, secs):
    dt = BASE + timedelta(seconds=secs)
    return Frame("S", lat, lon, alt, dt.timestamp(), dt)


def test_vertical_rate():
    a = mk(45, 7, 1000, 0)
    b = mk(45, 7, 1010, 2)
    s = segment(a, b)
    assert s.vertical_rate == pytest.approx(5.0)


def test_wind_north_component():
    # move 0.001 deg north in 1 s
    a = mk(45.0, 7.0, 1000, 0)
    b = mk(45.001, 7.0, 1000, 1)
    s = segment(a, b)
    expected_v = EARTH_RADIUS_M * math.radians(0.001) / 1.0
    assert s.wind_v == pytest.approx(expected_v, rel=1e-6)
    assert s.wind_u == pytest.approx(0, abs=1e-6)


def test_wind_east_component_uses_cos_lat():
    a = mk(60.0, 7.0, 1000, 0)
    b = mk(60.0, 7.001, 1000, 1)
    s = segment(a, b)
    expected_u = EARTH_RADIUS_M * math.cos(math.radians(60)) * math.radians(0.001)
    assert s.wind_u == pytest.approx(expected_u, rel=1e-6)


def test_nonpositive_dt_returns_none():
    a = mk(45, 7, 1000, 5)
    b = mk(45, 7, 1010, 5)
    assert segment(a, b) is None


def test_windowed_vertical_rate_recovers_slope():
    import random
    from windfall.kinematics import windowed_vertical_rate
    rng = random.Random(7)
    truth = -6.0
    # 25 s of 1 Hz points with +-3 m of GPS altitude noise
    pts = [(float(t), 5000.0 + truth * t + rng.uniform(-3, 3)) for t in range(25)]
    rate = windowed_vertical_rate(pts)
    assert rate == pytest.approx(truth, abs=0.3)
    # a single adjacent pair can be wildly off - that is the point
    pair_rate = pts[1][1] - pts[0][1]
    assert abs(pair_rate - truth) > abs(rate - truth)


def test_windowed_vertical_rate_needs_enough_window():
    from windfall.kinematics import windowed_vertical_rate
    assert windowed_vertical_rate([(0.0, 100.0), (1.0, 95.0)]) is None      # too few
    assert windowed_vertical_rate(
        [(0.0, 100.0), (1.0, 95.0), (2.0, 91.0)], min_span_seconds=4.0) is None
    assert windowed_vertical_rate(
        [(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)]) is None                 # zero span
