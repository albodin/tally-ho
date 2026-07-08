"""Tests for great-circle geometry."""

import math

import pytest

from windfall.geo import (
    BBox,
    bearing_deg,
    circles_bbox,
    compass_point,
    haversine_km,
    normalize_lon,
)


def test_normalize_lon():
    assert normalize_lon(0) == 0
    assert normalize_lon(190) == pytest.approx(-170)
    assert normalize_lon(-190) == pytest.approx(170)
    assert normalize_lon(360) == pytest.approx(0)


def test_haversine_known_distance():
    # Paris (48.8566, 2.3522) to London (51.5074, -0.1278) ~ 344 km
    d = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
    assert d == pytest.approx(344, abs=5)


def test_haversine_zero():
    assert haversine_km(45, 7, 45, 7) == pytest.approx(0, abs=1e-9)


def test_haversine_antimeridian():
    # 1 degree of lon either side of 180, near equator ~ 222 km, not ~40000
    d = haversine_km(0.0, 179.5, 0.0, -179.5)
    assert d == pytest.approx(111.3, abs=2)


def test_bearing_cardinals():
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0, abs=1)     # north
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(90, abs=1)    # east
    assert bearing_deg(1, 0, 0, 0) == pytest.approx(180, abs=1)   # south


def test_compass_point():
    assert compass_point(0) == "N"
    assert compass_point(90) == "E"
    assert compass_point(45) == "NE"
    assert compass_point(225) == "SW"


def test_bbox_contains_and_expand():
    box = BBox(44, 6, 46, 8)
    assert box.contains(45, 7)
    assert not box.contains(50, 7)
    grown = box.expanded_km(111.3)   # ~1 degree lat
    assert grown.min_lat == pytest.approx(43, abs=0.05)
    assert grown.max_lat == pytest.approx(47, abs=0.05)


def test_circles_bbox_covers_radius():
    box = circles_bbox([(45.0, 7.0, 111.3)])  # ~1 deg radius
    assert box is not None
    assert box.min_lat == pytest.approx(44, abs=0.05)
    assert box.max_lat == pytest.approx(46, abs=0.05)
    assert circles_bbox([]) is None
