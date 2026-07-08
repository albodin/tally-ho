"""Tests for two-tier ROI + geofence."""

import pytest

from tallyho.geofence import build_capture_roi, geofence_matches, in_capture_roi
from tallyho.models import Subscriber


def _sub(name, lat, lon, r, active=True):
    return Subscriber(id=hash(name) & 0xFFFF, name=name, lat=lat, lon=lon, radius_km=r,
                      ntfy_server="https://ntfy.sh", ntfy_topic=name, active=active)


def test_capture_roi_includes_margin():
    subs = [_sub("a", 45.0, 7.0, 30)]
    box = build_capture_roi(subs, margin_km=300)
    # capture box should extend well beyond the 30 km alert circle
    assert box.max_lat - 45.0 > 2.5    # ~300+ km north
    assert in_capture_roi(box, 47.0, 7.0)       # 200+ km away, still captured
    assert not in_capture_roi(box, 50.0, 7.0)   # ~550 km north, outside


def test_capture_roi_none_without_subs():
    assert build_capture_roi([], 300) is None
    assert in_capture_roi(None, 45, 7) is False


def test_geofence_match_within_radius():
    subs = [_sub("a", 45.0, 7.0, 30), _sub("b", 46.0, 8.0, 10)]
    # landing 5 km from a (north), far from b
    matches = geofence_matches(45.045, 7.0, subs)
    assert len(matches) == 1
    assert matches[0].subscriber.name == "a"
    assert matches[0].distance_km == pytest.approx(5.0, abs=0.5)
    assert matches[0].compass == "N"


def test_geofence_no_match_outside_radius():
    subs = [_sub("a", 45.0, 7.0, 5)]
    assert geofence_matches(45.5, 7.0, subs) == []   # ~55 km away


def test_inactive_subscriber_skipped():
    subs = [_sub("a", 45.0, 7.0, 30, active=False)]
    assert geofence_matches(45.0, 7.0, subs) == []
    assert build_capture_roi(subs, 300) is None
