"""Tests for tolerant telemetry parsing."""

from datetime import timezone

import pytest

from windfall.telemetry import (
    SENTINEL_THRESHOLD,
    FrameError,
    coerce_float,
    coerce_int,
    parse_datetime,
    parse_frame,
    try_parse_frame,
)


def _good_msg(**over):
    msg = {
        "serial": "S123",
        "lat": 44.20318,
        "lon": 7.5,
        "alt": 22010,
        "datetime": "2026-06-07T12:00:00.123456Z",
        "frame": 42,
        "type": "RS41",
    }
    msg.update(over)
    return msg


@pytest.mark.parametrize(
    "value,expected",
    [
        ("44.2", 44.2),
        (44.2, 44.2),
        ("22010", 22010.0),
        (22010, 22010.0),
        (" 1.5 ", 1.5),
        (None, None),
        ("", None),
        ("not-a-number", None),
        (-9999, None),
        ("-9999.0", None),
        (-9999.0, None),
        (-10000, None),
        (True, None),    # bool must not be treated as 1
        (False, None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_coerce_float(value, expected):
    got = coerce_float(value)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_sentinel_threshold_is_safe_for_real_negatives():
    # Real cold temps, below-sea-level alt, negative velocities must survive.
    assert coerce_float(-90.0) == -90.0     # very cold temp (C)
    assert coerce_float(-420) == -420.0     # Dead Sea-ish altitude
    assert coerce_float("-12.5") == -12.5   # descent vel
    assert SENTINEL_THRESHOLD < -9000


def test_coerce_int():
    assert coerce_int("42") == 42
    assert coerce_int(42.0) == 42
    assert coerce_int("-9999") is None
    assert coerce_int(None) is None


def test_parse_datetime_z_and_offset():
    dt = parse_datetime("2026-06-07T12:00:00.123456Z")
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2026 and dt.hour == 12
    dt2 = parse_datetime("2026-06-07T14:00:00+02:00")
    assert dt2.hour == 12  # normalised to UTC


def test_parse_datetime_bad():
    assert parse_datetime("nonsense") is None
    assert parse_datetime("") is None
    assert parse_datetime(None) is None


def test_parse_frame_happy_numbers():
    f = parse_frame(_good_msg())
    assert f.serial == "S123"
    assert f.lat == pytest.approx(44.20318)
    assert f.alt == pytest.approx(22010.0)
    assert f.frame == 42
    assert f.t == pytest.approx(f.dt.timestamp())


def test_parse_frame_happy_strings():
    # Wire reality: everything stringified.
    msg = _good_msg(lat="44.20318", lon="7.5", alt="22010", frame="42")
    f = parse_frame(msg)
    assert f.lat == pytest.approx(44.20318)
    assert f.alt == pytest.approx(22010.0)
    assert f.frame == 42


def test_parse_frame_sentinel_velocity_becomes_none():
    f = parse_frame(_good_msg(vel_h="-9999.0", vel_v=-9999))
    assert f.vel_h is None
    assert f.vel_v is None


def test_parse_frame_missing_mandatory_raises():
    with pytest.raises(FrameError):
        parse_frame(_good_msg(lat="-9999"))   # lat is sentinel → missing
    with pytest.raises(FrameError):
        parse_frame({k: v for k, v in _good_msg().items() if k != "serial"})
    with pytest.raises(FrameError):
        parse_frame(_good_msg(datetime="garbage"))


def test_parse_frame_lon_normalised():
    f = parse_frame(_good_msg(lon=190.0))
    assert f.lon == pytest.approx(-170.0)


def test_parse_frame_lat_out_of_range_raises():
    with pytest.raises(FrameError):
        parse_frame(_good_msg(lat=91.0))


def test_try_parse_frame_returns_none():
    assert try_parse_frame({"serial": "x"}) is None
    assert try_parse_frame(_good_msg()) is not None


def test_frame_dedup_key_fallback_to_datetime():
    f = parse_frame(_good_msg(frame="-9999"))   # frame sentinel → None
    assert f.frame is None
    assert f.dedup_key[0] == "S123"
    assert isinstance(f.dedup_key[1], str)
