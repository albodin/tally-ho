"""Tests for the HRRR source's pure pieces: the Lambert
projection, projected-grid WindCube sampling, HRRR-over-GFS composition, and
the wind-source factory. GRIB reading itself needs cfgrib + real files and is
exercised in deployment, like the GFS reader."""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from windfall.config import Config
from windfall.gfs import StaticGFSSource, WindCube, parse_grib_path
from windfall.hrrr import (
    CompositeWindSource,
    HerbieHRRRSource,
    LambertGrid,
    blend_fields,
    make_wind_source,
)
from windfall.profile import FlightProfile

WHEN = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _hrrr_grid() -> LambertGrid:
    return LambertGrid.from_attrs({}, nx=1799, ny=1059)


# ---- Lambert projection ------------------------------------------------------

def test_lambert_anchor_is_first_gridpoint():
    g = _hrrr_grid()
    y, x = g.proj(21.138123, 237.280472)
    assert abs(y) < 1.0 and abs(x) < 1.0


def test_lambert_central_meridian_hits_grid_centre():
    # The HRRR grid is built symmetric about LoV (-97.5°): the centre column is
    # index (1799-1)/2 = 899; the standard parallel row lands near 529.
    g = _hrrr_grid()
    y, x = g.proj(38.5, -97.5)
    assert x / g.dx == pytest.approx(899.0, abs=0.5)
    assert y / g.dy == pytest.approx(529.0, abs=0.5)


def test_lambert_local_scale_is_true_at_standard_parallel():
    g = _hrrr_grid()
    y1, x1 = g.proj(38.5, -97.5)
    y2, x2 = g.proj(38.5 + 3000.0 / 111_195.0, -97.5)   # ~3 km due north
    assert (y2 - y1) == pytest.approx(3000.0, rel=0.001)
    assert abs(x2 - x1) < 1.0
    # east of the central meridian x grows, west it shrinks
    assert g.proj(38.5, -96.5)[1] > x1 > g.proj(38.5, -98.5)[1]


def test_lambert_lon_wrap():
    g = _hrrr_grid()
    # the same meridian expressed as 262.5 or -97.5 must project identically
    assert g.proj(40.0, 262.5) == pytest.approx(g.proj(40.0, -97.5))


def test_window_slices_cover_query_and_clamp():
    g = _hrrr_grid()
    ysl, xsl = g.window(40.6, -111.9, 3.0)     # Salt Lake City
    assert 0 <= ysl.start < ysl.stop <= g.ny
    assert 0 <= xsl.start < xsl.stop <= g.nx
    # the window must contain the query point itself
    y, x = g.proj(40.6, -111.9)
    assert ysl.start * g.dy <= y <= (ysl.stop - 1) * g.dy
    assert xsl.start * g.dx <= x <= (xsl.stop - 1) * g.dx
    # ±3° of latitude is ~666 km → roughly 222 rows plus margin, well clamped
    assert 150 < (ysl.stop - ysl.start) < 400
    # a corner query clamps instead of going out of range
    ysl2, xsl2 = g.window(21.2, -122.6, 3.0)
    assert ysl2.start == 0 and xsl2.start == 0


# ---- projected-grid WindCube -------------------------------------------------

def test_windcube_projected_sampling():
    # A 3x3 grid in projected metres; proj is a fake equirectangular mapping so
    # expected values are easy to compute. u varies linearly with x.
    def proj(lat, lon):
        return (lat * 1000.0, lon * 1000.0)

    y_axis = np.array([0.0, 1000.0, 2000.0])
    x_axis = np.array([0.0, 1000.0, 2000.0])
    heights = np.tile(np.array([0.0, 5000.0])[:, None, None], (1, 3, 3))
    u = np.zeros((2, 3, 3))
    u[:, :, 1] = 10.0
    u[:, :, 2] = 20.0
    v = np.ones_like(u)
    cube = WindCube.from_grid(valid_time=WHEN, lats=y_axis, lons=x_axis,
                              heights=heights, u=u, v=v, proj=proj)
    assert cube.wind_at(0.0, 1.0, 1000.0)[0] == pytest.approx(10.0)
    assert cube.wind_at(0.0, 0.5, 1000.0)[0] == pytest.approx(5.0)   # bilinear x
    assert cube.wind_at(1.5, 2.0, 0.0)[0] == pytest.approx(20.0)
    assert cube.wind_at(0.0, 2.0, 9999.0)[1] == pytest.approx(1.0)


# ---- HRRR-over-GFS composition -------------------------------------------------

def _const_field(u, v):
    return lambda lat, lon, alt, sim_t=0.0: (u, v)


def test_blend_fields_ramp():
    f = blend_fields(_const_field(10.0, 0.0), _const_field(0.0, 10.0),
                     ceiling_m=20_000.0, ramp_m=2_000.0)
    assert f(0, 0, 5_000.0) == (10.0, 0.0)            # HRRR below the ramp
    assert f(0, 0, 25_000.0) == (0.0, 10.0)           # GFS above the ceiling
    u, v = f(0, 0, 19_000.0)                          # half-way up the ramp
    assert u == pytest.approx(5.0) and v == pytest.approx(5.0)


def _profile_const(u, v, top=30_000.0, step=500.0):
    p = FlightProfile(bin_size_m=250.0)
    alt = 100.0
    while alt < top:
        p.add_sample(alt, u, v)
        alt += step
    return p


def test_composite_source_field_profile_and_filler():
    hrrr = StaticGFSSource(_profile_const(10.0, 0.0, top=20_000.0))
    gfs = StaticGFSSource(_profile_const(0.0, 10.0))
    src = CompositeWindSource(hrrr, gfs, ceiling_m=20_000.0, ramp_m=1_000.0)

    field = src.wind_field(40.0, -111.0, WHEN)
    assert field(40.0, -111.0, 5_000.0, 0.0)[0] == pytest.approx(10.0)
    assert field(40.0, -111.0, 25_000.0, 0.0)[1] == pytest.approx(10.0)

    prof = src.profile_at(40.0, -111.0, WHEN)
    assert prof.wind(5_000.0)[0] == pytest.approx(10.0)    # HRRR bins below
    assert prof.wind(25_000.0)[1] == pytest.approx(10.0)   # GFS bins above

    fill = src.wind_filler(40.0, -111.0, WHEN)
    assert fill(5_000.0)[0] == pytest.approx(10.0)
    assert fill(25_000.0)[1] == pytest.approx(10.0)


class _DrySource(StaticGFSSource):
    """A source with no data for the requested time."""

    def __init__(self):
        super().__init__(FlightProfile())

    def profile_at(self, lat, lon, when):
        return None

    def wind_field(self, lat, lon, when):
        return None

    def wind_filler(self, lat, lon, when):
        return None


def test_composite_degrades_when_either_side_is_dry():
    gfs = StaticGFSSource(_profile_const(0.0, 10.0))
    src = CompositeWindSource(_DrySource(), gfs, ceiling_m=20_000.0)
    assert src.wind_field(40, -111, WHEN)(0, 0, 5_000.0, 0.0)[1] == pytest.approx(10.0)
    assert src.profile_at(40, -111, WHEN).wind(5_000.0)[1] == pytest.approx(10.0)

    hrrr = StaticGFSSource(_profile_const(10.0, 0.0, top=20_000.0))
    src2 = CompositeWindSource(hrrr, _DrySource(), ceiling_m=20_000.0)
    assert src2.wind_field(40, -111, WHEN)(0, 0, 5_000.0, 0.0)[0] == pytest.approx(10.0)
    assert src2.profile_at(40, -111, WHEN).wind(5_000.0)[0] == pytest.approx(10.0)


# ---- inventory + factory --------------------------------------------------------

def test_parse_grib_path_hrrr_names():
    info = parse_grib_path("hrrr/20260609/subset_ab12__hrrr.t06z.wrfprsf03.grib2")
    assert info is not None
    assert info.cycle.hour == 6 and info.fxx == 3
    assert info.valid_time.hour == 9


def test_make_wind_source_matrix(tmp_path):
    cfg = Config()
    cfg.gfs.path = str(tmp_path / "gfs")
    cfg.hrrr.path = str(tmp_path / "hrrr")

    # nothing enabled → None
    assert make_wind_source(cfg) is None

    # HRRR enabled but path missing → still None
    cfg.hrrr.enabled = True
    assert make_wind_source(cfg) is None

    # HRRR alone
    (tmp_path / "hrrr").mkdir()
    src = make_wind_source(cfg)
    assert isinstance(src, HerbieHRRRSource)

    # both → composite with HRRR primary, honouring the configured ceiling
    cfg.gfs.enabled = True
    (tmp_path / "gfs").mkdir()
    cfg.hrrr.ceiling_m = 19_000.0
    src = make_wind_source(cfg)
    assert isinstance(src, CompositeWindSource)
    assert isinstance(src.primary, HerbieHRRRSource)
    assert src.ceiling_m == 19_000.0

    # GFS alone when HRRR is disabled again
    cfg.hrrr.enabled = False
    from windfall.gfs import HerbieGFSSource
    assert isinstance(make_wind_source(cfg), HerbieGFSSource)
