"""Tests for DEM terrain termination.

The rasterio-backed lookups need real GLO-30 tiles and are not exercised here;
we test tile naming, the offline fallbacks, the factory degrade path, and that a
synthetic terrain actually moves the predicted landing (the terrain win)."""

import json

import pytest

from windfall.config import Config, DEMConfig
from windfall.dem import (
    CallableGround,
    FlatGround,
    ReloadableGround,
    download_dem_tiles,
    make_ground_model,
    tile_name,
    tiles_for_bbox,
)
from windfall.geo import BBox
from windfall.replay import replay_messages
from tests.conftest import fast_ensemble_cfg, simulate_flight


def test_tile_name_positive():
    assert tile_name(45.3, 7.8) == "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"


def test_tile_name_negative():
    assert tile_name(-1.2, -3.5) == "Copernicus_DSM_COG_10_S02_00_W004_00_DEM"


def test_tiles_for_bbox():
    names = tiles_for_bbox(BBox(44.5, 6.5, 45.5, 7.5))
    # spans lat 44..45, lon 6..7 → 4 tiles
    assert len(names) == 4
    assert "Copernicus_DSM_COG_10_N44_00_E006_00_DEM" in names
    assert "Copernicus_DSM_COG_10_N45_00_E007_00_DEM" in names


def test_flat_ground():
    g = FlatGround(123.0)
    assert g(45, 7) == 123.0


def test_callable_ground():
    g = CallableGround(lambda lat, lon: lat * 10)
    assert g(4, 0) == 40


def test_factory_degrades_to_flat_when_disabled():
    g = make_ground_model(DEMConfig(enabled=False))
    assert isinstance(g, FlatGround)


def test_factory_degrades_to_flat_when_path_missing():
    g = make_ground_model(DEMConfig(enabled=True, path="/no/such/dem/dir"))
    assert isinstance(g, FlatGround)


def test_terrain_changes_landing_and_eta():
    # Flight that lands over high terrain: ground at 1500 m must terminate the
    # descent above where flat sea level would, shortening time-of-flight.
    f = simulate_flight(serial="TERR1", ground_alt=1500.0, burst_alt=20000)
    cfg = fast_ensemble_cfg()
    sea = replay_messages(f.frames, cfg=cfg, ground_fn=FlatGround(0.0))
    hill = replay_messages(f.frames, cfg=cfg, ground_fn=FlatGround(1500.0))
    # With ground at 1500 m, predictions terminate sooner → shorter sim time
    assert hill.records[0].sim_seconds < sea.records[0].sim_seconds
    # And the hill prediction (correct ground) matches truth better than sea
    assert hill.final_error_km < sea.final_error_km


# ---- bounds-indexed tile sets (3DEP-style) ----------------------------------

class _FakeInv:
    def __init__(self, left, top, res):
        self.left, self.top, self.res = left, top, res

    def __mul__(self, xy):
        lon, lat = xy
        return ((lon - self.left) / self.res, (self.top - lat) / self.res)


class _FakeTransform:
    def __init__(self, left, top, res):
        self._inv = _FakeInv(left, top, res)

    def __invert__(self):
        return self._inv


class _FakeRaster:
    def __init__(self, left, bottom, right, top, res_deg, elevation,
                 geographic=True, n=12):
        import numpy as np
        from types import SimpleNamespace
        self.bounds = SimpleNamespace(left=left, bottom=bottom, right=right, top=top)
        self.res = (res_deg, res_deg)
        self.crs = SimpleNamespace(is_geographic=geographic)
        self.width = self.height = n
        self.nodata = None
        self.transform = _FakeTransform(left, top, res_deg)
        self._grid = np.full((n, n), float(elevation))

    def read(self, band, window):
        (r0, r1), (c0, c1) = window
        return self._grid[r0:r1, c0:c1]

    def close(self):
        pass


def _fake_tileset(tmp_path):
    """Two overlapping tiles (coarse 1800 m, fine 1200 m), one projected-CRS
    impostor, and the opener mapping filenames to fakes."""
    fakes = {
        "coarse.tif": _FakeRaster(6.0, 44.0, 8.0, 46.0, 1 / 1200.0, 1800.0, n=2400),
        "fine.tif": _FakeRaster(6.8, 44.8, 7.2, 45.2, 1 / 10800.0, 1200.0, n=4320),
        "utm.tif": _FakeRaster(6.0, 44.0, 8.0, 46.0, 1 / 1200.0, 999.0,
                               geographic=False, n=2400),
    }
    for name in fakes:
        (tmp_path / name).touch()
    return lambda p: fakes.get(str(p).rsplit("/", 1)[-1])


def test_indexed_dem_picks_finest_covering_tile(tmp_path):
    from windfall.dem import IndexedDEM

    dem = IndexedDEM(str(tmp_path), opener=_fake_tileset(tmp_path))
    # inside both tiles → the finer one wins; the projected impostor is skipped
    assert dem(45.0, 7.0) == pytest.approx(1200.0)
    # only the coarse tile covers this point
    assert dem(44.2, 6.2) == pytest.approx(1800.0)
    # outside everything → fallback
    assert dem(50.0, 20.0) == 0.0
    # memoised repeat lookups stay consistent
    assert dem(45.0, 7.0) == pytest.approx(1200.0)


def test_indexed_dem_empty_dir_falls_back(tmp_path):
    from windfall.dem import IndexedDEM

    dem = IndexedDEM(str(tmp_path), fallback_elevation_m=42.0,
                     opener=lambda p: None)
    assert dem(45.0, 7.0) == 42.0


def test_detect_source(tmp_path):
    from windfall.dem import detect_source

    (tmp_path / "N40_tile.tif").touch()
    assert detect_source(tmp_path) == "tiles"
    (tmp_path / "Copernicus_DSM_COG_10_N40_00_W112_00_DEM.tif").touch()
    assert detect_source(tmp_path) == "glo30"


# ---- auto-download + runtime reload ------------------------------------------

_BOX_2X2 = BBox(44.5, 6.5, 45.5, 7.5)   # spans lat 44..45, lon 6..7 → 4 tiles
_OCEAN = "Copernicus_DSM_COG_10_N44_00_E006_00_DEM"


def test_download_dem_tiles_fetches_missing_and_remembers_404(tmp_path):
    calls: list[str] = []

    def fetch(url, dest):
        calls.append(url)
        if _OCEAN in url:
            return False          # 404: ocean-only square, no tile exists
        dest.write_bytes(b"tile")
        return True

    cfg = DEMConfig(path=str(tmp_path / "dem"))
    skip: set[str] = set()
    new = download_dem_tiles(cfg, _BOX_2X2, skip=skip, fetch=fetch)

    assert len(new) == 3 and _OCEAN not in new
    for name in new:
        assert (tmp_path / "dem" / f"{name}.tif").read_bytes() == b"tile"
    assert skip == {_OCEAN}
    # the URL template is applied per tile name
    assert all(u.startswith("https://copernicus-dem-30m") for u in calls)
    # no .part leftovers, including for the 404
    assert not list((tmp_path / "dem").glob("*.part"))

    # second pass: everything on disk or known-absent → zero fetches
    calls.clear()
    assert download_dem_tiles(cfg, _BOX_2X2, skip=skip, fetch=fetch) == []
    assert calls == []


def test_download_dem_tiles_absent_list_survives_restart(tmp_path):
    calls: list[str] = []

    def fetch(url, dest):
        calls.append(url)
        if _OCEAN in url:
            return False
        dest.write_bytes(b"tile")
        return True

    cfg = DEMConfig(path=str(tmp_path / "dem"))
    download_dem_tiles(cfg, _BOX_2X2, skip=set(), fetch=fetch)
    absent_file = tmp_path / "dem" / "absent_tiles.json"
    assert json.loads(absent_file.read_text()) == [_OCEAN]
    # atomic write: no .part leftovers (also guards the tile downloads)
    assert not list((tmp_path / "dem").glob("*.part"))

    # "restart": a fresh skip set - the persisted 404 is merged in, not re-fetched
    calls.clear()
    skip: set[str] = set()
    assert download_dem_tiles(cfg, _BOX_2X2, skip=skip, fetch=fetch) == []
    assert calls == []
    assert _OCEAN in skip


def test_download_dem_tiles_absent_saved_even_when_pass_aborts(tmp_path):
    # 404s persist as discovered: a pass that dies partway (here the
    # unwritable-dir abort) keeps the ocean squares found before the failure
    def fetch(url, dest):
        if _OCEAN in url:
            return False
        raise PermissionError(13, "Permission denied", str(dest))

    # single worker: the ocean 404 (first in tile order) lands before the abort
    cfg = DEMConfig(path=str(tmp_path / "dem"), download_workers=1)
    assert download_dem_tiles(cfg, _BOX_2X2, fetch=fetch) == []
    assert json.loads((tmp_path / "dem" / "absent_tiles.json").read_text()) == [_OCEAN]


def test_download_dem_tiles_logs_roi_progress(tmp_path, caplog):
    def fetch(url, dest):
        if _OCEAN in url:
            return False
        dest.write_bytes(b"tile")
        return True

    # single worker: deterministic tile order, so the ratios are exact
    cfg = DEMConfig(path=str(tmp_path / "dem"), download_workers=1)
    import logging

    with caplog.at_level(logging.INFO, logger="windfall.dem"):
        download_dem_tiles(cfg, _BOX_2X2, fetch=fetch)
    got = [r.getMessage() for r in caplog.records
           if r.getMessage().startswith("downloaded DEM tile ")]
    # the ocean square is hit first (lat/lon-ascending order) and drops out of
    # the denominator, so the three land tiles count up to a complete 3/3
    assert [m[m.index("("):] for m in got] == [
        "(1/3 for ROI)", "(2/3 for ROI)", "(3/3 for ROI)"]


def test_download_dem_tiles_corrupt_absent_list_is_rebuilt(tmp_path):
    dem = tmp_path / "dem"
    dem.mkdir()
    (dem / "absent_tiles.json").write_text("{not json")

    def fetch(url, dest):
        if _OCEAN in url:
            return False
        dest.write_bytes(b"tile")
        return True

    cfg = DEMConfig(path=str(dem))
    # corrupt file reads as empty (one extra 404 round) and is rewritten intact
    assert len(download_dem_tiles(cfg, _BOX_2X2, fetch=fetch)) == 3
    assert json.loads((dem / "absent_tiles.json").read_text()) == [_OCEAN]


def test_download_dem_tiles_parallel_workers_fetch_concurrently(tmp_path):
    import threading

    # releases only when two fetches are in flight at once: a sequential
    # implementation trips the timeout, breaks the barrier, and downloads nothing
    barrier = threading.Barrier(2, timeout=10)

    def fetch(url, dest):
        barrier.wait()
        if _OCEAN in url:
            return False
        dest.write_bytes(b"tile")
        return True

    cfg = DEMConfig(path=str(tmp_path / "dem"), download_workers=4)
    new = download_dem_tiles(cfg, _BOX_2X2, fetch=fetch)
    assert len(new) == 3
    assert json.loads((tmp_path / "dem" / "absent_tiles.json").read_text()) == [_OCEAN]


def test_download_dem_tiles_no_roi_is_a_noop(tmp_path):
    cfg = DEMConfig(path=str(tmp_path / "dem"))
    assert download_dem_tiles(cfg, None, fetch=lambda u, d: True) == []
    assert not (tmp_path / "dem").exists()   # not even the dir is created


def test_download_dem_tiles_unwritable_dir_aborts_pass_quietly(tmp_path, caplog):
    # An unwritable DEM dir (bind-mount ownership mismatch) fails every tile
    # identically: one actionable error line, pass aborted - not a traceback
    # per tile per pass.
    calls = []

    def denied(url, dest):
        calls.append(url)
        raise PermissionError(13, "Permission denied", str(dest))

    cfg = DEMConfig(path=str(tmp_path), download_workers=1)
    import logging

    with caplog.at_level(logging.ERROR, logger="windfall.dem"):
        assert download_dem_tiles(cfg, _BOX_2X2, fetch=denied) == []
    assert len(calls) == 1                      # aborted after the first tile
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "not writable" in errors[0].getMessage()


def test_download_dem_tiles_transient_error_retries_next_pass(tmp_path):
    def boom(url, dest):
        dest.write_bytes(b"trunc")   # partial write before the failure
        raise OSError("network down")

    cfg = DEMConfig(path=str(tmp_path))
    skip: set[str] = set()
    box = BBox(45.2, 7.2, 45.3, 7.3)   # single tile
    assert download_dem_tiles(cfg, box, skip=skip, fetch=boom) == []
    assert skip == set()               # NOT recorded as absent - retried later
    assert list(tmp_path.iterdir()) == []   # partial .part cleaned up

    # the retry (network back) succeeds
    def ok(url, dest):
        dest.write_bytes(b"tile")
        return True

    assert len(download_dem_tiles(cfg, box, skip=skip, fetch=ok)) == 1


def test_reloadable_ground_picks_up_downloaded_tiles(tmp_path, monkeypatch):
    # Startup with no DEM dir → flat ground; after tiles arrive (the in-app
    # downloader's job), reload() swaps in a real GLO-30 model without the
    # tracker/predictor ever seeing a new ground_fn object. rasterio is stubbed
    # so the factory's import gate passes in environments without it.
    import sys
    import types

    monkeypatch.setitem(sys.modules, "rasterio", types.ModuleType("rasterio"))
    cfg = DEMConfig(path=str(tmp_path / "dem"))
    g = ReloadableGround(cfg)
    assert isinstance(g._model, FlatGround)
    assert g(45.0, 7.0) == 0.0

    (tmp_path / "dem").mkdir()
    (tmp_path / "dem" / "Copernicus_DSM_COG_10_N45_00_E007_00_DEM.tif").touch()
    g.reload()

    from windfall.dem import CopernicusDEM

    assert isinstance(g._model, CopernicusDEM)   # auto-detected as glo30
