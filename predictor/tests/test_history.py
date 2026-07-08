"""Tests for the SondeHub historical corpus (recovered flights + archive)."""

import json
from datetime import datetime, timedelta, timezone

from windfall.history import (CorpusFlight, download_corpus, fetch_corpus_winds,
                              flight_extent, iter_corpus, plan_wind_cycles)


def _report(serial, recovered=True, lat=47.5, lon=19.3):
    return {"serial": serial, "lat": lat, "lon": lon, "alt": 0,
            "recovered": recovered, "recovered_by": "test",
            "datetime": "2026-06-08T12:00:00"}


def _frames(serial, n=3):
    return [{"serial": serial, "lat": 47.0 + i * 0.01, "lon": 19.0, "alt": 1000.0 * i,
             "datetime": f"2026-06-08T10:0{i}:00.000000Z"} for i in range(n)]


def test_download_corpus_saves_and_caches(tmp_path):
    reports = [_report("AAA"), _report("BBB"), _report("AAA")]   # dup serial
    calls = []

    def telemetry(serial):
        calls.append(serial)
        return _frames(serial)

    paths = download_corpus(tmp_path, recovered_fn=lambda **kw: reports,
                            telemetry_fn=telemetry)
    assert sorted(p.name for p in paths) == ["AAA.json", "BBB.json"]
    assert calls == ["AAA", "BBB"]               # dup not re-fetched

    doc = json.loads((tmp_path / "AAA.json").read_text())
    assert doc["recovery"]["serial"] == "AAA"
    assert len(doc["frames"]) == 3

    # the corpus is a cache: a re-run downloads nothing new
    calls.clear()
    paths = download_corpus(tmp_path, recovered_fn=lambda **kw: reports,
                            telemetry_fn=telemetry)
    assert len(paths) == 2 and calls == []


def test_download_corpus_filters_and_limits(tmp_path):
    reports = [_report("NOPE", recovered=False), _report("R1"), _report("R2")]
    fetched = lambda **kw: reports
    telemetry = lambda s: _frames(s)

    paths = download_corpus(tmp_path / "a", recovered_fn=fetched, telemetry_fn=telemetry)
    assert [p.stem for p in paths] == ["R1", "R2"]    # not-found report skipped

    paths = download_corpus(tmp_path / "b", recovered_fn=fetched, telemetry_fn=telemetry,
                            only_recovered=False, limit=1)
    assert [p.stem for p in paths] == ["NOPE"]

    # archive has no file yet for R1 -> skipped, not fatal
    paths = download_corpus(tmp_path / "c", recovered_fn=fetched,
                            telemetry_fn=lambda s: None if s == "R1" else _frames(s))
    assert [p.stem for p in paths] == ["R2"]


def test_iter_corpus_loads_and_skips_junk(tmp_path):
    download_corpus(tmp_path, recovered_fn=lambda **kw: [_report("S1")],
                    telemetry_fn=lambda s: _frames(s))
    (tmp_path / "junk.json").write_text("{not json")
    (tmp_path / "empty.json").write_text(json.dumps({"recovery": {}, "frames": []}))

    flights = iter_corpus(tmp_path)
    assert len(flights) == 1
    cf = flights[0]
    assert cf.serial == "S1" and len(cf.frames) == 3
    assert cf.truth == (47.5, 19.3)

    assert iter_corpus(tmp_path / "missing") == []


def test_download_corpus_slims_frames(tmp_path):
    bloated = [dict(f, snr=12.5, uploaders=[{"callsign": "X", "antenna": "dipole"}])
               for f in _frames("FAT")]
    download_corpus(tmp_path, recovered_fn=lambda **kw: [_report("FAT")],
                    telemetry_fn=lambda s: bloated)
    saved = json.loads((tmp_path / "FAT.json").read_text())["frames"]
    assert all("uploaders" not in m and "snr" not in m for m in saved)
    assert all(m["lat"] is not None and m["datetime"] for m in saved)

    # slim=False keeps everything (the corpus as a full archive mirror)
    download_corpus(tmp_path / "full", recovered_fn=lambda **kw: [_report("FAT")],
                    telemetry_fn=lambda s: bloated, slim=False)
    saved = json.loads((tmp_path / "full" / "FAT.json").read_text())["frames"]
    assert all("uploaders" in m for m in saved)


def test_slim_corpus_rewrites_in_place(tmp_path):
    from windfall.history import slim_corpus

    bloated = [dict(f, uploaders=[{"callsign": "Y"}] * 50) for f in _frames("BIG")]
    download_corpus(tmp_path, recovered_fn=lambda **kw: [_report("BIG")],
                    telemetry_fn=lambda s: bloated, slim=False)
    (tmp_path / "junk.json").write_text("{not json")

    n, before, after = slim_corpus(tmp_path)
    assert n == 1 and after < before
    doc = json.loads((tmp_path / "BIG.json").read_text())
    assert doc["recovery"]["serial"] == "BIG"          # recovery kept whole
    assert all("uploaders" not in m for m in doc["frames"])
    # the engine sees identical flights before/after (lossless slim)
    flights = iter_corpus(tmp_path)
    assert len(flights) == 1 and len(flights[0].frames) == 3


def test_corpus_truth_guards_bad_positions():
    cf = CorpusFlight("X", {"serial": "X", "lat": 0.0, "lon": 0.0}, [{}], None)
    assert cf.truth is None
    cf = CorpusFlight("X", {"serial": "X"}, [{}], None)
    assert cf.truth is None


# ---- historical model-wind fetch planning -----------------------------------

def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _flight(serial, start="2026-06-08T13:00:00Z", minutes=120,
            lat=40.7, lon=-112.0, truth=(40.9, -111.5)):
    n = max(2, minutes // 10)
    t0 = _utc(start.rstrip("Z"))
    frames = []
    for i in range(n):
        t = t0 + timedelta(minutes=i * minutes / (n - 1))
        frames.append({"serial": serial, "lat": lat + i * 0.01, "lon": lon + i * 0.02,
                       "alt": 1000.0 * i, "datetime": t.isoformat().replace("+00:00", "Z")})
    recovery = {"serial": serial, "recovered": True}
    if truth is not None:
        recovery["lat"], recovery["lon"] = truth
    return CorpusFlight(serial, recovery, frames, None)


def test_flight_extent_spans_frames_and_truth():
    cf = _flight("EXT1", start="2026-06-08T13:00:00Z", minutes=120, truth=(40.9, -111.5))
    start, end, box = flight_extent(cf)
    assert start == _utc("2026-06-08T13:00:00")
    assert end == _utc("2026-06-08T15:00:00")
    assert box.contains(40.9, -111.5)          # recovered position included
    assert box.contains(40.7, -112.0)

    assert flight_extent(CorpusFlight("BAD", {}, [{"lat": 1.0}], None)) is None


def test_plan_wind_cycles_picks_published_cycle():
    # 13Z launch, 3.8 h latency -> cutoff 09:12 -> 06Z run (the 12Z GFS isn't
    # on the bucket until ~15:45). Hourly steps must cover 13:00 -> 15:00+pad.
    cf = _flight("P1", start="2026-06-08T13:00:00Z", minutes=120)
    plans = plan_wind_cycles([cf], cycle_hours=6.0, latency_hours=3.8)
    assert len(plans) == 1
    p = plans[0]
    assert p.cycle == _utc("2026-06-08T06:00:00")
    assert p.fxx[0] == 7                       # valid 13Z <= first frame
    assert p.fxx[-1] == 11                     # valid 17Z >= last frame + 2 h pad
    assert p.serials == ["P1"]
    assert p.box.contains(40.9, -111.5)        # margin swallows the truth point


def test_plan_wind_cycles_crosses_midnight():
    # 02Z launch with 3.8 h latency -> previous day's 18Z run.
    cf = _flight("P2", start="2026-06-08T02:00:00Z", minutes=60)
    (p,) = plan_wind_cycles([cf], cycle_hours=6.0, latency_hours=3.8)
    assert p.cycle == _utc("2026-06-07T18:00:00")
    assert p.fxx[0] == 8


def test_plan_wind_cycles_merges_and_dedups():
    a = _flight("A", start="2026-06-08T13:00:00Z", minutes=60, lat=40.0, lon=-112.0)
    b = _flight("B", start="2026-06-08T14:00:00Z", minutes=60, lat=41.5, lon=-111.0,
                truth=None)
    plans = plan_wind_cycles([a, b], cycle_hours=6.0, latency_hours=3.8)
    assert len(plans) == 1                     # same 06Z cycle
    p = plans[0]
    assert p.serials == ["A", "B"]
    assert p.fxx[0] == 7 and p.fxx[-1] == 11   # union of both flights' ranges
    assert p.box.contains(40.0, -112.0) and p.box.contains(41.5, -111.0)

    # already-cached hours drop out; a fully cached cycle disappears
    have = {(p.cycle, f) for f in range(7, 11)}
    (p2,) = plan_wind_cycles([a, b], cycle_hours=6.0, latency_hours=3.8, have=have)
    assert p2.fxx == [11]
    have.add((p.cycle, 11))
    assert plan_wind_cycles([a, b], cycle_hours=6.0, latency_hours=3.8, have=have) == []


def test_plan_wind_cycles_hourly_grid_and_clamp():
    cf = _flight("H1", start="2026-06-08T10:30:00Z", minutes=90)
    (p,) = plan_wind_cycles([cf], cycle_hours=1.0, latency_hours=1.0, max_fxx=18)
    assert p.cycle == _utc("2026-06-08T09:00:00")   # newest hourly run >=1 h old
    assert p.fxx[0] == 1                            # floor(10:30 - 09:00)
    assert p.fxx[-1] == 5                           # ceil(12:00 + 2 h pad - 09:00)

    # absurdly long span clamps to the model's forecast horizon
    long = _flight("H2", start="2026-06-08T10:00:00Z", minutes=30 * 60)
    (p,) = plan_wind_cycles([long], cycle_hours=1.0, latency_hours=1.0, max_fxx=18)
    assert p.fxx[-1] == 18

    # a flight with no parseable frames plans nothing
    bad = CorpusFlight("NOPE", {}, [{"lat": 1.0, "lon": 2.0}], None)
    assert plan_wind_cycles([bad], cycle_hours=6.0, latency_hours=3.8) == []


def test_fetch_corpus_winds_downloads_planned_cycles(tmp_path, monkeypatch):
    import windfall.gfs as gfs
    from windfall.config import Config

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cf = _flight("FW1", start="2026-06-08T13:00:00Z", minutes=120)
    (corpus / "FW1.json").write_text(json.dumps(
        {"recovery": cf.recovery, "frames": cf.frames}))

    calls = []
    monkeypatch.setattr(gfs, "download_gfs_cycle",
                        lambda cfg, box, fxx=None, levels=None, run_date=None:
                        calls.append((box, fxx, run_date)) or ["a.grib"])

    cfg = Config()
    cfg.gfs.path = str(tmp_path / "gfs-cache")
    lines = []
    assert fetch_corpus_winds(cfg, corpus, printer=lines.append) == 0
    assert len(calls) == 1
    box, fxx, run_date = calls[0]
    assert fxx == list(range(7, 12))
    assert run_date == datetime(2026, 6, 8, 6, 0)      # naive UTC for Herbie
    assert box.contains(40.9, -111.5)
    assert any("2026-06-08 06Z" in ln for ln in lines)

    # dry-run plans but never downloads
    calls.clear()
    assert fetch_corpus_winds(cfg, corpus, dry_run=True, printer=lines.append) == 0
    assert calls == []

    # empty corpus is a usage error
    assert fetch_corpus_winds(cfg, tmp_path / "nope", printer=lines.append) == 2


def test_fetch_corpus_winds_skips_cached_cycles(tmp_path, monkeypatch):
    import windfall.gfs as gfs
    from windfall.config import Config

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cf = _flight("FW2", start="2026-06-08T13:00:00Z", minutes=120)
    (corpus / "FW2.json").write_text(json.dumps(
        {"recovery": cf.recovery, "frames": cf.frames}))

    # cache already holds the whole planned cycle (Herbie-style names)
    cache = tmp_path / "gfs-cache" / "gfs" / "20260608"
    cache.mkdir(parents=True)
    for f in range(7, 12):
        (cache / f"subset_x__gfs.t06z.pgrb2.0p25.f{f:03d}").touch()

    monkeypatch.setattr(gfs, "download_gfs_cycle",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("downloaded")))
    cfg = Config()
    cfg.gfs.path = str(tmp_path / "gfs-cache")
    lines = []
    assert fetch_corpus_winds(cfg, corpus, printer=lines.append) == 0
    assert any("already in" in ln for ln in lines)
