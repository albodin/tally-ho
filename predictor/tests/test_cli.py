"""Tests for the standalone `windfall` CLI (replay / fetch-corpus / backtest /
ablate). Mirrors the tally-ho CLI tests for the commands both expose."""

import json

from windfall.cli import main
from tests.conftest import simulate_flight


def test_replay_file_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WINDFALL_ENSEMBLE_N_MEMBERS", "6")
    monkeypatch.setenv("WINDFALL_ENSEMBLE_MIN_INTERVAL_SECONDS", "300")
    f = simulate_flight(serial="WCLI1", burst_alt=24000)
    frames = tmp_path / "frames.json"
    frames.write_text(json.dumps(f.frames))
    assert main(["replay", "--file", str(frames)]) == 0
    out = capsys.readouterr().out
    assert "final landing error" in out

    # neither --file nor --serial
    assert main(["replay"]) == 2


def test_fetch_corpus_command(tmp_path, capsys, monkeypatch):
    import windfall.history as history

    f = simulate_flight(serial="WFC1", burst_alt=24000)
    monkeypatch.setattr(history, "fetch_recovered", lambda **kw: [
        {"serial": "WFC1", "lat": f.land_lat, "lon": f.land_lon, "recovered": True}])
    monkeypatch.setattr(history, "fetch_telemetry", lambda s: f.frames)

    out_dir = tmp_path / "corpus"
    rc = main(["fetch-corpus", "--out", str(out_dir),
               "--near", "45.0,7.0", "--duration", "3d"])
    assert rc == 0
    assert "corpus: 1 flights" in capsys.readouterr().out
    assert (out_dir / "WFC1.json").exists()

    assert main(["fetch-corpus", "--near", "oops"]) == 2


def _write_corpus(tmp_path, serial="WBT1"):
    f = simulate_flight(serial=serial, burst_alt=24000)
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / f"{serial}.json").write_text(json.dumps({
        "recovery": {"serial": serial, "lat": f.land_lat, "lon": f.land_lon,
                     "recovered": True},
        "frames": f.frames,
    }))
    return corpus


def test_fetch_gfs_command(tmp_path, capsys, monkeypatch):
    import windfall.gfs as gfs

    corpus = _write_corpus(tmp_path, serial="WFG1")
    monkeypatch.setenv("WINDFALL_GFS_PATH", str(tmp_path / "gfs-cache"))

    # dry-run prints the cycle plan without touching the downloader
    monkeypatch.setattr(gfs, "download_gfs_cycle",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("downloaded")))
    rc = main(["fetch-gfs", "--dir", str(corpus), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cycle(s) to download" in out and "WFG1" in out and "fxx" in out

    calls = []
    monkeypatch.setattr(gfs, "download_gfs_cycle",
                        lambda cfg, box, fxx=None, levels=None, run_date=None:
                        calls.append((fxx, run_date)) or ["x.grib"])
    assert main(["fetch-gfs", "--dir", str(corpus)]) == 0
    assert len(calls) == 1 and calls[0][1].tzinfo is None   # Herbie naive UTC

    assert main(["fetch-gfs", "--dir", str(tmp_path / "nothing")]) == 2


def test_slim_corpus_command(tmp_path, capsys):
    corpus = _write_corpus(tmp_path, serial="WSL1")
    assert main(["slim-corpus", "--dir", str(corpus)]) == 0
    assert "slimmed 1 flight(s)" in capsys.readouterr().out
    assert main(["slim-corpus", "--dir", str(tmp_path / "nothing")]) == 2


def test_backtest_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WINDFALL_ENSEMBLE_N_MEMBERS", "6")
    monkeypatch.setenv("WINDFALL_ENSEMBLE_MIN_INTERVAL_SECONDS", "300")
    corpus = _write_corpus(tmp_path)
    rc = main(["backtest", "--dir", str(corpus), "--no-gfs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WBT1" in out and "recovered" in out and "mean final error" in out

    # parallel path: same report shape from worker processes
    rc = main(["backtest", "--dir", str(corpus), "--no-gfs", "--jobs", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WBT1" in out and "mean final error" in out

    assert main(["backtest", "--dir", str(tmp_path / "nothing"), "--no-gfs"]) == 2


def test_ablate_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WINDFALL_ENSEMBLE_N_MEMBERS", "6")
    monkeypatch.setenv("WINDFALL_ENSEMBLE_MIN_INTERVAL_SECONDS", "300")
    corpus = _write_corpus(tmp_path, serial="WAB1")
    # without GFS data the model-backed variants degrade but still report; keep
    # the offline test on the two measured-capable modes for speed
    rc = main(["ablate", "--dir", str(corpus), "--modes", "measured-only,blend"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "measured-only" in out and "blend" in out and "WAB1" in out

    assert main(["ablate", "--dir", str(tmp_path / "nothing")]) == 2
