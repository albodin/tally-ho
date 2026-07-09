"""Tests for the CLI."""

import json

import pytest

from tallyho.cli import main
from tests.conftest import simulate_flight


def test_subscriber_add_list_deactivate(tmp_path, capsys):
    db = tmp_path / "t.db"
    rc = main(["--config", _cfg(tmp_path, db), "subscriber", "add",
               "--name", "bob", "--lat", "45", "--lon", "7", "--radius", "25",
               "--ntfy-topic", "bob-sondes", "--token-ref", "NTFY_BOB"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "added subscriber id=1" in out

    main(["--config", _cfg(tmp_path, db), "subscriber", "list"])
    out = capsys.readouterr().out
    assert "bob" in out and "bob-sondes" in out

    main(["--config", _cfg(tmp_path, db), "subscriber", "deactivate", "--id", "1"])
    main(["--config", _cfg(tmp_path, db), "subscriber", "list"])
    out = capsys.readouterr().out
    assert "inactive" in out


def test_token_set_list_delete(tmp_path, capsys, monkeypatch):
    import io

    db = tmp_path / "t.db"
    # piped stdin (not a tty) supplies the value - it is never a CLI argument
    monkeypatch.setattr("sys.stdin", io.StringIO("tk_secret_wxyz\n"))
    rc = main(["--config", _cfg(tmp_path, db), "token", "set", "home"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "saved token 'home'" in out and "tk_secret_wxyz" not in out

    main(["--config", _cfg(tmp_path, db), "token", "list"])
    out = capsys.readouterr().out
    assert "home" in out and "…wxyz" in out and "tk_secret_wxyz" not in out

    # a token referenced by a subscriber refuses deletion
    main(["--config", _cfg(tmp_path, db), "subscriber", "add",
          "--name", "bob", "--lat", "45", "--lon", "7", "--radius", "25",
          "--ntfy-topic", "bob-sondes", "--token-ref", "home"])
    capsys.readouterr()
    assert main(["--config", _cfg(tmp_path, db), "token", "delete", "home"]) == 2
    assert "used by 1 location" in capsys.readouterr().err

    monkeypatch.setattr("sys.stdin", io.StringIO("tk_other\n"))
    main(["--config", _cfg(tmp_path, db), "token", "set", "spare"])
    capsys.readouterr()
    assert main(["--config", _cfg(tmp_path, db), "token", "delete", "spare"]) == 0
    assert main(["--config", _cfg(tmp_path, db), "token", "delete", "spare"]) == 2


def test_token_set_rejects_bad_name_and_empty_value(tmp_path, capsys, monkeypatch):
    import io

    db = tmp_path / "t.db"
    assert main(["--config", _cfg(tmp_path, db), "token", "set", "bad name!"]) == 2
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    assert main(["--config", _cfg(tmp_path, db), "token", "set", "home"]) == 2
    assert "empty token" in capsys.readouterr().err


def test_replay_from_file(tmp_path, capsys, monkeypatch):
    # tiny ensemble via env override - this test is about the CLI plumbing
    monkeypatch.setenv("TALLYHO_ENSEMBLE_N_MEMBERS", "6")
    monkeypatch.setenv("TALLYHO_ENSEMBLE_MIN_INTERVAL_SECONDS", "300")
    f = simulate_flight(serial="CLI1", burst_alt=24000)
    frames_file = tmp_path / "frames.json"
    frames_file.write_text(json.dumps(f.frames))
    rc = main(["replay", "--file", str(frames_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "final landing error" in out
    assert "calibration" in out


def test_accuracy_command(tmp_path, capsys):
    from datetime import date, datetime, timezone

    from tallyho.models import Prediction, PredictionSource
    from tallyho.store import Store

    db = tmp_path / "t.db"
    store = Store(db)
    day = date(2026, 6, 7)
    store.record_landing("ACLI1", day, land_lat=45.5, land_lon=7.6, land_alt=210.0,
                         landed_at=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
                         detected_by="telemetry")
    store.save_prediction(Prediction(
        serial="ACLI1", launch_day=day,
        predicted_at=datetime(2026, 6, 7, 0, 39, tzinfo=timezone.utc),
        land_lat=45.51, land_lon=7.6,
        land_eta=datetime(2026, 6, 7, 0, 40, tzinfo=timezone.utc),
        source=PredictionSource.MEASURED, uncertainty_radius_km=5.0, alt_at_pred=500.0))
    store.close()

    rc = main(["--config", _cfg(tmp_path, db), "accuracy"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mean final error" in out
    assert "ACLI1" in out


def test_accuracy_command_empty(tmp_path, capsys):
    db = tmp_path / "empty.db"
    rc = main(["--config", _cfg(tmp_path, db), "accuracy"])
    assert rc == 0
    assert "no scored landings" in capsys.readouterr().out


def test_dem_tiles(tmp_path, capsys):
    db = tmp_path / "t.db"
    main(["--config", _cfg(tmp_path, db), "subscriber", "add",
          "--name", "c", "--lat", "45", "--lon", "7", "--radius", "20",
          "--ntfy-topic", "c"])
    capsys.readouterr()
    rc = main(["--config", _cfg(tmp_path, db), "dem-tiles"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Copernicus_DSM_COG_10_" in out


def test_health_command_heartbeat_fallback(tmp_path, capsys):
    """With the web UI disabled, `tallyho health` reads the heartbeat file."""
    from datetime import datetime, timedelta, timezone

    hb = tmp_path / "heartbeat"
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'health_file = "{hb}"\nhealth_stale_seconds = 120\n'
                   "[web]\nenabled = false\n")

    # no heartbeat yet → unhealthy
    assert main(["--config", str(cfg), "health"]) == 1

    # fresh heartbeat → healthy
    hb.write_text(datetime.now(timezone.utc).isoformat())
    assert main(["--config", str(cfg), "health"]) == 0

    # stale heartbeat → unhealthy
    hb.write_text((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
    assert main(["--config", str(cfg), "health"]) == 1


def test_health_command_probes_web_ui(tmp_path, capsys, monkeypatch):
    """With the web UI enabled (the default), health probes /api/health -
    200 = healthy, 503 = stale, connection refused = unhealthy."""
    import io
    import urllib.error
    import urllib.request
    from contextlib import contextmanager

    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    seen = {}

    @contextmanager
    def fake_ok(url, timeout):
        seen["url"] = url
        yield io.BytesIO(b'{"status": "ok"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_ok)
    assert main(["--config", str(cfg), "health"]) == 0
    assert seen["url"] == "http://127.0.0.1:8080/api/health"

    def fake_stale(url, timeout):
        raise urllib.error.HTTPError(url, 503, "unavailable", None,
                                     io.BytesIO(b'{"status": "stale"}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_stale)
    assert main(["--config", str(cfg), "health"]) == 1

    def fake_refused(url, timeout):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_refused)
    assert main(["--config", str(cfg), "health"]) == 1


def _cfg(tmp_path, db) -> str:
    p = tmp_path / "config.toml"
    p.write_text(f'db_path = "{db}"\n')
    return str(p)


def test_fetch_corpus_command(tmp_path, capsys, monkeypatch):
    import windfall.history as history

    f = simulate_flight(serial="FC1", burst_alt=24000)
    monkeypatch.setattr(history, "fetch_recovered", lambda **kw: [
        {"serial": "FC1", "lat": f.land_lat, "lon": f.land_lon, "recovered": True}])
    monkeypatch.setattr(history, "fetch_telemetry", lambda s: f.frames)

    out_dir = tmp_path / "corpus"
    db = tmp_path / "t.db"
    rc = main(["--config", _cfg(tmp_path, db), "fetch-corpus",
               "--out", str(out_dir), "--near", "45.0,7.0", "--duration", "3d"])
    assert rc == 0
    assert "corpus: 1 flights" in capsys.readouterr().out
    assert (out_dir / "FC1.json").exists()

    # bad --near
    rc = main(["--config", _cfg(tmp_path, db), "fetch-corpus", "--near", "oops"])
    assert rc == 2


def test_backtest_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TALLYHO_ENSEMBLE_N_MEMBERS", "6")
    monkeypatch.setenv("TALLYHO_ENSEMBLE_MIN_INTERVAL_SECONDS", "300")
    f = simulate_flight(serial="BTC1", burst_alt=24000)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "BTC1.json").write_text(json.dumps({
        "recovery": {"serial": "BTC1", "lat": f.land_lat, "lon": f.land_lon,
                     "recovered": True},
        "frames": f.frames,
    }))
    db = tmp_path / "t.db"
    rc = main(["--config", _cfg(tmp_path, db), "backtest",
               "--dir", str(corpus), "--no-gfs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "BTC1" in out and "recovered" in out and "mean final error" in out

    rc = main(["--config", _cfg(tmp_path, db), "backtest",
               "--dir", str(corpus), "--no-gfs", "--jobs", "2"])
    assert rc == 0
    assert "BTC1" in capsys.readouterr().out

    rc = main(["--config", _cfg(tmp_path, db), "backtest",
               "--dir", str(tmp_path / "nothing"), "--no-gfs"])
    assert rc == 2


def test_fetch_gfs_command(tmp_path, capsys, monkeypatch):
    import windfall.gfs as gfs

    f = simulate_flight(serial="FG1", burst_alt=24000)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "FG1.json").write_text(json.dumps({
        "recovery": {"serial": "FG1", "lat": f.land_lat, "lon": f.land_lon,
                     "recovered": True},
        "frames": f.frames,
    }))
    monkeypatch.setenv("TALLYHO_GFS_PATH", str(tmp_path / "gfs-cache"))
    monkeypatch.setattr(gfs, "download_gfs_cycle",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("downloaded")))
    db = tmp_path / "t.db"
    rc = main(["--config", _cfg(tmp_path, db), "fetch-gfs",
               "--dir", str(corpus), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cycle(s) to download" in out and "FG1" in out

    rc = main(["--config", _cfg(tmp_path, db), "fetch-gfs",
               "--dir", str(tmp_path / "nothing")])
    assert rc == 2
