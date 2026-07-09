"""Command-line interface.

    tallyho run                         # start the live monolith
    tallyho subscriber add ...          # onboard a friend (v1 onboarding)
    tallyho subscriber list
    tallyho subscriber deactivate --id N
    tallyho token set NAME              # save an ntfy bearer token (prompted/stdin)
    tallyho token list
    tallyho token delete NAME
    tallyho replay --file frames.json   # accuracy harness on archived frames
    tallyho replay --serial S1234567    # download from SondeHub, then replay
    tallyho accuracy                    # score saved predictions vs actual landings
    tallyho fetch-corpus --near 47.5,19.0 --duration 1m   # recovered flights -> data/corpus
    tallyho fetch-gfs                   # historical model cycles the corpus needs
    tallyho backtest                    # replay the corpus vs recovered ground truth
    tallyho dem-tiles                   # list GLO-30 tiles for the capture ROI
    tallyho web                         # local dashboard + onboarding UI (needs '.[api]')
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import load_config
from .models import Subscriber
from .store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tallyho", description="SondeHub landing predictor")
    parser.add_argument("--config", default=os.environ.get("TALLYHO_CONFIG", "data/config.toml"),
                        help="path to config TOML (seeded on first `run`; missing file = defaults)")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="start the live monolith")

    sp = sub.add_parser("subscriber", help="manage subscribers")
    ssub = sp.add_subparsers(dest="subcmd", required=True)
    add = ssub.add_parser("add")
    add.add_argument("--name", required=True)
    add.add_argument("--lat", type=float, required=True)
    add.add_argument("--lon", type=float, required=True)
    add.add_argument("--radius", type=float, required=True, help="alert radius (km)")
    add.add_argument("--ntfy-server", default="https://ntfy.sh")
    add.add_argument("--ntfy-topic", default="",
                     help="ntfy topic; omit (or leave blank) for a watch-only "
                          "location that is tracked/shown but never sends alerts")
    add.add_argument("--token-ref", default=None,
                     help="NAME of a saved ntfy token ('tallyho token set NAME'), "
                          "never the token itself")
    add.add_argument("--units", choices=("metric", "imperial"), default="metric",
                     help="alert display units: metric (km/m) or imperial (mi/ft)")
    ssub.add_parser("list")
    deact = ssub.add_parser("deactivate")
    deact.add_argument("--id", type=int, required=True)

    tk = sub.add_parser("token", help="manage ntfy bearer tokens (or use the web UI)")
    tsub = tk.add_subparsers(dest="subcmd", required=True)
    tset = tsub.add_parser("set", help="save/replace a token; the value is prompted "
                                       "(or piped on stdin), never a CLI argument")
    tset.add_argument("name")
    tsub.add_parser("list", help="saved token names (never values)")
    trm = tsub.add_parser("delete")
    trm.add_argument("name")

    rp = sub.add_parser("replay", help="run the accuracy harness")
    rp.add_argument("--file", help="JSON file: a list of raw telemetry dicts")
    rp.add_argument("--serial", help="download this serial's archive from SondeHub")

    ac = sub.add_parser("accuracy", help="score saved predictions against actual landings")
    ac.add_argument("--limit", type=int, default=200, help="how many recent landings to score")

    fc = sub.add_parser("fetch-corpus",
                        help="download recovered flights + their telemetry into a backtest corpus")
    fc.add_argument("--out", default="data/corpus", help="corpus directory (append-only cache)")
    fc.add_argument("--duration", default="7d", help="recovery lookback, SondeHub syntax (1d/7d/3m)")
    fc.add_argument("--near", default=None, metavar="LAT,LON", help="geo-filter centre")
    fc.add_argument("--distance-km", type=float, default=300.0, help="geo-filter radius (with --near)")
    fc.add_argument("--limit", type=int, default=None, help="stop after this many flights")
    fc.add_argument("--all", action="store_true",
                    help="include reports marked not-recovered (searched but not found)")

    fg = sub.add_parser("fetch-gfs",
                        help="download the historical GFS/HRRR cycles a corpus backtest needs")
    fg.add_argument("--dir", default="data/corpus", help="corpus directory")
    fg.add_argument("--model", choices=("gfs", "hrrr", "both"), default="gfs",
                    help="which model cache(s) to fill (HRRR: CONUS flights only)")
    fg.add_argument("--margin-km", type=float, default=75.0,
                    help="region margin around each flight's track")
    fg.add_argument("--pad-hours", type=float, default=2.0,
                    help="forecast-hour headroom past each flight's last frame")
    fg.add_argument("--dry-run", action="store_true",
                    help="print the cycle plan without downloading")

    bt = sub.add_parser("backtest",
                        help="replay a fetched corpus against recovered ground truth")
    bt.add_argument("--dir", default="data/corpus", help="corpus directory")
    bt.add_argument("--limit", type=int, default=None, help="replay at most this many flights")
    bt.add_argument("--no-gfs", action="store_true",
                    help="measured-winds only (skip the GFS cache at gfs.path)")
    bt.add_argument("--jobs", type=int, default=1,
                    help="worker processes - flights replay in parallel (default 1)")

    sub.add_parser("dem-tiles", help="list GLO-30 tiles covering the capture ROI")
    sub.add_parser("health", help="exit 0 if healthy: probes the web UI's /api/health, "
                                  "or the heartbeat file when the UI is disabled")

    wp = sub.add_parser("web", help="run the local dashboard web UI (needs '.[api]')")
    wp.add_argument("--host", default=None, help="bind address (default from [web] config)")
    wp.add_argument("--port", type=int, default=None, help="port (default 8080)")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(
        level=(args.log_level or cfg.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.cmd == "run":
        return _cmd_run(cfg, args.config)
    if args.cmd == "subscriber":
        return _cmd_subscriber(cfg, args)
    if args.cmd == "token":
        return _cmd_token(cfg, args)
    if args.cmd == "replay":
        return _cmd_replay(cfg, args)
    if args.cmd == "accuracy":
        return _cmd_accuracy(cfg, args)
    if args.cmd == "fetch-corpus":
        return _cmd_fetch_corpus(cfg, args)
    if args.cmd == "fetch-gfs":
        return _cmd_fetch_gfs(cfg, args)
    if args.cmd == "backtest":
        return _cmd_backtest(cfg, args)
    if args.cmd == "dem-tiles":
        return _cmd_dem_tiles(cfg)
    if args.cmd == "health":
        return _cmd_health(cfg)
    if args.cmd == "web":
        return _cmd_web(cfg, args)
    return 1


def _cmd_run(cfg, config_path) -> int:  # pragma: no cover - needs network
    from .app import App
    from .setup import ensure_setup

    # First run: seed the config template and serve the setup wizard; the
    # pipeline starts only once an account exists (or the web UI is disabled).
    cfg = ensure_setup(cfg, config_path)
    if cfg is None:
        return 1
    app = App(cfg)
    app.run()
    return 0


def _cmd_web(cfg, args) -> int:  # pragma: no cover - needs uvicorn/network
    from .web import run_web

    host = args.host or cfg.web.host
    port = args.port or cfg.web.port
    return run_web(cfg, host=host, port=port)


def _cmd_subscriber(cfg, args) -> int:
    store = Store(cfg.db_path)
    try:
        if args.subcmd == "add":
            sid = store.add_subscriber(Subscriber(
                name=args.name, lat=args.lat, lon=args.lon, radius_km=args.radius,
                ntfy_server=args.ntfy_server, ntfy_topic=args.ntfy_topic,
                ntfy_token_ref=args.token_ref, units=args.units))
            print(f"added subscriber id={sid}")
        elif args.subcmd == "list":
            for s in store.list_subscribers(active_only=False):
                flag = "" if s.active else " (inactive)"
                target = f"-> {s.ntfy_server}/{s.ntfy_topic}" if s.notify_enabled \
                    else "-> (watch-only, no ntfy)"
                print(f"[{s.id}] {s.name}{flag}  {s.lat:.4f},{s.lon:.4f}  r={s.radius_km}km  "
                      f"{target}")
        elif args.subcmd == "deactivate":
            store.set_subscriber_active(args.id, False)
            print(f"deactivated subscriber {args.id}")
        return 0
    finally:
        store.close()


def _cmd_token(cfg, args) -> int:
    """ntfy bearer tokens, stored in the DB and referenced by name from
    subscribers. `set` reads the value from a prompt (tty) or stdin (piped) -
    never from argv, which would land in shell history and `ps`."""
    import re

    store = Store(cfg.db_path)
    try:
        if args.subcmd == "set":
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.name):
                print("token name must be 1-64 chars of letters, digits, _ . -",
                      file=sys.stderr)
                return 2
            if sys.stdin.isatty():
                import getpass
                token = getpass.getpass(f"token for {args.name!r}: ")
            else:
                token = sys.stdin.readline().strip()
            if not token:
                print("empty token; nothing saved", file=sys.stderr)
                return 2
            store.set_ntfy_token(args.name, token)
            print(f"saved token {args.name!r}")
        elif args.subcmd == "list":
            toks = store.list_ntfy_tokens()
            if not toks:
                print("no tokens saved")
            for t in toks:
                print(f"{t['name']}  {t['hint']}  used by {t['refs']} location(s)")
        elif args.subcmd == "delete":
            refs = store.ntfy_token_refs(args.name)
            if refs:
                print(f"token {args.name!r} is used by {refs} location(s); "
                      "point them at another token first", file=sys.stderr)
                return 2
            if not store.delete_ntfy_token(args.name):
                print(f"no token named {args.name!r}", file=sys.stderr)
                return 2
            print(f"deleted token {args.name!r}")
        return 0
    finally:
        store.close()


def _cmd_replay(cfg, args) -> int:
    from windfall.replay import aggregate, replay_messages

    if args.file:
        with open(args.file) as fh:
            raw = json.load(fh)
    elif args.serial:
        raw = _download_serial(args.serial)
        if raw is None:
            return 2
    else:
        print("replay needs --file or --serial", file=sys.stderr)
        return 2
    result = replay_messages(raw, cfg=cfg)
    if result is None:
        print("no parseable frames", file=sys.stderr)
        return 2
    print(aggregate([result]).report())
    if result.final_error_km is not None:
        print(f"final landing error: {result.final_error_km:.2f} km")
    return 0


def _cmd_fetch_corpus(cfg, args) -> int:
    from windfall.history import download_corpus

    lat = lon = None
    if args.near:
        try:
            lat, lon = (float(x) for x in args.near.split(","))
        except ValueError:
            print("--near wants LAT,LON", file=sys.stderr)
            return 2
    paths = download_corpus(
        args.out, duration=args.duration, lat=lat, lon=lon,
        distance_km=args.distance_km if args.near else None,
        limit=args.limit, only_recovered=not args.all)
    print(f"corpus: {len(paths)} flights in {args.out}")
    return 0


def _cmd_fetch_gfs(cfg, args) -> int:
    from windfall.history import fetch_corpus_winds

    return fetch_corpus_winds(cfg, args.dir, model=args.model,
                              margin_km=args.margin_km, pad_hours=args.pad_hours,
                              dry_run=args.dry_run)


def _cmd_backtest(cfg, args) -> int:
    from windfall.hrrr import make_wind_source
    from windfall.replay import aggregate, backtest_corpus, backtest_corpus_parallel

    if args.jobs > 1:
        results = backtest_corpus_parallel(args.dir, cfg=cfg, with_gfs=not args.no_gfs,
                                           limit=args.limit, jobs=args.jobs)
    else:
        gfs = None if args.no_gfs else make_wind_source(cfg)
        results = backtest_corpus(args.dir, cfg=cfg, gfs_source=gfs, limit=args.limit)
    if not results:
        print(f"no flights in {args.dir} (run 'tallyho fetch-corpus' first)", file=sys.stderr)
        return 2
    print(aggregate(results).report())
    print("\nper flight (final prediction error, truth source):")
    for r in sorted(results, key=lambda r: (r.final_error_km is None, r.final_error_km)):
        fe = "-" if r.final_error_km is None else f"{r.final_error_km:.2f} km"
        print(f"  {r.serial:>12}: {fe:>10}  (n={r.n_predictions}, {r.truth_source})")
    return 0


def _cmd_accuracy(cfg, args) -> int:
    from windfall.replay import accuracy_from_store, aggregate

    store = Store(cfg.db_path)
    try:
        results = accuracy_from_store(store, limit=args.limit)
    finally:
        store.close()
    if not results:
        print("no scored landings yet (need a recorded landing with predictions)")
        return 0
    print(aggregate(results).report())
    print("\nper flight (final prediction error):")
    for r in sorted(results, key=lambda r: (r.final_error_km is None, r.final_error_km)):
        fe = "-" if r.final_error_km is None else f"{r.final_error_km:.2f} km"
        print(f"  {r.serial:>12}: {fe:>10}  (n={r.n_predictions})")
    return 0


def _download_serial(serial: str):  # pragma: no cover - needs network
    try:
        import sondehub
    except ImportError:
        print("sondehub not installed; pip install '.[ingest]'", file=sys.stderr)
        return None
    data = sondehub.download(serial=serial)
    return list(data)


def _cmd_health(cfg) -> int:
    """Healthy = the web UI answers /api/health with 200 (which itself checks
    heartbeat freshness, and reports "setup" while the wizard is waiting).
    With the UI disabled, fall back to reading the heartbeat file directly."""
    if cfg.web.enabled:
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{cfg.web.port}/api/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                print(resp.read(500).decode(errors="replace"))
                return 0
        except urllib.error.HTTPError as exc:
            print(f"unhealthy: {exc.read(500).decode(errors='replace')}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"web UI unreachable at {url}: {exc}", file=sys.stderr)
            return 1

    from .web import heartbeat_age

    age = heartbeat_age(cfg)
    if age is None:
        print("no heartbeat", file=sys.stderr)
        return 1
    if age < cfg.health_stale_seconds:
        print(f"ok (last frame {age:.0f}s ago)")
        return 0
    print(f"stale (last frame {age:.0f}s ago)", file=sys.stderr)
    return 1


def _cmd_dem_tiles(cfg) -> int:
    from windfall.dem import tiles_for_bbox
    from .geofence import build_capture_roi

    store = Store(cfg.db_path)
    try:
        subs = store.list_subscribers(active_only=True)
    finally:
        store.close()
    box = build_capture_roi(subs, cfg.roi.capture_margin_km)
    if box is None:
        print("no active subscribers; nothing to prefetch", file=sys.stderr)
        return 2
    for name in tiles_for_bbox(box):
        print(name)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
