"""Command-line interface of the standalone predictor.

    windfall replay --file frames.json      # accuracy harness on archived frames
    windfall replay --serial S1234567       # download from SondeHub, then replay
    windfall fetch-corpus --near 47.5,19.0 --duration 1m  # recovered flights -> corpus
    windfall fetch-gfs                      # historical model cycles the corpus needs
    windfall slim-corpus                    # strip frames to engine fields (10-20x smaller)
    windfall backtest                       # replay the corpus vs recovered truth
    windfall ablate                         # backtest per wind-assembly variant

Config comes from ``--config`` TOML + ``WINDFALL_*`` env overrides. The
DEM/GFS caches are plain directories (see ``dem.path`` / ``gfs.path``) shared
with - or independent of - a tally-ho deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="windfall", description="radiosonde landing predictor + accuracy harness")
    parser.add_argument("--config", default=os.environ.get("WINDFALL_CONFIG"),
                        help="path to config TOML")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("replay", help="run the accuracy harness on one flight")
    rp.add_argument("--file", help="JSON file: a list of raw telemetry dicts")
    rp.add_argument("--serial", help="download this serial's archive from SondeHub")

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
    _fetch_gfs_args(fg)

    sc = sub.add_parser("slim-corpus",
                        help="strip corpus frames to the fields the engine parses "
                             "(10-20x smaller files; lossless to backtest scores)")
    sc.add_argument("--dir", default="data/corpus", help="corpus directory")

    bt = sub.add_parser("backtest",
                        help="replay a fetched corpus against recovered ground truth")
    _backtest_args(bt)
    bt.add_argument("--no-gfs", action="store_true",
                    help="measured-winds only (skip the GFS/HRRR caches)")

    ab = sub.add_parser("ablate",
                        help="backtest once per wind-assembly variant and compare")
    _backtest_args(ab)
    ab.add_argument("--modes", default=None,
                    help="comma-separated subset of: %s" % ",".join(_mode_names()))

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(
        level=(args.log_level or cfg.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.cmd == "replay":
        return _cmd_replay(cfg, args)
    if args.cmd == "fetch-corpus":
        return _cmd_fetch_corpus(cfg, args)
    if args.cmd == "fetch-gfs":
        return _cmd_fetch_gfs(cfg, args)
    if args.cmd == "slim-corpus":
        return _cmd_slim_corpus(cfg, args)
    if args.cmd == "backtest":
        return _cmd_backtest(cfg, args)
    if args.cmd == "ablate":
        return _cmd_ablate(cfg, args)
    return 1


def _backtest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dir", default="data/corpus", help="corpus directory")
    p.add_argument("--limit", type=int, default=None, help="replay at most this many flights")
    p.add_argument("--dem", action="store_true",
                   help="terminate descents on the DEM at dem.path (default: flat ground)")
    p.add_argument("--jobs", type=int, default=1,
                   help="worker processes - flights replay in parallel (default 1)")


def _fetch_gfs_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dir", default="data/corpus", help="corpus directory")
    p.add_argument("--model", choices=("gfs", "hrrr", "both"), default="gfs",
                   help="which model cache(s) to fill (HRRR: CONUS flights only)")
    p.add_argument("--margin-km", type=float, default=75.0,
                   help="region margin around each flight's track")
    p.add_argument("--pad-hours", type=float, default=2.0,
                   help="forecast-hour headroom past each flight's last frame")
    p.add_argument("--dry-run", action="store_true",
                   help="print the cycle plan without downloading")


def _mode_names() -> list[str]:
    from .replay import ABLATION_MODES
    return [m.name for m in ABLATION_MODES]


def _ground_fn(cfg, args):
    if not getattr(args, "dem", False):
        return None
    from .dem import make_ground_model
    return make_ground_model(cfg.dem)


def _cmd_replay(cfg, args) -> int:
    from .replay import aggregate, replay_messages

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
    from .history import download_corpus

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
    from .history import fetch_corpus_winds

    return fetch_corpus_winds(cfg, args.dir, model=args.model,
                              margin_km=args.margin_km, pad_hours=args.pad_hours,
                              dry_run=args.dry_run)


def _cmd_slim_corpus(cfg, args) -> int:
    from .history import slim_corpus

    n, before, after = slim_corpus(args.dir)
    if n == 0:
        print(f"no flights in {args.dir}", file=sys.stderr)
        return 2
    print(f"slimmed {n} flight(s): {before / 1e9:.1f} -> {after / 1e9:.1f} GB")
    return 0


def _cmd_backtest(cfg, args) -> int:
    from .hrrr import make_wind_source
    from .replay import aggregate, backtest_corpus, backtest_corpus_parallel

    if args.jobs > 1:
        results = backtest_corpus_parallel(args.dir, cfg=cfg, with_dem=args.dem,
                                           with_gfs=not args.no_gfs,
                                           limit=args.limit, jobs=args.jobs)
    else:
        gfs = None if args.no_gfs else make_wind_source(cfg)
        results = backtest_corpus(args.dir, cfg=cfg, ground_fn=_ground_fn(cfg, args),
                                  gfs_source=gfs, limit=args.limit)
    if not results:
        print(f"no flights in {args.dir} (run 'windfall fetch-corpus' first)", file=sys.stderr)
        return 2
    print(aggregate(results).report())
    print("\nper flight (final prediction error, truth source):")
    for r in sorted(results, key=lambda r: (r.final_error_km is None, r.final_error_km)):
        fe = "-" if r.final_error_km is None else f"{r.final_error_km:.2f} km"
        print(f"  {r.serial:>12}: {fe:>10}  (n={r.n_predictions}, {r.truth_source})")
    return 0


def _cmd_ablate(cfg, args) -> int:
    from .replay import ablation_report, run_ablation

    modes = args.modes.split(",") if args.modes else None
    if args.jobs > 1:
        outcomes = run_ablation(args.dir, cfg=cfg, modes=modes, limit=args.limit,
                                jobs=args.jobs, with_dem=args.dem)
    else:
        outcomes = run_ablation(args.dir, cfg=cfg, ground_fn=_ground_fn(cfg, args),
                                modes=modes, limit=args.limit)
    if outcomes is None:
        print(f"no flights in {args.dir} (run 'windfall fetch-corpus' first)", file=sys.stderr)
        return 2
    print(ablation_report(outcomes))
    return 0


def _download_serial(serial: str):  # pragma: no cover - needs network
    try:
        import sondehub
    except ImportError:
        print("sondehub not installed; pip install 'windfall[archive]'", file=sys.stderr)
        return None
    data = sondehub.download(serial=serial)
    return list(data)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
