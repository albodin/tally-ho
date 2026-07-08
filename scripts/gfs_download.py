#!/usr/bin/env python3
"""Standalone GFS/HRRR downloader.

Pulls model wind GRIB (U/V + geopotential height + temperature on isobaric
levels) into the shared cache directories using Herbie. The engine's
:class:`windfall.gfs.HerbieGFSSource` / :class:`windfall.hrrr.HerbieHRRRSource`
read what this writes - no SondeHub prediction API, no external predictor
binary.

By default the monolith runs this same logic in-process on timers
(``download_in_process = true``), so you usually don't need this script.
It remains for running the download manually or from a host cron without Docker:

    python scripts/gfs_download.py [--config CONFIG] [--fxx 0 3 6]
    python scripts/gfs_download.py --model hrrr

For backtests, ``--date`` pins a historical cycle (NOAA's AWS archive keeps
GFS back to 2021, HRRR back to 2014) and ``--bbox`` overrides the
subscriber-derived region (GFS only - HRRR files are full-CONUS, windowed at
read time):

    python scripts/gfs_download.py --date 2026-06-01T06 --bbox 45,15,50,22
    python scripts/gfs_download.py --model hrrr --date 2026-06-01T06
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tallyho.config import load_config            # noqa: E402
from windfall.geo import BBox                       # noqa: E402
from tallyho.geofence import build_capture_roi    # noqa: E402
from windfall.gfs import download_gfs_cycle        # noqa: E402
from windfall.hrrr import download_hrrr_cycle      # noqa: E402
from tallyho.store import Store                    # noqa: E402

log = logging.getLogger("gfs_download")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="download model wind GRIB for tally-ho")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=("gfs", "hrrr"), default="gfs")
    ap.add_argument("--fxx", type=int, nargs="*", default=None,
                    help="forecast hours to fetch (default: from config)")
    ap.add_argument("--levels", default=None,
                    help="GRIB search regex (default: from config)")
    ap.add_argument("--date", default=None,
                    help="cycle run time (ISO, UTC; e.g. 2026-06-01T06) instead of latest")
    ap.add_argument("--bbox", default=None, metavar="MINLAT,MINLON,MAXLAT,MAXLON",
                    help="region override instead of the subscriber capture ROI")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    run_date = None
    if args.date:
        # Herbie wants naive-UTC datetimes (HerbieLatest().date is naive too).
        run_date = datetime.fromisoformat(args.date)
        if run_date.tzinfo is not None:
            run_date = run_date.astimezone(timezone.utc).replace(tzinfo=None)

    cfg = load_config(args.config)
    if args.model == "hrrr":
        paths = download_hrrr_cycle(cfg, fxx=args.fxx, levels=args.levels,
                                    run_date=run_date)
        log.info("downloaded %d HRRR file(s)", len(paths))
        return 0
    if args.bbox:
        box = BBox(*(float(x) for x in args.bbox.split(",")))
    else:
        store = Store(cfg.db_path)
        try:
            subs = store.list_subscribers(active_only=True)
        finally:
            store.close()
        box = build_capture_roi(subs, cfg.roi.capture_margin_km)
    paths = download_gfs_cycle(cfg, box, fxx=args.fxx, levels=args.levels, run_date=run_date)
    log.info("downloaded %d GFS file(s)", len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
