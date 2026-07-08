# tally-ho

[![CI](https://github.com/albodin/tally-ho/actions/workflows/ci.yml/badge.svg)](https://github.com/albodin/tally-ho/actions/workflows/ci.yml)

Self-hosted service that consumes the live **SondeHub** radiosonde telemetry
stream, predicts each sonde's landing location in real time, and pushes an
**ntfy** notification when a predicted (or confirmed) landing falls within a
subscriber's radius. Zero load on SondeHub's prediction API by design - it
consumes only the telemetry stream and predicts itself.

```
SondeHub MQTT ─► ingest ─► flight tracker ─► predictor ─► geofence + notifier ─► ntfy ─► friends
 (pysondehub)      │            │                │                  │
                   └────────────┴────────────────┴──────────────────┘
                                  SQLite (state, subscribers, dedup, predictions)
        NOAA GFS (Herbie) ───────────────────────────────────► fallback wind source only
```

The prediction engine - telemetry parsing through the Monte Carlo landing
ensemble, plus its replay/backtest/ablation harness - is the standalone
[`windfall`](predictor/README.md) package in [`predictor/`](predictor/).
tally-ho is the live wrapper: MQTT ingest, SQLite, geofence, ntfy, web
dashboard.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./predictor      # the prediction engine (`windfall`)
pip install -e ".[dev]"         # the service + tests; add ingest,dem,gfs,api as needed
pytest                          # offline; engine tests: (cd predictor && pytest)

tallyho subscriber add --name alice --lat 45.0 --lon 7.0 --radius 30 \
    --ntfy-server https://ntfy.sh --ntfy-topic alice-sondes --token-ref NTFY_ALICE
tallyho run                     # dashboard at http://127.0.0.1:8080 (api extra)
```

| Extra | Enables | Heavy deps |
|-------|---------|-----------|
| `ingest` | live SondeHub MQTT stream + archive download | `sondehub` |
| `dem` | Copernicus GLO-30 terrain termination | `rasterio` |
| `gfs` | GFS fallback wind source | `herbie-data`, `cfgrib`, `xarray` |
| `api` | `tallyho web` dashboard + onboarding UI | `fastapi`, `uvicorn` |

## Docker

```bash
cp .env.example .env                # host wiring: PUID/PGID, timezone, port, paths
cp secrets.env.example secrets.env  # ntfy tokens
docker compose up -d --build        # or the published image: uncomment TALLYHO_IMAGE
                                    #   in .env (ghcr.io/albodin/tally-ho:latest)
```

Open `http://<host>:8080` - the first-run wizard creates your account, picks
the wind/terrain sources, and starts the pipeline. The GFS/HRRR and DEM caches
fill themselves in-process. Volumes: `data/` (SQLite + `config.toml`), `dem/`,
`gfs/`, `hrrr/`.

Sessions travel plain HTTP, so a LAN bind is fine as-is; put a TLS reverse
proxy in front for anything wider.

## Configuration

`data/config.toml` is seeded on first run with every setting commented out at
its built-in default - uncomment a line to change it; every setting is
documented in the file itself. Any value can also be set with a
`TALLYHO_<SECTION>_<KEY>` env var, which beats the file.

Secrets never live in config or the DB: a subscriber stores only the *name* of
the environment variable holding its ntfy token (e.g. `NTFY_ALICE`), read at
send time.

## Accuracy

```bash
tallyho accuracy                # score saved predictions against observed landings
tallyho fetch-corpus --near 47.5,19.3 --distance-km 300 --duration 1m
tallyho fetch-gfs               # the archived model winds those flights need
tallyho backtest                # replay recovered flights through the production predictor
```

The harness - replay, backtest, ablation - is documented in
[`predictor/README.md`](predictor/README.md).

## License

[GNU AGPL v3](LICENSE) (or any later version) - covers both tally-ho and the
`windfall` engine. If you run a modified version as a network service, the
AGPL requires offering its users the modified source.
