# windfall

A radiosonde **descent/landing predictor** - the prediction engine of
[tally-ho](../README.md) as a standalone package.

## Install

```sh
pip install -e .                     # core - offline, numpy only
pip install -e ".[dem,gfs,archive]"  # + DEM terrain, GFS/HRRR winds, archive download
pip install -e ".[dev]"              # + pytest
```

## Usage

```sh
windfall fetch-corpus --near 40.6,-111.9 --distance-km 300 --duration 3m
                                   # ground-truthed flights from SondeHub recoveries
windfall fetch-gfs                 # the archived GFS/HRRR cycles the corpus needs
windfall backtest                  # replay the corpus through the production predictor
windfall ablate                    # score each wind-assembly variant separately
windfall replay --serial V1234567  # replay a single flight
```

Configuration is TOML plus `WINDFALL_<SECTION>_<KEY>` env overrides; every
knob is documented in [`src/windfall/config.py`](src/windfall/config.py).

## Tests

```sh
python -m pytest          # from predictor/ - offline, no network/GRIB/DEM data
```
