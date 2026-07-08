"""windfall - radiosonde descent/landing predictor.

The prediction engine extracted from tally-ho so it can be developed, tuned and
accuracy-tested as independent software. Telemetry parsing, flight tracking
(burst/float detection), measured ascent wind profiles, ballistic descent fit,
GFS 4-D wind fields with measured-wind blending, DEM ground termination, a
Monte Carlo landing ensemble - and the offline replay/backtest/ablation harness
that scores all of it against recovered real flights.

The heavy I/O extras (rasterio, herbie/cfgrib, sondehub) stay behind lazy
imports: the entire core runs and tests offline.
"""

__version__ = "0.1.0"

from .config import Config, load_config            # noqa: F401
from .models import (                              # noqa: F401
    Frame,
    FlightState,
    Landing,
    Prediction,
    PredictionSource,
)
from .predictor import GFSWindSource, Predictor    # noqa: F401
from .tracker import Flight, FlightTracker, TrackerEvent  # noqa: F401
