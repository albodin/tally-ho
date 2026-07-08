"""tally-ho - SondeHub radiosonde landing predictor.

A modular monolith: ingest the SondeHub telemetry stream, track each flight,
predict its landing in real time with measured ascent winds + a ballistic
descent model terminated on real terrain, geofence the prediction against
subscribers, and push lifecycle-aware ntfy alerts.

The prediction engine itself (tracking, wind profiles, descent fit, GFS,
DEM, ensemble, replay/backtest harness) is the standalone ``windfall``
package in ``predictor/`` - tally-ho is the live service wrapped around it.
"""

__version__ = "0.1.0"
