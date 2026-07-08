# tally-ho monolith: ingest + tracker + predictor + notifier.
FROM python:3.14-slim AS base

# System libs for rasterio (DEM) and cfgrib/eccodes (reading GFS GRIB).
# rasterio ships GDAL in its manylinux wheel, but that GDAL still dlopen()s a few
# host libs (libexpat) that python:*-slim omits - without them `import rasterio`
# fails and DEM termination silently degrades to flat ground. eccodes is for cfgrib.
# tzdata gives zoneinfo the IANA database so the `TZ` env var (set in
# docker-compose) resolves to a real zone for human-facing times; slim omits it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libeccodes0 \
        libexpat1 \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY predictor ./predictor
COPY src ./src

# Install the standalone prediction engine (windfall) and the service on top,
# with the live-ingest, DEM, GFS-read and web-UI extras. Listing both lets pip
# resolve tally-ho's `windfall` dependency to the local package.
RUN pip install --no-cache-dir "./predictor[dem,gfs]" ".[ingest,dem,gfs,api]"

# Non-root runtime user.
RUN useradd --create-home --uid 10001 tallyho \
    && mkdir -p /data /dem /gfs /hrrr \
    && chown -R tallyho /data /dem /gfs /hrrr
USER tallyho

ENV TALLYHO_CONFIG=/data/config.toml \
    TALLYHO_DB_PATH=/data/tallyho.db \
    TALLYHO_HEALTH_FILE=/data/heartbeat \
    TALLYHO_DEM_PATH=/dem \
    TALLYHO_GFS_PATH=/gfs \
    TALLYHO_HRRR_PATH=/hrrr \
    TALLYHO_WEB_HOST=0.0.0.0

# The dashboard / onboarding UI is served in-process by `tallyho run`.
EXPOSE 8080

# Healthcheck: last frame must be recent.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["tallyho", "health"]

ENTRYPOINT ["tallyho"]
CMD ["run"]
