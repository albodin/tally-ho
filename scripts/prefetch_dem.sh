#!/usr/bin/env bash
# Prefetch Copernicus GLO-30 DEM tiles for the current capture ROI into ./dem.
#
# NORMALLY NOT NEEDED: while `[dem] enabled` and `download_in_process` are true
# (the defaults), the app's own timer thread downloads these exact tiles and
# hot-reloads the ground model. This script is the manual alternative for
# deployments that opt out (`TALLYHO_DEM_DOWNLOAD_IN_PROCESS=false`, e.g. to
# keep /dem a read-only mount) - it asks the running container which tiles the
# ROI covers (the ROI is built from your active subscribers, so add a watched
# location first), then pulls each from the AWS Open Data mirror into ./dem.
#
# Terrain is static, so this is a ONE-SHOT: run it once after onboarding, and
# again only if you add a location that extends the ROI into new tiles.
# DEM only affects the landing *elevation* at termination; with no tiles the app
# falls back to flat sea level. 404s on ocean-only squares are expected - those
# tiles don't exist and correctly fall back to sea level.
#
# Usage:  scripts/prefetch_dem.sh [DEST_DIR]   (default: ./dem)
set -euo pipefail

DEST="${1:-./dem}"
BUCKET="https://copernicus-dem-30m.s3.amazonaws.com"
mkdir -p "$DEST"

docker compose exec -T tallyho tallyho dem-tiles | while read -r tile; do
  [ -z "$tile" ] && continue
  out="$DEST/$tile.tif"
  if [ -f "$out" ]; then
    echo "have  $tile"
    continue
  fi
  echo "fetch $tile"
  if ! curl -fsSL "$BUCKET/$tile/$tile.tif" -o "$out"; then
    echo "skip  $tile (ocean-only tile - no DEM; falls back to sea level)"
    rm -f "$out"
  fi
done
