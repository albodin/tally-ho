#!/bin/sh
set -eu

if [ "$(id -u)" = "0" ]; then
    PUID="${PUID:-10001}"
    PGID="${PGID:-10001}"
    chown "$PUID:$PGID" /data /dem /gfs /hrrr
    chown -R "$PUID:$PGID" /data
    exec setpriv --reuid "$PUID" --regid "$PGID" --clear-groups tallyho "$@"
fi

exec tallyho "$@"
