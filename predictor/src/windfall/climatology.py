"""Learned priors from this receiver's own flight history.

Launch sites are highly consistent - same balloon batch, same fill, same
payload, twice daily - so the median *observed* burst altitude of past flights
launched nearby beats the static per-type table substantially (burst altitude
is the dominant pre-burst error after wind). Likewise the median fitted
ballistic constant of past descents of the same sonde family beats the generic
``default_b``.

Both queries are tiny SQLite scans, cached briefly so the per-sweep predictor
doesn't hammer the store.
"""

from __future__ import annotations

import time
from typing import Protocol


class ClimatologyStore(Protocol):
    """What climatology needs from a flight-history store (duck-typed)."""

    def site_burst_alts(self, lat: float, lon: float, *, box_deg: float,
                        sonde_type: str | None) -> list[float]: ...
    def type_descent_bs(self, family: str) -> list[float]: ...


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2]


class Climatology:
    def __init__(
        self,
        store: ClimatologyStore,
        min_samples: int = 3,
        ttl_seconds: float = 3600.0,
        site_box_deg: float = 0.5,
    ):
        self.store = store
        self.min_samples = min_samples
        self.ttl = ttl_seconds
        self.site_box_deg = site_box_deg
        self._cache: dict[tuple, tuple[float, float | None]] = {}  # key -> (at, value)

    def _cached(self, key: tuple, compute) -> float | None:
        hit = self._cache.get(key)
        now = time.monotonic()
        if hit is not None and now - hit[0] < self.ttl:
            return hit[1]
        value = compute()
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = (now, value)
        return value

    def burst_alt(self, lat: float | None, lon: float | None,
                  sonde_type: str | None = None) -> float | None:
        """Median observed burst altitude of past flights launched near
        (lat, lon), or None below ``min_samples``."""
        if lat is None or lon is None:
            return None
        key = ("burst", round(lat, 1), round(lon, 1), _family(sonde_type))

        def compute() -> float | None:
            alts = self.store.site_burst_alts(
                lat, lon, box_deg=self.site_box_deg, sonde_type=sonde_type)
            if len(alts) < self.min_samples:
                return None
            return _median(alts)

        return self._cached(key, compute)

    def descent_b(self, sonde_type: str | None) -> float | None:
        """Median fitted ballistic constant of past flights of this sonde
        family, or None below ``min_samples``."""
        fam = _family(sonde_type)
        if fam is None:
            return None

        def compute() -> float | None:
            bs = self.store.type_descent_bs(fam)
            if len(bs) < self.min_samples:
                return None
            return _median(bs)

        return self._cached(("b", fam), compute)


def _family(sonde_type: str | None) -> str | None:
    """Sonde family key: 'RS41-SGP' and 'RS41-SG' share chute behaviour."""
    if not sonde_type:
        return None
    return sonde_type.upper().split("-")[0]
