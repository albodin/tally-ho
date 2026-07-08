"""Tests for learned priors from flight history."""

from datetime import date

from windfall.climatology import Climatology, _family
from tallyho.store import Store


def _flight_row(serial, burst_alt=None, descent_b=None, lat=45.0, lon=7.0,
                sonde_type="RS41-SGP", day="2026-06-01"):
    return {
        "serial": serial, "launch_day": day, "type": sonde_type,
        "state": "LANDED", "first_seen": None, "last_seen": None,
        "launch_lat": lat, "launch_lon": lon, "burst_alt": burst_alt,
        "descent_b": descent_b, "max_alt": burst_alt,
        "last_lat": lat, "last_lon": lon, "last_alt": 300.0,
    }


def _store_with_history():
    store = Store(":memory:")
    for i, (burst, b) in enumerate(
            [(31000, 5.0), (33000, 5.6), (32000, 5.2), (34000, 6.0), (32500, 5.4)]):
        store.upsert_flight(_flight_row(f"S{i}", burst_alt=burst, descent_b=b))
    # a different site far away - must not contaminate the local prior
    store.upsert_flight(_flight_row("FAR1", burst_alt=20000, lat=52.0, lon=13.0))
    # a different sonde family
    store.upsert_flight(_flight_row("M20A", burst_alt=27000, descent_b=9.0,
                                    sonde_type="M20"))
    return store


def test_site_burst_alt_median():
    clim = Climatology(_store_with_history())
    # median of the five local RS41 bursts (M20 filtered out by family)
    assert clim.burst_alt(45.05, 7.05, "RS41") == 32500
    # far site has only one flight → below min_samples
    assert clim.burst_alt(52.0, 13.0, "RS41") is None
    # unknown position → None
    assert clim.burst_alt(None, None, "RS41") is None


def test_type_descent_b_median():
    clim = Climatology(_store_with_history())
    assert clim.descent_b("RS41-SG") == 5.4      # family match across subtypes
    assert clim.descent_b("M20") is None         # one sample < min_samples
    assert clim.descent_b(None) is None


def test_min_samples_gate():
    store = Store(":memory:")
    store.upsert_flight(_flight_row("A", burst_alt=30000))
    store.upsert_flight(_flight_row("B", burst_alt=31000, day="2026-06-02"))
    clim = Climatology(store, min_samples=3)
    assert clim.burst_alt(45.0, 7.0, "RS41") is None


def test_cache_serves_repeat_lookups():
    store = _store_with_history()
    clim = Climatology(store)
    assert clim.burst_alt(45.0, 7.0, "RS41") == 32500
    store.close()   # a closed store would raise - cached value must be served
    assert clim.burst_alt(45.0, 7.0, "RS41") == 32500


def test_family():
    assert _family("RS41-SGP") == "RS41"
    assert _family("m20") == "M20"
    assert _family(None) is None
