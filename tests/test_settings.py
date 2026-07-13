"""The reflective settings schema (settings_meta) + the web settings editor.

The schema half is pure stdlib and runs offline; the endpoint half is skipped
without the `api` extra, like the rest of the web tests.
"""

from dataclasses import fields, is_dataclass

import pytest

from tallyho import settings_meta
from tallyho.config import Config
from tallyho.settings_meta import (RESTART_REQUIRED, apply_values, coerce,
                                   describe, dotted_keys, field_specs,
                                   restart_required_in, validate_update)


def _leaves():
    """(section|None, key) -> default for every Config knob (nesting is one
    level deep by construction)."""
    cfg = Config()
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if is_dataclass(v):
            out.update({(f.name, g.name): getattr(v, g.name) for g in fields(v)})
        else:
            out[(None, f.name)] = v
    return out


def _spec(dotted):
    return next(s for s in field_specs() if s.dotted == dotted)


# ---- schema completeness ----------------------------------------------------
def test_every_knob_has_a_spec_and_nothing_else():
    assert {(s.section, s.key) for s in field_specs()} == set(_leaves())


def test_spec_defaults_and_kinds_match_the_dataclasses():
    known = {"str", "float", "int", "bool", "enum", "int_list", "opt_int", "color"}
    leaves = _leaves()
    for s in field_specs():
        assert s.kind in known, s.dotted
        assert s.default == leaves[(s.section, s.key)], s.dotted
        if s.kind == "enum":
            assert s.default in s.choices, s.dotted


def test_restart_required_names_real_knobs():
    """A renamed/removed knob must not linger in RESTART_REQUIRED unnoticed."""
    assert RESTART_REQUIRED <= set(_leaves())


def test_unmapped_default_type_raises():
    with pytest.raises(TypeError, match="unmapped type"):
        settings_meta._kind_of("x", "y", object())


# ---- help text mined from the template ---------------------------------------
def test_inline_help():
    assert "alt below running max" in _spec("tracker.burst_drop_m").help


def test_continuation_lines_join():
    assert "68%" in _spec("uncertainty.radius_scale").help


def test_prose_block_attaches_to_next_field():
    assert "radio horizon" in _spec("tracker.descent_lost_timeout_seconds").help


def test_value_containing_hash_is_not_split():
    """map_url_template's value embeds '#map=...' - splitting on the first '#'
    would truncate the URL into bogus help text."""
    assert "map=" not in _spec("notify.map_url_template").help


def test_section_help():
    _, section_help, _ = settings_meta._template_info()
    assert "Monte Carlo" in section_help["ensemble"]
    assert "CONUS" in section_help["hrrr"]     # prose block after the header


# ---- env-override detection ---------------------------------------------------
def test_env_override_flags_exactly_the_set_vars(monkeypatch):
    monkeypatch.setenv("TALLYHO_GFS_ENABLED", "1")
    monkeypatch.setenv("TALLYHO_DB_PATH", "/data/tallyho.db")
    flagged = {f"{sec['name']}.{f['key']}" if sec["name"] else f["key"]
               for sec in describe(Config())["sections"] for f in sec["fields"]
               if f["env_overridden"]}
    assert flagged == {"gfs.enabled", "db_path"}


# ---- coercion -------------------------------------------------------------------
@pytest.mark.parametrize("dotted,value,expected", [
    ("tracker.burst_drop_m", 350, 350.0),          # int into a float knob
    ("tracker.burst_consecutive", 4.0, 4),         # JS integral float into int
    ("ensemble.enabled", False, False),
    ("ensemble.seed", None, None),
    ("ensemble.seed", 123, 123),
    ("profile.correction_mode", "bias", "bias"),
    ("gfs.download_fxx", [0, 6.0, 12], [0, 6, 12]),
    ("display_tz", "Europe/Rome", "Europe/Rome"),
    ("dem.download_workers", 16, 16),              # at the range bound
    ("dem.download_check_seconds", 30, 30.0),
    ("colors.track", "#5AD1C8", "#5ad1c8"),        # normalized to lowercase
    ("colors.landing", "#7bd88f", "#7bd88f"),
    ("colors.track_opacity", 0.5, 0.5),            # opacities are plain floats
    ("colors.watch_fill_opacity", 1, 1.0),         # at the range bound
])
def test_coerce_accepts(dotted, value, expected):
    assert coerce(_spec(dotted), value) == expected


@pytest.mark.parametrize("dotted,value", [
    ("tracker.burst_consecutive", 1.5),     # fractional into int
    ("tracker.burst_consecutive", True),    # bool is not a number
    ("ensemble.enabled", "true"),           # string is not a bool
    ("tracker.burst_drop_m", "300"),
    ("profile.correction_mode", "magic"),   # not in choices
    ("gfs.download_fxx", ["a"]),
    ("gfs.download_fxx", 3),
    ("ensemble.seed", 1.5),
    ("display_tz", 3),
    ("dem.download_workers", 0),            # below the range
    ("dem.download_workers", 64),           # above the range (runtime clamps at 16)
    ("dem.download_check_seconds", 5),      # below the loop's 30 s floor
    ("colors.track", "red"),                # named colors aren't accepted
    ("colors.track", "#abc"),               # 3-digit hex breaks the JS alpha suffixing
    ("colors.track", "#12345g"),            # not hex
    ("colors.track", 0xffffff),             # not a string
    ("colors.track_opacity", 1.5),          # opacity outside [0, 1]
    ("colors.track_opacity", -0.1),
    ("colors.track_opacity", "0.5"),        # string is not a number
])
def test_coerce_rejects(dotted, value):
    with pytest.raises(ValueError):
        coerce(_spec(dotted), value)


def test_ranges_name_real_numeric_knobs_and_ship_in_schema():
    """A renamed knob must not linger in _RANGES; the UI gets min/max."""
    assert set(settings_meta._RANGES) <= set(_leaves())
    for sk in settings_meta._RANGES:
        assert _spec(f"{sk[0]}.{sk[1]}" if sk[0] else sk[1]).kind in ("int", "float")
    fields = {f"{sec['name']}.{f['key']}" if sec["name"] else f["key"]: f
              for sec in describe(Config())["sections"] for f in sec["fields"]}
    assert fields["dem.download_workers"]["min"] == 1
    assert fields["dem.download_workers"]["max"] == 16
    assert fields["dem.download_check_seconds"]["max"] is None


# ---- validate_update / apply_values ---------------------------------------------
def test_validate_update_diffs_against_live_config():
    cfg = Config()
    changed, errors = validate_update(cfg, {
        "tracker.burst_drop_m": 350.0,       # a change
        "tracker.burst_consecutive": 3,      # already the default -> dropped
        "log_level": "DEBUG",                # top-level change
    })
    assert errors == {}
    assert changed == {"tracker": {"burst_drop_m": 350.0}, "log_level": "DEBUG"}
    assert dotted_keys(changed) == ["log_level", "tracker.burst_drop_m"]


def test_validate_update_collects_errors_per_field():
    changed, errors = validate_update(Config(), {
        "tracker.nope": 1,
        "nonsense": 2,
        "tracker.burst_consecutive": 1.5,
        "tracker.burst_drop_m": 350.0,       # the valid one still lands
    })
    assert set(errors) == {"tracker.nope", "nonsense", "tracker.burst_consecutive"}
    assert changed == {"tracker": {"burst_drop_m": 350.0}}


def test_validate_update_rejects_env_overridden(monkeypatch):
    monkeypatch.setenv("TALLYHO_GFS_ENABLED", "1")
    changed, errors = validate_update(Config(), {"gfs.enabled": True})
    assert changed == {}
    assert "TALLYHO_GFS_ENABLED" in errors["gfs.enabled"]


def test_apply_values_mutates_in_place():
    cfg = Config()
    tracker = cfg.tracker    # components hold sub-configs by reference too
    apply_values(cfg, {"tracker": {"burst_drop_m": 350.0}, "log_level": "DEBUG"})
    assert tracker.burst_drop_m == 350.0
    assert cfg.log_level == "DEBUG"


def test_restart_required_in_flags_only_sticky_keys():
    changed = {"web": {"port": 9090}, "tracker": {"burst_drop_m": 350.0},
               "db_path": "x.db", "tick_seconds": 20.0}
    assert restart_required_in(changed) == ["db_path", "web.port"]


# =============================================================================
# The web endpoints (skipped without the `api` extra, like test_web.py)
# =============================================================================
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

import tallyho.web as web_module  # noqa: E402
from tallyho.auth import hash_password  # noqa: E402
from tallyho.config import load_config  # noqa: E402
from tallyho.setup import seed_config  # noqa: E402
from tallyho.store import Store  # noqa: E402
from tallyho.web import create_app  # noqa: E402

PASSWORD = "correct-horse-battery"
PASSWORD_HASH = hash_password(PASSWORD)   # scrypt is slow; hash once per module


@pytest.fixture
def client(tmp_path):
    store = Store(":memory:")
    path = tmp_path / "config.toml"
    seed_config(path)
    cfg = Config()
    c = TestClient(create_app(cfg, store, config_path=path))
    c.store, c.cfg, c.config_path = store, cfg, path
    store.add_user("admin", PASSWORD_HASH)
    r = c.post("/api/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 200
    yield c
    store.close()


def test_settings_require_auth(client):
    fresh = TestClient(client.app)
    assert fresh.get("/api/settings").status_code == 401
    assert fresh.put("/api/settings", json={"values": {}}).status_code == 401
    assert fresh.post("/api/restart").status_code == 401
    r = fresh.get("/settings", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/login"


def test_settings_page_served(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store"


def test_get_settings_ships_the_full_schema(client):
    body = client.get("/api/settings").json()
    assert body["writable"] is True
    assert body["pending_restart"] == []
    got = {f"{s['name']}.{f['key']}" if s["name"] else f["key"]
           for s in body["sections"] for f in s["fields"]}
    want = {f"{sec}.{key}" if sec else key for sec, key in _leaves()}
    assert got == want
    fields_by = {f["key"]: f for s in body["sections"] if s["name"] == "web"
                 for f in s["fields"]}
    assert fields_by["port"]["value"] == 8080
    assert fields_by["port"]["restart_required"] is True


def test_put_settings_writes_file_and_hot_applies(client):
    r = client.put("/api/settings", json={"values": {
        "tracker.burst_drop_m": 350.0, "gfs.keep_hours": 48.0}})
    assert r.status_code == 200
    body = r.json()
    assert body["changed"] == ["gfs.keep_hours", "tracker.burst_drop_m"]
    assert body["restart_required"] == []
    # hot-applied to the very object the daemon's components share
    assert client.cfg.tracker.burst_drop_m == 350.0
    # persisted, comments intact (same guarantee test_setup checks for the wizard)
    assert load_config(client.config_path).tracker.burst_drop_m == 350.0
    text = client.config_path.read_text()
    assert "burst_drop_m = 350.0" in text
    assert "# download_cadence_hours = 6.0" in text


def test_put_settings_validation_errors_are_per_field(client):
    r = client.put("/api/settings", json={"values": {
        "tracker.nope": 1, "tracker.burst_drop_m": "high"}})
    assert r.status_code == 422
    errors = r.json()["detail"]["errors"]
    assert set(errors) == {"tracker.nope", "tracker.burst_drop_m"}
    # nothing was written or applied
    assert client.cfg.tracker.burst_drop_m == 300.0
    assert "burst_drop_m = " not in [
        line.split("#")[0].strip() for line in client.config_path.read_text().splitlines()]


def test_put_settings_rejects_env_overridden(client, monkeypatch):
    monkeypatch.setenv("TALLYHO_GFS_ENABLED", "1")
    r = client.put("/api/settings", json={"values": {"gfs.enabled": True}})
    assert r.status_code == 422
    assert "TALLYHO_GFS_ENABLED" in r.json()["detail"]["errors"]["gfs.enabled"]


def test_put_settings_tracks_pending_restart(client):
    r = client.put("/api/settings", json={"values": {
        "web.port": 9090, "tracker.burst_drop_m": 350.0}})
    body = r.json()
    assert body["restart_required"] == ["web.port"]
    assert body["pending_restart"] == ["web.port"]
    # a later hot-only save keeps the pending flag
    r = client.put("/api/settings", json={"values": {"tick_seconds": 20.0}})
    assert r.json()["restart_required"] == []
    assert r.json()["pending_restart"] == ["web.port"]
    assert client.get("/api/settings").json()["pending_restart"] == ["web.port"]


def test_put_settings_noop_touches_nothing(client):
    before = client.config_path.read_text()
    r = client.put("/api/settings", json={"values": {"tracker.burst_drop_m": 300.0}})
    assert r.status_code == 200
    assert r.json()["changed"] == []
    assert client.config_path.read_text() == before


def test_put_settings_color_roundtrip(client):
    """A map-color edit hot-applies, persists, and reaches the dashboard's
    bootstrap endpoint (which is where the map actually reads it from)."""
    r = client.put("/api/settings", json={"values": {"colors.track": "#FF0000"}})
    assert r.status_code == 200
    assert r.json()["changed"] == ["colors.track"]
    assert r.json()["restart_required"] == []
    assert client.cfg.colors.track == "#ff0000"    # normalized lowercase
    assert client.get("/api/config").json()["colors"]["track"] == "#ff0000"
    assert load_config(client.config_path).colors.track == "#ff0000"


def test_put_settings_seed_roundtrip(client):
    client.put("/api/settings", json={"values": {"ensemble.seed": 123}})
    assert load_config(client.config_path).ensemble.seed == 123
    client.put("/api/settings", json={"values": {"ensemble.seed": None}})
    assert client.cfg.ensemble.seed is None
    assert load_config(client.config_path).ensemble.seed is None


def test_settings_read_only_without_config_path():
    store = Store(":memory:")
    try:
        c = TestClient(create_app(Config(), store))   # no config_path (tests/embeds)
        store.add_user("admin", PASSWORD_HASH)
        c.post("/api/login", json={"username": "admin", "password": PASSWORD})
        body = c.get("/api/settings").json()
        assert body["writable"] is False
        r = c.put("/api/settings", json={"values": {"tick_seconds": 20.0}})
        assert r.status_code == 503
    finally:
        store.close()


def test_restart_endpoint_fires_the_trigger(client, monkeypatch):
    fired = []
    monkeypatch.setattr(web_module, "_trigger_restart", lambda: fired.append(True))
    r = client.post("/api/restart")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert fired == [True]


def test_trigger_restart_sends_sigint(monkeypatch):
    """The real trigger must deliver SIGINT to this process (delayed) - patch
    os.kill so the suite doesn't shut itself down."""
    import os
    import signal

    calls = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))
    web_module._trigger_restart(delay=0.0)
    for _ in range(100):
        if calls:
            break
        import time
        time.sleep(0.01)
    assert calls == [(os.getpid(), signal.SIGINT)]
