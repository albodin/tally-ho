"""First-run setup: config seeding, comment-preserving writes, the wizard app,
and the `tallyho run` gate."""

import pytest

from tallyho.config import Config, load_config
from tallyho.setup import ensure_setup, seed_config, template_text, write_config_values
from tallyho.store import Store


# ---- seeding ---------------------------------------------------------------
def test_seed_config_writes_template_once(tmp_path):
    path = tmp_path / "data" / "config.toml"
    assert seed_config(path) is True
    assert path.read_text() == template_text()

    path.write_text("# mine now\n")
    assert seed_config(path) is False   # never touches an existing file
    assert path.read_text() == "# mine now\n"


def test_seeded_template_loads_as_pure_defaults(tmp_path):
    path = tmp_path / "config.toml"
    seed_config(path)
    assert load_config(path) == load_config(tmp_path / "nonexistent.toml")


# ---- comment-preserving writes ----------------------------------------------
def test_write_config_values_preserves_comments(tmp_path):
    pytest.importorskip("tomlkit")
    path = tmp_path / "config.toml"
    seed_config(path)

    write_config_values(path, {
        "gfs": {"enabled": True, "keep_hours": 48.0},
        "dem": {"enabled": False},
        "log_level": "DEBUG",
    })
    cfg = load_config(path)
    assert cfg.gfs.enabled is True and cfg.gfs.keep_hours == 48.0
    assert cfg.dem.enabled is False
    assert cfg.log_level == "DEBUG"
    # the commented reference lines are still there for later hand-editing
    text = path.read_text()
    assert "# burst_drop_m = 300.0" in text
    assert "# download_cadence_hours = 6.0" in text


# ---- the wizard app ----------------------------------------------------------
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from tallyho.setup import create_setup_app  # noqa: E402


def _payload(**over):
    body = {"username": "admin", "password": "hunter2hunter2",
            "gfs_enabled": True, "hrrr_enabled": False,
            "dem_enabled": True, "gfs_keep_hours": 48.0}
    body.update(over)
    return body


@pytest.fixture
def wizard(tmp_path):
    path = tmp_path / "config.toml"
    seed_config(path)
    store = Store(":memory:")
    completed = []
    app = create_setup_app(Config(), store, path,
                           on_complete=lambda: completed.append(True))
    c = TestClient(app)
    c.store, c.config_path, c.completed = store, path, completed
    yield c
    store.close()


def test_wizard_health_reports_setup(wizard):
    r = wizard.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "setup"}


def test_wizard_serves_page_everywhere(wizard):
    r = wizard.get("/")
    assert "setup" in r.text.lower()
    # no-store: after the handover "/" serves the dashboard, and a heuristically
    # cached wizard page would keep showing until a forced reload
    assert r.headers["cache-control"] == "no-store"
    r = wizard.get("/anything/else", follow_redirects=False)
    assert r.status_code == 307 or r.status_code == 302 or r.status_code == 200


def test_wizard_serves_static_assets(wizard):
    """The wizard page links the shared stylesheet; the /static mount must win
    over the catch-all redirect."""
    r = wizard.get("/static/theme.css", follow_redirects=False)
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]


def test_wizard_completes_setup(wizard):
    r = wizard.post("/api/setup", json=_payload(hrrr_enabled=True))
    assert r.status_code == 200
    assert wizard.completed == [True]
    assert wizard.store.count_users() == 1

    cfg = load_config(wizard.config_path)
    assert cfg.gfs.enabled is True and cfg.gfs.keep_hours == 48.0
    assert cfg.hrrr.enabled is True and cfg.dem.enabled is True
    # the wizard signed the browser in for the handover to the dashboard
    assert wizard.cookies.get("session")


def test_wizard_refuses_second_run(wizard):
    assert wizard.post("/api/setup", json=_payload()).status_code == 200
    r = wizard.post("/api/setup", json=_payload(username="intruder"))
    assert r.status_code == 409
    assert wizard.store.count_users() == 1


@pytest.mark.parametrize("bad", [
    {"password": "short"},          # under the minimum length
    {"username": ""},
    {"gfs_keep_hours": -1},
])
def test_wizard_validates_input(wizard, bad):
    assert wizard.post("/api/setup", json=_payload(**bad)).status_code == 422
    assert wizard.store.count_users() == 0


# ---- the run gate -------------------------------------------------------------
def test_ensure_setup_ready_when_account_exists(tmp_path):
    db = tmp_path / "t.db"
    s = Store(db)
    s.add_user("admin", "scrypt$whatever")
    s.close()

    path = tmp_path / "config.toml"
    cfg = Config(db_path=str(db))
    out = ensure_setup(cfg, path)
    assert out is cfg          # no wizard, no config reload needed
    assert path.exists()       # but the template was still seeded


def test_ensure_setup_headless_without_web(tmp_path):
    cfg = Config(db_path=str(tmp_path / "t.db"))
    cfg.web.enabled = False
    out = ensure_setup(cfg, tmp_path / "config.toml")
    assert out is cfg          # no accounts needed when nothing is served
