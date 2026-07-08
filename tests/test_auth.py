"""Web-UI auth: scrypt hashing, the login throttle, and the session guard."""

import pytest

from tallyho.auth import LoginLimiter, hash_password, verify_password


# ---- password hashing ------------------------------------------------------
def test_hash_roundtrip():
    stored = hash_password("hunter2hunter2")
    assert stored.startswith("scrypt$")
    assert verify_password("hunter2hunter2", stored)
    assert not verify_password("hunter2hunter3", stored)


def test_hashes_are_salted():
    assert hash_password("same") != hash_password("same")


@pytest.mark.parametrize("stored", ["", "plain", "scrypt$bad", "md5$1$2$3$YQ==$YQ=="])
def test_malformed_stored_hash_is_rejected(stored):
    assert not verify_password("anything", stored)


# ---- login throttle --------------------------------------------------------
def test_limiter_locks_after_max_fails_and_recovers():
    now = [0.0]
    lim = LoginLimiter(max_fails=3, lockout_seconds=30.0, clock=lambda: now[0])
    key = "1.2.3.4"
    for _ in range(3):
        assert lim.retry_after(key) == 0.0
        lim.record_failure(key)
    assert lim.retry_after(key) == pytest.approx(30.0)
    now[0] = 29.0
    assert lim.retry_after(key) > 0
    now[0] = 31.0
    assert lim.retry_after(key) == 0.0   # lockout expired
    lim.reset(key)
    assert lim.retry_after(key) == 0.0


def test_limiter_is_per_key():
    lim = LoginLimiter(max_fails=1, lockout_seconds=30.0, clock=lambda: 0.0)
    lim.record_failure("a")
    assert lim.retry_after("a") > 0
    assert lim.retry_after("b") == 0.0


# ---- session guard (needs the api extra) -----------------------------------
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from tallyho.config import Config  # noqa: E402
from tallyho.store import Store  # noqa: E402
from tallyho.web import create_app  # noqa: E402

PASSWORD = "correct-horse-battery"
PASSWORD_HASH = hash_password(PASSWORD)


@pytest.fixture
def store():
    s = Store(":memory:")
    s.add_user("admin", PASSWORD_HASH)
    yield s
    s.close()


@pytest.fixture
def client(store):
    return TestClient(create_app(Config(), store))


def test_api_requires_session(client):
    r = client.get("/api/flights")
    assert r.status_code == 401


def test_pages_redirect_to_login(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_login_page_is_public(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert "password" in r.text.lower()


def test_wrong_credentials_rejected(client):
    assert client.post("/api/login", json={"username": "admin",
                                           "password": "wrong-password"}).status_code == 401
    assert client.post("/api/login", json={"username": "nobody",
                                           "password": PASSWORD}).status_code == 401


def test_login_logout_cycle(client):
    r = client.post("/api/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 200 and r.json()["user"] == "admin"
    assert client.get("/api/flights").status_code == 200
    assert client.get("/", follow_redirects=False).status_code == 200

    client.post("/api/logout")
    assert client.get("/api/flights").status_code == 401


def test_lockout_after_repeated_failures(client):
    for _ in range(5):
        client.post("/api/login", json={"username": "admin", "password": "nope-nope"})
    # even the right password is refused while locked out
    r = client.post("/api/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 429


def test_session_survives_app_restart(store):
    """The signing secret persists in the DB, so a cookie issued before a
    restart is still valid after (and the setup wizard's login carries over)."""
    c1 = TestClient(create_app(Config(), store))
    c1.post("/api/login", json={"username": "admin", "password": PASSWORD})

    c2 = TestClient(create_app(Config(), store))
    c2.cookies.set("session", c1.cookies.get("session"))
    assert c2.get("/api/flights").status_code == 200
