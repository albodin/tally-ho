"""Web-UI authentication: scrypt password hashing + signed-cookie sessions.

The dashboard is a single-admin app, so this stays deliberately small: the
stdlib scrypt KDF for the password hash (no extra dependency), Starlette's
``SessionMiddleware`` for the cookie, and a pure-ASGI guard that rejects
unauthenticated requests. The account + the session signing secret live in the
DB (``users`` / ``kv``), created by the first-run setup wizard.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

# scrypt cost parameters (CPU/memory hardness). 2^14/8/1 ≈ 16 MB and tens of
# milliseconds per hash.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32

SESSION_SECRET_KEY = "session_secret"   # kv key holding the cookie signing secret
SESSION_MAX_AGE = 14 * 86400
MIN_PASSWORD_LEN = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_b64, dk_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
        dk = hashlib.scrypt(password.encode(), salt=salt,
                            n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)


def session_secret(store) -> str:
    """The cookie signing secret, created once and persisted so sessions
    survive restarts (and the setup wizard's login carries into the app)."""
    secret = store.get_kv(SESSION_SECRET_KEY)
    if secret is None:
        secret = secrets.token_urlsafe(32)
        store.set_kv(SESSION_SECRET_KEY, secret)
    return secret


class LoginLimiter:
    """Per-client failed-login throttle: after ``max_fails`` consecutive
    failures, further attempts are refused for ``lockout_seconds``."""

    def __init__(self, max_fails: int = 5, lockout_seconds: float = 30.0,
                 clock=time.monotonic):
        self.max_fails = max_fails
        self.lockout_seconds = lockout_seconds
        self._clock = clock
        self._fails: dict[str, tuple[int, float]] = {}   # key -> (count, last_at)

    def retry_after(self, key: str) -> float:
        """Seconds until this client may try again (0 = allowed now)."""
        count, last_at = self._fails.get(key, (0, 0.0))
        if count < self.max_fails:
            return 0.0
        return max(0.0, self.lockout_seconds - (self._clock() - last_at))

    def record_failure(self, key: str) -> None:
        if len(self._fails) > 1024:   # unbounded-growth guard
            self._fails.clear()
        count, _ = self._fails.get(key, (0, 0.0))
        self._fails[key] = (count + 1, self._clock())

    def reset(self, key: str) -> None:
        self._fails.pop(key, None)


# Paths reachable without a session: the healthcheck, the login page and its
# POST. Everything else - pages and API alike - needs the cookie.
PUBLIC_PATHS = frozenset({"/api/health", "/api/login", "/login", "/favicon.ico"})


class RequireSession:
    """Pure-ASGI guard (runs inside SessionMiddleware): unauthenticated API
    calls get a 401 JSON body, page requests a redirect to /login."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        if scope.get("session", {}).get("user"):
            await self.app(scope, receive, send)
            return
        from starlette.responses import JSONResponse, RedirectResponse

        if scope["path"].startswith("/api/"):
            resp = JSONResponse({"detail": "authentication required"}, status_code=401)
        else:
            resp = RedirectResponse("/login", status_code=302)
        await resp(scope, receive, send)
