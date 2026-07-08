"""First-run setup: seed the config file, then gate the pipeline on a wizard.

``tallyho run`` will not start ingesting until setup is complete: the config
file has been seeded into the data dir and (when the web UI is enabled) an
account exists. Until then only a setup page is served - the wizard creates
the admin account, writes its choices into the config file, and the process
continues straight into the normal pipeline, no restart needed.

The seeded template ships in the package (``config.example.toml``) with every
setting commented out at its default, so an untouched file always tracks the
code's defaults; the app writes it once and never modifies it afterwards -
the user owns the file. Programmatic writes (the wizard) go through tomlkit,
which preserves the comments.

NOTE: no ``from __future__ import annotations`` here - like :mod:`tallyho.web`,
the request model is defined inside ``create_setup_app`` and FastAPI cannot
resolve stringized annotations for locally-scoped classes.
"""

import logging
import threading
from importlib import resources
from pathlib import Path

from .auth import MIN_PASSWORD_LEN, hash_password
from .config import Config, load_config
from .store import Store

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "data/config.toml"
_SETUP_HTML = Path(__file__).resolve().parent / "static" / "setup.html"


def template_text() -> str:
    return (resources.files("tallyho") / "config.example.toml").read_text()


def seed_config(path: str | Path) -> bool:
    """Write the commented template to ``path`` if missing. Never touches an
    existing file."""
    p = Path(path)
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(template_text())
    log.info("seeded config template at %s", p)
    return True


def write_config_values(path: str | Path, values: dict) -> None:
    """Set keys in the config file, preserving everything else (comments
    included). ``values`` maps a section name to a ``{key: value}`` dict, or a
    top-level key straight to its value."""
    import tomlkit

    p = Path(path)
    doc = tomlkit.parse(p.read_text())
    for key, val in values.items():
        if isinstance(val, dict):
            section = doc.setdefault(key, tomlkit.table())
            for k, v in val.items():
                section[k] = v
        else:
            doc[key] = val
    p.write_text(tomlkit.dumps(doc))


def create_setup_app(cfg: Config, store: Store, config_path: str | Path,
                     on_complete=None):
    """The wizard-only FastAPI app served while setup is incomplete."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, RedirectResponse
        from pydantic import BaseModel, ConfigDict, Field
        from starlette.middleware.sessions import SessionMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "first-run setup needs the web UI; install it with: pip install '.[api]' "
            "- or edit the config file, set [web] enabled = false, and rerun"
        ) from exc

    from .auth import SESSION_MAX_AGE, session_secret

    class SetupIn(BaseModel):
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        username: str = Field(min_length=1, max_length=64)
        password: str = Field(min_length=MIN_PASSWORD_LEN)
        gfs_enabled: bool = True
        hrrr_enabled: bool = False
        dem_enabled: bool = True
        gfs_keep_hours: float = Field(default=48.0, ge=0)

    app = FastAPI(title="tally-ho setup", docs_url=None, openapi_url=None)
    # Same signing secret as the main app, so the wizard's login survives the
    # handover to the real dashboard.
    app.add_middleware(SessionMiddleware, secret_key=session_secret(store),
                       max_age=SESSION_MAX_AGE, same_site="lax")

    @app.get("/", include_in_schema=False)
    def wizard():
        return FileResponse(str(_SETUP_HTML), media_type="text/html")

    @app.get("/api/health")
    def health():
        # 200: the container is healthy - it is doing what it should be doing
        # (waiting for the user), the pipeline just hasn't been configured yet.
        return {"status": "setup"}

    @app.post("/api/setup")
    def complete(payload: SetupIn, request: Request):
        if store.count_users() > 0:
            raise HTTPException(status_code=409, detail="setup already completed")
        write_config_values(config_path, {
            "gfs": {"enabled": payload.gfs_enabled,
                    "keep_hours": payload.gfs_keep_hours},
            "hrrr": {"enabled": payload.hrrr_enabled},
            "dem": {"enabled": payload.dem_enabled},
        })
        store.add_user(payload.username, hash_password(payload.password))
        request.session["user"] = payload.username
        log.info("setup complete: account %r created, config written to %s",
                 payload.username, config_path)
        if on_complete is not None:
            on_complete()
        return {"ok": True}

    @app.get("/{path:path}", include_in_schema=False)
    def anywhere(path: str):
        return RedirectResponse("/")

    return app


def run_setup(cfg: Config, store: Store, config_path: str | Path) -> bool:  # pragma: no cover - needs uvicorn/network
    """Serve the wizard (blocking) until setup completes. False = the server
    stopped without finishing (e.g. SIGTERM)."""
    import uvicorn

    done = threading.Event()
    server = None

    def on_complete():
        done.set()
        server.should_exit = True

    app = create_setup_app(cfg, store, config_path, on_complete=on_complete)
    config = uvicorn.Config(app, host=cfg.web.host, port=cfg.web.port,
                            log_level=cfg.log_level.lower())
    server = uvicorn.Server(config)
    log.info("first-run setup: open http://%s:%d to create your account",
             cfg.web.host, cfg.web.port)
    server.run()
    return done.is_set()


def ensure_setup(cfg: Config, config_path: str | Path | None) -> Config | None:
    """Seed the config template and, if setup is incomplete, serve the wizard
    until it finishes. Returns the config to run with (reloaded when the wizard
    wrote the file), or None when setup could not complete."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    seed_config(path)
    store = Store(cfg.db_path)
    try:
        if not cfg.web.enabled or store.count_users() > 0:
            return cfg
        try:
            if not run_setup(cfg, store, path):
                log.error("setup did not complete; exiting")
                return None
        except RuntimeError as exc:   # api extra missing
            log.error("%s", exc)
            return None
    finally:
        store.close()
    return load_config(path)
