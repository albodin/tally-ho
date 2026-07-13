"""Reflective settings schema for the web UI's settings editor.

Walks the :class:`~tallyho.config.Config` dataclasses to enumerate every knob
(section, key, type, default), pairs each with help text parsed from the
packaged ``config.example.toml`` comments, and validates/applies edits coming
from the browser. Pure stdlib, so the core stays importable offline; the web
layer is the only consumer.

The schema can't drift from the code: it *is* the dataclasses, and the drift
test (tests/test_config_template.py) already forces every knob into the
template this module mines for help text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields, is_dataclass
from functools import cache

from .config import Config

ENV_PREFIX = "TALLYHO_"

# Keys whose runtime consumers capture the value at startup (constructor bake,
# thread-start decision, server bind), so an in-place edit of the shared Config
# can't reach them - they need a process restart. Everything else is read live
# off the shared object each loop/call and hot-applies. Each entry names the
# capture point that makes it sticky.
RESTART_REQUIRED: frozenset[tuple[str | None, str]] = frozenset({
    (None, "db_path"),                      # Store(cfg.db_path) at App init
    (None, "health_file"),                  # heartbeat path wired at startup
    (None, "log_level"),                    # logging.basicConfig at CLI start
    (None, "display_tz"),                   # AlertManager freezes its tzinfo
    ("ingest", "reorder_hold_seconds"),     # baked into ReorderBuffer
    ("ingest", "dedup_ttl_seconds"),        # baked into ReorderBuffer
    ("notify", "request_timeout_seconds"),  # captured by HttpNtfySink
    ("web", "enabled"),                     # server built once at run()
    ("web", "host"),
    ("web", "port"),
    ("dem", "path"),                        # ReloadableGround built at App init
    ("dem", "enabled"),
    ("dem", "source"),
    ("dem", "download_in_process"),         # downloader thread-start decision
    ("dem", "download_check_seconds"),      # cadence computed before the loop
    ("gfs", "path"),                        # wind source built at App init
    ("gfs", "enabled"),
    ("gfs", "download_in_process"),
    ("gfs", "download_cadence_hours"),
    ("hrrr", "path"),
    ("hrrr", "enabled"),
    ("hrrr", "download_in_process"),
    ("hrrr", "download_cadence_hours"),
    ("hrrr", "ceiling_m"),                  # CompositeWindSource constructor
    ("hrrr", "blend_ramp_m"),
})

# Fields whose kind can't be inferred from the default value alone.
_SPECIAL_KINDS: dict[tuple[str | None, str], str] = {
    ("ensemble", "seed"): "opt_int",        # int | None; None = per-flight seed
    ("gfs", "download_fxx"): "int_list",
    ("hrrr", "download_fxx"): "int_list",
}
_CHOICES: dict[tuple[str | None, str], tuple[str, ...]] = {
    ("profile", "correction_mode"): ("blend", "bias"),
    ("dem", "source"): ("auto", "glo30", "tiles"),
}
# (min, max) bounds, None = unbounded on that side. Each entry mirrors a hard
# clamp at the consuming site - validate here so the UI never accepts a value
# the runtime would silently override.
_RANGES: dict[tuple[str | None, str], tuple[float | None, float | None]] = {
    ("dem", "download_workers"): (1, 16),          # clamp in dem.download_dem_tiles
    ("dem", "download_check_seconds"): (30, None),  # clamp in App._dem_loop
    # CSS/Leaflet clamp opacity to [0, 1]; validate so the map never gets a
    # value it would silently reinterpret
    **{("colors", f.name): (0, 1) for f in fields(Config().colors)
       if f.name.endswith("_opacity")},
}


@dataclass(frozen=True)
class FieldSpec:
    section: str | None      # None = top-level Config key
    key: str
    kind: str                # str | float | int | bool | enum | int_list | opt_int | color
    default: object
    help: str                # from the template's comments; may be ""
    choices: tuple[str, ...] | None
    restart_required: bool
    env_var: str             # e.g. "TALLYHO_GFS_ENABLED"
    minimum: float | None = None   # numeric kinds only; None = unbounded
    maximum: float | None = None

    @property
    def dotted(self) -> str:
        return f"{self.section}.{self.key}" if self.section else self.key


def _kind_of(section: str | None, key: str, default: object) -> str:
    sk = (section, key)
    # [colors] fields are "#rrggbb" hex colors by construction (ColorsConfig),
    # except the *_opacity floats, which fall through to the numeric kinds
    if section == "colors" and not key.endswith("_opacity"):
        return "color"
    if sk in _SPECIAL_KINDS:
        return _SPECIAL_KINDS[sk]
    if sk in _CHOICES:
        return "enum"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    raise TypeError(
        f"config field {'.'.join(filter(None, (section, key)))} has a default of "
        f"unmapped type {type(default).__name__}; teach settings_meta about it")


@cache
def _template_info() -> tuple[tuple, dict, dict]:
    """``(section_order, section_help, field_help)`` mined from the packaged
    template's comments. ``section_order`` starts with ``None`` (the top-level
    keys) and follows the file; help text is best-effort - the drift test
    guarantees every knob has a definition line, not that it has prose."""
    from .setup import template_text

    cfg = Config()
    known: dict[str | None, set[str]] = {
        None: {f.name for f in fields(cfg) if not is_dataclass(getattr(cfg, f.name))}}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if is_dataclass(v):
            known[f.name] = {g.name for g in fields(v)}

    section: str | None = None
    order: list[str | None] = [None]
    section_help: dict[str | None, str] = {None: ""}
    field_help: dict[tuple[str | None, str], str] = {}
    pending: list[str] = []       # prose lines waiting for the next field
    last_field: tuple | None = None  # target for "#   # more text" continuations
    after_header = False          # prose right after [section] describes it

    for line in template_text().splitlines():
        if not line.strip():
            # a blank line detaches any accumulated prose (e.g. the file header)
            pending.clear()
            last_field = None
            after_header = False
            continue
        m = re.match(r"^\[([a-z_]+)\]\s*(?:#\s*(.*))?$", line)
        if m:
            section = m.group(1)
            order.append(section)
            section_help[section] = (m.group(2) or "").strip()
            pending.clear()
            last_field = None
            after_header = True
            continue
        # continuation of the previous field's inline help: "#<spaces># text"
        m = re.match(r"^#\s+#\s*(.*)$", line)
        if m and last_field is not None:
            field_help[last_field] = f"{field_help[last_field]} {m.group(1).strip()}".strip()
            continue
        # a field definition (commented-out or active)
        m = re.match(r"^#?\s*([a-z_][a-z0-9_]*)\s*=\s*(.*)$", line)
        if m and m.group(1) in known.get(section, set()):
            key = m.group(1)
            # inline help sits after 2+ spaces then '#'; a bare '#' can be part
            # of the value itself (map_url_template's "#map=..."), so the
            # single-space form is never split on
            parts = re.split(r"\s{2,}#\s*", m.group(2), maxsplit=1)
            help_text = parts[1].strip() if len(parts) > 1 else ""
            if pending:
                help_text = f"{' '.join(pending)} {help_text}".strip()
                pending.clear()
            field_help[(section, key)] = help_text
            last_field = (section, key)
            after_header = False
            continue
        # anything else that's a comment is prose
        m = re.match(r"^#\s*(.*)$", line)
        if m:
            text = m.group(1).strip()
            if text and after_header:
                section_help[section] = f"{section_help[section]} {text}".strip()
            elif text:
                pending.append(text)
            last_field = None
            continue

    return tuple(order), section_help, field_help


@cache
def field_specs() -> tuple[FieldSpec, ...]:
    """Every config knob as a :class:`FieldSpec`, top-level keys first, then
    sections in dataclass order (which matches the template)."""
    cfg = Config()
    _, _, field_help = _template_info()
    flat: list[tuple[str | None, str, object]] = []
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if is_dataclass(v):
            flat.extend((f.name, g.name, getattr(v, g.name)) for g in fields(v))
        else:
            flat.append((None, f.name, v))
    return tuple(FieldSpec(
        section=section, key=key,
        kind=_kind_of(section, key, default),
        default=default,
        help=field_help.get((section, key), ""),
        choices=_CHOICES.get((section, key)),
        restart_required=(section, key) in RESTART_REQUIRED,
        env_var=ENV_PREFIX + (f"{section}_{key}" if section else key).upper(),
        minimum=_RANGES.get((section, key), (None, None))[0],
        maximum=_RANGES.get((section, key), (None, None))[1],
    ) for section, key, default in flat)


def describe(cfg: Config) -> dict:
    """JSON-ready schema + the live effective values, grouped by section in
    template order. ``env_overridden`` is computed per call - the environment
    beats the file (see config.load_config), so those fields are read-only."""
    order, section_help, _ = _template_info()
    by_section: dict[str | None, list[FieldSpec]] = {}
    for spec in field_specs():
        by_section.setdefault(spec.section, []).append(spec)
    sections = []
    for name in list(order) + [s for s in by_section if s not in order]:
        specs = by_section.get(name)
        if not specs:
            continue
        target = cfg if name is None else getattr(cfg, name)
        sections.append({
            "name": name,
            "title": name or "general",
            "help": section_help.get(name, ""),
            "fields": [{
                "key": s.key,
                "kind": s.kind,
                "value": getattr(target, s.key),
                "default": s.default,
                "help": s.help,
                "choices": list(s.choices) if s.choices else None,
                "restart_required": s.restart_required,
                "env_var": s.env_var,
                "env_overridden": s.env_var in os.environ,
                "min": s.minimum,
                "max": s.maximum,
            } for s in specs],
        })
    return {"sections": sections}


def _in_range(spec: FieldSpec, value: float) -> float:
    lo, hi = spec.minimum, spec.maximum
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        if lo is not None and hi is not None:
            raise ValueError(f"must be between {lo} and {hi}")
        raise ValueError(f"must be at least {lo}" if lo is not None
                         else f"must be at most {hi}")
    return value


def coerce(spec: FieldSpec, value: object) -> object:
    """Validate a JSON-decoded value against ``spec``; raises ValueError with a
    human message. Integral floats pass as ints (every JS number is a float),
    but bool is never a number and "true" is never a bool. Numeric kinds also
    honor the ``_RANGES`` bounds."""
    kind = spec.kind
    if kind == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError("expected true or false")
    if kind == "str":
        if isinstance(value, str):
            return value
        raise ValueError("expected a string")
    if kind == "color":
        # exactly 6 hex digits: the map JS builds "#rrggbbaa" alpha variants by
        # suffixing, and <input type="color"> only speaks this form
        if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            return value.lower()
        raise ValueError('expected a hex color like "#4ea1ff"')
    if kind == "enum":
        if isinstance(value, str) and value in (spec.choices or ()):
            return value
        raise ValueError("expected one of: " + ", ".join(spec.choices or ()))
    if kind == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _in_range(spec, float(value))
        raise ValueError("expected a number")
    if kind in ("int", "opt_int"):
        if kind == "opt_int" and value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("expected an integer")
        return int(_in_range(spec, int(value)))
    if kind == "int_list":
        if not isinstance(value, list):
            raise ValueError("expected a list of integers")
        out = []
        for v in value:
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or (isinstance(v, float) and not v.is_integer()):
                raise ValueError("expected a list of integers")
            out.append(int(v))
        return out
    raise ValueError(f"unknown setting kind {kind!r}")   # pragma: no cover


def validate_update(cfg: Config, values: dict) -> tuple[dict, dict[str, str]]:
    """Check an edit payload (``{dotted_key: value}``) against the schema and
    the live config. Returns ``(changed, errors)``: ``changed`` is the nested
    ``{section: {key: value}} / {key: value}`` dict of values that actually
    differ from ``cfg`` (the shape write_config_values/apply_values take -
    unchanged keys are dropped so their template lines stay commented), and
    ``errors`` maps dotted keys to messages. Env-overridden keys are rejected:
    a file write would be silently masked by the environment."""
    by_dotted = {s.dotted: s for s in field_specs()}
    changed: dict = {}
    errors: dict[str, str] = {}
    for dotted, raw in values.items():
        spec = by_dotted.get(dotted)
        if spec is None:
            errors[str(dotted)] = "unknown setting"
            continue
        if spec.env_var in os.environ:
            errors[dotted] = (f"set by the environment variable {spec.env_var}, "
                              "which overrides the config file - unset it to edit here")
            continue
        try:
            val = coerce(spec, raw)
        except ValueError as exc:
            errors[dotted] = str(exc)
            continue
        target = cfg if spec.section is None else getattr(cfg, spec.section)
        if getattr(target, spec.key) == val:
            continue
        if spec.section is None:
            changed[spec.key] = val
        else:
            changed.setdefault(spec.section, {})[spec.key] = val
    return changed, errors


def apply_values(cfg: Config, changed: dict) -> None:
    """Overlay ``changed`` (validate_update's nested dict) onto the live Config
    in place. The daemon's components hold this object by reference and read
    most knobs per iteration, so hot-appliable fields take effect immediately.
    Each assignment is one (GIL-atomic) setattr; a multi-field save is not
    transactional across fields - fine for tuning knobs."""
    for key, val in changed.items():
        if isinstance(val, dict):
            section = getattr(cfg, key)
            for k, v in val.items():
                setattr(section, k, v)
        else:
            setattr(cfg, key, val)


def dotted_keys(changed: dict) -> list[str]:
    """Flatten validate_update's nested dict back to sorted dotted names."""
    out = []
    for key, val in changed.items():
        if isinstance(val, dict):
            out.extend(f"{key}.{k}" for k in val)
        else:
            out.append(key)
    return sorted(out)


def restart_required_in(changed: dict) -> list[str]:
    """The subset of ``changed`` (nested dict) that needs a restart, dotted."""
    out = []
    for key, val in changed.items():
        if isinstance(val, dict):
            out.extend(f"{key}.{k}" for k in val if (key, k) in RESTART_REQUIRED)
        elif (None, key) in RESTART_REQUIRED:
            out.append(key)
    return sorted(out)
