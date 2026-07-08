"""The packaged config template must track the Config dataclasses.

Every knob must appear in the template (commented out), and every commented
value must equal the code default - so an untouched seeded file always means
"defaults", and a new knob can't be forgotten in the docs.
"""

import re
import tomllib
from dataclasses import fields, is_dataclass

from tallyho.config import Config
from tallyho.setup import template_text

# Shown in the template with an example value, not the default (None has no
# TOML spelling).
EXAMPLE_ONLY = {("ensemble", "seed")}


def _leaves(obj, prefix=()):
    for f in fields(obj):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            yield from _leaves(v, prefix + (f.name,))
        else:
            yield prefix + (f.name,), v


def _template_values() -> dict:
    """Uncomment every `# key = value` line whose key is a real field of the
    active section, then parse the result as TOML. Prose comments never name
    a field followed by '=', so they can't collide."""
    cfg = Config()
    known = {None: {f.name for f in fields(cfg) if not is_dataclass(getattr(cfg, f.name))}}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if is_dataclass(v):
            known[f.name] = {g.name for g in fields(v)}

    section = None
    lines = []
    for line in template_text().splitlines():
        m = re.match(r"\[([a-z_]+)\]", line)
        if m:
            section = m.group(1)
            assert section in known, f"template has unknown section [{section}]"
            lines.append(f"[{section}]")
            continue
        m = re.match(r"#\s*([a-z_][a-z0-9_]*)\s*=\s*(.+)$", line)
        if m and m.group(1) in known.get(section, set()):
            lines.append(f"{m.group(1)} = {m.group(2)}")
    return tomllib.loads("\n".join(lines))


def test_template_is_valid_toml_and_fully_commented():
    doc = tomllib.loads(template_text())
    active = {(s, k) for s, v in doc.items() if isinstance(v, dict) for k in v} \
        | {(k,) for k, v in doc.items() if not isinstance(v, dict)}
    assert not active, f"template must ship fully commented, found active: {active}"


def test_every_knob_is_in_the_template_at_its_default():
    flat = {}
    for k, v in _template_values().items():
        if isinstance(v, dict):
            flat.update({(k, k2): v2 for k2, v2 in v.items()})
        else:
            flat[(k,)] = v
    leaves = dict(_leaves(Config()))

    missing = set(leaves) - set(flat)
    assert not missing, f"knobs missing from the template: {sorted(missing)}"
    for key, default in leaves.items():
        if key in EXAMPLE_ONLY:
            continue
        assert flat[key] == default, \
            f"{'.'.join(key)}: template says {flat[key]!r}, code default is {default!r}"
