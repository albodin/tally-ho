// Settings editor: renders the reflective schema from GET /api/settings as
// one form section per config section, PUTs back only the edited fields
// (keyed by dotted name), maps per-field validation errors onto the form, and
// drives the restart flow for startup-captured ("restart required") keys.
import { $, api, status } from "./util.js";

const fields = new Map();   // dotted key -> {spec, input, initial, errEl}
let writable = true;

const dotted = (section, key) => (section ? `${section}.${key}` : key);

// ---- value <-> input ------------------------------------------------------
function renderValue(spec, value) {
  if (spec.kind === "int_list") return (value || []).join(", ");
  if (spec.kind === "opt_int") return value === null ? "" : String(value);
  return String(value);
}

// The input's current content as the JSON value the API expects; throws a
// user-facing message on unparseable input (server-side validation still has
// the final word - this just catches "abc" in a number box early).
function parseValue(spec, input) {
  if (spec.kind === "bool") return input.checked;
  const raw = input.value.trim();
  if (spec.kind === "str" || spec.kind === "enum") return spec.kind === "str" ? input.value : raw;
  if (spec.kind === "opt_int") {
    if (raw === "") return null;
    const n = Number(raw);
    if (!Number.isInteger(n)) throw new Error("expected an integer (or blank for unset)");
    return n;
  }
  if (spec.kind === "int" || spec.kind === "float") {
    const n = Number(raw);
    if (raw === "" || !Number.isFinite(n)) throw new Error("expected a number");
    if (spec.kind === "int" && !Number.isInteger(n)) throw new Error("expected an integer");
    return n;
  }
  if (spec.kind === "int_list") {
    if (raw === "") return [];
    return raw.split(",").map((part) => {
      const n = Number(part.trim());
      if (part.trim() === "" || !Number.isInteger(n)) {
        throw new Error("expected comma-separated integers, e.g. 0, 1, 2");
      }
      return n;
    });
  }
  throw new Error(`unknown kind ${spec.kind}`);
}

function isDirty(f) {
  try { return JSON.stringify(parseValue(f.spec, f.input)) !== JSON.stringify(f.initial); }
  catch { return true; }   // unparseable = edited (and will be flagged on save)
}

// ---- rendering ------------------------------------------------------------
function makeInput(spec) {
  let input;
  if (spec.kind === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = spec.value;
  } else if (spec.kind === "enum") {
    input = document.createElement("select");
    for (const c of spec.choices) {
      const opt = document.createElement("option");
      opt.value = opt.textContent = c;
      opt.selected = c === spec.value;
      input.appendChild(opt);
    }
  } else {
    input = document.createElement("input");
    if (spec.kind === "int") { input.type = "number"; input.step = "1"; }
    else if (spec.kind === "float") { input.type = "number"; input.step = "any"; }
    else input.type = "text";
    input.value = renderValue(spec, spec.value);
    input.placeholder = renderValue(spec, spec.default);
  }
  return input;
}

function badge(cls, text, title) {
  const b = document.createElement("span");
  b.className = `badge ${cls}`;
  b.textContent = text;
  if (title) b.title = title;
  return b;
}

function renderSection(sec, frag) {
  const el = document.createElement("section");
  const h2 = document.createElement("h2");
  h2.textContent = sec.name ? `[${sec.name}]` : "general";
  el.appendChild(h2);
  if (sec.help) {
    const p = document.createElement("p");
    p.className = "section-help";
    p.textContent = sec.help;
    el.appendChild(p);
  }
  const grid = document.createElement("div");
  grid.className = "settings-grid";
  for (const f of sec.fields) {
    const key = dotted(sec.name, f.key);
    const skey = document.createElement("div");
    skey.className = "skey";
    const code = document.createElement("code");
    code.textContent = f.key;
    skey.appendChild(code);
    if (f.restart_required) {
      skey.appendChild(badge("badge-restart", "restart",
                             "applied on the next app restart"));
    }
    if (f.env_overridden) {
      skey.appendChild(badge("badge-env", "env",
                             `set by ${f.env_var}, which overrides the config file`));
    }
    const input = makeInput(f);
    input.dataset.key = key;
    if (f.env_overridden || !writable) input.disabled = true;
    const help = document.createElement("div");
    help.className = "setting-help";
    help.textContent = f.help || "";
    if (f.env_overridden) {
      // visible, not tooltip-only: an editable-looking-but-locked field with
      // no stated reason reads as a bug
      const note = document.createElement("div");
      note.className = "env-note";
      note.textContent = `set by ${f.env_var} in the environment - edit it there`;
      help.appendChild(note);
    }
    const errEl = document.createElement("div");
    errEl.className = "field-error";
    grid.append(skey, input, help, errEl);
    fields.set(key, { spec: f, input, initial: f.value, errEl });
  }
  el.appendChild(grid);
  frag.appendChild(el);
}

function refreshDirty() {
  let n = 0;
  for (const f of fields.values()) {
    const dirty = !f.input.disabled && isDirty(f);
    f.input.classList.toggle("dirty", dirty);
    if (dirty) n += 1;
  }
  $("savebar").hidden = n === 0;
  $("dirty-count").textContent = `${n} unsaved change${n === 1 ? "" : "s"}`;
  return n;
}

function clearFieldErrors() {
  for (const f of fields.values()) {
    f.errEl.textContent = "";
    f.input.classList.remove("bad");
  }
}

function showFieldErrors(errors) {
  let firstBad = null;
  for (const [key, msg] of Object.entries(errors)) {
    const f = fields.get(key);
    if (!f) continue;
    f.errEl.textContent = msg;
    f.input.classList.add("bad");
    if (!firstBad) firstBad = f.input;
  }
  if (firstBad) firstBad.scrollIntoView({ block: "center" });
}

function showRestartBanner(pending) {
  const banner = $("restart-banner");
  if (!pending || pending.length === 0) { banner.hidden = true; return; }
  $("restart-banner-text").textContent =
    `Saved but waiting for a restart to take effect: ${pending.join(", ")} - ` +
    `use "Restart app" above when ready.`;
  banner.hidden = false;
}

// ---- save -----------------------------------------------------------------
async function save() {
  clearFieldErrors();
  const values = {};
  const parseErrors = {};
  for (const [key, f] of fields) {
    if (f.input.disabled || !isDirty(f)) continue;
    try { values[key] = parseValue(f.spec, f.input); }
    catch (e) { parseErrors[key] = e.message; }
  }
  if (Object.keys(parseErrors).length) {
    showFieldErrors(parseErrors);
    status("Fix the marked fields before saving.");
    return;
  }
  if (Object.keys(values).length === 0) { status("Nothing to save.", "ok"); return; }
  $("save").disabled = true;
  try {
    // plain fetch (not api()): a 422 body carries the per-field error map
    const r = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    if (r.status === 401) { location.href = "/login"; return; }
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      if (body.detail && body.detail.errors) {
        showFieldErrors(body.detail.errors);
        status("Some settings were rejected - see the marked fields.");
      } else {
        status(`Save failed: ${r.status} ${JSON.stringify(body.detail || body)}`);
      }
      return;
    }
    for (const [key, v] of Object.entries(values)) {
      const f = fields.get(key);
      f.initial = v;
      f.input.classList.remove("dirty");
    }
    status(body.changed.length
      ? `Saved: ${body.changed.join(", ")}`
      : "No changes (values already match).", "ok");
    showRestartBanner(body.pending_restart);
    refreshDirty();
  } catch (e) {
    status("Save failed: " + e.message);
  } finally {
    $("save").disabled = false;
  }
}

function revert() {
  clearFieldErrors();
  for (const f of fields.values()) {
    if (f.input.disabled) continue;
    if (f.spec.kind === "bool") f.input.checked = f.initial;
    else f.input.value = renderValue(f.spec, f.initial);
  }
  refreshDirty();
  status(null);
}

// ---- restart --------------------------------------------------------------
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

async function healthUp() {
  try { await fetch("/api/health", { cache: "no-store" }); return true; }
  catch { return false; }   // any HTTP response (even 503) means it's serving
}

async function restartApp() {
  if (refreshDirty() > 0 &&
      !confirm("There are unsaved edits - restart anyway? They will be lost.")) return;
  if (!confirm("Restart tally-ho now?\nUnder Docker (restart: unless-stopped) it is "
               + "back in seconds; without a supervisor the process just stops.")) return;
  try { await api("/api/restart", { method: "POST" }); }
  catch (e) { status("Restart request failed: " + e.message); return; }
  $("restart-overlay").hidden = false;
  // phase 1: wait for the old process to actually go down, so an early health
  // "ok" from the not-yet-exited server isn't mistaken for the new one
  for (let i = 0; i < 20 && await healthUp(); i++) await sleep(1000);
  // phase 2: wait for the new process to come up
  $("overlay-text").textContent = "Waiting for the server to come back…";
  for (let i = 0; i < 90; i++) {
    await sleep(2000);
    if (await healthUp()) { location.reload(); return; }
  }
  $("overlay-title").textContent = "Still down";
  $("overlay-text").textContent =
    "The server has not come back - if it runs without a supervisor it must "
    + "be started by hand. Reload this page once it's up.";
}

// ---- boot -----------------------------------------------------------------
async function load() {
  const body = await api("/api/settings");
  writable = body.writable;
  $("intro").textContent = writable
    ? `Edits are saved to ${body.config_path}. Most settings apply immediately; `
      + `those marked "restart" apply on the next restart, and "env" fields are `
      + `controlled by an environment variable.`
    : "Read-only: this server was started without a config file to write.";
  const frag = document.createDocumentFragment();
  for (const sec of body.sections) renderSection(sec, frag);
  const root = $("sections");
  root.textContent = "";
  root.appendChild(frag);
  $("save").disabled = !writable;
  showRestartBanner(body.pending_restart);
  refreshDirty();
}

$("logout").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST" }); } catch {}
  location.href = "/login";
});
$("save").addEventListener("click", save);
$("revert").addEventListener("click", revert);
$("restart").addEventListener("click", restartApp);
$("sections").addEventListener("input", refreshDirty);
$("sections").addEventListener("change", refreshDirty);

load().catch((e) => status("Failed to load settings: " + e.message));
