// Small helpers shared by the dashboard modules.

export const $ = (id) => document.getElementById(id);

// The palette lives in theme.css / dashboard.css as CSS variables; this is how
// JS-drawn things (map lines, the history chart) read them, so a color is
// never defined twice.
export const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
// numeric CSS variable (the configurable map opacities); a missing/broken
// value degrades to fully opaque rather than an invisible NaN
export const cssNum = (name) => {
  const v = parseFloat(cssVar(name));
  return Number.isFinite(v) ? v : 1;
};

export const fnum = (v, d=4) => (v === null || v === undefined) ? "-" : Number(v).toFixed(d);
export const esc = (s) => (s ?? "").toString().replace(/[&<>"]/g, c =>
  ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;" }[c]));

// IANA timezone all times are shown in - server-provided (from TZ /
// TALLYHO_DISPLAY_TZ via /api/config), so the dashboard clock matches the
// server's configured zone, not each viewer's. Defaults to UTC until loaded.
let DISPLAY_TZ = "UTC";
export function setDisplayTz(tz) { DISPLAY_TZ = tz; }

// HH:MM:SS in DISPLAY_TZ with a zone-name suffix (e.g. "14:23:05 EDT"). When
// the timestamp falls on a different calendar day than now (in DISPLAY_TZ) the
// date is prefixed (e.g. "15 Jun 14:23:05 EDT", "15 Jun 2024 …" across years)
// so cross-day times aren't mistaken for today. Formatters are rebuilt only
// when DISPLAY_TZ changes (after /api/config).
let _timeFmt = null, _dateFmt = null, _dayKeyFmt = null, _timeFmtTz = null;
export const hhmm = (iso) => {
  if (!iso) return "-";
  try {
    if (_timeFmtTz !== DISPLAY_TZ) {
      const tz = DISPLAY_TZ;
      _timeFmt = new Intl.DateTimeFormat("en-GB", { timeZone: tz,
        hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
        timeZoneName: "short" });
      _dateFmt = new Intl.DateTimeFormat("en-GB", { timeZone: tz,
        day: "2-digit", month: "short" });
      // en-CA yields a sortable YYYY-MM-DD day key for same-day comparison.
      _dayKeyFmt = new Intl.DateTimeFormat("en-CA", { timeZone: tz,
        year: "numeric", month: "2-digit", day: "2-digit" });
      _timeFmtTz = tz;
    }
    const d = new Date(iso), time = _timeFmt.format(d);
    const key = _dayKeyFmt.format(d), nowKey = _dayKeyFmt.format(new Date());
    if (key === nowKey) return time;
    const date = key.slice(0, 4) === nowKey.slice(0, 4)
      ? _dateFmt.format(d) : `${_dateFmt.format(d)} ${key.slice(0, 4)}`;
    return `${date} ${time}`;
  } catch { return iso; }
};

export async function api(path, opts) {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = JSON.stringify(j.detail || j); } catch {}
    throw new Error(`${r.status} ${detail}`);
  }
  return r.status === 204 ? null : r.json();
}

// Visibility is class-driven (.notice is display:none until .ok/.err), so
// clearing only needs to drop the class. Success confirmations dismiss
// themselves; errors persist until the next successful refresh clears them.
let statusTimer = null;
export function status(msg, kind) {
  const el = $("status");
  clearTimeout(statusTimer);
  if (!msg) { el.className = "notice"; return; }
  el.textContent = msg; el.className = `notice ${kind || "err"}`;
  if (kind === "ok") statusTimer = setTimeout(() => status(null), 4000);
}
