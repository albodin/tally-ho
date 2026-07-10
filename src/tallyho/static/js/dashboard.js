// Dashboard entry point: header wiring, the clear-history buttons, and the
// refresh loop that drives every module; pushed by SSE (see events.js), with
// the old 15 s poll kept as an automatic fallback while SSE is down.
import { $, api, setDisplayTz, status } from "./util.js";
import { refreshMap } from "./map.js";
import { refreshHistory } from "./history.js";
import { refreshAccuracy, refreshAlerts, refreshFlights, refreshSubs } from "./tables.js";
import { initSubscribers } from "./subscribers.js";
import { initTokens, refreshTokens } from "./tokens.js";
import { connectEvents } from "./events.js";

const REFRESH_MS = 15000;

async function refreshHealth() {
  try {
    const h = await api("/api/stats");
    $("health").textContent = `${h.active_flights} active flight(s) · ${h.subscribers} location(s) · ${h.db_path}`;
  } catch (e) { $("health").textContent = "offline"; }
}

// One-time: learn which timezone to render times in (see util.js). On
// failure we keep the UTC default rather than block the dashboard.
async function loadConfig() {
  try { const c = await api("/api/config"); if (c && c.tz) setDisplayTz(c.tz); }
  catch (e) { /* keep UTC */ }
}

async function refreshAll() {
  try {
    await Promise.all([refreshHealth(), refreshMap(), refreshFlights(), refreshAlerts(),
                       refreshAccuracy(), refreshSubs(), refreshTokens(), refreshHistory()]);
    if ($("status").classList.contains("err")) status(null);
  } catch (e) { status("Refresh failed: " + e.message); }
}

$("logout").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST" }); } catch {}
  location.href = "/login";
});

$("acc-clear").addEventListener("click", async () => {
  if (!confirm("Clear prediction-accuracy history?\nDeletes all recorded landings and "
               + "finished flights' prediction records. Active flights are unaffected.")) return;
  try {
    await api("/api/accuracy", { method: "DELETE" });
    status("Accuracy history cleared.", "ok");
    await refreshAll();
  } catch (err) { status("Clear failed: " + err.message); }
});

$("alerts-clear").addEventListener("click", async () => {
  if (!confirm("Clear the sent-alert history?\nAlerts for flights still in the air "
               + "are kept so they aren't re-sent.")) return;
  try {
    await api("/api/alerts", { method: "DELETE" });
    status("Alert history cleared.", "ok");
    await refreshAlerts();
  } catch (err) { status("Clear failed: " + err.message); }
});

initSubscribers(refreshAll);
initTokens();

// event -> refetchers (subscribers refreshes tokens too: the tokens table shows
// a per-token reference count that subscriber edits change).
const EVENT_REFETCH = {
  flights:     [refreshFlights, refreshMap, refreshHistory, refreshHealth],
  accuracy:    [refreshAccuracy, refreshMap, refreshFlights],
  alerts:      [refreshAlerts],
  subscribers: [refreshSubs, refreshMap, refreshTokens, refreshHealth],
  tokens:      [refreshTokens],
  stats:       [refreshHealth],
  changed:     [refreshAll],
};

let fallbackTimer = null;
const startFallbackPoll = () => { if (!fallbackTimer) fallbackTimer = setInterval(refreshAll, REFRESH_MS); };
const stopFallbackPoll  = () => { if (fallbackTimer) { clearInterval(fallbackTimer); fallbackTimer = null; } };

// Load DISPLAY_TZ before the first paint so times never flash in UTC first,
// then hand the refresh loop to SSE (poll only while it's down).
loadConfig().then(() => {
  refreshAll();
  connectEvents(EVENT_REFETCH, {
    onUp:   () => { stopFallbackPoll(); refreshAll(); },  // resync anything missed
    onDown: () => { startFallbackPoll(); },              // degrade to today's behavior
  });
});
// Slow safety net even while connected: cheap insurance for the hhmm() "today vs
// dated" labels crossing midnight and any missed doorbell.
setInterval(refreshAll, 10 * 60 * 1000);
