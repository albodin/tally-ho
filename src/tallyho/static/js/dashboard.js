// Dashboard entry point: header wiring, the clear-history buttons, and the
// 15 s refresh loop that drives every module.
import { $, api, setDisplayTz, status } from "./util.js";
import { refreshMap } from "./map.js";
import { refreshHistory } from "./history.js";
import { refreshAccuracy, refreshAlerts, refreshFlights, refreshSubs } from "./tables.js";
import { initSubscribers } from "./subscribers.js";
import { initTokens, refreshTokens } from "./tokens.js";

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
// Load DISPLAY_TZ before the first paint so times never flash in UTC first.
loadConfig().then(refreshAll);
setInterval(refreshAll, REFRESH_MS);
