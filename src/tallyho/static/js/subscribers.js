// Watched-location (subscriber) form: add/edit, pick-on-map with a draft
// pin + radius preview, test-ntfy, and the table's row actions.
import { $, api, cssVar, status } from "./util.js";
import { map } from "./map.js";

// own layer - refreshMap never clears it, so the draft survives map rebuilds
const draftLayer = L.layerGroup().addTo(map);
let draftMarker = null, draftCircle = null, picking = false;

const radiusMeters = () => {
  const r = parseFloat($("f-radius").value);
  return Number.isFinite(r) && r > 0 ? r * 1000 : 0;
};
function setLatLon(lat, lon) {
  $("f-lat").value = Number(lat).toFixed(5);
  $("f-lon").value = Number(lon).toFixed(5);
}
function clearDraft() { draftLayer.clearLayers(); draftMarker = null; draftCircle = null; }
function drawDraft(lat, lon) {
  clearDraft();
  draftCircle = L.circle([lat, lon], { radius:radiusMeters(), color:cssVar("--accent"),
    weight:1, dashArray:"5,5", fillOpacity:.06 }).addTo(draftLayer);
  draftMarker = L.marker([lat, lon], { draggable:true }).addTo(draftLayer);
  draftMarker.on("drag", (e) => { setLatLon(e.latlng.lat, e.latlng.lng); draftCircle.setLatLng(e.latlng); });
}
function setPicking(on) {
  picking = on;
  $("f-pick").textContent = on ? "📍 Click the map…" : "📍 Pick on map";
  $("map").style.cursor = on ? "crosshair" : "";
}

function fillForm(s) {
  $("f-id").value = s.id; $("f-name").value = s.name; $("f-lat").value = s.lat;
  $("f-lon").value = s.lon; $("f-radius").value = s.radius_km; $("f-server").value = s.ntfy_server;
  $("f-topic").value = s.ntfy_topic; $("f-token").value = s.ntfy_token_ref || "";
  $("f-submit").textContent = "Save changes"; $("f-cancel").style.display = "";
  setPicking(false); drawDraft(s.lat, s.lon); updateTestBtn();
  map.setView([s.lat, s.lon], Math.max(map.getZoom(), 9));
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}
function resetForm() {
  $("sub-form").reset(); $("f-id").value = ""; $("f-server").value = "https://ntfy.sh";
  $("f-submit").textContent = "Add location"; $("f-cancel").style.display = "none";
  clearDraft(); setPicking(false); updateTestBtn();
}

// ---- Test ntfy: unlocks only once a server AND topic are filled ----
function updateTestBtn() {
  const ready = !!($("f-server").value.trim() && $("f-topic").value.trim());
  const b = $("f-test");
  b.disabled = !ready;
  b.title = ready ? "Send a sample alert to this ntfy server/topic"
                  : "Fill in the ntfy server and topic to enable";
}

export function initSubscribers(refreshAll) {
  $("f-pick").addEventListener("click", () => setPicking(!picking));
  map.on("click", (e) => {
    if (!picking) return;
    setLatLon(e.latlng.lat, e.latlng.lng);
    drawDraft(e.latlng.lat, e.latlng.lng);
    setPicking(false);
  });
  // keep the draft pin/circle in sync as the user types coordinates or radius
  ["f-lat", "f-lon", "f-radius"].forEach((id) => $(id).addEventListener("input", () => {
    const lat = parseFloat($("f-lat").value), lon = parseFloat($("f-lon").value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    if (!draftMarker) drawDraft(lat, lon);
    else { draftMarker.setLatLng([lat, lon]); draftCircle.setLatLng([lat, lon]);
           draftCircle.setRadius(radiusMeters()); }
  }));

  $("f-cancel").addEventListener("click", resetForm);
  ["f-server", "f-topic"].forEach((id) => $(id).addEventListener("input", updateTestBtn));
  updateTestBtn();

  $("f-test").addEventListener("click", async () => {
    const b = $("f-test"), label = b.textContent;
    const body = {
      ntfy_server: $("f-server").value.trim(),
      ntfy_topic: $("f-topic").value.trim(),
      ntfy_token_ref: $("f-token").value.trim() || null,
    };
    b.disabled = true; b.textContent = "Sending…";
    try {
      const r = await api("/api/test-ntfy", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      status(r && r.note ? "Test sent - " + r.note
                         : "Test notification sent - check your ntfy device.", "ok");
    } catch (err) {
      status("Test failed: " + err.message);
    } finally {
      b.textContent = label; updateTestBtn();  // restore label + correct enabled state
    }
  });

  $("sub-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = $("f-id").value;
    const body = {
      name: $("f-name").value.trim(),
      lat: parseFloat($("f-lat").value),
      lon: parseFloat($("f-lon").value),
      radius_km: parseFloat($("f-radius").value),
      ntfy_server: $("f-server").value.trim(),
      ntfy_topic: $("f-topic").value.trim(),
      ntfy_token_ref: $("f-token").value.trim() || null,
      active: true,
    };
    try {
      if (id) await api(`/api/subscribers/${id}`, { method:"PUT",
        headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
      else await api("/api/subscribers", { method:"POST",
        headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
      resetForm(); status(id ? "Location updated." : "Location added.", "ok");
      await refreshAll();
    } catch (err) { status("Save failed: " + err.message); }
  });

  $("subs").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const id = btn.dataset.id, act = btn.dataset.act;
    try {
      if (act === "edit") { fillForm(await api(`/api/subscribers/${id}`)); return; }
      if (act === "toggle") {
        const active = btn.dataset.active !== "true";
        await api(`/api/subscribers/${id}/active`, { method:"POST",
          headers:{"Content-Type":"application/json"}, body:JSON.stringify({ active }) });
      } else if (act === "del") {
        if (!confirm("Delete this watched location?")) return;
        await api(`/api/subscribers/${id}`, { method:"DELETE" });
      }
      await refreshAll();
    } catch (err) { status("Action failed: " + err.message); }
  });
}
