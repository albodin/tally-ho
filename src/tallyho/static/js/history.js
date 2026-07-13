// Per-sonde prediction history: the panel (table + SVG chart) and its map
// overlay. Distance semantics: vs the actual landing once recorded, else vs
// the current latest prediction (drift) - the backend picks and reports which.
import { $, api, cssNum, cssVar, esc, fnum, hhmm } from "./util.js";
import { burstIcon, historyLayer, landingStyle, map, predIcon,
         sourceColor, trackColor } from "./map.js";

const LINE = cssVar("--line"), MUTED = cssVar("--muted"),
      ACCENT = cssVar("--accent"), BAD = cssVar("--bad");

let historyKey = null;          // { serial, day } while the panel is open
let lastHistJson = "";
// A LANDED flight with its landing recorded can't gain predictions or move -
// once we've rendered it, stop re-fetching on the 15 s poll (reopening the
// panel fetches fresh). Active flights keep polling: their history grows.
let histFinal = false;
const histDistLabel = (ref) => ref === "landing"
  ? "error vs actual landing" : "drift vs latest prediction";

// concise hover line shared by chart points and map dots
const histTip = (p) => `${hhmm(p.predicted_at)} · `
  + `${p.alt_at_pred==null ? "alt -" : fnum(p.alt_at_pred,0)+" m"} · `
  + `${fnum(p.distance_km,2)} km`;

function renderHistChart(preds, ref) {
  const el = $("hist-chart");
  const pts = preds.filter(p => p.distance_km != null && p.predicted_at);
  if (!pts.length) { el.innerHTML = ""; return; }
  const W = 640, H = 170, padL = 46, padR = 46, padT = 16, padB = 24;
  const ts = pts.map(p => Date.parse(p.predicted_at));
  const t0 = Math.min(...ts), t1 = Math.max(...ts);
  // Linear km scale: early high-altitude predictions can sit far out, which
  // compresses the late convergence tail - accepted, it shows the spread honestly.
  const dmax = Math.max(...pts.map(p => p.distance_km)) || 1;
  const x = t => t1 === t0 ? padL + (W - padL - padR) / 2
                           : padL + (W - padL - padR) * (t - t0) / (t1 - t0);
  const y = d => padT + (H - padT - padB) * (1 - d / dmax);
  const grid = [0, dmax / 2, dmax].map(v =>
    `<line x1="${padL}" y1="${y(v)}" x2="${W - padR}" y2="${y(v)}" stroke="${LINE}"/>
     <text x="${padL - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="${MUTED}">${fnum(v, 1)}</text>`
  ).join("");
  // altitude-at-prediction on its own right-hand scale (km), so the error's
  // convergence can be read against where in the descent each prediction ran
  const apts = pts.filter(p => p.alt_at_pred != null);
  let altLine = "", altAxis = "";
  if (apts.length) {
    const trackC = trackColor();
    const amax = Math.max(...apts.map(p => p.alt_at_pred)) || 1;
    const ya = a => padT + (H - padT - padB) * (1 - a / amax);
    altLine = apts.length > 1 ? `<polyline fill="none" stroke="${trackC}"
      stroke-width="1.2" stroke-dasharray="4,3" opacity=".8"
      points="${apts.map(p => `${x(Date.parse(p.predicted_at))},${ya(p.alt_at_pred)}`).join(" ")}"/>` : "";
    altAxis = [0, amax / 2, amax].map(v =>
      `<text x="${W - padR + 6}" y="${ya(v) + 4}" font-size="11" fill="${trackC}">${fnum(v / 1000, 1)}</text>`
    ).join("") + `<text x="${W - padR}" y="${padT - 5}" text-anchor="end"
      font-size="11" fill="${trackC}">altitude km</text>`;
  }
  const line = pts.length > 1 ? `<polyline fill="none" stroke="${ACCENT}" stroke-width="1.5"
    points="${pts.map((p, i) => `${x(ts[i])},${y(p.distance_km)}`).join(" ")}"/>` : "";
  // each dot pairs with a larger invisible hit circle carrying a native
  // <title> tooltip - hover anywhere near a point to read its values
  const dots = pts.map((p, i) =>
    `<g><circle cx="${x(ts[i])}" cy="${y(p.distance_km)}" r="${i === pts.length - 1 ? 4 : 2.5}"
      fill="${sourceColor(p.source)}"/>
    <circle cx="${x(ts[i])}" cy="${y(p.distance_km)}" r="7" fill="transparent"
      ><title>${histTip(p)}</title></circle></g>`).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg"
      style="width:100%; max-width:760px; display:block">
    ${grid}${altLine}${line}${dots}${altAxis}
    <text x="${padL}" y="${H - 6}" font-size="11" fill="${MUTED}">${hhmm(pts[0].predicted_at)}</text>
    <text x="${W - padR}" y="${H - 6}" text-anchor="end" font-size="11" fill="${MUTED}">${hhmm(pts[pts.length - 1].predicted_at)}</text>
    <text x="${padL}" y="${padT - 5}" font-size="11" fill="${ACCENT}">km ${histDistLabel(ref)}</text>
  </svg>`;
}

function renderHistPanel(d) {
  const state = esc(d.flight.state);
  $("hist-title").innerHTML =
    `<a href="https://sondehub.org/${encodeURIComponent(d.serial)}" target="_blank">${esc(d.serial)}</a>
     <span class="pill ${state}">${state}</span>
     · ${d.predictions.length} prediction(s) · distance = ${histDistLabel(d.distance_reference)}`;
  $("hist-dist-col").textContent = d.distance_reference === "landing" ? "Error" : "Drift";
  const tb = $("hist-rows");
  if (!d.predictions.length) {
    tb.innerHTML = `<tr><td colspan="7" class="empty">No predictions recorded for this flight.</td></tr>`;
  } else {
    // newest first - the latest prediction is what you act on
    tb.innerHTML = [...d.predictions].reverse().map(p => `<tr>
      <td class="muted">${hhmm(p.predicted_at)}</td>
      <td>${fnum(p.alt_at_pred,0)} m</td>
      <td><a href="https://www.openstreetmap.org/?mlat=${p.land_lat}&mlon=${p.land_lon}#map=11/${p.land_lat}/${p.land_lon}" target="_blank">${fnum(p.land_lat,4)}, ${fnum(p.land_lon,4)}</a></td>
      <td>${hhmm(p.land_eta)}</td>
      <td>${p.uncertainty_radius_km==null?"-":"±"+fnum(p.uncertainty_radius_km,1)+" km"}</td>
      <td class="muted">${esc(p.source)}</td>
      <td>${p.distance_km==null?"-":fnum(p.distance_km,2)+" km"}</td>
    </tr>`).join("");
  }
  renderHistChart(d.predictions, d.distance_reference);
}

function drawHistoryOverlay(d) {
  historyLayer.clearLayers();
  const preds = d.predictions, n = preds.length, pts = [];
  if (d.track.length > 1) {
    // flown track of a LANDED flight - active flights' tracks are already on the map
    L.polyline(d.track.map(p => [p[0], p[1]]),
      { color: trackColor(), weight:2, opacity: cssNum("--track-opacity") })
      .bindPopup(`<b>${esc(d.serial)}</b> flown track`).addTo(historyLayer);
  }
  if (d.burst) {
    // where the ascent actually ended (the track's apogee)
    L.marker([d.burst.lat, d.burst.lon], { icon: burstIcon })
      .bindTooltip(`burst · ${fnum(d.burst.alt,0)} m`).bindPopup(
      `<b>${esc(d.serial)}</b> burst<br>${fnum(d.burst.lat,5)}, ${fnum(d.burst.lon,5)}`
      + `<br>${fnum(d.burst.alt,0)} m`
    ).addTo(historyLayer);
  }
  if (n > 1)
    L.polyline(preds.map(p => [p.land_lat, p.land_lon]),
      { color: MUTED, weight:1.5, opacity:.5 }).addTo(historyLayer);
  preds.forEach((p, i) => {
    const fade = .25 + .75 * (n > 1 ? i / (n - 1) : 1);   // older → fainter
    L.circleMarker([p.land_lat, p.land_lon], {
      radius:4, color: sourceColor(p.source), weight:1.5,
      opacity: fade, fillOpacity: .5 * fade,
    }).bindTooltip(histTip(p)).bindPopup(
      `<b>${esc(d.serial)}</b> prediction at ${hhmm(p.predicted_at)}`
      + `<br>alt ${fnum(p.alt_at_pred,0)} m · ±${fnum(p.uncertainty_radius_km,1)} km · ${esc(p.source)}`
      + `<br>${fnum(p.distance_km,2)} km ${histDistLabel(d.distance_reference)}`
    ).addTo(historyLayer);
    pts.push([p.land_lat, p.land_lon]);
  });
  if (n) {
    const last = preds[n - 1];
    L.marker([last.land_lat, last.land_lon], { icon: predIcon() })
      .bindTooltip(`latest prediction · ${histTip(last)}`).bindPopup(
      `<b>${esc(d.serial)}</b> latest prediction (${hhmm(last.predicted_at)})`
    ).addTo(historyLayer);
  }
  if (d.landing) {
    const lnd = d.landing;
    L.circleMarker([lnd.land_lat, lnd.land_lon], landingStyle())
      .bindTooltip(`actual landing · ${hhmm(lnd.landed_at)}`).bindPopup(
      `<b>${esc(d.serial)}</b> actual landing<br>${fnum(lnd.land_lat,5)}, ${fnum(lnd.land_lon,5)}`
      + `<br>${hhmm(lnd.landed_at)} · ${esc(lnd.detected_by||"")}`
    ).addTo(historyLayer);
    pts.push([lnd.land_lat, lnd.land_lon]);
    if (n) {
      const last = preds[n - 1];
      L.polyline([[last.land_lat, last.land_lon], [lnd.land_lat, lnd.land_lon]],
        { color: BAD, weight:1.5, dashArray:"3,5" })
        .bindTooltip(`final error ${fnum(last.distance_km,2)} km`, { sticky:true })
        .bindPopup(`final error ${fnum(last.distance_km,2)} km`).addTo(historyLayer);
    }
  }
  return pts;
}

export async function refreshHistory(fit) {
  if (!historyKey || histFinal) return;
  let d;
  try {
    d = await api(`/api/flights/${encodeURIComponent(historyKey.serial)}/${encodeURIComponent(historyKey.day)}/history`);
  } catch (e) {
    closeHistory();   // flight gone (history cleared / restart) - fold quietly
    return;
  }
  histFinal = d.flight.state === "LANDED" && !!d.landing;
  const j = JSON.stringify(d);
  // unchanged data: skip the rebuild so open popups survive (same trick as the map)
  if (!fit && j === lastHistJson) return;
  lastHistJson = j;
  renderHistPanel(d);
  const pts = drawHistoryOverlay(d);
  if (fit && pts.length) map.fitBounds(pts, { padding:[40,40], maxZoom:12 });
}
async function openHistory(serial, day) {
  historyKey = { serial, day };
  lastHistJson = ""; histFinal = false;
  $("history-section").style.display = "";
  await refreshHistory(true);
  if (historyKey) $("history-section").scrollIntoView({ behavior:"smooth", block:"start" });
}
function closeHistory() {
  historyKey = null; lastHistJson = ""; histFinal = false;
  $("history-section").style.display = "none";
  historyLayer.clearLayers();
}
$("hist-close").addEventListener("click", closeHistory);
// the flights + accuracy tables render "history" links (data-hist); delegate
// on the (static) tbodys so re-rendered rows keep working
for (const id of ["flights", "accuracy"]) {
  $(id).addEventListener("click", (e) => {
    const a = e.target.closest("a[data-hist]");
    if (!a) return;
    e.preventDefault();
    openHistory(a.dataset.serial, a.dataset.day);
  });
}
