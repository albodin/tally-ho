// The Leaflet map: layers, drawn icons and the 15 s rebuild (refreshMap).
// Leaflet itself is vendored (static/vendor/leaflet) and loaded as a classic
// script before the modules, so `L` is a global here.
import { api, cssVar, esc, fnum, hhmm } from "./util.js";

// preferCanvas: tracks/paths/circles render to one canvas instead of an SVG
// node per shape - SVG re-projection of thousands of points is what made
// zooming laggy.
export const map = L.map("map", { worldCopyJump: true, preferCanvas: true }).setView([20, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "© OpenStreetMap",
  updateWhenZooming: false, keepBuffer: 4
}).addTo(map);

const launchLayer = L.layerGroup().addTo(map);
const trackLayer = L.layerGroup().addTo(map);
const flightLayer = L.layerGroup().addTo(map);
const pathLayer = L.layerGroup().addTo(map);
const predLayer = L.layerGroup().addTo(map);
const landingLayer = L.layerGroup().addTo(map);
const subLayer = L.layerGroup().addTo(map);
// per-sonde prediction-history overlay - refreshMap never clears it (like the
// subscriber form's draft layer), so an open history trail survives the 15 s
// map rebuilds
export const historyLayer = L.layerGroup().addTo(map);
L.control.layers(null, {
  "Launch sites": launchLayer, "Flown tracks": trackLayer, "Sondes": flightLayer,
  "Predicted paths": pathLayer, "Predicted landings": predLayer,
  "Actual landings": landingLayer, "Watched locations": subLayer,
  "Prediction history": historyLayer
}, { collapsed: false }).addTo(map);

export const SOURCE_COLOR = {
  measured: cssVar("--accent"), gfs: cssVar("--gfs"), extrapolation: cssVar("--muted"),
};
// solid line = where the sonde has actually been (vs the dashed predicted path)
export const TRACK_COLOR = cssVar("--track");
export const WARN_COLOR = cssVar("--warn");
// actual-landing dot, shared with the history overlay
export const LANDING_STYLE = { radius:5, color:cssVar("--ok"), weight:2,
                               fillColor:"#244a30", fillOpacity:.9 };

// ---- drawn SVG icons (consistent everywhere, unlike OS emoji fonts) ----
// The fills inside are illustration colors (balloon shades, parachute canvas),
// not theme - they stay literal.
const svgIcon = (svg, w, h, anchor, popup) => L.divIcon({
  className: "svg-icon", html: svg, iconSize: [w, h],
  iconAnchor: anchor, popupAnchor: popup,
});
// balloon + payload box; the GPS fix is the payload, so anchor there
const balloonSvg = (fill, stroke) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="30" viewBox="0 0 24 30">
    <ellipse cx="12" cy="9" rx="7" ry="8" fill="${fill}" stroke="${stroke}" stroke-width="1.2"/>
    <ellipse cx="9.5" cy="6" rx="2.2" ry="3" fill="#fff" opacity=".25"/>
    <path d="M12 17 l-1.6 2.2 h3.2 z" fill="${stroke}"/>
    <line x1="12" y1="19.2" x2="12" y2="23.5" stroke="#5b6878" stroke-width="1"/>
    <rect x="9.6" y="23.5" width="4.8" height="4.5" rx="1" fill="#dbe4ee" stroke="#5b6878" stroke-width="1"/>
  </svg>`;
const parachuteSvg =
  `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="28" viewBox="0 0 24 28">
    <path d="M2.5 11 a9.5 8.5 0 0 1 19 0 z" fill="#ffb454" stroke="#a05a12" stroke-width="1.2"/>
    <path d="M2.5 11 L9.7 20.5 M12 11 L12 20.5 M21.5 11 L14.3 20.5" stroke="#a05a12" stroke-width=".9" fill="none"/>
    <rect x="9.6" y="20.5" width="4.8" height="4.8" rx="1" fill="#dbe4ee" stroke="#5b6878" stroke-width="1"/>
  </svg>`;
const landedPinSvg =
  `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="28" viewBox="0 0 22 28">
    <path d="M11 1.5 C5.8 1.5 1.8 5.6 1.8 10.7 c0 6.8 9.2 15.6 9.2 15.6 s9.2-8.8 9.2-15.6 C20.2 5.6 16.2 1.5 11 1.5 z"
          fill="#39b362" stroke="#1c6e3a" stroke-width="1.2"/>
    <path d="M7 10.5 l2.6 2.8 L15 7.6" fill="none" stroke="#fff" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
const rocketSvg =
  `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="30" viewBox="0 0 22 30">
    <path d="M11 1 C14 4 15.5 8.5 15.5 12.5 c0 2.6 -.4 5 -1.2 6.8 H7.7 C6.9 17.5 6.5 15.1 6.5 12.5 C6.5 8.5 8 4 11 1 z"
          fill="#cdd7e3" stroke="#4e5c6e" stroke-width="1.2"/>
    <circle cx="11" cy="10" r="2" fill="#4ea1ff" stroke="#1f5e9e" stroke-width=".8"/>
    <path d="M6.6 13.5 L3 19.5 L6.9 18 z" fill="#e5484d" stroke="#8c1d22" stroke-width=".8"/>
    <path d="M15.4 13.5 L19 19.5 L15.1 18 z" fill="#e5484d" stroke="#8c1d22" stroke-width=".8"/>
    <path d="M9 19.3 h4 l-.5 2 h-3 z" fill="#8a97a8"/>
    <path d="M11 21.5 c1.6 2.2 1.6 4.3 0 7 c-1.6 -2.7 -1.6 -4.8 0 -7 z" fill="#ffb454" stroke="#e07b1f" stroke-width=".7"/>
  </svg>`;
const targetSvg =
  `<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 26 26">
    <circle cx="13" cy="13" r="9" fill="rgba(255,180,84,.18)" stroke="#ffb454" stroke-width="2"/>
    <circle cx="13" cy="13" r="3.2" fill="#ffb454"/>
    <path d="M13 1 v5 M13 20 v5 M1 13 h5 M20 13 h5" stroke="#ffb454" stroke-width="2" stroke-linecap="round"/>
  </svg>`;
const FLIGHT_ICONS = {
  ASCENT: svgIcon(balloonSvg("#e5484d", "#8c1d22"), 24, 30, [12, 26], [0, -24]),
  FLOAT:  svgIcon(balloonSvg("#a78bfa", "#6d4fc4"), 24, 30, [12, 26], [0, -24]),
  DESCENT: svgIcon(parachuteSvg, 24, 28, [12, 23], [0, -21]),
  LANDED: svgIcon(landedPinSvg, 22, 28, [11, 27], [0, -25]),
};
const flightIcon = (state) => FLIGHT_ICONS[state] || FLIGHT_ICONS.ASCENT;
const launchIcon = svgIcon(rocketSvg, 22, 30, [11, 28], [0, -26]);
// predicted landing spot - a crosshair, not Leaflet's oversized default pin
export const predIcon = svgIcon(targetSvg, 26, 26, [13, 13], [0, -12]);

let didFit = false;
let lastMapJson = "";
export async function refreshMap() {
  const fc = await api("/api/map");
  // Rebuilding every layer is the expensive part (and it closes open popups);
  // skip it entirely when nothing on the map has changed since last refresh.
  const fcJson = JSON.stringify(fc);
  if (fcJson === lastMapJson) return;
  lastMapJson = fcJson;
  launchLayer.clearLayers(); trackLayer.clearLayers();
  flightLayer.clearLayers(); pathLayer.clearLayers(); predLayer.clearLayers();
  landingLayer.clearLayers(); subLayer.clearLayers();
  const pts = [];
  for (const f of fc.features) {
    const p = f.properties;
    if (p.kind === "track") {
      // solid line of where the sonde has actually flown (launch → now)
      const line = f.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
      L.polyline(line, { color: TRACK_COLOR, weight:2, opacity:.85 }).bindPopup(
        `<b>${esc(p.serial)}</b> flown track`
      ).addTo(trackLayer);
      for (const ll of line) pts.push(ll);
      continue;
    }
    if (p.kind === "path") {
      // GeoJSON LineString: [[lon, lat], ...] → Leaflet [lat, lon]
      const line = f.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
      L.polyline(line, { color: SOURCE_COLOR[p.source] || WARN_COLOR, weight:2,
        opacity:.8, dashArray:"6,6" }).bindPopup(
        `<b>${esc(p.serial)}</b> predicted path<br>source ${esc(p.source)} · land ${hhmm(p.eta)}`
      ).addTo(pathLayer);
      for (const ll of line) pts.push(ll);
      continue;
    }
    const [lon, lat] = f.geometry.coordinates;  // GeoJSON is [lon, lat]
    if (p.kind === "launch") {
      L.marker([lat, lon], { icon: launchIcon }).bindPopup(
        `<b>${esc(p.serial)}</b> launch site<br>${fnum(lat,5)}, ${fnum(lon,5)}`
        + `<br>first seen ${hhmm(p.first_seen)}`
      ).addTo(launchLayer);
      pts.push([lat, lon]);
    } else if (p.kind === "flight") {
      L.marker([lat, lon], { icon: flightIcon(p.state) }).bindPopup(
        `<b>${esc(p.serial)}</b> (${esc(p.ftype||"?")})<br>${esc(p.state)} · ${fnum(p.alt,0)} m`
        + `<br><a href="https://sondehub.org/${encodeURIComponent(p.serial)}" target="_blank">track ↗</a>`
      ).addTo(flightLayer);
      pts.push([lat, lon]);
    } else if (p.kind === "prediction") {
      L.circle([lat, lon], { radius:(p.uncertainty_radius_km||0)*1000,
        color: WARN_COLOR, weight:1, fillOpacity:.08 }).addTo(predLayer);
      L.marker([lat, lon], { icon: predIcon }).bindPopup(
        `<b>${esc(p.serial)}</b> predicted landing<br>${fnum(lat,5)}, ${fnum(lon,5)}`
        + `<br>ETA ${hhmm(p.eta)} · ±${fnum(p.uncertainty_radius_km,1)} km · ${esc(p.source)}`
      ).addTo(predLayer);
      pts.push([lat, lon]);
    } else if (p.kind === "landing") {
      L.circleMarker([lat, lon], LANDING_STYLE).bindPopup(
        `<b>${esc(p.serial)}</b> (${esc(p.ftype||"?")}) landed<br>${fnum(lat,5)}, ${fnum(lon,5)}`
        + `<br>${hhmm(p.landed_at)} · ${fnum(p.alt,0)} m · ${esc(p.detected_by||"")}`
      ).addTo(landingLayer);
      pts.push([lat, lon]);
    } else if (p.kind === "subscriber") {
      // pure range ring - non-interactive so clicks pass through to sondes underneath
      L.circle([lat, lon], { radius:(p.radius_km||0)*1000,
        color: p.active ? "#a8f0bd" : "#5a6a7a", weight: p.active ? 3 : 1.5, dashArray:"5,5",
        opacity: p.active ? .95 : .5,
        fillColor: p.active ? "#7bd88f" : "#5a6a7a",
        fillOpacity: p.active ? .15 : .04, interactive:false }).addTo(subLayer);
      pts.push([lat, lon]);
    }
  }
  if (!didFit && pts.length) { map.fitBounds(pts, { padding:[40,40], maxZoom:10 }); didFit = true; }
}
