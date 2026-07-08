// The dashboard tables: pure fetch-and-render, re-run on the 15 s poll.
// Row actions are wired by delegation elsewhere: "history" links (data-hist)
// in history.js, subscriber buttons (data-act) in subscribers.js.
import { $, api, esc, fnum, hhmm } from "./util.js";

export async function refreshFlights() {
  const rows = await api("/api/flights");
  const tb = $("flights");
  if (!rows.length) { tb.innerHTML = `<tr><td colspan="10" class="empty">No active flights.</td></tr>`; return; }
  tb.innerHTML = rows.map(f => {
    const pr = f.prediction;
    const land = pr ? `<a href="https://www.openstreetmap.org/?mlat=${pr.land_lat}&mlon=${pr.land_lon}#map=11/${pr.land_lat}/${pr.land_lon}" target="_blank">${fnum(pr.land_lat,4)}, ${fnum(pr.land_lon,4)}</a>` : "-";
    return `<tr>
      <td><a href="https://sondehub.org/${encodeURIComponent(f.serial)}" target="_blank">${esc(f.serial)}</a></td>
      <td>${esc(f.type||"?")}</td>
      <td><span class="pill ${esc(f.state)}">${esc(f.state)}</span></td>
      <td>${fnum(f.last_alt,0)} m</td>
      <td>${land}</td>
      <td>${pr ? hhmm(pr.land_eta) : "-"}</td>
      <td>${pr ? "±"+fnum(pr.uncertainty_radius_km,1)+" km" : "-"}</td>
      <td class="muted">${pr ? esc(pr.source) : "-"}</td>
      <td class="muted">${hhmm(f.last_seen)}</td>
      <td><a href="#" data-hist data-serial="${esc(f.serial)}" data-day="${esc(f.launch_day)}">history</a></td>
    </tr>`;
  }).join("");
}

export async function refreshAlerts() {
  const rows = await api("/api/alerts");
  const tb = $("alerts");
  if (!rows.length) { tb.innerHTML = `<tr><td colspan="6" class="empty">No alerts yet.</td></tr>`; return; }
  tb.innerHTML = rows.map(a => `<tr>
    <td class="muted">${hhmm(a.sent_at)}</td>
    <td>${esc(a.alert_type)}</td>
    <td>${esc(a.subscriber_name||("#"+a.subscriber_id))}</td>
    <td>${esc(a.serial)}</td>
    <td>${a.distance_km==null?"-":fnum(a.distance_km,1)+" km"}</td>
    <td>${a.land_lat==null?"-":fnum(a.land_lat,4)+", "+fnum(a.land_lon,4)}</td>
  </tr>`).join("");
}

export async function refreshSubs() {
  const rows = await api("/api/subscribers");
  const tb = $("subs");
  if (!rows.length) { tb.innerHTML = `<tr><td colspan="7" class="empty">No watched locations yet - add one below.</td></tr>`; return; }
  tb.innerHTML = rows.map(s => `<tr>
    <td>${esc(s.name)}</td>
    <td>${fnum(s.lat,4)}, ${fnum(s.lon,4)}</td>
    <td>${fnum(s.radius_km,0)} km</td>
    <td class="muted">${s.notify ? esc(s.ntfy_server)+"/"+esc(s.ntfy_topic) : "watch-only (no ntfy)"}</td>
    <td class="muted">${esc(s.ntfy_token_ref||"-")}</td>
    <td>${s.active ? '<span class="ok">yes</span>' : '<span class="muted">no</span>'}</td>
    <td style="white-space:nowrap">
      <button class="tiny secondary" data-act="edit" data-id="${s.id}">Edit</button>
      <button class="tiny secondary" data-act="toggle" data-id="${s.id}" data-active="${s.active}">${s.active?"Deactivate":"Activate"}</button>
      <button class="tiny danger" data-act="del" data-id="${s.id}">Delete</button>
    </td>
  </tr>`).join("");
}

export async function refreshAccuracy() {
  const data = await api("/api/accuracy");
  const sum = data.summary, tb = $("accuracy");
  $("acc-summary").textContent = sum
    ? `mean final error ${fnum(sum.mean_final_error_km,2)} km · calibration `
      + `${sum.calibration_rate==null?"-":Math.round(sum.calibration_rate*100)+"%"} `
      + `· ${sum.n_flights} flight(s)`
    : "";
  // Error by altitude-at-prediction (high → low): predictions are noisy up high
  // and converge as the sonde descends. "Final error" above is just the last,
  // lowest-altitude prediction - this row shows the whole convergence.
  const bkEl = $("acc-buckets");
  if (sum && sum.bucket_mean_error_km) {
    const order = ["20-99km", "10-20km", "5-10km", "2-5km", "0-2km"];
    const chips = order.filter(k => (sum.bucket_counts[k] || 0) > 0).map(k =>
      `${k.replace("km"," km")} alt: <b>${fnum(sum.bucket_mean_error_km[k],1)} km</b>`
      + ` <span class="muted">(n=${sum.bucket_counts[k]})</span>`);
    bkEl.innerHTML = chips.length
      ? "error by altitude-at-prediction - " + chips.join(" · ") : "";
  } else { bkEl.innerHTML = ""; }
  if (!data.flights.length) {
    tb.innerHTML = `<tr><td colspan="5" class="empty">No landings scored yet - predictions are scored once a sonde lands.</td></tr>`;
    return;
  }
  tb.innerHTML = data.flights.map(r => `<tr>
    <td><a href="https://sondehub.org/${encodeURIComponent(r.serial)}" target="_blank">${esc(r.serial)}</a></td>
    <td><a href="https://www.openstreetmap.org/?mlat=${r.truth_lat}&mlon=${r.truth_lon}#map=12/${r.truth_lat}/${r.truth_lon}" target="_blank">${fnum(r.truth_lat,4)}, ${fnum(r.truth_lon,4)}</a></td>
    <td>${r.final_error_km==null?"-":fnum(r.final_error_km,2)+" km"}</td>
    <td class="muted">${r.n_predictions}</td>
    <td>${r.launch_day ? `<a href="#" data-hist data-serial="${esc(r.serial)}" data-day="${esc(r.launch_day)}">history</a>` : ""}</td>
  </tr>`).join("");
}
