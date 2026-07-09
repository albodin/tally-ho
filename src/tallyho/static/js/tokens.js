// ntfy token manager: the form PUTs a value, the table and the watched-location
// dropdown only ever show names + a last-4 hint - the server never returns a
// saved token (write-only API).
import { $, api, esc, hhmm, status } from "./util.js";

let names = [];

// The subscriber form's token <select>. Editing a location whose token was
// deleted (or saved under a name we don't know) gets an explicit "(missing)"
// option instead of silently clearing the reference on the next save.
function rebuildSelect(selected) {
  const sel = $("f-token");
  sel.innerHTML = `<option value="">none (public topic)</option>` +
    names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (selected && !names.includes(selected)) {
    sel.insertAdjacentHTML("beforeend",
      `<option value="${esc(selected)}">${esc(selected)} (missing - save it below)</option>`);
  }
  sel.value = selected || "";
}
export function setTokenSelect(name) { rebuildSelect(name); }

export async function refreshTokens() {
  const rows = await api("/api/tokens");
  names = rows.map(t => t.name);
  rebuildSelect($("f-token").value);
  const tb = $("tokens");
  if (!rows.length) {
    tb.innerHTML = `<tr><td colspan="5" class="empty">No tokens saved - only needed for private ntfy topics.</td></tr>`;
    return;
  }
  tb.innerHTML = rows.map(t => `<tr>
    <td>${esc(t.name)}</td>
    <td class="muted">${esc(t.hint)}</td>
    <td class="muted">${t.refs ? t.refs + " location(s)" : "-"}</td>
    <td class="muted">${hhmm(t.updated_at)}</td>
    <td style="white-space:nowrap">
      <button class="tiny danger" data-tok-del="${esc(t.name)}">Delete</button>
    </td>
  </tr>`).join("");
}

export function initTokens() {
  $("tok-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("t-name").value.trim();
    try {
      await api(`/api/tokens/${encodeURIComponent(name)}`, { method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: $("t-value").value }) });
      $("tok-form").reset();
      status(`Token "${name}" saved.`, "ok");
      await refreshTokens();
    } catch (err) { status("Save failed: " + err.message); }
  });

  $("tokens").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-tok-del]");
    if (!btn) return;
    const name = btn.dataset.tokDel;
    if (!confirm(`Delete token "${name}"?`)) return;
    try {
      await api(`/api/tokens/${encodeURIComponent(name)}`, { method: "DELETE" });
      status(`Token "${name}" deleted.`, "ok");
      await refreshTokens();
    } catch (err) { status("Delete failed: " + err.message); }
  });
}
