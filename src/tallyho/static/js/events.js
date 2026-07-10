// Doorbell listener. Reconnect handling is ours; refetching stays in the
// existing modules. `map` is eventName -> [refresher, ...].
import { api } from "./util.js";

export function connectEvents(map, { onUp, onDown }) {
  let es = null, retryMs = 2000;
  const open = () => {
    es = new EventSource("/api/events");
    es.onopen = () => { retryMs = 2000; onUp(); };
    for (const [name, fns] of Object.entries(map))
      es.addEventListener(name, () => fns.forEach(f => f().catch(() => {})));
    es.onerror = async () => {
      if (es.readyState !== EventSource.CLOSED) return;  // browser is auto-retrying
      // Permanent failure (spec: any non-200 - e.g. 401 after the 14-day session
      // expires; stops EventSource retrying forever). Probe with fetch:
      // api() redirects to /login on 401; otherwise back off and retry ourselves.
      onDown();
      try { await api("/api/stats"); } catch {}
      setTimeout(open, retryMs = Math.min(retryMs * 2, 30000));
    };
  };
  open();
}
