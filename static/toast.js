// One feedback mechanism for every page (UI contract V7).
// Two sources, one sink: htmx responses carry an HX-Trigger header
// {"toast": {"message", "kind"}} that htmx turns into a "toast" DOM event, and
// full-page responses render Django messages into #server-toasts as JSON.
// Messages are inserted with textContent, so they are text, never markup.
(function () {
  const TIMEOUT_MS = { ok: 2500, info: 2500, error: 5000 };

  function show(message, kind) {
    const slot = document.getElementById("toast-slot");
    if (!slot) return;
    const safeKind = TIMEOUT_MS[kind] ? kind : "info";
    const el = document.createElement("div");
    el.className = `toast toast--${safeKind}`;
    el.setAttribute("role", safeKind === "error" ? "alert" : "status");
    el.textContent = message;
    slot.replaceChildren(el); // one toast at a time; the newest wins
    setTimeout(() => {
      el.classList.add("toast--hide");
      setTimeout(() => el.remove(), 300);
    }, TIMEOUT_MS[safeKind]);
  }

  document.body.addEventListener("toast", (e) => {
    const detail = e.detail || {};
    if (detail.message) show(String(detail.message), detail.kind);
  });

  const serverToasts = document.getElementById("server-toasts");
  if (serverToasts) {
    try {
      JSON.parse(serverToasts.textContent).forEach((t) => show(t.message, t.kind));
    } catch (_) {
      // Malformed JSON only means no toast; the page itself is unaffected.
    }
  }

  window.Toast = { show };
})();
