/*
 * share-sheet.js — Singleton bottom-sheet channel picker.
 *
 * Opens a slide-up panel listing notification channels as checkboxes.
 * All channels are pre-checked (faster on mobile: uncheck to skip).
 * POSTs selected PKs to `shareUrl` and calls `onResult(html)` on success.
 *
 * window.ShareSheet.open({ channels, shareUrl, onResult })
 * window.ShareSheet.close()
 */
(function () {
  "use strict";

  let sheet, backdrop, channelList, sendBtn, triggerEl, inFlight;

  function resetSendBtn() {
    if (!sendBtn) return;
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  }

  function hideSheet() {
    if (!sheet || sheet.hidden) return;
    sheet.hidden = true;
    document.body.classList.remove("share-sheet--open");
    if (triggerEl) {
      triggerEl.focus();
      triggerEl = null;
    }
  }

  function getCsrf() {
    const v = `; ${document.cookie}`;
    const p = v.split("; csrftoken=");
    return p.length === 2 ? p.pop().split(";").shift() : "";
  }

  function build() {
    sheet = document.createElement("div");
    sheet.className = "share-sheet";
    sheet.setAttribute("role", "dialog");
    sheet.setAttribute("aria-modal", "true");
    sheet.setAttribute("aria-label", "Share");
    sheet.hidden = true;

    backdrop = document.createElement("div");
    backdrop.className = "share-sheet-backdrop";
    backdrop.addEventListener("click", close);

    const inner = document.createElement("div");
    inner.className = "share-sheet-inner";

    const title = document.createElement("div");
    title.className = "share-sheet-title";
    title.textContent = "Share to…";

    channelList = document.createElement("div");
    channelList.className = "share-sheet-channels";

    sendBtn = document.createElement("button");
    sendBtn.type = "button";
    sendBtn.className = "share-sheet-send";
    sendBtn.textContent = "Send";

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "share-sheet-cancel";
    cancelBtn.textContent = "Cancel";
    cancelBtn.addEventListener("click", close);

    inner.appendChild(title);
    inner.appendChild(channelList);
    inner.appendChild(sendBtn);
    inner.appendChild(cancelBtn);
    sheet.appendChild(backdrop);
    sheet.appendChild(inner);
    document.body.appendChild(sheet);

    document.addEventListener("keydown", (e) => {
      if (sheet.hidden) return;
      if (e.key === "Escape") { e.preventDefault(); close(); }
    });
  }

  function close() {
    if (inFlight) {
      inFlight.abort();
      inFlight = null;
    }
    resetSendBtn();
    hideSheet();
  }

  function open({ channels, shareUrl, onResult }) {
    if (!sheet) build();

    channelList.innerHTML = "";
    channels.forEach(({ pk, name }) => {
      const label = document.createElement("label");
      label.className = "share-sheet-channel";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = String(pk);
      cb.checked = false;
      cb.className = "share-sheet-checkbox";

      const nameEl = document.createElement("span");
      nameEl.textContent = name;

      label.appendChild(cb);
      label.appendChild(nameEl);
      channelList.appendChild(label);
    });

    // Swap sendBtn to drop stale listeners from previous open.
    const newSend = sendBtn.cloneNode(true);
    sendBtn.replaceWith(newSend);
    sendBtn = newSend;
    resetSendBtn();

    sendBtn.addEventListener("click", () => {
      const selected = Array.from(
        channelList.querySelectorAll("input:checked")
      ).map((i) => i.value);
      if (!selected.length) return;

      sendBtn.disabled = true;
      sendBtn.textContent = "Sending…";

      const params = new URLSearchParams();
      selected.forEach((pk) => params.append("channels", pk));

      const ac = new AbortController();
      inFlight = ac;

      fetch(shareUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          "X-CSRFToken": getCsrf(),
        },
        body: params,
        signal: ac.signal,
      })
        .then((r) => r.text())
        .then((html) => {
          inFlight = null;
          resetSendBtn();
          hideSheet();
          if (onResult) onResult(html);
        })
        .catch((err) => {
          if (err.name === "AbortError") return;
          resetSendBtn();
        })
        .finally(() => {
          if (inFlight === ac) inFlight = null;
        });
    });

    triggerEl = document.activeElement;
    sheet.hidden = false;
    document.body.classList.add("share-sheet--open");

    // Move focus to the first checkbox for keyboard/screen-reader users.
    const firstCb = channelList.querySelector("input");
    if (firstCb) firstCb.focus();
  }

  window.ShareSheet = { open, close };
})();
