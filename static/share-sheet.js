/*
 * share-sheet.js — Singleton bottom-sheet channel picker.
 *
 * Opens a slide-up panel listing notification channels as checkboxes and
 * POSTs the selected PKs to `shareUrl` through htmx.ajax, so the response's
 * HX-Trigger toast is shown by toast.js exactly like a share from the
 * single-channel form. `onResult()` is called after a completed request.
 *
 * window.ShareSheet.open({ channels, shareUrl, onResult })
 * window.ShareSheet.close()
 */
(function () {
  "use strict";

  let sheet, channelList, sendBtn, triggerEl;

  function resetSendBtn() {
    if (!sendBtn) return;
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  }

  function hideSheet() {
    if (!sheet || sheet.hidden) return;
    sheet.hidden = true;
    document.body.classList.remove("sheet-open");
    if (triggerEl) {
      triggerEl.focus();
      triggerEl = null;
    }
  }

  function build() {
    sheet = document.createElement("div");
    sheet.className = "sheet";
    sheet.setAttribute("role", "dialog");
    sheet.setAttribute("aria-modal", "true");
    sheet.setAttribute("aria-label", "Share");
    sheet.hidden = true;

    const backdrop = document.createElement("div");
    backdrop.className = "sheet-backdrop";
    backdrop.addEventListener("click", close);

    const inner = document.createElement("div");
    inner.className = "sheet-inner";

    const title = document.createElement("div");
    title.className = "sheet-title";
    title.textContent = "Share to…";

    channelList = document.createElement("div");
    channelList.className = "share-sheet-channels";

    sendBtn = document.createElement("button");
    sendBtn.type = "button";
    sendBtn.className = "sheet-send";
    sendBtn.textContent = "Send";

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "sheet-cancel";
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

    // Swap sendBtn to drop stale listeners from the previous open.
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

      htmx.ajax("POST", shareUrl, { source: sheet, swap: "none", values: { channels: selected } })
        .then(() => { if (onResult) onResult(); })
        .finally(() => { resetSendBtn(); hideSheet(); });
    });

    triggerEl = document.activeElement;
    sheet.hidden = false;
    document.body.classList.add("sheet-open");

    const firstCb = channelList.querySelector("input");
    if (firstCb) firstCb.focus();
  }

  // One picker trigger for every page: any .share-btn--picker (review card,
  // lightbox) opens the sheet with the channels base.html embeds.
  const shareChannels = JSON.parse(
    document.getElementById("share-channels-data")?.textContent || "[]"
  );
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".share-btn--picker");
    if (!btn || !btn.dataset.shareUrl || !shareChannels.length) return;
    open({ channels: shareChannels, shareUrl: btn.dataset.shareUrl });
  });

  window.ShareSheet = { open, close };
})();
