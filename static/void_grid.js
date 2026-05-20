(function () {
  const grid      = document.getElementById("void-grid");
  const toolbar   = document.getElementById("void-toolbar");
  const rescueBtn    = document.getElementById("void-rescue");
  const rescueFavBtn = document.getElementById("void-rescue-fav");
  const selectAllBtn = document.getElementById("void-select-all");

  if (!grid) return;

  const bulkUrl = grid.dataset.bulkUrl;
  let items = Array.from(grid.querySelectorAll(".void-item"));
  const selected = new Set(); // content hashes

  // ── Lightbox ─────────────────────────────────────────────────────────────

  const lb = document.createElement("div");
  lb.id = "lightbox";
  lb.innerHTML = `
    <div class="lb-backdrop"></div>
    <div class="lb-inner">
      <button class="lb-close" aria-label="Close">×</button>
      <button class="lb-prev" aria-label="Previous">‹</button>
      <button class="lb-next" aria-label="Next">›</button>
      <img class="lb-img" src="" alt="">
      <div class="lb-bar">
        <span class="lb-pos"></span>
      </div>
      <div class="lb-actions">
        <button class="lb-action-rescue">✓ Rescue</button>
        <button class="lb-action-rescue-fav">★ Rescue</button>
        <button class="lb-action-purge">⊘ Purge</button>
      </div>
    </div>
  `;
  document.body.appendChild(lb);

  const lbImg          = lb.querySelector(".lb-img");
  const lbPos          = lb.querySelector(".lb-pos");
  const lbPrev         = lb.querySelector(".lb-prev");
  const lbNext         = lb.querySelector(".lb-next");
  const lbRescue       = lb.querySelector(".lb-action-rescue");
  const lbRescueFav    = lb.querySelector(".lb-action-rescue-fav");
  const lbPurge        = lb.querySelector(".lb-action-purge");

  let current = 0;

  function getCsrf() {
    const val = `; ${document.cookie}`;
    const parts = val.split("; csrftoken=");
    return parts.length === 2 ? parts.pop().split(";").shift() : "";
  }

  function postAction(url, body) {
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-CSRFToken": getCsrf(),
      },
      body: new URLSearchParams(body),
    }).then((r) => r.json());
  }

  function showLb(idx) {
    current = idx;
    const item = items[idx];
    lbImg.src = item.dataset.src;
    lbPos.textContent = `${idx + 1} / ${items.length}`;
    lbPrev.disabled = idx === 0;
    lbNext.disabled = idx === items.length - 1;
  }

  function openLb(idx) {
    showLb(idx);
    lb.classList.add("lb-open");
    document.documentElement.style.overflow = "hidden";
  }

  function closeLb() {
    lb.classList.remove("lb-open");
    lbImg.src = "";
    document.documentElement.style.overflow = "";
  }

  function prevLb() { if (current > 0) showLb(current - 1); }
  function nextLb() { if (current < items.length - 1) showLb(current + 1); }

  function removeItem(item) {
    const hash = item.dataset.hash;
    selected.delete(hash);
    const idx = items.indexOf(item);
    items.splice(idx, 1);
    item.remove();
    updateToolbar();
    updateSelectAllBtn();

    if (items.length === 0) {
      closeLb();
      // Replace grid with empty-state message
      grid.innerHTML = "";
      grid.insertAdjacentHTML("afterend",
        `<div class="done">
          <p class="done-title">Trash empty</p>
          <p class="done-hint">Nothing in the void.</p>
          <div class="done-actions"><a href="/rate/inbox/" class="done-btn">← Inbox</a></div>
        </div>`
      );
      grid.remove();
      if (selectAllBtn) selectAllBtn.remove();
    } else {
      showLb(Math.min(idx, items.length - 1));
    }
  }

  function lbAction(action) {
    const item = items[current];
    postAction(item.dataset.actionUrl, { action }).then((data) => {
      if (data.rescued || data.purged) removeItem(item);
    });
  }

  lbRescue.addEventListener("click",    (e) => { e.stopPropagation(); lbAction("rescue"); });
  lbRescueFav.addEventListener("click", (e) => { e.stopPropagation(); lbAction("rescue_fav"); });
  lbPurge.addEventListener("click",     (e) => { e.stopPropagation(); lbAction("purge"); });
  lb.querySelector(".lb-backdrop").addEventListener("click", closeLb);
  lb.querySelector(".lb-close").addEventListener("click", closeLb);
  lbImg.addEventListener("click", closeLb);
  lbPrev.addEventListener("click", (e) => { e.stopPropagation(); prevLb(); });
  lbNext.addEventListener("click", (e) => { e.stopPropagation(); nextLb(); });

  // ── Selection ─────────────────────────────────────────────────────────────

  function updateToolbar() {
    const n = selected.size;
    toolbar.classList.toggle("void-toolbar--visible", n > 0);
    toolbar.querySelectorAll(".void-sel-count").forEach((el) => { el.textContent = n; });
  }

  function updateSelectAllBtn() {
    if (!selectAllBtn) return;
    const allSelected = items.length > 0 && selected.size === items.length;
    selectAllBtn.textContent = allSelected ? "deselect all" : "select all";
  }

  function setSelected(item, on) {
    const hash = item.dataset.hash;
    const cb   = item.querySelector(".void-check");
    if (on) {
      selected.add(hash);
      item.classList.add("void-item--selected");
      if (cb) cb.checked = true;
    } else {
      selected.delete(hash);
      item.classList.remove("void-item--selected");
      if (cb) cb.checked = false;
    }
  }

  function toggleSelected(item) {
    setSelected(item, !selected.has(item.dataset.hash));
    updateToolbar();
    updateSelectAllBtn();
  }

  items.forEach((item, i) => {
    const selArea = item.querySelector(".void-item-select");
    const cb      = item.querySelector(".void-check");

    // Checkbox click → select only (no lightbox)
    if (selArea) {
      selArea.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleSelected(item);
      });
    }
    if (cb) {
      cb.addEventListener("change", () => {
        setSelected(item, cb.checked);
        updateToolbar();
        updateSelectAllBtn();
      });
    }

    // Click elsewhere → open lightbox
    item.addEventListener("click", (e) => {
      if (e.target === cb || (selArea && selArea.contains(e.target))) return;
      openLb(i);
    });

    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openLb(i); }
    });
  });

  // ── Select all ───────────────────────────────────────────────────────────

  if (selectAllBtn) {
    selectAllBtn.addEventListener("click", () => {
      const allSelected = selected.size === items.length;
      items.forEach((item) => setSelected(item, !allSelected));
      updateToolbar();
      updateSelectAllBtn();
    });
  }

  // ── Bulk rescue ──────────────────────────────────────────────────────────

  function bulkRescue(fav) {
    const hashes = Array.from(selected);
    if (!hashes.length) return;
    const body = hashes.map((h) => `hashes=${encodeURIComponent(h)}`).join("&")
      + `&fav=${fav ? "1" : "0"}`;
    fetch(bulkUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-CSRFToken": getCsrf(),
      },
      body,
    }).then((r) => r.json()).then((data) => {
      if (data.rescued > 0) {
        // Remove all rescued items from DOM
        const toRemove = items.filter((item) => selected.has(item.dataset.hash));
        toRemove.forEach((item) => {
          selected.delete(item.dataset.hash);
          item.classList.remove("void-item--selected");
          const cb = item.querySelector(".void-check");
          if (cb) cb.checked = false;
          item.remove();
        });
        items = items.filter((item) => !toRemove.includes(item));
        updateToolbar();
        updateSelectAllBtn();

        if (items.length === 0) {
          grid.innerHTML = "";
          grid.insertAdjacentHTML("afterend",
            `<div class="done">
              <p class="done-title">Trash empty</p>
              <p class="done-hint">Nothing in the void.</p>
              <div class="done-actions"><a href="/rate/inbox/" class="done-btn">← Inbox</a></div>
            </div>`
          );
          grid.remove();
          if (selectAllBtn) selectAllBtn.remove();
        }
      }
    });
  }

  rescueBtn.addEventListener("click",    () => bulkRescue(false));
  rescueFavBtn.addEventListener("click", () => bulkRescue(true));

  // ── Keyboard shortcuts ───────────────────────────────────────────────────

  document.addEventListener("keydown", (e) => {
    if (!lb.classList.contains("lb-open")) return;
    if (e.key === "Escape")     { closeLb(); }
    if (e.key === "ArrowLeft")  { e.preventDefault(); prevLb(); }
    if (e.key === "ArrowRight") { e.preventDefault(); nextLb(); }
  });

  // Swipe to navigate in lightbox
  let startX = 0;
  lbImg.addEventListener("touchstart", (e) => {
    startX = e.touches[0].clientX;
  }, { passive: true });
  lbImg.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].clientX - startX;
    if (Math.abs(dx) > 40) { dx < 0 ? nextLb() : prevLb(); } else { closeLb(); }
  }, { passive: true });
})();
