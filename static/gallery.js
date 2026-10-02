// Gallery lightbox (UI contract V9). The panel is a server-rendered partial in
// #lightbox-content that reuses the review card's score row and tag editor;
// prev/next walk the grid in DOM order (random sort and tag filters included),
// and each action's response swaps the grid card and the nav badges out of
// band, so the grid never shows a stale score.
(function () {
  const grid = document.querySelector(".gallery-grid");
  const lightbox = document.getElementById("lightbox");
  const content = document.getElementById("lightbox-content");
  if (!grid || !lightbox || !content) return;

  let currentHash = null;
  let currentIndex = -1; // position at load time, for the step after a purge

  function items() {
    return Array.from(grid.querySelectorAll(".gallery-item"));
  }

  function indexOfCurrent() {
    return items().findIndex((item) => item.dataset.hash === currentHash);
  }

  function load(item) {
    currentHash = item.dataset.hash;
    currentIndex = items().indexOf(item);
    htmx.ajax("GET", item.dataset.lightboxUrl, { target: content, swap: "innerHTML" });
  }

  function open(item) {
    lightbox.hidden = false;
    document.body.classList.add("lightbox-open");
    load(item);
  }

  function close() {
    lightbox.hidden = true;
    document.body.classList.remove("lightbox-open");
    content.innerHTML = "";
    currentHash = null;
    currentIndex = -1;
    window.ShareSheet?.close();
    window.TagModal?.close();
  }

  function step(delta) {
    const list = items();
    const target = list[indexOfCurrent() + delta];
    if (target) load(target);
  }

  function refreshPosition() {
    const list = items();
    const index = indexOfCurrent();
    const pos = content.querySelector("[data-pos]");
    if (pos && index >= 0) pos.textContent = `${index + 1} / ${list.length}`;
    const prev = content.querySelector("[data-action='prev']");
    const next = content.querySelector("[data-action='next']");
    if (prev) prev.disabled = index <= 0;
    if (next) next.disabled = index < 0 || index >= list.length - 1;
  }

  grid.addEventListener("click", (e) => {
    const item = e.target.closest(".gallery-item");
    if (!item) return;
    e.preventDefault();
    open(item);
  });

  lightbox.querySelector(".lb-backdrop").addEventListener("click", close);

  content.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    if (btn.dataset.action === "prev") step(-1);
    else if (btn.dataset.action === "next") step(1);
    else if (btn.dataset.action === "close") close();
  });

  document.body.addEventListener("htmx:afterSwap", (e) => {
    if (e.detail.target !== content || lightbox.hidden) return;
    if (content.querySelector(".lb-img")) {
      refreshPosition();
      return;
    }
    // An empty panel means the image was purged and its card has left the grid.
    const list = items();
    if (!list.length || currentIndex < 0) {
      close();
      return;
    }
    load(list[Math.min(currentIndex, list.length - 1)]);
  });

  document.addEventListener("keydown", (e) => {
    if (lightbox.hidden) return;
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (document.body.classList.contains("sheet-open")) return;
    const key = e.key;
    if (key === "Escape") { close(); return; }
    if (key === "ArrowLeft") { e.preventDefault(); step(-1); return; }
    if (key === "ArrowRight") { e.preventDefault(); step(1); return; }
    if (key >= "0" && key <= "6") {
      e.preventDefault();
      content.querySelector(`[data-score="${key}"]`)?.click();
    } else if (key === "n" || key === "N") {
      e.preventDefault();
      content.querySelector("[data-action='nsfw']")?.click();
    } else if (key === "s" || key === "S") {
      e.preventDefault();
      content.querySelector("[data-action='share']")?.click();
    }
  });

  // Swipe on the picture steps through the grid; a plain tap closes.
  let startX = 0;
  content.addEventListener("touchstart", (e) => {
    if (e.target.classList.contains("lb-img")) startX = e.touches[0].clientX;
  }, { passive: true });
  content.addEventListener("touchend", (e) => {
    if (!e.target.classList.contains("lb-img")) return;
    const dx = e.changedTouches[0].clientX - startX;
    if (dx < -40) step(1);
    else if (dx > 40) step(-1);
    else close();
  }, { passive: true });
})();
