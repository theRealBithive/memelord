(function () {
  const SWIPE_THRESHOLD = 55;

  function trigger(selector) {
    const el = document.querySelector(selector);
    if (el) { el.click(); el.blur(); }
  }

  // Touch swipe on the image (mobile). Swipe left → trash, matching the old
  // swipe.js "bad" gesture — many users still muscle-memory that motion even
  // though arrow keys are now prev/next for scored corpus review.
  let startX = 0;
  let startY = 0;
  let swipeActive = false;

  document.addEventListener("touchstart", (e) => {
    if (!e.target.closest(".image-wrap")) return;
    startX = e.touches[0].clientX;
    startY = e.touches[0].clientY;
    swipeActive = true;
  }, { passive: true });

  document.addEventListener("touchend", (e) => {
    if (!swipeActive) return;
    swipeActive = false;
    const dx = e.changedTouches[0].clientX - startX;
    const dy = e.changedTouches[0].clientY - startY;
    if (Math.abs(dx) <= Math.abs(dy) || Math.abs(dx) < SWIPE_THRESHOLD) return;
    if (dx < 0) trigger("[data-action='trash']");
  }, { passive: true });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    const key = e.key;

    if (key >= "1" && key <= "6") {
      e.preventDefault();
      trigger(`[data-score="${key}"]`);
    } else if (key === "ArrowLeft") {
      e.preventDefault();
      trigger("[data-action='prev']");
    } else if (key === "ArrowRight") {
      e.preventDefault();
      trigger("[data-action='next']");
    } else if (key === "Backspace" || key === "Delete") {
      e.preventDefault();
      trigger("[data-action='trash']");
    } else if (key === "f" || key === "F") {
      e.preventDefault();
      trigger("[data-action='fav']");
    } else if (key === "n" || key === "N") {
      e.preventDefault();
      document.querySelector(".nsfw-toggle-btn")?.click();
    }
  });
})();
