(function () {
  let startX = 0, startY = 0;
  let swipeActive = false;
  const THRESHOLD = 55;

  function trigger(selector) {
    const el = document.querySelector(selector);
    if (el) el.click();
  }

  // Swipe on the image: left = next, right = prev (gallery convention).
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
    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > THRESHOLD) {
      if (dx < 0) trigger("[data-action='next']");
      else trigger("[data-action='prev']");
    }
  }, { passive: true });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        trigger("[data-action='prev']");
        break;
      case "ArrowRight":
        e.preventDefault();
        trigger("[data-action='next']");
        break;
      case "n": case "N":
        e.preventDefault();
        document.querySelector(".nsfw-toggle-btn")?.click();
        break;
    }
  });
})();
