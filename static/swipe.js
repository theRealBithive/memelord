(function () {
  let startX = 0, startY = 0;
  let swipeActive = false;
  const THRESHOLD = 55;

  function trigger(action) {
    if (action === "nsfw_toggle") {
      const btn = document.querySelector(".nsfw-toggle-btn");
      if (btn) { btn.click(); btn.blur(); }
      return;
    }
    const btn = document.querySelector(`[data-action="${action}"]`);
    if (btn) { btn.click(); btn.blur(); }
  }

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
    if (Math.abs(dx) > Math.abs(dy)) {
      if (dx < -THRESHOLD) trigger("bad");
      else if (dx > THRESHOLD) trigger("good");
    } else {
      if (dy < -THRESHOLD) trigger("fav");
      else if (dy > THRESHOLD) trigger("skip");
    }
  }, { passive: true });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    const map = {
      ArrowLeft: "bad", ArrowRight: "good", ArrowUp: "fav", ArrowDown: "skip",
      n: "nsfw_toggle", N: "nsfw_toggle",
    };
    if (map[e.key]) { e.preventDefault(); trigger(map[e.key]); }
  });
})();
