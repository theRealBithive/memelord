(function () {
  function trigger(selector) {
    const el = document.querySelector(selector);
    if (el) { el.click(); el.blur(); }
  }

  // Close an in-flight share when the review card swaps (prev/next/score).
  document.body.addEventListener("htmx:afterSwap", (e) => {
    if (e.detail.target?.id === "review-card") window.ShareSheet?.close();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    // An open sheet (confirm or share) owns the keyboard.
    if (document.body.classList.contains("sheet-open")) return;

    const key = e.key;

    if (key >= "0" && key <= "6") {
      // 0 = trash (the first score button carries data-score="0"); 1-6 = score.
      e.preventDefault();
      trigger(`[data-score="${key}"]`);
    } else if (key === "ArrowLeft") {
      e.preventDefault();
      trigger("[data-action='prev']");
    } else if (key === "ArrowRight") {
      e.preventDefault();
      trigger("[data-action='next']");
    } else if (key === "n" || key === "N") {
      e.preventDefault();
      trigger("[data-action='nsfw']");
    } else if (key === "s" || key === "S") {
      e.preventDefault();
      trigger("[data-action='share']");
    }
  });
})();
