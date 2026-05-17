(function () {
  function trigger(selector) {
    const el = document.querySelector(selector);
    if (el) el.click();
  }

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
    } else if (key === "n" || key === "N") {
      e.preventDefault();
      document.querySelector(".nsfw-toggle-btn")?.click();
    }
  });
})();
