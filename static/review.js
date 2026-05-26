(function () {
  // Channels are injected by review.html into a <script type="application/json">.
  // Read once on load — channels don't change while the review page is open.
  const shareChannels = JSON.parse(
    document.getElementById("share-channels-data")?.textContent || "[]"
  );

  function trigger(selector) {
    const el = document.querySelector(selector);
    if (el) { el.click(); el.blur(); }
  }

  // Open the share sheet when a multi-channel picker button is clicked.
  // Delegated on document because the button lives inside #review-card (HTMX-swapped).
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".share-btn--picker");
    if (!btn) return;
    const shareUrl = btn.dataset.shareUrl;
    if (!shareUrl || !shareChannels.length) return;
    window.ShareSheet?.open({
      channels: shareChannels,
      shareUrl,
      onResult: (html) => {
        const toast = document.getElementById("share-toast");
        if (toast) toast.outerHTML = html;
      },
    });
  });


  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    const key = e.key;

    if (key >= "0" && key <= "6") {
      // 0 = trash (the trash button carries data-score="0"); 1-6 = score.
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
      document.querySelector(".review-nsfw-btn")?.click();
    } else if (key === "s" || key === "S") {
      e.preventDefault();
      // Click whichever share button is present (single-channel form submit or picker).
      const shareEl = document.querySelector(".share-btn, .share-btn--picker");
      if (shareEl) shareEl.click();
    }
  });
})();
