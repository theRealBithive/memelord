(function () {
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
        <span class="lb-score"></span>
        <span class="lb-fav" aria-hidden="true">★</span>
        <span class="lb-pos"></span>
        <a class="lb-review" href="">review →</a>
      </div>
    </div>
  `;
  document.body.appendChild(lb);

  const lbImg    = lb.querySelector(".lb-img");
  const lbScore  = lb.querySelector(".lb-score");
  const lbFav    = lb.querySelector(".lb-fav");
  const lbPos    = lb.querySelector(".lb-pos");
  const lbReview = lb.querySelector(".lb-review");
  const lbPrev   = lb.querySelector(".lb-prev");
  const lbNext   = lb.querySelector(".lb-next");

  const items = Array.from(document.querySelectorAll(".gallery-item"));
  let current = 0;

  function show(idx) {
    current = idx;
    const item  = items[idx];
    const score = item.dataset.score;
    lbImg.src             = item.querySelector("img").src;
    lbReview.href         = item.href;
    lbScore.textContent   = score || "";
    lbScore.className     = score ? `lb-score lb-score--${score}` : "lb-score";
    lbFav.hidden          = item.dataset.fav !== "1";
    lbPos.textContent     = `${idx + 1} / ${items.length}`;
    lbPrev.disabled       = idx === 0;
    lbNext.disabled       = idx === items.length - 1;
  }

  function open(idx) {
    show(idx);
    lb.classList.add("lb-open");
    document.documentElement.style.overflow = "hidden";
  }

  function close() {
    lb.classList.remove("lb-open");
    lbImg.src = "";
    document.documentElement.style.overflow = "";
  }

  function prev() { if (current > 0) show(current - 1); }
  function next() { if (current < items.length - 1) show(current + 1); }

  items.forEach((a, i) => {
    a.addEventListener("click", (e) => { e.preventDefault(); open(i); });
  });

  lb.querySelector(".lb-backdrop").addEventListener("click", close);
  lb.querySelector(".lb-close").addEventListener("click", close);
  lbImg.addEventListener("click", close);
  lbPrev.addEventListener("click", (e) => { e.stopPropagation(); prev(); });
  lbNext.addEventListener("click", (e) => { e.stopPropagation(); next(); });

  // Keyboard
  document.addEventListener("keydown", (e) => {
    if (!lb.classList.contains("lb-open")) return;
    if (e.key === "Escape")      { close(); return; }
    if (e.key === "ArrowLeft")   { e.preventDefault(); prev(); }
    if (e.key === "ArrowRight")  { e.preventDefault(); next(); }
  });

  // Touch swipe (left = next, right = prev)
  let startX = 0;
  lbImg.addEventListener("touchstart", (e) => {
    startX = e.touches[0].clientX;
  }, { passive: true });
  lbImg.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].clientX - startX;
    if (Math.abs(dx) > 40) {
      dx < 0 ? next() : prev();
    } else {
      close(); // tap without real swipe = close
    }
  }, { passive: true });
})();
