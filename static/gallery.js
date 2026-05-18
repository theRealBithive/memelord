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
      <div class="lb-actions">
        <button class="lb-action-score lb-action-score--1" data-score="1">1</button>
        <button class="lb-action-score lb-action-score--2" data-score="2">2</button>
        <button class="lb-action-score lb-action-score--3" data-score="3">3</button>
        <button class="lb-action-score lb-action-score--4" data-score="4">4</button>
        <button class="lb-action-score lb-action-score--5" data-score="5">5</button>
        <button class="lb-action-score lb-action-score--6" data-score="6">6</button>
        <button class="lb-action-fav" aria-label="Toggle favourite">★</button>
        <button class="lb-action-trash" aria-label="Trash">🗑</button>
      </div>
    </div>
  `;
  document.body.appendChild(lb);

  const lbImg          = lb.querySelector(".lb-img");
  const lbScore        = lb.querySelector(".lb-score");
  const lbFav          = lb.querySelector(".lb-fav");
  const lbPos          = lb.querySelector(".lb-pos");
  const lbReview       = lb.querySelector(".lb-review");
  const lbPrev         = lb.querySelector(".lb-prev");
  const lbNext         = lb.querySelector(".lb-next");
  const lbActionFav    = lb.querySelector(".lb-action-fav");
  const lbActionTrash  = lb.querySelector(".lb-action-trash");
  const lbActionScores = Array.from(lb.querySelectorAll(".lb-action-score"));

  const items = Array.from(document.querySelectorAll(".gallery-item"));
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

  function show(idx) {
    current = idx;
    const item  = items[idx];
    const score = item.dataset.score;
    const fav   = item.dataset.fav === "1";
    lbImg.src           = item.querySelector("img").src;
    lbReview.href       = item.href;
    lbScore.textContent = score || "";
    lbScore.className   = score ? `lb-score lb-score--${score}` : "lb-score";
    lbFav.hidden        = !fav;
    lbPos.textContent   = `${idx + 1} / ${items.length}`;
    lbPrev.disabled     = idx === 0;
    lbNext.disabled     = idx === items.length - 1;
    lbActionScores.forEach((btn) => {
      btn.classList.toggle("lb-action-score--active", btn.dataset.score === score);
    });
    lbActionFav.classList.toggle("lb-action-fav--on", fav);
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

  lbActionScores.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const item = items[current];
      postAction(item.dataset.actionUrl, { action: "score", score: btn.dataset.score }).then((data) => {
        if (data.score != null) {
          item.dataset.score = String(data.score);
          const overlay = item.querySelector(".gallery-item-score");
          if (overlay) {
            overlay.textContent = data.score;
            overlay.className = `gallery-item-score gallery-item-score--${data.score}`;
          }
          show(current);
        }
      });
    });
  });

  lbActionFav.addEventListener("click", (e) => {
    e.stopPropagation();
    const item = items[current];
    postAction(item.dataset.actionUrl, { action: "fav" }).then((data) => {
      item.dataset.fav = data.fav ? "1" : "0";
      let favEl = item.querySelector(".gallery-item-fav");
      if (data.fav && !favEl) {
        favEl = document.createElement("span");
        favEl.className = "gallery-item-fav";
        favEl.textContent = "★";
        item.querySelector(".gallery-item-overlay").appendChild(favEl);
      } else if (!data.fav && favEl) {
        favEl.remove();
      }
      show(current);
    });
  });

  lbActionTrash.addEventListener("click", (e) => {
    e.stopPropagation();
    const item = items[current];
    postAction(item.dataset.actionUrl, { action: "trash" }).then((data) => {
      if (data.deleted) {
        const removedIdx = current;
        items.splice(removedIdx, 1);
        item.remove();
        if (items.length === 0) {
          close();
        } else {
          show(Math.min(removedIdx, items.length - 1));
        }
      }
    });
  });

  document.addEventListener("keydown", (e) => {
    if (!lb.classList.contains("lb-open")) return;
    if (e.key === "Escape")     { close(); return; }
    if (e.key === "ArrowLeft")  { e.preventDefault(); prev(); }
    if (e.key === "ArrowRight") { e.preventDefault(); next(); }
  });

  let startX = 0;
  lbImg.addEventListener("touchstart", (e) => {
    startX = e.touches[0].clientX;
  }, { passive: true });
  lbImg.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].clientX - startX;
    if (Math.abs(dx) > 40) {
      dx < 0 ? next() : prev();
    } else {
      close();
    }
  }, { passive: true });
})();
