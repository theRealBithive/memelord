(function () {
  function getCsrf() {
    const val = `; ${document.cookie}`;
    const parts = val.split("; csrftoken=");
    return parts.length === 2 ? parts.pop().split(";").shift() : "";
  }

  function saveTags(url, tags) {
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-CSRFToken": getCsrf(),
      },
      body: new URLSearchParams({ tags: tags.join(",") }),
    }).then((r) => r.json());
  }

  function getTagsFromContainer(container) {
    return Array.from(container.querySelectorAll(".tag-pill[data-tag]"))
      .map((p) => p.dataset.tag);
  }

  function getSuggestionsFromContainer(container) {
    return Array.from(container.querySelectorAll(".tag-suggestion-pill[data-tag]"))
      .map((p) => p.dataset.tag);
  }

  function renderPills(container, tagList) {
    const input = container.querySelector(".tag-input");
    const addBtn = container.querySelector(".tag-add-btn");
    const modalBtn = container.querySelector(".tag-modal-open");

    container.querySelectorAll(".tag-pill").forEach((p) => p.remove());

    const tagUrl = input ? input.dataset.tagUrl : container.dataset.tagUrl;
    const ref = addBtn || modalBtn || input;

    tagList.forEach((tag) => {
      const pill = document.createElement("span");
      pill.className = "tag-pill";
      pill.dataset.tag = tag;
      const label = document.createTextNode("#" + tag + " ");
      const btn = document.createElement("button");
      btn.className = "tag-remove";
      btn.setAttribute("aria-label", "remove tag");
      btn.textContent = "×";
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const updated = getTagsFromContainer(container).filter((t) => t !== tag);
        saveTags(tagUrl, updated).then((data) => renderPills(container, data.tags));
      });
      pill.appendChild(label);
      pill.appendChild(btn);
      container.insertBefore(pill, ref || null);
    });
  }

  // ── Inline tag editor (desktop) ───────────────────────────────────────

  function initCardTags(container) {
    const input = container.querySelector(".tag-input");
    if (!input || input._tagInit) return;
    input._tagInit = true;

    const tagUrl = input.dataset.tagUrl;
    const acUrl  = input.dataset.autocompleteUrl;
    const dropdown = container.querySelector(".tag-autocomplete");
    let acTimer;

    function commit() {
      const name = input.value.trim().toLowerCase();
      input.value = "";
      if (!name) return;
      const existing = getTagsFromContainer(container);
      if (existing.includes(name)) return;
      saveTags(tagUrl, [...existing, name]).then((data) => renderPills(container, data.tags));
    }

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") { e.preventDefault(); commit(); }
    });

    input.addEventListener("input", () => {
      const q = input.value.trim();
      clearTimeout(acTimer);
      if (!q || !acUrl || !dropdown) return;
      acTimer = setTimeout(() => {
        fetch(`${acUrl}?q=${encodeURIComponent(q)}`)
          .then((r) => r.json())
          .then(({ tags }) => {
            dropdown.innerHTML = "";
            const existing = getTagsFromContainer(container);
            const filtered = tags.filter((t) => !existing.includes(t));
            if (!filtered.length) { dropdown.hidden = true; return; }
            filtered.forEach((tag) => {
              const item = document.createElement("div");
              item.className = "tag-autocomplete-item";
              item.textContent = "#" + tag;
              item.addEventListener("mousedown", (e) => {
                e.preventDefault();
                input.value = "";
                const cur = getTagsFromContainer(container);
                if (!cur.includes(tag)) {
                  saveTags(tagUrl, [...cur, tag]).then((data) => renderPills(container, data.tags));
                }
                dropdown.hidden = true;
              });
              dropdown.appendChild(item);
            });
            dropdown.hidden = false;
          });
      }, 150);
    });

    input.addEventListener("blur", () => {
      setTimeout(() => { if (dropdown) dropdown.hidden = true; }, 200);
    });
  }

  function initSuggestionPills(container) {
    container.querySelectorAll(".tag-suggestion-pill:not([data-init])").forEach((pill) => {
      pill.dataset.init = "1";
      pill.addEventListener("click", (e) => {
        e.preventDefault();
        const tag = pill.dataset.tag;
        const url = pill.dataset.tagUrl;
        if (!tag || !url) return;
        const existing = getTagsFromContainer(container);
        if (existing.includes(tag)) { pill.remove(); return; }
        saveTags(url, [...existing, tag]).then((data) => {
          renderPills(container, data.tags);
          pill.remove();
        });
      });
    });
  }

  // ── Tag modal (mobile + opt-in elsewhere) ─────────────────────────────
  //
  // One global modal element, lazily built. Opened with a generic config
  // so both the inline card editor and the gallery lightbox can reuse it.

  let modalEl = null;
  let modalState = null;  // { appliedTags, suggestedTags, autocompleteUrl, onSave }

  function buildModal() {
    if (modalEl) return modalEl;
    modalEl = document.createElement("div");
    modalEl.className = "tag-modal";
    modalEl.hidden = true;
    modalEl.innerHTML = `
      <div class="tag-modal-backdrop"></div>
      <div class="tag-modal-inner">
        <div class="tag-modal-header">
          <span class="tag-modal-title">Tags</span>
          <button type="button" class="tag-modal-close" aria-label="Close">×</button>
        </div>
        <div class="tag-modal-input-wrap">
          <input type="text" class="tag-modal-input" placeholder="type a tag…" autocomplete="off" autocapitalize="none">
          <div class="tag-modal-autocomplete" hidden></div>
        </div>
        <div class="tag-modal-body">
          <div class="tag-modal-section tag-modal-applied">
            <div class="tag-modal-section-title">Applied</div>
            <div class="tag-modal-pills tag-modal-applied-pills"></div>
          </div>
          <div class="tag-modal-section tag-modal-suggested" hidden>
            <div class="tag-modal-section-title">Suggested</div>
            <div class="tag-modal-pills tag-modal-suggested-pills"></div>
          </div>
        </div>
        <button type="button" class="tag-modal-done">Done</button>
      </div>
    `;
    document.body.appendChild(modalEl);

    const input    = modalEl.querySelector(".tag-modal-input");
    const dropdown = modalEl.querySelector(".tag-modal-autocomplete");

    modalEl.querySelector(".tag-modal-close").addEventListener("click", closeModal);
    modalEl.querySelector(".tag-modal-done").addEventListener("click", closeModal);
    modalEl.querySelector(".tag-modal-backdrop").addEventListener("click", closeModal);

    let acTimer;

    function commit() {
      const name = input.value.trim().toLowerCase();
      input.value = "";
      dropdown.hidden = true;
      if (!name || !modalState) return;
      const current = modalState.appliedTags();
      if (current.includes(name)) return;
      modalState.onSave([...current, name]).then(refreshModal);
    }

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") { e.preventDefault(); commit(); }
      else if (e.key === "Escape") { e.preventDefault(); closeModal(); }
    });

    input.addEventListener("input", () => {
      const q = input.value.trim();
      clearTimeout(acTimer);
      if (!q || !modalState?.autocompleteUrl) { dropdown.hidden = true; return; }
      acTimer = setTimeout(() => {
        fetch(`${modalState.autocompleteUrl}?q=${encodeURIComponent(q)}`)
          .then((r) => r.json())
          .then(({ tags }) => {
            dropdown.innerHTML = "";
            const existing = modalState.appliedTags();
            const filtered = tags.filter((t) => !existing.includes(t));
            if (!filtered.length) { dropdown.hidden = true; return; }
            filtered.forEach((tag) => {
              const item = document.createElement("div");
              item.className = "tag-autocomplete-item";
              item.textContent = "#" + tag;
              item.addEventListener("mousedown", (e) => {
                e.preventDefault();
                input.value = "";
                dropdown.hidden = true;
                const cur = modalState.appliedTags();
                if (!cur.includes(tag)) {
                  modalState.onSave([...cur, tag]).then(refreshModal);
                }
              });
              dropdown.appendChild(item);
            });
            dropdown.hidden = false;
          });
      }, 150);
    });

    return modalEl;
  }

  function refreshModal(tagsOverride) {
    if (!modalEl || !modalState) return;
    const applied = tagsOverride || modalState.appliedTags();
    const appliedEl = modalEl.querySelector(".tag-modal-applied-pills");
    appliedEl.innerHTML = "";
    if (!applied.length) {
      const hint = document.createElement("span");
      hint.className = "tag-modal-empty";
      hint.textContent = "No tags yet.";
      appliedEl.appendChild(hint);
    } else {
      applied.forEach((tag) => {
        const pill = document.createElement("span");
        pill.className = "tag-pill";
        pill.dataset.tag = tag;
        pill.appendChild(document.createTextNode("#" + tag + " "));
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "tag-remove";
        btn.setAttribute("aria-label", "remove tag");
        btn.textContent = "×";
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          const next = modalState.appliedTags().filter((t) => t !== tag);
          modalState.onSave(next).then(refreshModal);
        });
        pill.appendChild(btn);
        appliedEl.appendChild(pill);
      });
    }

    const suggested = (modalState.suggestedTags || []).filter((t) => !applied.includes(t));
    const suggestedSection = modalEl.querySelector(".tag-modal-suggested");
    const suggestedEl = modalEl.querySelector(".tag-modal-suggested-pills");
    suggestedEl.innerHTML = "";
    if (!suggested.length) {
      suggestedSection.hidden = true;
    } else {
      suggestedSection.hidden = false;
      suggested.forEach((tag) => {
        const pill = document.createElement("button");
        pill.type = "button";
        pill.className = "tag-suggestion-pill";
        pill.dataset.tag = tag;
        pill.textContent = tag;
        pill.addEventListener("click", (e) => {
          e.preventDefault();
          const cur = modalState.appliedTags();
          if (cur.includes(tag)) { pill.remove(); return; }
          modalState.onSave([...cur, tag]).then((next) => {
            pill.remove();
            refreshModal(next);
          });
        });
        suggestedEl.appendChild(pill);
      });
    }
  }

  function openModal(opts) {
    buildModal();
    modalState = opts;
    refreshModal();
    modalEl.hidden = false;
    document.documentElement.style.overflow = "hidden";
    // Defer focus so iOS Safari actually pops the keyboard on the tap that opened us.
    setTimeout(() => {
      const input = modalEl.querySelector(".tag-modal-input");
      input.value = "";
      input.focus();
    }, 30);
  }

  function closeModal() {
    if (!modalEl) return;
    modalEl.hidden = true;
    modalState = null;
    document.documentElement.style.overflow = "";
    const dropdown = modalEl.querySelector(".tag-modal-autocomplete");
    if (dropdown) dropdown.hidden = true;
  }

  function initModalButtons(container) {
    container.querySelectorAll(".tag-modal-open:not([data-init])").forEach((btn) => {
      btn.dataset.init = "1";
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const input = container.querySelector(".tag-input");
        if (!input) return;
        openModal({
          appliedTags: () => getTagsFromContainer(container),
          suggestedTags: getSuggestionsFromContainer(container),
          autocompleteUrl: input.dataset.autocompleteUrl,
          onSave: (tags) =>
            saveTags(input.dataset.tagUrl, tags).then((data) => {
              renderPills(container, data.tags);
              return data.tags;
            }),
        });
      });
    });
  }

  // Expose for gallery.js (lightbox uses its own state, not a card-tags container).
  window.TagModal = { open: openModal, close: closeModal };

  function initAll() {
    document.querySelectorAll(".card-tags").forEach((c) => {
      initCardTags(c);
      initSuggestionPills(c);
      initModalButtons(c);
    });
  }

  initAll();
  document.body.addEventListener("htmx:afterSettle", initAll);
  document.body.addEventListener("htmx:beforeSwap", closeModal);
})();
