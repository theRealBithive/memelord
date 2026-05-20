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

  function renderPills(container, tagList) {
    const input = container.querySelector(".tag-input");
    const acEl  = container.querySelector(".tag-autocomplete");
    const addBtn = container.querySelector(".tag-add-btn");

    // Remove existing pills only
    container.querySelectorAll(".tag-pill").forEach((p) => p.remove());

    const tagUrl = input ? input.dataset.tagUrl : container.dataset.tagUrl;
    const acUrl  = input ? input.dataset.autocompleteUrl : "";

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
      // Insert before add button or input
      const ref = addBtn || input;
      container.insertBefore(pill, ref || null);
    });
  }

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

  function initAll() {
    document.querySelectorAll(".card-tags").forEach(initCardTags);
  }

  initAll();
  document.body.addEventListener("htmx:afterSettle", initAll);
})();
