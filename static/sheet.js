// One confirmation sheet for every irreversible action (UI contract V5).
// Declarative: an htmx element says hx-confirm="question" and a plain form says
// data-confirm="question". Optional data-confirm-label sets the red button's
// text and data-confirm-word demands that the word is typed first.
(function () {
  const sheet = document.getElementById("confirm-sheet");
  if (!sheet) return;
  const text = sheet.querySelector(".sheet-text");
  const wordWrap = sheet.querySelector(".sheet-word");
  const wordLabel = sheet.querySelector(".sheet-word-label");
  const wordInput = sheet.querySelector(".sheet-word-input");
  const okBtn = sheet.querySelector(".sheet-confirm");
  const cancelBtn = sheet.querySelector(".sheet-cancel");
  let resolveOpen = null; // resolver of the Promise returned by open()

  function close(result) {
    sheet.hidden = true;
    document.body.classList.remove("sheet-open");
    const resolve = resolveOpen;
    resolveOpen = null;
    if (resolve) resolve(result);
  }

  function open({ message, confirmLabel, word }) {
    if (resolveOpen) close(false);
    text.textContent = message;
    okBtn.textContent = confirmLabel || "Delete";
    wordWrap.hidden = !word;
    wordInput.value = "";
    wordInput.dataset.word = word || "";
    okBtn.disabled = Boolean(word);
    if (word) wordLabel.textContent = `Type ${word} to continue`;
    sheet.hidden = false;
    document.body.classList.add("sheet-open");
    (word ? wordInput : cancelBtn).focus();
    return new Promise((resolve) => { resolveOpen = resolve; });
  }

  wordInput.addEventListener("input", () => {
    okBtn.disabled = wordInput.value.trim() !== wordInput.dataset.word;
  });
  okBtn.addEventListener("click", () => {
    if (okBtn.disabled) return;
    // Resolve with the typed word when one was demanded, so the caller can
    // send it to the server, which checks it again (the UI is not the guard).
    close(wordInput.dataset.word ? wordInput.value.trim() : true);
  });
  cancelBtn.addEventListener("click", () => close(false));
  sheet.querySelector(".sheet-backdrop").addEventListener("click", () => close(false));
  document.addEventListener("keydown", (e) => {
    if (!sheet.hidden && e.key === "Escape") { e.preventDefault(); close(false); }
  });

  // htmx asks before every request; only a question (hx-confirm) opens the sheet.
  document.body.addEventListener("htmx:confirm", (e) => {
    const question = e.detail.question;
    if (!question) return;
    e.preventDefault();
    const el = e.detail.elt;
    open({ message: question, confirmLabel: el.dataset.confirmLabel, word: el.dataset.confirmWord })
      .then((confirmed) => { if (confirmed) e.detail.issueRequest(true); });
  });

  // Plain forms: form.submit() bypasses this listener, so it does not ask twice.
  document.addEventListener("submit", (e) => {
    const form = e.target;
    if (!(form instanceof HTMLFormElement) || !form.dataset.confirm) return;
    e.preventDefault();
    open({ message: form.dataset.confirm, confirmLabel: form.dataset.confirmLabel, word: form.dataset.confirmWord })
      .then((confirmed) => {
        if (!confirmed) return;
        const field = form.querySelector("[data-confirm-field]");
        if (field && typeof confirmed === "string") field.value = confirmed;
        form.submit();
      });
  });

  window.ConfirmSheet = { open };
})();
