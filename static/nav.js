// The nav menus are <details> elements so they work without JS and on tap.
// This only adds the two behaviours <details> lacks: closing on an outside
// tap / Escape, and closing when htmx swaps the nav out from under them.
(function () {
  function openMenus() {
    return document.querySelectorAll("#main-nav details[open]");
  }
  function closeAll(except) {
    openMenus().forEach((menu) => { if (menu !== except) menu.removeAttribute("open"); });
  }
  document.addEventListener("click", (e) => {
    const inside = e.target.closest("#main-nav details");
    closeAll(inside);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAll(null);
  });
  document.body.addEventListener("htmx:beforeSwap", () => closeAll(null));
})();
