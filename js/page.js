// Section index in the left gutter, built from h1/h2[data-nav].
(function () {
  const { el } = window.U;
  const nav = document.querySelector(".section-index");
  const list = document.getElementById("section-index-list");
  const heads = [...document.querySelectorAll("main h1[data-nav], main h2[data-nav]")];

  const links = heads.map((h) => {
    const a = el("a", { href: `#${h.id}`, text: h.dataset.nav });
    a.dataset.level = h.tagName === "H1" ? "1" : "2";
    list.appendChild(el("li", null, a));
    return a;
  });

  function update() {
    const y = innerHeight * 0.3;
    let active = -1;
    heads.forEach((h, i) => { if (h.getBoundingClientRect().top < y) active = i; });
    links.forEach((a, i) => a.classList.toggle("is-active", i === active));
    // Also light the parent h1 of an active h2.
    if (active >= 0 && links[active].dataset.level === "2") {
      for (let i = active; i >= 0; i--) {
        if (links[i].dataset.level === "1") { links[i].classList.add("is-active"); break; }
      }
    }
    nav.classList.toggle("visible", heads[0].getBoundingClientRect().top < innerHeight);
  }
  addEventListener("scroll", update, { passive: true });
  addEventListener("resize", update);
  update();
})();
