// Section 2: pi0.5 (no context) vs. VICTR on a chosen task.
(function () {
  const { el, taskPicker, media, taskById } = window.U;
  const root = document.getElementById("why-widget");
  const items = window.VICTR_SITE.whyContext;

  // whyContext is ordered strongest contrast first; open on the first one.
  const start = items[0].task;
  window.U.shareTask(start);
  const picker = taskPicker(items.map((it) => it.task), start, render);
  const body = el("div");
  root.append(picker, body);


  // Badge comes from the eval results, so it stays honest on every task.
  function column(label, pol, clip, task) {
    const cls = !pol ? "partial" : pol.success === 0 ? "fail" : pol.sr >= 50 ? "success" : "partial";
    const word = pol ? `${pol.success}/${pol.n} success${pol.note ? ` · ${pol.note}` : ""}` : "not evaluated";
    const stats = pol ? `${pol.progress.toFixed(0)}% task progress` : "";
    const m = media(clip, `${label} rollout`, task.name);
    if (m.isVideo) { m.video.loop = true; m.video.autoplay = true; m.video.play().catch(() => {}); }
    return el("div.versus-col", null,
      el("h4", null, label, el("span.badge." + cls, { text: word })),
      m,
      el("div.versus-stats", { text: stats }));
  }

  function render(id) {
    const it = items.find((x) => x.task === id);
    const task = taskById(id);
    body.replaceChildren(
      el("div.versus", null,
        column("π0.5 (no context)", task.policies.pi05, it.pi05, task),
        column("VICTR (retrieved context)", task.policies.victr, it.victr, task)));
  }
  render(start);
})();
