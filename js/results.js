// Section 4: aggregate chart, per-task eval browser, and analysis figures.
(function () {
  const { el, svg, chips, toggle, media, showTip, hideTip, policyById } = window.U;
  const SITE = window.VICTR_SITE;
  const TASKS = window.VICTR_RESULTS.tasks;

  // ---------------------------------------------------------------------------
  // Stacked OOD/InD bar chart shared by Fig. 4, 6 and 8.
  // groups: [{name, fill, ink, ind: [sr, progress], ood: [sr, progress], sub?}]
  // domain: "ind" or "ood" draws one bar per group in the policy colour.
  function barChart(root, groups, { title, height = 260, highlight, width = 800, domain } = {}) {
    const W = width, m = { l: 40, r: 10, t: 16, b: groups.some((g) => g.sub) ? (domain ? 46 : 58) : (domain ? 30 : 44) };
    const Hh = height + m.t + m.b;
    const pw = W - m.l - m.r;
    const s = svg("svg", { viewBox: `0 0 ${W} ${Hh}` });
    const g = svg("g", { transform: `translate(${m.l},${m.t})` });
    s.appendChild(g);
    const Y = (v) => height * (1 - v / 100);

    const axis = svg("g.axis");
    [0, 20, 40, 60, 80, 100].forEach((v) => {
      axis.appendChild(svg("line", { x1: 0, x2: pw, y1: Y(v), y2: Y(v) }));
      axis.appendChild(svg("text", { x: -6, y: Y(v) + 4, "text-anchor": "end", text: v }));
    });
    g.appendChild(axis);
    g.appendChild(svg("text", { x: -30, y: height / 2, transform: `rotate(-90 -30 ${height / 2})`, "text-anchor": "middle", "font-size": 11, fill: "#777", text: "% of rollouts" }));

    const gw = pw / groups.length, bw = Math.min(56, gw * (domain ? 0.5 : 0.3));
    const bars = domain ? [[domain, 0, null]] : [["ood", -1, "OOD"], ["ind", 1, "InD"]];
    groups.forEach((grp, i) => {
      const cx = gw * (i + 0.5);
      bars.forEach(([dom, dir, lab]) => {
        const [sr, prog] = grp[dom];
        const x = cx + dir * (bw / 2 + 3) - bw / 2;
        const isInd = dom === "ind" && !domain;
        const solid = isInd ? "#d9d9d9" : grp.fill;
        const stroke = isInd ? "#9a9a9a" : grp.ink;
        const tip = `<b>${U.nameHTML(grp.name)}</b> · ${dom === "ood" ? "out-of-domain" : "in-domain"}<br>success ${sr.toFixed(1)}% · progress ${prog.toFixed(1)}%`;
        const gg = svg("g", { onmousemove: (e) => showTip(e, tip), onmouseleave: hideTip, style: "cursor: default" });
        gg.appendChild(svg("rect.bar", { x, y: Y(prog), width: bw, height: height - Y(prog), fill: solid, "fill-opacity": 0.38, stroke, "stroke-width": 1, "stroke-dasharray": "3 2" }));
        gg.appendChild(svg("rect.bar", { x, y: Y(sr), width: bw, height: height - Y(sr), fill: solid, stroke, "stroke-width": 1.2 }));
        const bold = highlight && grp.name === highlight && (domain || dom === "ood");
        gg.appendChild(svg("text.bar-label", { x: x + bw / 2, y: Y(prog) - 4, "text-anchor": "middle", "font-weight": bold ? 700 : 400, text: prog.toFixed(1) }));
        if (prog - sr > 7) gg.appendChild(svg("text.bar-label", { x: x + bw / 2, y: Y(sr) - 4, "text-anchor": "middle", fill: "#666", text: sr.toFixed(1) }));
        if (lab) gg.appendChild(svg("text.sub-label", { x: x + bw / 2, y: height + 13, "text-anchor": "middle", text: lab }));
        g.appendChild(gg);
      });
      g.appendChild(svg("text.group-label", { x: cx, y: height + (domain ? 18 : 32), "text-anchor": "middle" }, ...U.svgName(grp.name)));
      if (grp.sub) g.appendChild(svg("text.sub-label", { x: cx, y: height + (domain ? 33 : 47), "text-anchor": "middle", text: grp.sub }));
    });

    if (title) root.appendChild(el("div.widget-title", { text: title }));
    root.append(
      s,
      el("div.legend", null,
        el("span", null, el("i", { style: { background: "#888" } }), "binary success"),
        el("span", null, el("i", { style: { background: "#888", opacity: 0.35 } }), "+ partial credit (task progress)"),
        domain ? null : el("span", null, el("i", { style: { background: "#d9d9d9" } }), "in-domain shown in grey")));
  }

  // Fig. 4
  // Pooled over episodes per domain, precomputed by tools/build_results.py.
  const POOLED = window.VICTR_RESULTS.pooled;
  const mainGroups = SITE.policies.map((p) => ({ name: p.label, fill: p.fill, ink: p.ink, ind: POOLED[p.id].ind, ood: POOLED[p.id].ood }));
  // In-domain and out-of-domain side by side.
  function splitBars(root, groups, opts) {
    const ind = el("div"), ood = el("div");
    root.appendChild(el("div.chart-pair", null, ind, ood));
    barChart(ind, groups, { ...opts, title: "In-domain tasks", domain: "ind", width: 480, height: 220 });
    barChart(ood, groups, { ...opts, title: "Out-of-domain tasks", domain: "ood", width: 480, height: 220 });
  }
  splitBars(document.getElementById("main-chart"), mainGroups, { highlight: "VICTR" });

  // ---------------------------------------------------------------------------
  // Per-task browser
  (function evalBrowser() {
    const root = document.getElementById("eval-browser");
    let domain = "all", metric = "sr", selected = null, selPolicy = "victr", selRollout = 0;
    const table = el("table.eval-table");
    const panel = el("div.rollout-panel", { style: { display: "none" } });

    root.append(
      el("div.controls-row", null,
        el("span.label", { text: "Tasks" }),
        toggle([["all", "All"], ["ind", "In-domain"], ["ood", "Out-of-domain"]], domain, (d) => { domain = d; renderTable(); }),
        el("span.label", { text: "Metric" }),
        toggle([["sr", "Success"], ["progress", "Task progress"]], metric, (mm) => { metric = mm; renderTable(); })),
      table, panel);

    function cellBar(p, pol) {
      if (!p) return el("div.cell-bar.na", null, el("span", { text: "not run" }));
      const v = p[metric];
      return el("div.cell-bar", { title: `${p.success} success · ${p.partial} partial · ${p.failure} fail (n=${p.n})` },
        el("div.p", { style: { width: `${p.progress}%`, background: pol.fill } }),
        el("div.s", { style: { width: `${p.sr}%`, background: pol.fill, borderRight: p.sr > 0 ? `2px solid ${pol.ink}` : "none" } }),
        el("span", { text: `${v.toFixed(0)}%` }));
    }

    function renderTable() {
      const head = el("tr", null, el("th", { text: "Task" }), ...SITE.policies.map((p) => el("th", null, ...U.name(p.label))));
      const rows = [head];
      ["ind", "ood"].forEach((d) => {
        if (domain !== "all" && domain !== d) return;
        rows.push(el("tr.domain-row", null, el("td", { colspan: 5, text: d === "ind" ? "In-domain" : "Out-of-domain" })));
        TASKS.filter((t) => t.domain === d).forEach((t) => {
          const tr = el("tr.task-row" + (selected === t.id ? ".selected" : ""), { onclick: () => select(t.id) },
            el("td", { text: t.name }),
            ...SITE.policies.map((p) => el("td", null, cellBar(t.policies[p.id], p))));
          rows.push(tr);
        });
      });
      table.replaceChildren(...rows);
    }

    // Real rollouts in dot order (success, partial, failure), from data/rollouts.js.
    const ROLLOUTS = window.VICTR_ROLLOUTS || {};
    const OUTCOME = { success: "success", partial_success: "partial", failure: "failure" };
    function rollouts(taskId, pol) {
      return (ROLLOUTS[taskId]?.[pol] || []).map((r) => ({ ...r, outcome: OUTCOME[r.outcome] }));
    }

    function select(id) {
      selected = id; selRollout = 0;
      renderTable(); renderPanel();
      panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
      U.shareTask(id, follow);
    }
    const follow = U.onSharedTask((id) => {
      if (id === selected || !TASKS.some((t) => t.id === id)) return;
      selected = id; selRollout = 0;
      if (domain !== "all" && TASKS.find((t) => t.id === id).domain !== domain) domain = "all";
      renderTable(); renderPanel();
    });

    function renderPanel() {
      const t = TASKS.find((x) => x.id === selected);
      if (!t) return;
      panel.style.display = "";
      const p = t.policies[selPolicy];
      const rs = rollouts(t.id, selPolicy);
      const pol = policyById(selPolicy);
      const dots = el("div.rollout-dots", null, ...rs.map((r, k) => el("button." + r.outcome + (k === selRollout ? ".active" : ""), {
        type: "button", text: k + 1, title: r.outcome, onclick: () => { selRollout = k; renderPanel(); } })));
      const r = rs[selRollout] || { outcome: "failure", video: null };
      const out = r.outcome;
      const clip = media(r.video, "No recording", `${U.plainName(pol.label)} · rollout ${selRollout + 1}`);
      if (clip.isVideo) {
        clip.video.controls = true; clip.video.loop = true; clip.video.play().catch(() => {});
        if (r.speed > 1) clip.appendChild(el("span.speed-tag", { text: `${r.speed}×` }));
      }
      panel.replaceChildren(
        el("h3", null, t.name, " ", el("span.badge." + (t.domain === "ood" ? "partial" : "success"), { text: t.domain === "ood" ? "OOD" : "InD" })),
        el("p", { style: { fontSize: "14px", color: "#666", margin: "0 0 10px" }, text: `“${t.instruction}”` }),
        chips(SITE.policies.map((pp) => ({ id: pp.id, label: U.name(pp.label) })), selPolicy, (id) => { selPolicy = id; selRollout = 0; renderPanel(); }),
        p ? dots : el("p", { style: { fontSize: "14px", color: "#999" }, text: "This policy was not evaluated on this task." }),
        p ? el("div", null, clip,
          el("div.versus-stats", null, el("span.badge." + (out === "failure" ? "fail" : out), { text: out }),
            `  ${p.success}/${p.n} success · ${p.progress.toFixed(1)}% task progress`)) : null);
    }

    renderTable();
  })();

  // ---------------------------------------------------------------------------
  // Table I, sortable.
  (function vfeTable() {
    const root = document.getElementById("vfe-table");
    const rows = SITE.vfeTable;
    let sortKey = null, dom = "ood";
    const cols = [["ind", 0, "InD τ_b"], ["ind", 1, "InD r"], ["ood", 0, "OOD τ_b"], ["ood", 1, "OOD r"]];
    const table = el("table.vfe-table");
    root.append(el("div.widget-title", { text: "Agreement with human progress annotations (Table I)" }), table,
      el("p.caption", { text: "Click a metric to sort. Underlined: best online value model. Offline models use privileged future frames and can't run during a rollout." }));

    function best(group, d, i) { return Math.max(...rows.filter((r) => r.group === group).map((r) => r[d][i])); }

    function render() {
      const head = el("tr", null, el("th", { text: "Method" }),
        ...cols.map(([d, i, lab], ci) => el("th", { text: lab + (sortKey === ci ? " ↓" : ""), onclick: () => { sortKey = sortKey === ci ? null : ci; render(); } })));
      const out = [head];
      ["Online", "Offline (privileged)"].forEach((group) => {
        out.push(el("tr", null, el("td.group", { colspan: 5, text: group })));
        let rs = rows.filter((r) => r.group === group);
        if (sortKey != null) { const [d, i] = cols[sortKey]; rs = [...rs].sort((a, b) => b[d][i] - a[d][i]); }
        rs.forEach((r) => {
          out.push(el("tr" + (r.ours ? ".ours" : ""), null,
            el("td", { text: r.name.trim(), style: { paddingLeft: r.name.startsWith("  ") ? "22px" : null, color: r.name.startsWith("  ") ? "#777" : null } }),
            ...cols.map(([d, i]) => {
              const v = r[d][i];
              const isBest = group === "Online" && v === best("Online", d, i);
              return el("td.metric", null, el("div.bg", { style: { width: `${v * 100}%` } }), el("span" + (isBest ? ".best" : ""), { text: v.toFixed(2) }));
            })));
        });
      });
      table.replaceChildren(...out);
    }
    render();
  })();

  // Fig. 6
  splitBars(document.getElementById("vfe-deploy-chart"), SITE.vfeDeploy, { highlight: "IC-VFE" });

  // ---------------------------------------------------------------------------
  // Fig. 7: success and progress vs. pool size, one chart per domain.
  (function demoCount() {
    const root = document.getElementById("demo-count-chart");
    const d = SITE.demoCount;
    const ind = el("div"), ood = el("div");
    root.appendChild(el("div.chart-pair", null, ind, ood));
    [[ind, "ind", "In-domain tasks"], [ood, "ood", "Out-of-domain tasks"]].forEach(([box, dom, title]) => {
      const W = 480, Hh = 250, m = { l: 40, r: 16, t: 16, b: 38 };
      const pw = W - m.l - m.r, ph = Hh - m.t - m.b;
      const X = (n) => (Math.log(n) / Math.log(10)) * pw;
      const Y = (v) => ph * (1 - v / 100);
      const s = svg("svg", { viewBox: `0 0 ${W} ${Hh}` });
      const g = svg("g", { transform: `translate(${m.l},${m.t})` });
      s.appendChild(g);
      const axis = svg("g.axis");
      [0, 20, 40, 60, 80, 100].forEach((v) => { axis.appendChild(svg("line", { x1: 0, x2: pw, y1: Y(v), y2: Y(v) })); axis.appendChild(svg("text", { x: -6, y: Y(v) + 4, "text-anchor": "end", text: v })); });
      d.n.forEach((n) => axis.appendChild(svg("text", { x: X(n), y: ph + 15, "text-anchor": "middle", text: n })));
      axis.appendChild(svg("text", { x: pw / 2, y: ph + 32, "text-anchor": "middle", text: "demonstrations in pool (log scale)" }));
      g.appendChild(axis);
      const color = "#b8901a";
      [["sr", null, "success"], ["progress", "4 3", "progress"]].forEach(([k, dash, kl]) => {
        const pts = d.n.map((n, i) => [X(n), Y(d[dom][k][i])]);
        g.appendChild(svg("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(""), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash }));
        pts.forEach((p, i) => {
          g.appendChild(svg("circle", { cx: p[0], cy: p[1], r: 4, fill: dash ? "#fff" : color, stroke: color, "stroke-width": 1.5,
            onmousemove: (e) => showTip(e, `${d.n[i]} demo${d.n[i] > 1 ? "s" : ""}<br>${kl}: ${d[dom][k][i].toFixed(1)}%`), onmouseleave: hideTip }));
          // Progress labels above the point, success labels below, so the two series never collide.
          const anchor = i === 0 ? "start" : i === d.n.length - 1 ? "end" : "middle";
          g.appendChild(svg("text.bar-label", { x: p[0] + (i === 0 ? 6 : 0), y: dash ? p[1] - 9 : p[1] + 17, "text-anchor": anchor, text: d[dom][k][i].toFixed(1) }));
        });
      });
      box.append(el("div.widget-title", { text: title }), s, el("div.legend", null,
        el("span", null, el("i.line", { style: { background: color } }), "binary success"),
        el("span", null, el("i.dash"), "task progress")));
    });
  })();

  // Fig. 8
  splitBars(document.getElementById("train-deploy-chart"),
    SITE.trainDeploy.map((x) => ({ ...x, name: x.name.split(" + ")[0], sub: `+ ${x.deploy} at test` })), {});
})();
