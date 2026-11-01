// Section 3.25: IC-VFE attention, measured on the deployed model (data/attention.js,
// scratch/attention/extract.py). Rows are query blocks, columns key blocks: one block
// per second of the reference demo, then the current observation Oₜ, the instruction ℓ
// and the value token r. Each cell is the head-averaged attention a row's tokens pay to
// that block. Context seconds read only themselves, except at the block-causal layers
// (6, 12, 18), where they also read earlier seconds.
(function () {
  const { el, svg, taskPicker, whenVisible, showTip, hideTip } = window.U;
  const root = document.getElementById("attn-widget");
  const DATA = window.VICTR_ATTN;
  const ids = Object.keys(DATA);
  const CAUSAL = new Set([6, 12, 18]);
  const SIZE = 560, labelW = 40, labelH = 30;

  // 0 = masked, 1 = always attended, 2 = extra block-causal edge
  function edge(q, k, T, causal) {
    if (q === T + 2) return 1;            // r reads everything
    if (k === T + 2) return 0;
    if (q >= T) return 1;                 // Oₜ and ℓ read all context and each other
    if (k >= T) return 0;                 // context never reads the query
    if (q === k) return 1;
    return causal && k < q ? 2 : 0;
  }
  const label = (i, T) => (i < T ? `${i}s` : ["Oₜ", "ℓ", "r"][i - T]);

  const s = svg("svg", { viewBox: `0 0 ${labelW + SIZE + 2} ${labelH + SIZE + 2}`, width: labelW + SIZE + 2, height: labelH + SIZE + 2 });
  const layerName = el("div.layer-name");
  const layerKind = el("div.layer-kind");
  const strip = el("div.layer-strip");
  const layerBtns = [];
  for (let l = 1; l <= 18; l++) {
    const b = el("button" + (CAUSAL.has(l) ? ".causal" : ""), { type: "button", text: l, title: `Layer ${l}`, onclick: () => { auto = false; setLayer(l); } });
    layerBtns.push(b); strip.appendChild(b);
  }
  let mode = "act";
  const meta = el("p", { style: { fontSize: "13px", color: "#777", marginTop: "12px" } });
  const side = el("div.attn-side", null,
    layerName, layerKind, strip,
    el("div.controls-row", null, U.toggle([["act", "Attention"], ["mask", "Mask only"]], mode, (m) => { mode = m; paint(); })),
    el("div.legend", null,
      el("span", null, el("i", { style: { background: "var(--attend)" } }), "attended in every layer"),
      el("span", null, el("i", { style: { background: "var(--causal)" } }), "block-causal attention (layers 6, 12, 18)"),
      el("span", null, el("i", { style: { background: "#fff" } }), "masked")),
    meta);
  root.append(el("div.widget-title", { text: "IC-VFE attention, 18-layer Gemma 3 270M" }),
    taskPicker(ids, window.U.sharedTask(ids, ids[0]), (id) => setTask(id)),
    el("div.attn-wrap", { style: { marginTop: "12px" } }, s, side));

  let task, T, N, cell, cells = [], rowHi, layer = 1, hoverRow = -1, auto = true;

  function setTask(id) {
    task = DATA[id];
    T = task.T; N = T + 3; cell = SIZE / N;
    s.replaceChildren();
    const grid = svg("g", { transform: `translate(${labelW},${labelH})` });
    s.appendChild(grid);
    const every = T > 40 ? 10 : 5;
    for (let i = 0; i < N; i++) {
      if (i < T && i % every) continue;
      const mid = (i + 0.5) * cell, fs = i < T ? 9 : 12;
      const style = i < T ? {} : { "font-style": "italic", "font-family": "Times New Roman, serif" };
      s.appendChild(svg("text", { x: labelW + mid, y: labelH - 8, "text-anchor": "middle", "font-size": fs, fill: "#777", ...style, text: label(i, T) }));
      s.appendChild(svg("text", { x: labelW - 6, y: labelH + mid + 3, "text-anchor": "end", "font-size": fs, fill: "#777", ...style, text: label(i, T) }));
    }
    grid.appendChild(svg("line", { x1: T * cell, x2: T * cell, y1: 0, y2: N * cell, stroke: "#999", "stroke-width": 1 }));
    grid.appendChild(svg("line", { y1: T * cell, y2: T * cell, x1: 0, x2: N * cell, stroke: "#999", "stroke-width": 1 }));
    cells = [];
    const gap = cell > 8 ? 1 : 0;
    for (let q = 0; q < N; q++) {
      cells.push([]);
      for (let k = 0; k < N; k++) {
        const r = svg("rect.attn-cell", { x: k * cell + gap / 2, y: q * cell + gap / 2, width: cell - gap, height: cell - gap, rx: Math.min(3, cell / 5) });
        r.addEventListener("mousemove", (e) => hover(q, k, e));
        grid.appendChild(r);
        cells[q].push(r);
      }
    }
    rowHi = svg("rect", { x: -2, width: N * cell + 4, height: cell, rx: 3, fill: "none", stroke: "#2c2f31", "stroke-width": 1.5, opacity: 0 });
    grid.appendChild(rowHi);
    meta.textContent = `Rows attend to columns: each second of the reference demo, then the observation Oₜ, the instruction ℓ and the value token r. ` +
      `Example query taken from a held-out demonstration ${task.query_sec} s in (progress ${task.progress.toFixed(2)}).`;
    paint();
  }
  s.addEventListener("mouseleave", () => { hoverRow = -1; if (rowHi) rowHi.setAttribute("opacity", 0); hideTip(); paint(); });

  function paint() {
    const causal = CAUSAL.has(layer);
    const A = task.layers[layer - 1];
    for (let q = 0; q < N; q++) {
      const maxW = Math.max(...A[q]) || 1;
      for (let k = 0; k < N; k++) {
        const e = edge(q, k, T, causal);
        const r = cells[q][k];
        r.setAttribute("fill", e === 2 ? "var(--causal)" : e === 1 ? "var(--attend)" : "#fff");
        r.setAttribute("stroke", e ? "none" : "#f0f0f0");
        let op = e ? (mode === "act" ? 0.04 + 0.96 * Math.sqrt(A[q][k] / maxW) : 1) : 1;
        if (hoverRow >= 0 && q !== hoverRow) op *= 0.25;
        r.setAttribute("fill-opacity", op.toFixed(3));
      }
    }
    layerName.textContent = `Layer ${layer}`;
    layerKind.textContent = causal ? "Block-causal: each context second also reads earlier seconds" : "Within-block: context seconds are isolated from each other";
    layerBtns.forEach((b, i) => b.classList.toggle("active", i + 1 === layer));
  }

  function hover(q, k, e) {
    hoverRow = q;
    rowHi.setAttribute("y", q * cell); rowHi.setAttribute("opacity", 1);
    const w = task.layers[layer - 1][q][k];
    const on = edge(q, k, T, CAUSAL.has(layer));
    showTip(e, `<b>${label(q, T)}</b> → <b>${label(k, T)}</b>: ${on ? `${(w * 100).toFixed(1)}% of attention` : "masked"}`);
    paint();
  }

  function setLayer(l) { layer = l; paint(); }

  // Auto-steps through layers until the user picks one.
  let timer = null;
  whenVisible(root,
    () => { if (!timer) timer = setInterval(() => { if (auto) setLayer((layer % 18) + 1); }, 1400); },
    () => { clearInterval(timer); timer = null; });
  setTask(window.U.sharedTask(ids, ids[0]));
})();
