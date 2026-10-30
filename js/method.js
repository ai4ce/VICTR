// Section 3: overview pipeline, demo-pool builder, IC-VFE annotation,
// retrieval replay, Fig. 5 examples and the IC-VLA action panel. 3.2 and 3.3 share one task + playback clock.
(function () {
  const { el, svg, chips, taskPicker, media, Clock, playerBar, whenVisible, taskById, clamp, lerp } = window.U;
  const SITE = window.VICTR_SITE;
  const SIM = window.SIM;

  const scrollTo = (sel) => document.querySelector(sel).scrollIntoView({ behavior: "smooth", block: "start" });

  // Value colour ramp: white -> gold, used wherever a progress key is drawn.
  const valueColor = (v) => {
    const a = [255, 251, 238], b = [224, 164, 0];
    return `rgb(${a.map((x, i) => Math.round(lerp(x, b[i], clamp(v, 0, 1)))).join(",")})`;
  };

  function arrowDefs(id, color) {
    return svg("defs", null,
      svg("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5.6, markerHeight: 5.6, orient: "auto-start-reverse" },
        svg("path", { d: "M0,0 L10,5 L0,10 z", fill: color })));
  }

  // ---------------------------------------------------------------------------
  // Shared task + clock for 3.2 and 3.3.
  const state = {
    task: SIM.task(window.U.sharedTask(SIM.ids, SIM.ids[0])),
    clock: new Clock(SIM.task(window.U.sharedTask(SIM.ids, SIM.ids[0])).length),
    taskListeners: new Set(),
    pickers: [],
    setTask(id) {
      this.task = SIM.task(id);
      this.clock.pause();
      this.clock.setLength(this.task.length);
      this.clock.seek(0);
      this.pickers.forEach((p) => p.set(id));
      this.taskListeners.forEach((fn) => fn(this.task));
      this.clock.play();
    },
    onTask(fn) { this.taskListeners.add(fn); fn(this.task); },
    picker() {
      const p = taskPicker(SIM.ids, this.task.id, (id) => this.setTask(id));
      this.pickers.push(p);
      return p;
    },
  };

  // ===========================================================================
  // 3. Overview: the paper's Fig. 2 layout. Stage 1 retrieves a context chunk;
  // in stage 2 the pi0.5 VLM and the action expert sit side by side, with their
  // input tokens, encoders and the raw observation stacked underneath. Steps
  // auto-advance and highlight one part of the loop at a time.
  (function pipeline() {
    const root = document.getElementById("pipeline");
    const W = 820, Hh = 292;
    const s = svg("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "VICTR architecture" });
    s.append(arrowDefs("arrow", "#9aa0a4"), arrowDefs("arrow-v", "#e0a400"), arrowDefs("arrow-c", "#7a5bb5"));
    s.append(
      svg("text.stage-label", { x: 14, y: 14, text: "Stage 1 · context retrieval" }),
      svg("text.stage-label", { x: 356, y: 14, text: "Stage 2 · action generation" }),
      svg("line", { x1: 341, y1: 4, x2: 341, y2: Hh - 4, stroke: "#e3e5e7", "stroke-dasharray": "4 4" }));

    const wire = (d, cls) => svg("path.flow" + (cls ? "." + cls : ""), {
      d, "marker-end": `url(#${cls === "value" ? "arrow-v" : cls === "chunk" ? "arrow-c" : "arrow"})` });
    const W_ = {
      obsVfe: wire("M84,232 L84,169"),
      refVfe: wire("M214,232 L214,206 L130,206 L130,169"),
      value: wire("M154,111 L182,111", "value"),
      poolRet: wire("M290,232 L290,169"),
      chunk: wire("M326,153 L354,153", "chunk"),
      tokVla: wire("M406,139 L406,130 M490,139 L490,130 M550,139 L550,130 M616,139 L616,130", "chunk"),
      noiseAe: wire("M733,139 L733,130"),
      aeAct: wire("M733,56 L733,48"),
      enc: wire("M490,186 L490,169 M550,186 L550,169 M616,186 L616,169"),
      raw: wire("M490,232 L490,212 M550,232 L550,212 M616,232 L616,212"),
    };
    s.append(...Object.values(W_));
    s.append(svg("text", { x: 138, y: 202, "font-size": 10, fill: "#888", text: "reference demo" }),
      svg("text", { x: 160, y: 103, "font-size": 12, fill: "#b88600", "font-style": "italic", text: "v̂ₜ" }));

    const N = {};
    function node(key, x, y, w, h, t, sub, opt = {}) {
      const g = svg("g.node", opt.go ? { tabindex: 0, role: "link", onclick: () => scrollTo(opt.go),
        onkeydown: (e) => { if (e.key === "Enter") scrollTo(opt.go); } } : { style: "cursor: default" });
      g.appendChild(svg("rect.box", { x, y, width: w, height: h, rx: opt.r ?? 9,
        fill: opt.fill || "#fff", stroke: opt.stroke || "#cfd4d7", "stroke-width": 1.5 }));
      g.appendChild(svg("text", { x: x + w / 2, y: y + h / 2 + (sub ? -3 : 4), "text-anchor": "middle",
        "font-weight": 600, "font-size": opt.fs || 13, text: t }));
      if (sub != null) g.subT = g.appendChild(svg("text.sub", { x: x + w / 2, y: y + h / 2 + 13, "text-anchor": "middle", text: sub }));
      s.appendChild(g);
      N[key] = g;
    }
    const CH = { fill: "#f3eefc", stroke: "#7a5bb5" }, TOK = { fill: "#f6f7f8", fs: 11, r: 5 };
    node("obs", 14, 232, 140, 52, "Observation oₜ", "3 cams · joints · task", { go: "#icvfe" });
    node("pool", 184, 232, 142, 52, "Demo pool", "5–12 demos · 1 s chunks", { go: "#pool" });
    node("vfe", 14, 56, 140, 111, "IC-VFE", "SigLIP + Gemma 3 270M", { go: "#icvfe", fill: "#fff8db", stroke: "#e0a400" });
    node("ret", 184, 56, 142, 111, "Retrieve", "argmin |v̂ₜ − v̂ᵢ|", { go: "#vfe-widget" });
    node("act", 660, 20, 146, 28, "Action chunk", null, { fs: 11.5, r: 6 });
    node("vla", 356, 56, 294, 72, "IC-VLA · π0.5 VLM", "block-causal: query reads cₜ, cₜ can't read query", CH);
    node("ae", 660, 56, 146, 72, "Action expert", "flow matching", { fill: "#eaf4f7", stroke: "#4b93a8" });
    node("ctx", 356, 139, 100, 28, "Context cₜ", null, { go: "#vfe-widget", ...CH, fs: 11.5, r: 5 });
    node("img", 462, 139, 56, 28, "images", null, TOK);
    node("jnt", 524, 139, 52, 28, "joints", null, TOK);
    node("lang", 582, 139, 68, 28, "task ℓ", null, TOK);
    node("noise", 660, 139, 146, 28, "noise", null, TOK);
    node("sig", 462, 186, 56, 24, "SigLIP", null, { ...TOK, fill: "#fff" });
    node("mlp", 524, 186, 52, 24, "MLP", null, { ...TOK, fill: "#fff" });
    node("tkz", 582, 186, 68, 24, "tokenizer", null, { ...TOK, fill: "#fff", fs: 9.5 });
    node("raw", 462, 232, 188, 52, "Observation oₜ", "3 cams · joint states · task", {});

    const steps = [
      { on: ["obs", "pool", "vfe"], w: ["obsVfe", "refVfe"],
        text: "IC-VFE reads a reference demonstration from the pool and estimates how far along the task the robot is: v̂ₜ." },
      { on: ["vfe", "pool", "ret"], w: ["value", "poolRet"],
        text: "The pool chunk whose progress key is closest to v̂ₜ is retrieved." },
      { on: ["ret", "ctx", "img", "jnt", "lang", "sig", "mlp", "tkz", "raw", "vla"], w: ["chunk", "tokVla", "enc", "raw"],
        text: "The chunk's images, joint states and FAST action tokens are placed in the VLM's context, ahead of the current observation and task." },
      { on: ["vla", "ae", "noise", "act"], w: ["noiseAe", "aeAct"],
        text: "The action expert denoises the next one-second action chunk. Then the loop repeats." },
    ];
    const text = el("div.step-text");
    const dots = el("div.step-dots");
    const dotBtns = steps.map((_, i) => dots.appendChild(el("button", { type: "button", "aria-label": `Step ${i + 1}`, onclick: () => { go(i); restart(); } })));

    let step = 0, timer = null, textTimer = null;
    function live() {
      const c = state.task.retrieve(state.clock.t);
      N.vfe.subT.textContent = step === 0 ? `v̂ₜ = ${c.query.toFixed(2)}` : "SigLIP + Gemma 3 270M";
      N.ret.subT.textContent = step === 1 ? `demo ${c.demo + 1}, ${c.start.toFixed(1)}–${c.end.toFixed(1)} s` : "argmin |v̂ₜ − v̂ᵢ|";
    }
    function go(i) {
      step = i;
      const st = steps[i];
      Object.entries(N).forEach(([k, g]) => g.classList.toggle("dim", !st.on.includes(k)));
      Object.entries(W_).forEach(([k, p]) => p.classList.toggle("on", st.w.includes(k)));
      text.textContent = `${i + 1}. ${st.text}`;
      dotBtns.forEach((b, k) => b.classList.toggle("active", k === i));
      live();
    }
    function restart() { stop(); timer = setInterval(() => go((step + 1) % steps.length), 3600); }
    function stop() { clearInterval(timer); timer = null; }

    root.append(s, text, dots);
    go(0);
    state.clock.on(live);
    state.onTask(live);
    whenVisible(root, restart, stop);
  })();

  // ===========================================================================
  // 3.1 Demonstration pool builder (auto-playing step animation)
  (function poolBuilder() {
    const root = document.getElementById("poolbuilder");
    const task = SIM.task(SIM.ids[0]);
    const demos = task.pool;
    const maxLen = Math.max(...demos.map((d) => d.length));
    const W = 800, rowH = 22, top = 14, left = 70, barW = 470;
    const Hh = top + demos.length * rowH + 14;
    const sx = barW / maxLen;

    const s = svg("svg", { viewBox: `0 0 ${W} ${Hh}` });
    const rows = demos.map((d) => {
      const g = svg("g.demo-bar");
      const label = svg("text", { x: left - 10, y: rowH / 2 + 4, "text-anchor": "end", "font-size": 11, fill: "#777", text: `demo ${d.index + 1}` });
      const base = svg("rect", { x: left, y: 3, width: d.length * sx, height: rowH - 6, rx: 4, fill: "#e9ecee" });
      const cells = svg("g", { opacity: 0 });
      const ticks = svg("g", { opacity: 0 });
      for (let k = 0; k < d.length; k++) {
        cells.appendChild(svg("rect", { x: left + k * sx + 0.5, y: 3, width: sx - 1, height: rowH - 6, rx: 2, fill: valueColor(d.values[k]) }));
        if (k > 0) ticks.appendChild(svg("line", { x1: left + k * sx, x2: left + k * sx, y1: 3, y2: rowH - 3, stroke: "#fff", "stroke-width": 1.2 }));
      }
      const refTag = svg("text", { x: left + d.length * sx + 8, y: rowH / 2 + 4, "font-size": 11, "font-weight": 700, fill: "#b88600", opacity: 0, text: "reference" });
      g.append(base, cells, ticks, label, refTag);
      s.appendChild(g);
      return { d, g, base, cells, ticks, refTag };
    });

    // Key-value store box, shown in every step.
    const store = svg("g");
    const sx0 = 600;
    store.appendChild(svg("rect", { x: sx0, y: top, width: 190, height: Hh - top - 14, rx: 10, fill: "#fbfbfc", stroke: "#cfd4d7" }));
    store.appendChild(svg("text", { x: sx0 + 95, y: top + 22, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "#333", text: "Chunk index" }));
    const rowsTxt = [
      ["key", "#999", 600], ["progress v̂ᵢ", "#b88600", 400], ["DINOv2 hᵢ", "#4b93a8", 400],
      ["value", "#999", 600], ["3 camera images", "#555", 400], ["proprio state", "#555", 400], ["FAST action tokens", "#7a5bb5", 400],
    ];
    rowsTxt.forEach(([t, c, w], i) => store.appendChild(svg("text", {
      x: sx0 + 18 + (w === 600 ? 0 : 10), y: top + 52 + i * 22, "font-size": w === 600 ? 10 : 12.5,
      "font-weight": w, fill: c, "letter-spacing": w === 600 ? ".08em" : 0, text: w === 600 ? t.toUpperCase() : t })));
    s.appendChild(store);

    const steps = [
      `${demos.length} demonstrations are recorded for the task at deployment time. None are used for training.`,
      "One of them is the reference IC-VFE is conditioned on.",
      "Every demonstration is cut into one-second chunks, the IC-VLA's action horizon (H = 30 steps), starting every 5 frames.",
      "IC-VFE labels each chunk with its progress v̂ᵢ (darker gold = further along).",
      "Each chunk is indexed by its progress (and optionally a DINOv2 embedding) and stores the observation and action tokens to place in context.",
    ];
    const text = el("div.step-text");
    const dots = el("div.step-dots");
    const dotBtns = steps.map((_, i) => { const b = el("button", { type: "button", "aria-label": `Step ${i + 1}`, onclick: () => { go(i); restart(); } }); dots.appendChild(b); return b; });

    let step = 0, timer = null, textTimer = null;
    function go(i) {
      step = i;
      rows.forEach((r) => {
        const y = top + demos.indexOf(r.d) * rowH;
        r.g.style.transform = `translate(0px, ${y}px)`;
        r.ticks.style.opacity = i >= 2 ? 1 : 0;
        r.cells.style.opacity = i >= 3 ? 1 : 0;
        r.refTag.style.opacity = i >= 1 && r.d.isReference ? 1 : 0;
        r.base.style.fill = i >= 1 && r.d.isReference ? "#ffe89f" : "#e9ecee";
        r.base.style.stroke = i >= 1 && r.d.isReference ? "#e0a400" : "transparent";
      });
      // Soft fade between step captions.
      text.style.opacity = 0;
      clearTimeout(textTimer);
      textTimer = setTimeout(() => { text.textContent = `${i + 1}. ${steps[i]}`; text.style.opacity = 1; }, text.textContent ? 450 : 0);
      dotBtns.forEach((b, k) => b.classList.toggle("active", k === i));
    }
    function restart() { stop(); timer = setInterval(() => go((step + 1) % steps.length), 6400); }
    function stop() { clearInterval(timer); timer = null; }

    root.append(el("div.widget-title", { text: `Building the pool · ${window.U.taskById(task.id).name.toLowerCase()}` }), s, text, dots);
    go(0);
    whenVisible(root, restart, stop);
  })();

  // ===========================================================================
  // 3.2 IC-VFE annotating a rollout + the chunk it retrieves
  (function vfeWidget() {
    const root = document.getElementById("vfe-widget");
    const W = 800, Hh = 190, m = { l: 46, r: 12, t: 12, b: 26 };
    const pw = W - m.l - m.r, ph = Hh - m.t - m.b;

    const qSlot = el("div"), cSlot = el("div");
    const cHead = el("span");
    const refStrip = el("div.ref-strip");
    const refLabel = el("div.ref-label", null, el("span", { text: "Reference demo" }), el("span.ref-len"));
    const readout = el("div.readout");
    const s = svg("svg", { viewBox: `0 0 ${W} ${Hh}` });
    const mapSvg = svg("svg");

    root.append(
      el("div.widget-title", { text: "IC-VFE annotating a rollout" }),
      state.picker(),
      el("div.retrieval-grid", { style: { marginTop: "12px" } },
        el("div", null, el("h4", null, el("span", { text: "Current observation oₜ" })), qSlot),
        el("div", null, el("h4", null, el("span", { text: "Retrieved chunk cₜ" }), cHead), cSlot)),
      el("div.vfe-plot", { style: { marginTop: "12px" } }, s, readout),
      refStrip, refLabel,
      playerBar(state.clock),
      el("div.pool-map", null, mapSvg),
      el("p.caption", { html: "IC-VFE's estimates on a rollout. The highlighted second of the reference demo has the closest progress. Rows are demos; each tick is a chunk at its progress key <span class='math'>v̂ᵢ</span>. Gold: current <span class='math'>v̂ₜ</span>; purple: retrieved chunk." }));

    // Static axes
    const g = svg("g", { transform: `translate(${m.l},${m.t})` });
    s.appendChild(g);
    const axis = svg("g.axis");
    g.appendChild(axis);
    g.appendChild(svg("text", { x: -38, y: ph / 2, transform: `rotate(-90 -38 ${ph / 2})`, "text-anchor": "middle", "font-size": 11, fill: "#777", text: "progress v̂ₜ" }));
    const xTicks = svg("g.axis");
    g.appendChild(xTicks);
    const live = svg("path", { fill: "none", stroke: "#e0a400", "stroke-width": 2.4, "stroke-linejoin": "round" });
    const cursor = svg("line", { y1: 0, y2: ph, stroke: "#2c2f31", "stroke-width": 1, opacity: 0.35 });
    const dot = svg("circle", { r: 4.5, fill: "#e0a400", stroke: "#fff", "stroke-width": 1.5 });
    g.append(live, cursor, dot);

    let task, qMedia, cMedia, refCells = [], paths = [], lo = 0, hi = 1, lastChunk = null;
    const X = (t) => (t / task.length) * pw;
    const Y = (v) => ph * (1 - (v - lo) / (hi - lo));
    const MW = 800, rowH = 14, mm = { l: 74, r: 14, t: 18, b: 22 };
    const mpw = MW - mm.l - mm.r;
    let valLine, valLabel, hit, tickEls = new Map();

    state.onTask((tk) => {
      task = tk;
      const vs = task.ticks.map((x) => x.v);
      lo = Math.min(...vs); hi = Math.max(...vs);
      if (hi - lo < 1e-6) { lo -= 0.05; hi += 0.05; }
      axis.replaceChildren();
      [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
        const v = lo + f * (hi - lo);
        axis.appendChild(svg("line", { x1: 0, x2: pw, y1: Y(v), y2: Y(v) }));
        axis.appendChild(svg("text", { x: -6, y: Y(v) + 4, "text-anchor": "end", text: v.toFixed(2) }));
      });
      qMedia = media(tk.cfg.rollout.video, "Query rollout", "");
      qSlot.replaceChildren(qMedia);
      cMedia = media(tk.cfg.rollout.context, "Retrieved chunk", "");
      cMedia.classList.add("chunk-frame");
      cSlot.replaceChildren(cMedia);
      lastChunk = null;
      xTicks.replaceChildren();
      const step = task.length > 15 ? 5 : 2;
      for (let s2 = 0; s2 <= task.length; s2 += step) {
        xTicks.appendChild(svg("text", { x: X(s2), y: ph + 14, "text-anchor": "middle", text: `${s2}s` }));
      }
      refStrip.replaceChildren();
      refCells = task.ref.values.slice(0, task.ref.length).map((v) => {
        const c = el("div", { style: { background: valueColor(v) } });
        refStrip.appendChild(c);
        return c;
      });
      // The curve up to each tick, built once per task.
      paths = task.ticks.map((_, k) => task.ticks.slice(0, k + 1)
        .map((tk2, i) => `${i ? "L" : "M"}${X(tk2.t).toFixed(1)},${Y(tk2.v).toFixed(1)}`).join(""));
      root.querySelector(".ref-len").textContent = `demo ${task.ref.index + 1} · ${task.ref.length} s, one frame per second in context`;

      // Pool map
      mapSvg.setAttribute("viewBox", `0 0 ${MW} ${mm.t + tk.pool.length * rowH + mm.b}`);
      mapSvg.replaceChildren();
      const mg = svg("g", { transform: `translate(${mm.l},${mm.t})` });
      mapSvg.appendChild(mg);
      tk.pool.forEach((d, i) => {
        mg.appendChild(svg("text", { x: -8, y: i * rowH + rowH / 2 + 4, "text-anchor": "end", "font-size": 10.5,
          fill: d.isReference ? "#b88600" : "#888", "font-weight": d.isReference ? 700 : 400, text: `demo ${d.index + 1}${d.isReference ? " ★" : ""}` }));
        mg.appendChild(svg("line", { x1: 0, x2: mpw, y1: i * rowH + rowH / 2, y2: i * rowH + rowH / 2, stroke: "#f0f1f2" }));
      });
      tickEls = new Map(tk.chunks.map((c) => {
        const r = svg("rect", { x: c.key * mpw - 1.5, y: c.demo * rowH + 3, width: 3, height: rowH - 6, rx: 1, fill: "#c8cdd0" });
        mg.appendChild(r);
        return [`${c.demo}:${c.start}`, r];
      }));
      const axisY = tk.pool.length * rowH + 4;
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => mg.appendChild(svg("text", { x: v * mpw, y: axisY + 12, "text-anchor": "middle", "font-size": 10, fill: "#999", text: v })));
      valLine = svg("line", { y1: -6, y2: tk.pool.length * rowH, stroke: "#e0a400", "stroke-width": 2 });
      valLabel = svg("text", { y: -8, "text-anchor": "middle", "font-size": 11, fill: "#b88600", "font-weight": 700 });
      hit = svg("rect", { width: 9, height: rowH, rx: 2, fill: "#7a5bb5" });
      mg.append(valLine, valLabel, hit);
    });

    function renderChunk(c) {
      const chunk = `${c.demo}:${c.start}`;
      if (chunk === lastChunk) return;
      tickEls.get(lastChunk)?.setAttribute("fill", "#c8cdd0");
      lastChunk = chunk;
      tickEls.get(lastChunk)?.setAttribute("fill", "#7a5bb5");
      cHead.textContent = `demo ${c.demo + 1} · ${c.start.toFixed(1)}–${c.end.toFixed(1)} s · v̂ᵢ = ${c.key.toFixed(2)}`;
    }

    state.clock.on((clk) => {
      if (!task) return;
      const t = clk.t, k = task.tickIndex(t), tick = task.ticks[k];
      live.setAttribute("d", paths[k]);
      cursor.setAttribute("x1", X(t)); cursor.setAttribute("x2", X(t));
      dot.setAttribute("cx", X(tick.t)); dot.setAttribute("cy", Y(tick.v));
      refCells.forEach((cell, i) => cell.classList.toggle("hot", i === tick.refSec));
      readout.innerHTML = `v̂ₜ = <b>${tick.v.toFixed(2)}</b> · matches ref. ${tick.refSec}–${tick.refSec + 1} s`;
      const c = task.retrieve(t);
      renderChunk(c);
      qMedia.sync(t, clk.playing);
      cMedia.sync(t, clk.playing);
      qMedia.setSub(`${task.cfg.task} · t = ${t.toFixed(1)} s`);
      qMedia.setProgress(t / task.length);
      valLine.setAttribute("x1", c.query * mpw); valLine.setAttribute("x2", c.query * mpw);
      valLabel.setAttribute("x", c.query * mpw); valLabel.textContent = `v̂ₜ = ${c.query.toFixed(2)}`;
      hit.setAttribute("x", c.key * mpw - 4.5); hit.setAttribute("y", c.demo * rowH);
    });

    let started = false;
    whenVisible(root, () => { if (!started) { started = true; state.clock.play(); } });
  })();

  // ===========================================================================
  // 3.2c Real retrieval examples (paper Fig. 5)
  (function fig5() {
    const root = document.getElementById("fig5-widget");
    const ex = SITE.retrievalExamples;
    const exTask = window.U.sharedTask(ex.map((e) => e.task), ex[0].task);
    let exId = ex.find((e) => e.task === exTask).id, cam = "top";
    const pickTask = (task) => { exId = ex.find((e) => e.task === task).id; render(); };
    const grid = el("div.fig5-grid");
    const cols = [
      ["original", "Query", "query"],
      ["value", "IC-VFE progress (VICTR)", "value"],
      ["vision", "DINOv2 similarity (RICL)", ""],
      ["vision_value", "Progress + vision", ""],
    ];
    function render() {
      grid.replaceChildren(...cols.map(([sig, label, cls]) =>
        el("figure" + (cls ? "." + cls : ""), null,
          el("img", { src: `assets/retrieval/${exId}_${cam}_${sig}.png`, alt: `${label}, ${cam} camera`, loading: "lazy" }),
          el("figcaption", { text: label }))));
    }
    root.append(
      el("div.widget-title", { text: "Retrieved chunks by signal · examples" }),
      el("div.controls-row", null,
        taskPicker(ex.map((e) => e.task), exTask, pickTask),
        U.toggle([["top", "Head cam"], ["left", "Left wrist"], ["right", "Right wrist"]], cam, (c) => { cam = c; render(); })),
      grid,
      el("p.caption", { text: "Retrieval uses the head camera only." }));
    render();
  })();

})();
