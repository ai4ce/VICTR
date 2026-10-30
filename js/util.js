// Small shared helpers. No framework; every widget builds its own DOM.
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";

  // el("div.cls#id", {attr...}, children...)
  function el(spec, attrs, ...kids) {
    const [, tag, rest] = spec.match(/^([a-z0-9]+)(.*)$/i);
    const node = document.createElement(tag);
    applySpec(node, rest);
    applyAttrs(node, attrs);
    append(node, kids);
    return node;
  }

  function svg(spec, attrs, ...kids) {
    const [, tag, rest] = spec.match(/^([a-z]+)(.*)$/i);
    const node = document.createElementNS(SVG_NS, tag);
    applySpec(node, rest);
    applyAttrs(node, attrs);
    append(node, kids);
    return node;
  }

  function applySpec(node, rest) {
    (rest.match(/[.#][^.#]+/g) || []).forEach((p) => {
      if (p[0] === ".") node.classList.add(p.slice(1));
      else node.id = p.slice(1);
    });
  }

  function applyAttrs(node, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
      else node.setAttribute(k, v);
    }
  }

  function append(node, kids) {
    kids.flat().forEach((k) => {
      if (k == null || k === false) return;
      node.appendChild(typeof k === "string" ? document.createTextNode(k) : k);
    });
  }

  // Deterministic RNG so simulated placeholders look the same on every load.
  function rng(seedStr) {
    let h = 1779033703 ^ seedStr.length;
    for (let i = 0; i < seedStr.length; i++) {
      h = Math.imul(h ^ seedStr.charCodeAt(i), 3432918353);
      h = (h << 13) | (h >>> 19);
    }
    return function () {
      h = Math.imul(h ^ (h >>> 16), 2246822507);
      h = Math.imul(h ^ (h >>> 13), 3266489909);
      h ^= h >>> 16;
      return (h >>> 0) / 4294967296;
    };
  }

  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  const fmtTime = (s) => {
    const m = Math.floor(s / 60);
    return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
  };

  // Policy names: "VICTR_vision" -> VICTR with "vision" as a subscript.
  const nameParts = (s) => { const i = s.indexOf("_"); return i < 0 ? [s, null] : [s.slice(0, i), s.slice(i + 1)]; };
  function name(s) { const [a, b] = nameParts(s); return b ? [a, el("sub", { text: b })] : [a]; }
  function svgName(s) { const [a, b] = nameParts(s); return b ? [a, svg("tspan", { dy: "0.3em", "font-size": "0.75em", text: b })] : [a]; }
  function nameHTML(s) { const [a, b] = nameParts(s); return b ? `${a}<sub>${b}</sub>` : a; }
  const plainName = (s) => s.replace("_", " ");

  // Segmented chip selector. items: [{id, label, tag}]
  function chips(items, activeId, onPick) {
    const wrap = el("div.chips");
    const buttons = items.map((it) => {
      const b = el("button.chip", { type: "button", onclick: () => pick(it.id) },
        it.label, it.tag ? el("span.tag", { text: it.tag }) : null);
      b.dataset.id = it.id;
      wrap.appendChild(b);
      return b;
    });
    function pick(id, silent) {
      buttons.forEach((b) => b.classList.toggle("active", b.dataset.id === id));
      if (!silent) onPick(id);
    }
    pick(activeId, true);
    wrap.set = (id) => pick(id, true);
    return wrap;
  }

  // In-domain / out-of-domain toggle plus a dropdown of that domain's tasks.
  // ids: task ids from data/results.js, in display order. Switching domain
  // returns to the task last picked there (or its first task).
  // Every task selector on the page follows the last task picked in any of them.
  const taskListeners = new Set();
  let sharedId = null;
  function onSharedTask(fn) { taskListeners.add(fn); return fn; }
  function shareTask(id, from) { sharedId = id; taskListeners.forEach((fn) => { if (fn !== from) fn(id); }); }
  // The shared task if `ids` has it, else the fallback (for a widget's first render).
  const sharedTask = (ids, fallback) => (ids.includes(sharedId) ? sharedId : fallback);

  function taskPicker(ids, activeId, onPick) {
    const byDomain = { ind: [], ood: [] };
    ids.map(taskById).forEach((t) => byDomain[t.domain].push(t));
    const last = { ind: byDomain.ind[0]?.id, ood: byDomain.ood[0]?.id };
    let domain = null, current = null;
    const pick = () => { current = select.value; onPick(current); shareTask(current, listener); };

    const select = el("select.task-select", { "aria-label": "Task", onchange: () => { last[domain] = select.value; pick(); } });
    const domains = toggle([["ind", "In-domain"], ["ood", "Out-of-domain"]], null, (d) => {
      if (d === domain) return;
      fill(d);
      pick();
    });
    const listener = onSharedTask((id) => {
      if (id === current || !ids.includes(id)) return;
      set(id);
      onPick(id);
    });

    function fill(d) {
      domain = d;
      select.replaceChildren(...byDomain[d].map((t) => el("option", { value: t.id, text: t.name })));
      select.value = last[d];
    }
    function set(id) {
      current = id;
      const d = taskById(id).domain;
      last[d] = id;
      domains.set(d);
      fill(d);
    }
    domains.querySelectorAll("button").forEach((b) => {
      if (!byDomain[b.dataset.id].length) { b.disabled = true; b.title = "No examples for this domain"; }
    });
    set(activeId);

    const wrap = el("div.task-picker", null, domains, select);
    wrap.set = set;
    return wrap;
  }

  function toggle(options, activeId, onPick) {
    const wrap = el("div.toggle");
    const btns = options.map(([id, label]) => {
      const b = el("button", { type: "button", text: label, onclick: () => set(id, true) });
      b.dataset.id = id;
      wrap.appendChild(b);
      return b;
    });
    function set(id, fire) {
      btns.forEach((b) => b.classList.toggle("active", b.dataset.id === id));
      if (fire) onPick(id);
    }
    set(activeId, false);
    wrap.set = (id) => set(id, false);
    return wrap;
  }

  // A video when `src` is set, otherwise a labelled placeholder. Placeholders
  // expose the same seek/play surface so widgets can drive either.
  function media(src, title, sub) {
    if (src) {
      const v = el("video", { src, muted: "", playsinline: "", preload: "metadata" });
      v.muted = true;
      const frame = el("div.media-frame", null, v);
      frame.isVideo = true;
      frame.video = v;
      // Follow an external clock: play natively, seek only to correct drift.
      frame.sync = (t, playing) => {
        if (playing && v.paused) v.play().catch(() => {});
        if (!playing && !v.paused) v.pause();
        if (Math.abs(v.currentTime - t) > (playing ? 0.3 : 0.05)) v.currentTime = t;
      };
      frame.setSub = () => {};
      frame.setProgress = () => {};
      return frame;
    }
    const subEl = el("div.ph-sub", { text: sub || "" });
    const bar = el("div.ph-bar");
    const box = el("div.media-placeholder", null,
      el("div.ph-title", { text: title }), subEl, bar);
    box.sync = () => {};
    box.setSub = (s) => { subEl.textContent = s; };
    box.setProgress = (f) => { bar.style.width = `${clamp(f, 0, 1) * 100}%`; };
    return box;
  }

  // Shared playback clock: drives a time t in [0, length] and notifies
  // listeners. Multiple widgets subscribe so 3.2 and 3.3 stay in lockstep.
  class Clock {
    constructor(length) {
      this.length = length;
      this.t = 0;
      this.playing = false;
      this.rate = 1;
      this.listeners = new Set();
      this._last = null;
      this._tick = this._tick.bind(this);
    }
    on(fn) { this.listeners.add(fn); fn(this); return () => this.listeners.delete(fn); }
    emit() { this.listeners.forEach((fn) => fn(this)); }
    setLength(len) { this.length = len; this.t = Math.min(this.t, len); this.emit(); }
    seek(t) { this.t = clamp(t, 0, this.length); this.emit(); }
    play() {
      if (this.playing) return;
      if (this.t >= this.length - 0.01) this.t = 0;
      this.playing = true; this._last = null;
      requestAnimationFrame(this._tick); this.emit();
    }
    pause() { this.playing = false; this.emit(); }
    toggle() { this.playing ? this.pause() : this.play(); }
    _tick(now) {
      if (!this.playing) return;
      if (this._last != null) {
        this.t += ((now - this._last) / 1000) * this.rate;
        if (this.t >= this.length) { this.t = 0; } // loop
      }
      this._last = now;
      this.emit();
      requestAnimationFrame(this._tick);
    }
  }

  const PLAY_ICON = '<svg viewBox="0 0 16 16"><path d="M4 2v12l10-6z"/></svg>';
  const PAUSE_ICON = '<svg viewBox="0 0 16 16"><path d="M3 2h4v12H3zM9 2h4v12H9z"/></svg>';

  function playerBar(clock) {
    const btn = el("button.play-btn", { type: "button", "aria-label": "Play/pause", onclick: () => clock.toggle() });
    const range = el("input.scrubber", { type: "range", min: 0, max: 1000, value: 0 });
    const time = el("span.timecode");
    range.addEventListener("input", () => { clock.pause(); clock.seek((range.value / 1000) * clock.length); });
    clock.on((c) => {
      btn.innerHTML = c.playing ? PAUSE_ICON : PLAY_ICON;
      range.value = Math.round((c.t / c.length) * 1000);
      time.textContent = `${fmtTime(c.t)} / ${fmtTime(c.length)}`;
    });
    return el("div.player-bar", null, btn, range, time);
  }

  // Start/stop a callback only while `node` is on screen.
  function whenVisible(node, onEnter, onLeave) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? onEnter() : onLeave && onLeave()));
    }, { threshold: 0.15 });
    io.observe(node);
  }

  const tooltipEl = () => document.getElementById("tooltip");
  function showTip(evt, html) {
    const t = tooltipEl();
    t.innerHTML = html;
    t.style.opacity = 1;
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const r = t.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    t.style.left = `${x}px`; t.style.top = `${y}px`;
  }
  function hideTip() { tooltipEl().style.opacity = 0; }

  const taskById = (id) => window.VICTR_RESULTS.tasks.find((t) => t.id === id);
  const policyById = (id) => window.VICTR_SITE.policies.find((p) => p.id === id);

  window.U = {
    el, svg, rng, clamp, lerp, smoothstep, fmtTime, chips, taskPicker, onSharedTask, shareTask, sharedTask, toggle, media,
    name, svgName, nameHTML, plainName,
    Clock, playerBar, whenVisible, showTip, hideTip, taskById, policyById,
  };
})();
