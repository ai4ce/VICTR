// Per-task model used by the method widgets (3.2 / 3.3), read from the real
// VICTR rollouts in data/vfe.js (tools/build_vfe.py). Everything shown at an
// inference tick (progress, retrieved chunk + its clip, matching reference
// second, action chunks) is precomputed; the page only finds the current tick.
(function () {
  const H = 30; // action chunk length (steps), one second at 30 Hz
  const DATA = window.VICTR_VFE;

  function buildTask(id) {
    const cfg = { task: id, ...DATA[id] };
    const ticks = cfg.ticks;
    const pool = cfg.pool.map((d, i) => ({ ...d, index: i, isReference: i === cfg.reference }));
    const chunks = [];
    pool.forEach((d) => d.keys.forEach(([start, key]) => chunks.push({ demo: d.index, start, key })));

    // Index of the tick in effect at time t (the last one at or before t).
    const tickIndex = (t) => {
      let k = 0;
      while (k + 1 < ticks.length && ticks[k + 1].t <= t) k++;
      return k;
    };

    function retrieve(t) {
      const tick = ticks[tickIndex(t)];
      return { ...tick, end: tick.start + 1, query: tick.v, demoObj: pool[tick.demo], tick };
    }

    return {
      id, cfg, length: cfg.rollout.length, ticks, tickIndex, pool, ref: pool[cfg.reference], chunks, retrieve,
      demoActions: (c) => c.tick.ctx || [0, 1].map(() => [0, 1, 2].map(() => Array(H).fill(0))),
      predictedActions: (t, c) => c.tick.pred,
    };
  }

  const cache = new Map();
  window.SIM = {
    H,
    // Tasks with real rollouts, in site order.
    ids: window.VICTR_RESULTS.tasks.map((t) => t.id).filter((id) => DATA[id]),
    task(id) {
      if (!cache.has(id)) cache.set(id, buildTask(id));
      return cache.get(id);
    },
  };
})();
