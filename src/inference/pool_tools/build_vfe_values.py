"""VFE progress values for the demo retrieval pool, as one sidecar per task pool and model:
assets/victr_icl_pool_demo/<task_slug>.<model>.npz, for --model vfe (the IC-VFE) or vfe_recap (the
RECAP-style no-context VFE; yor_vfe.VFE_MODELS). The VFE alternatives to the Robometer key_values
stored inside the pools; openpi.policies.yor_retrieval uses them when value_estimator=<model>.

In-domain pools (assets/victr_icl_pool_indomain, --pool-dir) are scored the same way from the
icl-dataset videos, with contexts drawn only from the task's training-pool episodes.

One value per stride-5 chunk start (the pool's retrieval key frame) of every keep=true episode,
keyed by (episode_index, start_frame) like build_canonical_context.py, so it does not depend on
chunk order. Each frame is scored from the demo dataset's own videos and 15-D joint state with the
same context rule as live queries (yor_vfe.YorVfePredictor.context_episode): the first keep=true
episode of the task other than the frame's own. vfe_recap has no context (context_episode -1).

Run in lerobot_env (resumable: finished tasks are skipped):
  lerobot_env/bin/python build_vfe_values.py [--model vfe|vfe_recap] [--pool-dir assets/victr_icl_pool_demo] [--tasks ...]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "openpi/src/openpi/policies"))
import yor_vfe  # noqa: E402  (plain module import: lerobot_env has no openpi package)

CHUNK_SIZE = 10
STRIDE = CHUNK_SIZE // 2  # viktr.retrieval.chunk_dictionary.chunk_starts


def task_slug(task: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:80]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", default="assets/victr_icl_pool_demo")
    parser.add_argument("--model", choices=tuple(yor_vfe.VFE_MODELS), default="vfe")
    parser.add_argument("--checkpoint", default=None, help="default: the --model checkpoint in yor_vfe.VFE_MODELS")
    parser.add_argument("--tasks", nargs="*", help="only these tasks (default: every task with a pool)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pool_dir = pathlib.Path(args.pool_dir)
    predictor = yor_vfe.make_predictor(args.model, checkpoint=args.checkpoint)  # demo + in-domain sources (yor_vfe.default_sources)
    names = {tid: entries[0]["task_name"] for tid, entries in predictor.entries_by_task.items()}
    tasks = args.tasks or sorted(t for t in names.values() if (pool_dir / f"{task_slug(t)}.pkl").exists())
    pooled = {p.stem for p in pool_dir.glob("*.pkl")}
    unscored = sorted(pooled - {task_slug(t) for t in names.values()})
    if unscored:
        print(f"no VFE values possible (fewer than two keep=true demo episodes): {unscored}", flush=True)
    total_frames = sum(
        (e["demo_length"] - CHUNK_SIZE) // STRIDE + 1
        for t in tasks for e in predictor.entries_by_task[predictor._task_id(t)] if e["demo_length"] >= CHUNK_SIZE
    )
    done_frames, started = 0, time.monotonic()
    for task in tasks:
        out = pool_dir / f"{task_slug(task)}.{args.model}.npz"
        entries = predictor.entries_by_task[predictor._task_id(task)]
        n_task = sum((e["demo_length"] - CHUNK_SIZE) // STRIDE + 1 for e in entries if e["demo_length"] >= CHUNK_SIZE)
        if out.exists() and not args.overwrite:
            print(f"skip (exists): {out.name}", flush=True)
            total_frames -= n_task
            continue
        rows = []
        task_started = time.monotonic()
        for entry in entries:
            ep = entry["episode_index"]
            for start in range(0, entry["demo_length"] - CHUNK_SIZE + 1, STRIDE):
                pred = predictor.predict_dataset_frame(task, ep, start)
                context = pred["context_episode"]
                rows.append((ep, start, pred["value"], pred["raw_value"], -1 if context is None else context))
                done_frames += 1
            rate = done_frames / (time.monotonic() - started)
            print(f"  {task!r} episode {ep}: {len(rows)}/{n_task} | all tasks {done_frames}/{total_frames}, "
                  f"{rate:.1f} frames/s, ~{(total_frames - done_frames) / rate / 60:.0f} min left", flush=True)
        ep_idx, start_frame, value, raw, ctx = (np.array(c) for c in zip(*rows))
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez(
            tmp,
            task=np.array(task),
            episode_index=ep_idx.astype(np.int64),
            start_frame=start_frame.astype(np.int64),
            value=value.astype(np.float32),
            raw_value=raw.astype(np.float32),
            context_episode=ctx.astype(np.int64),
            model=np.array(args.model),
            checkpoint=np.array(str(predictor.checkpoint)),
            info=np.array(json.dumps(predictor.info)),
        )
        tmp.rename(out)
        print(f"{out.name}: {len(rows)} chunk starts in {time.monotonic() - task_started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
