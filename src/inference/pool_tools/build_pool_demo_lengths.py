"""Per-demonstration metadata for each retrieval pool, as one sidecar per task pool:
<pool_dir>/<task_slug>.demos.json. openpi.policies.yor_retrieval reads it when a run keeps only the
N shortest demonstrations in the buffer (RetrievalContextInputs.pool_demos, server --pool-demos,
experiments.py EXP E).

For every episode in the pool: its length, keep and success flags and uid, from the LeRobot
meta/episodes.jsonl of the dataset the pool was built from. Pool chunks store no episode length
(the last stride-5 chunk ends up to 5 frames before the episode does, so chunk ends do not rank
episodes correctly: 195/196/197/198-frame demos all end their last chunk at frame 195).

The pool's episodes and chunk starts come from its <task>.canonical.npz or <task>.vfe.npz sidecar
(both keyed per pool chunk, so no 1-32 GB pool pickle is loaded). The build refuses to write a task
unless every pool episode is in the dataset with the pool's task and its last chunk ends within
the final 5 frames of that episode's length. The loader repeats the chunk-end check on the real pool.

Run with any Python that has numpy:
  lerobot_env/bin/python build_pool_demo_lengths.py            # both default pool directories
  lerobot_env/bin/python build_pool_demo_lengths.py --pool-dir assets/victr_icl_pool_demo --dataset assets/icl-demo-dataset
"""

import argparse
import json
import pathlib

import numpy as np

CHUNK_SIZE = 10
STRIDE = CHUNK_SIZE // 2  # viktr.retrieval.chunk_dictionary.chunk_starts

# (pool directory, the LeRobot v2.1 dataset its episodes came from). The demo pools were built from
# adityx23/icl-demo-dataset (assets/icl-demo-dataset is the copy yor_vfe.DEFAULT_DATASET reads); the
# in-domain pools from icl-dataset (build_indomain_pool.py's VIDEO_DATASET, assets/icl-dataset-full).
DEFAULT_SOURCES = (
    ("assets/victr_icl_pool_demo", "assets/icl-demo-dataset"),
    ("assets/victr_icl_pool_indomain", "assets/icl-dataset-full"),
)


def pool_chunk_starts(pool_dir: pathlib.Path, slug: str):
    """(task, {episode_index: [start_frame, ...]}) from the task's per-chunk sidecar."""
    for suffix in (".canonical.npz", ".vfe.npz"):
        path = pool_dir / f"{slug}{suffix}"
        if path.is_file():
            with np.load(path) as z:
                starts = {}
                for e, s in zip(z["episode_index"].tolist(), z["start_frame"].tolist(), strict=True):
                    starts.setdefault(int(e), []).append(int(s))
                return str(z["task"]), starts, path.name
    raise FileNotFoundError(f"{pool_dir / slug}.pkl has no .canonical.npz or .vfe.npz sidecar to read its chunks from")


def build(pool_dir: pathlib.Path, dataset: pathlib.Path, tasks=None):
    episodes = {}
    for line in (dataset / "meta/episodes.jsonl").read_text().splitlines():
        ep = json.loads(line)
        episodes[int(ep["episode_index"])] = ep
    for pkl in sorted(pool_dir.glob("*.pkl")):
        slug = pkl.stem
        task, starts, source = pool_chunk_starts(pool_dir, slug)
        if tasks and task not in tasks and slug not in tasks:
            continue
        rows = []
        for e in sorted(starts):
            ep = episodes.get(e)
            if ep is None:
                raise ValueError(f"{slug}: pool episode {e} is not in {dataset}")
            if list(ep.get("tasks") or []) != [task]:
                raise ValueError(f"{slug}: pool episode {e} is {ep.get('tasks')!r} in {dataset}, not {task!r}")
            end = max(starts[e]) + CHUNK_SIZE
            length = int(ep["length"])
            if not (length - STRIDE < end <= length):
                raise ValueError(f"{slug}: episode {e} has {length} frames in {dataset} but its last pool chunk "
                                 f"ends at frame {end}; this dataset does not describe the pool")
            rows.append({"episode_index": e, "length": length, "keep": bool(ep.get("keep", True)),
                         "success": ep.get("success"), "episode_uid": ep.get("episode_uid")})
        out = pool_dir / f"{slug}.demos.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"task": task, "dataset": str(dataset), "chunks_from": source,
                                   "episodes": rows}, indent=1) + "\n")
        tmp.replace(out)
        shortest = sorted((r["length"], r["episode_index"]) for r in rows if r["keep"])
        print(f"{out}: {len(rows)} episodes ({sum(not r['keep'] for r in rows)} keep=false); "
              f"shortest keep=true {[e for _, e in shortest[:5]]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", help="one pool directory (default: both default pool directories)")
    parser.add_argument("--dataset", help="LeRobot v2.1 dataset root the --pool-dir pools were built from")
    parser.add_argument("--tasks", nargs="*", help="only these tasks or task slugs")
    args = parser.parse_args()
    if bool(args.pool_dir) != bool(args.dataset):
        parser.error("--pool-dir and --dataset go together")
    sources = [(args.pool_dir, args.dataset)] if args.pool_dir else DEFAULT_SOURCES
    for pool_dir, dataset in sources:
        build(pathlib.Path(pool_dir), pathlib.Path(dataset), set(args.tasks or ()))


if __name__ == "__main__":
    main()
