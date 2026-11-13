"""Retrieval pools for in-domain (icl-dataset) evaluation tasks, with the SAME pool episodes the
VICTR training pool used, written to assets/victr_icl_pool_indomain/ in the demo pools' format.

Episode selection: the training pool (victr_icl_pool_expanded_224 -> _canonical_224 ->
_canonical_3cam_224, the ctx3cam checkpoints' pool) holds each task's "pool" episodes from
viktr.data.icl_dataset.build_splits(expanded_tasks(), seed=42), default 85/10/5 fractions
(viktr scripts/build_retrieval_pool.py). That split is reproduced here from v2.1 metadata with the
same algorithm and inputs, and checked before anything is built against four numbers recorded
from the real training artifacts:
  - 1,784 episodes in the 31 expanded tasks and 1,186 in the 20 canonical tasks (viktr
    notes/ICRA_plan.md Sec 1b);
  - 10,336 chunks in the 20 canonical task pools, 9,856 of them with a full 30-step window
    (notes/context_encoding_redesign.md Sec 4, exhaustive scan of the training pools).
The random stream is shared across tasks in sorted-task order, so a different episode set, task
list, excluded set or fraction for any task changes these totals.
  - Excluded episodes: viktr's icl_dataset_v2_excluded_episodes.json (the Hub keep=False flags,
    identical to this dataset's `keep`).
  - Task strings: the split was made with the original on-disk labels. The Hub later relabelled
    the two "orange cube ... on a cluttered table" tasks from "on the plate" to "in the box"
    (viktr EXPANDED_PROMPT_OVERRIDES notes); that changes the sorted task order, so those two
    labels are mapped back before splitting.

Pool contents, per training (viktr build_retrieval_pool.py, resize_pool_images.py,
build_pool_canonical_3cam.py):
  - 10-frame chunks at stride 5 over each pool episode (chunk_dictionary.chunk_starts);
  - DINOv2 ViT-B/14 CLS keys of each chunk's first frame at NATIVE 480x640 resolution
    (embeddings.embed_frames: bilinear stretch to 224, ImageNet normalization). Head keys are
    training's; wrist keys are computed the same way. Queries must therefore use raw
    (not letterboxed) preprocessing; the pool records key_image_mode="raw";
  - stored frames for all three cameras, openpi resize_with_pad_torch(224, 224);
  - proprio = 15-D observation.state, actions = the 20-D action columns (as the demo pools);
  - canonical context fields come from build_canonical_context.py's sidecar.
One deliberate difference: training's key_values were RoboDopamine progress
(meta/value_estimates on the training cluster's dataset copy), which is not published. key_values
here are zero-shot Robometer-4B progress, computed exactly like the demo pools'
(estimate_episode over head frames letterboxed to 224), so value retrieval compares a Robometer
query with Robometer keys. VFE values come from build_vfe_values.py.

Sources: Hannibal52Barca/icl-dataset-end-effector-space (v2.1 parquet: joint state, actions, FK
end-effector poses; the dataset training's canonical pool used as icl-dataset-fixed-obs) and
assets/icl-dataset-full (the v2.1 icl-dataset copy with videos, identical metadata and columns).

Stages (each resumable per task):
  lerobot_env/bin/python build_indomain_pool.py --stage values   # Robometer, needs lerobot_env
  openpi/.venv/bin/python build_indomain_pool.py --stage pool    # decode, DINOv2, write .pkl
then the canonical and VFE sidecars (see RETRIEVAL.md).
"""

import argparse
import ast
import json
import pathlib
import pickle
import random
import sys
import time

import numpy as np

WORKSPACE = pathlib.Path(__file__).resolve().parent
VIKTR = pathlib.Path("/home/lairlab/viktr")
EE_DATASET = WORKSPACE / "datasets/icl-dataset-end-effector-space"
VIDEO_DATASET = WORKSPACE / "assets/icl-dataset-full"
OUT_DIR = WORKSPACE / "assets/victr_icl_pool_indomain"
CAMERAS = ("observation.images.zed", "observation.images.fish0", "observation.images.fish1")
CHUNK_SIZE = 10
STRIDE = CHUNK_SIZE // 2
ACTION_COLUMNS = ("action.left_ee", "action.right_ee", "action.gripper", "action.base_vel", "action.lift_cmd")
SEED, TRAIN_FRAC, POOL_FRAC, TEST_FRAC = 42, 0.85, 0.10, 0.05

DEFAULT_TASKS = (
    "clean the plate",
    "hit the yellow cube with the mallet using the right arm",
    "pass the red chilli from the left arm to the right arm",
    "pick up the eggplant and place it on the plate with the left arm",
    "place the red chilli in the cup",
    "uncap the blue marker",
)
# notes/ICRA_plan.md Sec 1b: the 20 tasks of the canonical training set.
CANONICAL_TASKS = (
    "clean the plate", "hit the yellow cube with the mallet using the right arm",
    "pass the red chilli from the left arm to the right arm", "pass the salt shaker from the left arm to the right arm",
    "pass the yellow cube from the left arm to the right arm",
    "pick up the eggplant and place it on the plate with the left arm",
    "pick up the eggplant and place it on the plate with the right arm",
    "pick up the orange cube and place it on the plate with the left arm on a cluttered table",
    "pick up the orange cube and place it on the plate with the right arm on a cluttered table",
    "pick up the pumpkin and place it on the plate with the left arm", "pick up the pumpkin and place it on the plate with the right arm",
    "pick up the red chilli and place it on the plate with the left arm",
    "pick up the red chilli and place it on the plate with the right arm",
    "place the red chilli in the cup", "place the red chilli in the cup with the left arm", "put the circle on the peg",
    "put the eggplant into the box with the left arm", "put the eggplant into the box with the right arm",
    "uncap the blue marker", "uncap the red marker",
)
RECORDED = {"expanded_episodes": 1784, "canonical_episodes": 1186, "canonical_pool_chunks": 10336, "canonical_pool_windows30": 9856}
# Hub relabel -> the label the split was made with.
ORIGINAL_LABELS = {
    f"pick up the orange cube and place it in the box with the {arm} arm on a cluttered table":
        f"pick up the orange cube and place it on the plate with the {arm} arm on a cluttered table"
    for arm in ("left", "right")
}


def task_slug(task: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:80]


def chunk_starts(num_frames: int) -> list[int]:
    return list(range(0, num_frames - CHUNK_SIZE + 1, STRIDE)) if num_frames >= CHUNK_SIZE else []


def _viktr_expanded_excluded_tasks() -> set[str]:
    """EXPANDED_TASKS_EXCLUDED read from viktr's source, so this cannot drift from it."""
    tree = ast.parse((VIKTR / "src/viktr/data/icl_dataset.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EXPANDED_TASKS_EXCLUDED" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("EXPANDED_TASKS_EXCLUDED not found in viktr")


def episode_meta(root=EE_DATASET) -> dict[int, dict]:
    return {e["episode_index"]: e for e in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())}


def training_splits() -> dict[str, dict[str, list[int]]]:
    """viktr build_splits(expanded_tasks(), seed=42) on the original labels, verified."""
    meta = episode_meta()
    excluded = set(json.loads((VIKTR / "src/viktr/data/icl_dataset_v2_excluded_episodes.json").read_text())["excluded_episodes"])
    if excluded != {i for i, e in meta.items() if not e["keep"]}:
        raise RuntimeError("viktr's excluded episodes differ from this dataset's keep=False flags")
    label = {i: ORIGINAL_LABELS.get(e["tasks"][0], e["tasks"][0]) for i, e in meta.items()}
    all_tasks = sorted({label[i] for i in meta if i not in excluded})  # icl_dataset.all_tasks
    tasks = [t for t in all_tasks if t not in _viktr_expanded_excluded_tasks()]  # expanded_tasks
    rng = random.Random(SEED)
    splits = {}
    for task in tasks:  # icl_dataset.build_splits, verbatim
        eps = sorted(i for i in meta if label[i] == task and i not in excluded)
        rng.shuffle(eps)
        n = len(eps)
        n_pool = max(1, round(n * POOL_FRAC))
        n_test = max(1, round(n * TEST_FRAC))
        n_train = n - n_pool - n_test
        splits[task] = {"train": sorted(eps[:n_train]), "pool": sorted(eps[n_train:n_train + n_pool]),
                        "test": sorted(eps[n_train + n_pool:])}
    length = {i: e["length"] for i, e in meta.items()}
    got = {
        "expanded_episodes": sum(sum(map(len, s.values())) for s in splits.values()),
        "canonical_episodes": sum(sum(map(len, splits[t].values())) for t in CANONICAL_TASKS),
        "canonical_pool_chunks": sum(len(chunk_starts(length[e])) for t in CANONICAL_TASKS for e in splits[t]["pool"]),
        "canonical_pool_windows30": sum(1 for t in CANONICAL_TASKS for e in splits[t]["pool"]
                                        for s in chunk_starts(length[e]) if s + 30 <= length[e]),
    }
    if len(tasks) != 31 or got != RECORDED:
        raise RuntimeError(f"Reproduced split does not match the training pool's recorded totals: {got} vs {RECORDED}")
    return splits


def _decode(path: pathlib.Path, expected: int) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    if len(frames) != expected:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames, metadata says {expected}")
    return np.stack(frames)


def _video(episode: int, camera: str) -> pathlib.Path:
    return VIDEO_DATASET / f"videos/chunk-{episode // 100:03d}/{camera}/episode_{episode:06d}.mp4"


def _parquet(episode: int) -> pathlib.Path:
    return EE_DATASET / f"data/chunk-{episode // 100:03d}/episode_{episode:06d}.parquet"


def stage_values(tasks, splits, build_dir):
    """Robometer-4B per-frame progress for each pool episode, as build_victr_pool.py annotated the
    demo pools (head frames letterboxed to 224, estimate_episode)."""
    sys.path.insert(0, str(VIKTR / "src"))
    from viktr.value.robometer import RobometerValueEstimator

    meta = episode_meta()
    estimator = None
    for task in tasks:
        for ep in splits[task]["pool"]:
            out = build_dir / f"robometer_episode_{ep:06d}.npy"
            if out.exists():
                continue
            if estimator is None:
                estimator = RobometerValueEstimator(image_key=CAMERAS[0], device="cuda")
            t0 = time.monotonic()
            frames = _decode(_video(ep, CAMERAS[0]), meta[ep]["length"])
            letterboxed = np.stack([_letterbox_cv2(f) for f in frames])
            del frames
            values = estimator.estimate_episode(letterboxed, task)
            np.save(out, values.astype(np.float32))
            print(f"  robometer {task!r} episode {ep}: {len(values)} frames in {time.monotonic() - t0:.0f}s", flush=True)


def _letterbox_cv2(img, size=224):
    """build_victr_pool._resize_with_pad (the demo pools' Robometer input), verbatim math."""
    import cv2

    h, w = img.shape[:2]
    ratio = max(w / size, h / size)
    rh, rw = int(h / ratio), int(w / ratio)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_h0, rem_h = divmod(size - rh, 2)
    pad_w0, rem_w = divmod(size - rw, 2)
    return cv2.copyMakeBorder(resized, pad_h0, pad_h0 + rem_h, pad_w0, pad_w0 + rem_w, cv2.BORDER_CONSTANT, value=0)


def _episode_cache(ep, n, starts, build_dir, embed, resize_with_pad_torch, torch):
    """Per-episode stored frames (224, all cameras) and first-frame keys, cached in build_dir so an
    interrupted build resumes per episode and the pickle is assembled from memory maps."""
    done = build_dir / f"episode_{ep:06d}.keys.npz"
    if not done.exists():
        keys = {}
        for cam in CAMERAS:
            native = _decode(_video(ep, cam), n)
            keys[cam] = embed(np.ascontiguousarray(native[starts]))
            np.save(build_dir / f"episode_{ep:06d}.{cam}.npy", resize_with_pad_torch(torch.from_numpy(native), 224, 224).numpy())
            del native
        np.savez(build_dir / "tmp_keys.npz", **keys)
        (build_dir / "tmp_keys.npz").rename(done)
    with np.load(done) as z:
        keys = {cam: z[cam] for cam in CAMERAS}
    frames = {cam: np.load(build_dir / f"episode_{ep:06d}.{cam}.npy", mmap_mode="r") for cam in CAMERAS}
    if any(len(k) != len(starts) for k in keys.values()) or any(f.shape != (n, 224, 224, 3) for f in frames.values()):
        raise RuntimeError(f"episode {ep}: cached frames/keys do not match the episode")
    return frames, keys


def stage_pool(tasks, splits, build_dir, out_dir):
    import pandas as pd
    import torch

    sys.path.insert(0, str(WORKSPACE / "openpi/src"))
    from openpi.policies import yor_retrieval
    from openpi.shared.image_tools import resize_with_pad_torch

    model = None
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def embed(frames):  # viktr embed_frames == yor_retrieval.embed_frame, batched
        nonlocal model
        model = model or yor_retrieval.load_dinov2()
        out = []
        with torch.inference_mode():
            for i in range(0, len(frames), 32):
                x = torch.from_numpy(frames[i:i + 32]).permute(0, 3, 1, 2).float() / 255
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                out.append(model.forward_features((x - mean) / std)["x_norm_clstoken"].float().numpy())
        return np.concatenate(out)

    meta = episode_meta()
    for task in tasks:
        out_path = out_dir / f"{task_slug(task)}.pkl"
        if out_path.exists():
            print(f"skip (exists): {out_path.name}", flush=True)
            continue
        t0 = time.monotonic()
        chunks, keys = [], {cam: [] for cam in CAMERAS}
        for ep in splits[task]["pool"]:
            n = meta[ep]["length"]
            if meta[ep]["tasks"][0] != task:
                raise RuntimeError(f"episode {ep} is labelled {meta[ep]['tasks'][0]!r}, not {task!r}")
            df = pd.read_parquet(_parquet(ep))
            if len(df) != n or not (df["episode_index"] == ep).all():
                raise RuntimeError(f"episode {ep}: parquet does not match metadata")
            proprio = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            actions = np.concatenate([np.stack(df[c].to_numpy()).reshape(n, -1) for c in ACTION_COLUMNS], axis=1).astype(np.float32)
            values = np.load(build_dir / f"robometer_episode_{ep:06d}.npy")
            if values.shape != (n,):
                raise RuntimeError(f"episode {ep}: Robometer values have shape {values.shape}")
            starts = chunk_starts(n)
            frames, ep_keys = _episode_cache(ep, n, starts, build_dir, embed, resize_with_pad_torch, torch)
            for cam in CAMERAS:
                keys[cam].append(ep_keys[cam])
            for s in starts:
                chunks.append({
                    "task": task, "episode_index": ep, "start_frame": s, "end_frame": s + CHUNK_SIZE,
                    "images": {cam: np.asarray(frames[cam][s:s + CHUNK_SIZE]) for cam in CAMERAS},
                    "proprio": proprio[s:s + CHUNK_SIZE], "actions": actions[s:s + CHUNK_SIZE],
                    "value": values[s:s + CHUNK_SIZE],
                })
            print(f"  {task!r} episode {ep}: {n} frames, {len(starts)} chunks ({time.monotonic() - t0:.0f}s)", flush=True)
        key_embeddings_by_camera = {cam: np.concatenate(v).astype(np.float32) for cam, v in keys.items()}
        pool = {
            "chunks": chunks,
            "key_embeddings": key_embeddings_by_camera[CAMERAS[0]],
            "key_embeddings_by_camera": key_embeddings_by_camera,
            "key_values": np.array([c["value"][0] for c in chunks], dtype=np.float32),
            "key_image_mode": "raw",
            "key_value_source": "robometer",
            "source": {
                "episodes": splits[task]["pool"],
                "split": "viktr build_splits(expanded_tasks(), seed=42, 85/10/5) pool split, verified against the training pool's recorded totals",
                "datasets": [str(EE_DATASET), str(VIDEO_DATASET)],
            },
        }
        tmp = out_path.with_name(out_path.name + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(pool, f, protocol=5)
        tmp.rename(out_path)
        print(f"{out_path.name}: {len(splits[task]['pool'])} episodes, {len(chunks)} chunks, "
              f"{out_path.stat().st_size / 1e9:.1f} GB in {time.monotonic() - t0:.0f}s", flush=True)
        del pool, chunks, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("split", "values", "pool"), required=True)
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()
    splits = training_splits()
    unknown = [t for t in args.tasks if t not in splits]
    if unknown:
        raise SystemExit(f"Not training tasks (expanded set): {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = args.out_dir / "build"
    build_dir.mkdir(exist_ok=True)
    (args.out_dir / "splits.json").write_text(json.dumps(
        {"note": "viktr build_splits(expanded_tasks(), seed=42, 85/10/5), reproduced and verified by build_indomain_pool.py",
         "recorded_totals": RECORDED, "splits": splits}, indent=1))
    for t in args.tasks:
        s = splits[t]
        print(f"{t!r}: pool {s['pool']} (train {len(s['train'])}, test {s['test']})", flush=True)
    if args.stage == "values":
        stage_values(args.tasks, splits, build_dir)
    elif args.stage == "pool":
        stage_pool(args.tasks, splits, build_dir, args.out_dir)


if __name__ == "__main__":
    main()
