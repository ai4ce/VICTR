"""Build a VICTR retrieval pool (one ChunkDictionary pickle per task) from
adityx23/icl-demo-dataset, following the exact logic in ~/viktr's real, working
implementation (src/viktr/retrieval/{embeddings,chunk_dictionary}.py and
src/viktr/data/icl_dataset.py's load_episode_arrays/task_slug) -- not a reimplementation
from prose, the actual algorithm, adapted only where the demo dataset's v2.1 metadata
layout (meta/episodes.jsonl) differs from icl-dataset's v3 layout (meta/episodes/chunk-*/
file-*.parquet) that load_episode_arrays was written against.

Output: one <task_slug>.pkl per task under --out-dir, matching the format
chunk_dictionary.save_pool()/load_pool() use (pickled ChunkDictionary: chunks + first-
frame DINOv2 CLS embeddings), so openpi.policies.yor_retrieval.RetrievalContextInputs
can load these with viktr's own load_pool()/load_pools() unchanged.

This pool's source (285 demo episodes, 27 tasks, recorded 3+6 Aug 2026) is NOT
byte-identical to whatever pool the actual checkpoint (yor_icl_fast_victr_vision_expanded,
job 16429151) was trained against (assets/victr_icl_pool_expanded_224/, 31 tasks, built
from the full icl-dataset's "pool" split per icl_dataset.py's build_splits) -- this is a
stand-in built from what we actually have access to, per the user's explicit direction
(2026-08-28: "this is the context pool we're using"), not a claim of exact equivalence.
"""

import argparse
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

# Reuse viktr's actual retrieval code directly rather than reimplementing it.
sys.path.insert(0, "/home/lairlab/viktr/src")
from viktr.retrieval.chunk_dictionary import (  # noqa: E402
    ChunkDictionary,
    load_pool,
    split_episode_into_chunks,
)
from viktr.retrieval.embeddings import embed_frames, load_dinov2  # noqa: E402

PRIMARY_CAMERA = "observation.images.zed"
# 2026-09-13: pools store every camera view per chunk (all 10 frames, letterboxed to 224),
# plus one first-frame DINOv2 key per camera in pool.key_embeddings_by_camera. key_embeddings
# stays the head-camera key, so readers that only know the original format are unaffected.
CAMERAS = (PRIMARY_CAMERA, "observation.images.fish0", "observation.images.fish1")
WRIST_CAMERAS = CAMERAS[1:]
CHUNK_SIZE = 10  # matches Pi0FastVictrConfig.context_chunk_size / VictrConfig.context_chunk_size
FRAME_SIZE = 224  # see _resize_with_pad's docstring for why frames are stored at this size


def _resize_with_pad(img: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """Numpy/cv2 port of openpi.shared.image_tools.resize_with_pad(img, size, size) --
    aspect-preserving resize + zero-pad (letterbox), not a plain distorting resize.
    Applied at frame-load time here as a memory fix: confirmed 2026-08-28 that 3 of 27
    pool-build tasks (the longest-duration ones in the dataset -- up to 19.6 min/episode)
    were OOM-killed even in their own isolated subprocess. Root cause: at native
    480x640x3 uint8 (~0.92MB/frame), a long episode's full frame array alone could reach
    tens of GB, and chunk_dictionary.split_episode_into_chunks then duplicates 10 frames
    per chunk across potentially thousands of overlapping chunks before any embedding
    happens -- both terms compound. Storing frames at this target resolution instead
    provided zero fidelity cost: DINOv2's own embed_frames always resizes its input to
    224x224 internally (embeddings.py's _to_dinov2_input), and
    openpi.policies.yor_retrieval.RetrievalContextInputs's context-image extraction does
    the identical resize_with_pad(224, 224) on these frames before feeding the model --
    so pre-resizing here is a no-op from the model's perspective, just ~6x less memory
    held per frame at every stage in between (float32/int intermediate `img` here is
    still uint8 in, uint8 out)."""
    cur_height, cur_width = img.shape[:2]
    ratio = max(cur_width / size, cur_height / size)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_h0, rem_h = divmod(size - resized_height, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(size - resized_width, 2)
    pad_w1 = pad_w0 + rem_w
    return cv2.copyMakeBorder(resized, pad_h0, pad_h1, pad_w0, pad_w1, cv2.BORDER_CONSTANT, value=0)


def task_slug(task: str) -> str:
    """Verbatim copy of viktr.data.icl_dataset.task_slug -- kept local (not imported)
    since icl_dataset.py's other top-level code assumes the v3/scratch icl-dataset
    layout this script doesn't use."""
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:80]


def _build_chunk_dictionary_streaming(
    dataset, episode_indices: list[int], task: str, chunk_size: int, primary_camera: str, embed_model,
    value_estimator=None,
) -> ChunkDictionary:
    """Same end result as viktr.retrieval.chunk_dictionary.build_chunk_dictionary, but
    processes one episode at a time (load -> split_episode_into_chunks -> discard the
    full-episode frame array) instead of materializing every episode's full frame array
    for the whole task before any chunking/embedding happens. Combined with
    _resize_with_pad (frames stored at 224x224, not native 480x640), this fixes the OOM
    that killed 3 of 27 pool-build tasks (confirmed 2026-08-28: all 3 were this
    dataset's longest-duration tasks, up to 19.6 min/episode -- the memory these two
    fixes target scales directly with total frame count, so long episodes were the
    exact failure mode). split_episode_into_chunks is viktr's own unmodified function;
    only the loop structure around it changed.

    value_estimator (2026-09-06, optional): a viktr.value.base.ValueEstimator (in
    practice viktr.value.robometer.RobometerValueEstimator) -- when given, each
    episode's per-frame value/progress is computed via estimate_episode() and threaded
    into split_episode_into_chunks' values= param, so the resulting pool gets a real
    key_values array (needed for viktr.retrieval.metrics.value_retrieve -- see
    yor_icl_victr_value_expanded's TrainConfig entry). None (default) reproduces the
    original vision-only pool exactly -- this is additive, not a behavior change for
    existing callers."""
    from lerobot.datasets.video_utils import FrameTimestampError

    all_chunks = []
    for ep_idx in episode_indices:
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        try:
            episode = _load_one_episode(dataset, ep_idx, from_idx, to_idx, list(CAMERAS))
        except FrameTimestampError as e:
            print(f"  SKIP episode {ep_idx} (frame timestamp desync): {e}", flush=True)
            continue
        except RuntimeError as e:
            # torchcodec's own boundary case ("Requested next frame while there are no
            # more frames left to decode") -- confirmed 2026-08-28 to fatally crash the
            # whole batch job if uncaught (killed a real run partway through task 3/27).
            # Same class of issue as FrameTimestampError (video/metadata length mismatch
            # for this one episode) but torchcodec raises plain RuntimeError for it
            # instead, so it needs its own catch -- skip the episode, keep going.
            print(f"  SKIP episode {ep_idx} (video decode error): {e}", flush=True)
            continue
        values = None
        if value_estimator is not None:
            values = value_estimator.estimate_episode(episode["images"][primary_camera], task)
        all_chunks.extend(
            split_episode_into_chunks(
                task=task,
                episode_index=ep_idx,
                images=episode["images"],
                proprio=episode["proprio"],
                actions=episode["actions"],
                chunk_size=chunk_size,
                values=values,
            )
        )
        del episode  # release this episode's full-resolution frame array before loading the next

    if not all_chunks:
        raise ValueError(f"no chunks produced for task={task!r}; check episode lengths vs chunk_size={chunk_size}")
    key_embeddings_by_camera = _embed_chunk_keys(all_chunks, CAMERAS, embed_model)
    key_values = None
    if all(c.value is not None for c in all_chunks):
        # First-frame value per chunk -- mirrors build_chunk_dictionary's own convention
        # (chunk_dictionary.py) exactly, so pools built either way are interchangeable.
        key_values = np.array([c.value[0] for c in all_chunks], dtype=np.float32)
    pool = ChunkDictionary(
        chunks=all_chunks, key_embeddings=key_embeddings_by_camera[primary_camera], key_values=key_values
    )
    pool.key_embeddings_by_camera = key_embeddings_by_camera
    return pool


def save_pool(pool: ChunkDictionary, path: pathlib.Path) -> None:
    """viktr's save_pool (write to .tmp, then atomic rename) with pickle protocol 5.

    Python < 3.14 defaults to protocol 4, which copies every array into a bytes object that
    the pickler's memo keeps alive until the dump ends: a second full copy of the pool in RAM.
    That OOM-killed the ~32 GB sort_the_items_into_their_containers pool. Protocol 5 writes
    array buffers directly (measured: 913 MB vs 328 MB peak RSS for a 0.6 GB list of chunks).
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(pool, f, protocol=5)
    os.replace(tmp_path, path)


def _embed_chunk_keys(chunks, cameras, embed_model) -> dict[str, np.ndarray]:
    """First-frame DINOv2 key per chunk, per camera -- the same frame and embedding call
    the head-camera key has always used."""
    return {
        cam: embed_frames(np.stack([c.images[cam][0] for c in chunks], axis=0), model=embed_model)
        for cam in cameras
    }


def _frame_uint8_224(frame) -> np.ndarray:
    arr = frame.permute(1, 2, 0).numpy() if hasattr(frame, "permute") else np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return _resize_with_pad(arr)  # see _resize_with_pad's docstring


def _action_vector(item) -> np.ndarray:
    # This lerobot version doesn't expose a combined "action" key -- concatenate the
    # sub-fields ourselves, in the same left_ee/right_ee/gripper/base_vel/lift_cmd
    # order used everywhere else this session (collect.py's original column order,
    # confirmed against the lerobot-format fine-tuned checkpoints' own config.json
    # action_feature_names earlier this session).
    return np.concatenate(
        [
            np.asarray(item["action.left_ee"], dtype=np.float32).reshape(-1),
            np.asarray(item["action.right_ee"], dtype=np.float32).reshape(-1),
            np.asarray(item["action.gripper"], dtype=np.float32).reshape(-1),
            np.asarray(item["action.base_vel"], dtype=np.float32).reshape(-1),
            np.asarray(item["action.lift_cmd"], dtype=np.float32).reshape(-1),
        ]
    )


def _load_one_episode(dataset, ep_idx: int, from_idx: int, to_idx: int, camera_keys: list[str]) -> dict:
    # Preallocated rather than list-then-stack: with three cameras, the stack's transient
    # second copy of the longest episodes (~20 min) would be tens of GB.
    n = to_idx - from_idx
    images = {cam: np.empty((n, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8) for cam in camera_keys}
    proprio, actions = [], []
    for j, i in enumerate(range(from_idx, to_idx)):
        item = dataset[i]
        for cam in camera_keys:
            images[cam][j] = _frame_uint8_224(item[cam])
        proprio.append(np.asarray(item["observation.state"], dtype=np.float32))
        actions.append(_action_vector(item))
    return {
        "episode_index": ep_idx,
        "images": images,
        "proprio": np.stack(proprio, axis=0),
        "actions": np.stack(actions, axis=0),
    }


def _add_wrist_cameras(root: pathlib.Path, pool_path: pathlib.Path) -> None:
    """Adds the wrist views (all frames of every chunk) and per-camera keys to an existing
    head-camera pool, instead of rebuilding it: the head frames, proprio, actions, Robometer
    values and head keys stay byte-for-byte unchanged, and the hours-long value annotation
    is not redone.

    Alignment is verified, not assumed: every re-decoded head frame, state and action that a
    chunk covers must equal what the pool already stores, or the task aborts without writing.
    Each chunk's wrist arrays are views into one per-episode array while building (overlapping
    chunks share memory); pickling writes them out as independent arrays, like the head view.
    """
    t0 = time.time()
    pool = load_pool(pool_path)
    by_camera = getattr(pool, "key_embeddings_by_camera", None) or {}
    if all(cam in pool.chunks[0].images for cam in WRIST_CAMERAS) and all(cam in by_camera for cam in CAMERAS):
        print(f"SKIP (already has wrist cameras): {pool_path.name}", flush=True)
        return
    old_bytes = pool_path.stat().st_size
    # Frame data dominates the file; each added camera stores as much as the head view.
    estimate = old_bytes * len(CAMERAS)
    free = shutil.disk_usage(pool_path.parent).free
    margin = 20 * 1024**3
    if free < estimate + margin:
        print(f"ABORT (disk): {pool_path.name} needs ~{estimate / 1e9:.1f} GB plus a 20 GB margin, "
              f"{free / 1e9:.1f} GB free", flush=True)
        sys.exit(3)

    dataset = _load_dataset(root)
    episodes: dict[int, list[int]] = {}
    for ci, c in enumerate(pool.chunks):
        episodes.setdefault(int(c.episode_index), []).append(ci)
    print(f"Adding wrist cameras to {pool_path.name}: {len(pool)} chunks, {len(episodes)} episodes", flush=True)

    frames_checked = 0
    for ep_idx, chunk_ids in sorted(episodes.items()):
        meta = dataset.meta.episodes[ep_idx]
        from_idx, to_idx = meta["dataset_from_index"], meta["dataset_to_index"]
        n = to_idx - from_idx
        # The first chunk covering each frame is its reference copy (chunks overlap).
        reference: dict[int, tuple[int, int]] = {}
        for ci in chunk_ids:
            c = pool.chunks[ci]
            if c.end_frame > n:
                raise RuntimeError(f"episode {ep_idx}: chunk ends at {c.end_frame} but the episode has {n} frames")
            for offset in range(c.end_frame - c.start_frame):
                reference.setdefault(c.start_frame + offset, (ci, offset))
        wrist = {cam: np.empty((n, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8) for cam in WRIST_CAMERAS}
        for j in range(n):
            item = dataset[from_idx + j]
            for cam in WRIST_CAMERAS:
                wrist[cam][j] = _frame_uint8_224(item[cam])
            if j in reference:
                ci, offset = reference[j]
                c = pool.chunks[ci]
                head = _frame_uint8_224(item[PRIMARY_CAMERA])
                if not np.array_equal(head, c.images[PRIMARY_CAMERA][offset]):
                    diff = np.abs(head.astype(np.int16) - c.images[PRIMARY_CAMERA][offset].astype(np.int16)).mean()
                    raise RuntimeError(f"episode {ep_idx} frame {j}: re-decoded head frame differs from the pool "
                                       f"(mean abs diff {diff:.3f}); refusing to attach wrist frames")
                if not (np.array_equal(np.asarray(item["observation.state"], dtype=np.float32), c.proprio[offset])
                        and np.array_equal(_action_vector(item), c.actions[offset])):
                    raise RuntimeError(f"episode {ep_idx} frame {j}: state/action differs from the pool")
                frames_checked += 1
        for ci in chunk_ids:
            c = pool.chunks[ci]
            for cam in WRIST_CAMERAS:
                c.images[cam] = wrist[cam][c.start_frame : c.end_frame]
        print(f"  episode {ep_idx}: {n} frames, {len(chunk_ids)} chunks", flush=True)

    embed_model = load_dinov2()
    new_keys = _embed_chunk_keys(pool.chunks, WRIST_CAMERAS, embed_model)
    # The stored head keys are kept; recomputing a sample shows they are the same embedding.
    sample = pool.chunks[:64]
    head_check = _embed_chunk_keys(sample, (PRIMARY_CAMERA,), embed_model)[PRIMARY_CAMERA]
    head_diff = float(np.abs(head_check - pool.key_embeddings[: len(sample)]).max())
    if head_diff > 1e-2:
        raise RuntimeError(f"recomputed head keys differ from the stored ones by {head_diff}; embedding changed")
    pool.key_embeddings_by_camera = {PRIMARY_CAMERA: pool.key_embeddings, **new_keys}
    save_pool(pool, pool_path)
    print(f"  -> {pool_path.name}: verified {frames_checked} head frames/states/actions bit-exact, head-key "
          f"recompute max diff {head_diff:.2e}, {old_bytes / 1e9:.1f} GB -> {pool_path.stat().st_size / 1e9:.1f} GB "
          f"in {time.time() - t0:.0f}s", flush=True)


def _load_dataset(root: pathlib.Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # tolerance_s: default 1e-4 rejected ~every frame here with a consistent ~0.033s
    # (~1 frame @ 30fps) gap between queried and loaded timestamps -- confirmed 2026-08-28
    # this is systematic (same ~33ms gap across dozens of episodes/tasks, not a handful of
    # genuinely corrupted ones), so it's a video-encoding/timestamp-convention quirk of
    # this specific dataset, not real desync -- unlike icl_dataset.py's own tolerance_s=2e-4
    # bump (which absorbs sub-millisecond float-rounding boundary noise on a handful of
    # episodes), this needs a tolerance comfortably above one whole frame period.
    return LeRobotDataset("icl-demo-dataset", root=root, tolerance_s=0.05)


def _tasks_to_episodes(dataset) -> dict[str, list[int]]:
    # Group kept episode indices by task string. dataset.meta.episodes is an arrow-backed
    # HF Dataset here (this lerobot version), not a dict -- iterate rows directly.
    task_to_episodes: dict[str, list[int]] = {}
    for ep_meta in dataset.meta.episodes:
        if not ep_meta.get("keep", False):
            continue
        tasks = ep_meta.get("tasks") or []
        if not tasks:
            continue
        task = tasks[0]
        task_to_episodes.setdefault(task, []).append(ep_meta["episode_index"])
    return task_to_episodes


def _build_one_task(root: pathlib.Path, out_dir: pathlib.Path, task: str, annotate_value: bool = False) -> None:
    """Builds and saves the pool for exactly one task, then returns (process exits
    right after in --single-task mode). Isolating this per-task in its own subprocess
    (see main()) is the actual point: confirmed 2026-08-28 the orchestrating process's
    RSS grew unboundedly across tasks (OOM-killed by the kernel partway through task
    22/27, ~50GB RSS at time of death, no Python-level exception -- the video
    decode/DINOv2-embedding working set for each task was never being fully released
    between iterations, whether from a real leak or just allocator fragmentation wasn't
    worth root-causing when a fresh-process-per-task sidesteps it entirely and is the
    more robust pattern for a batch job like this regardless).

    annotate_value (2026-09-06): also load a RobometerValueEstimator (lerobot/Robometer-4B)
    and attach per-frame value estimates, so the saved pool gets a real key_values array --
    needed for yor_icl_victr_value_expanded's value_retrieve path (see this project's
    history: RetrievalContextInputs previously always used vision_retrieve regardless of
    retrieval_metric because no pool had key_values at all)."""
    dataset = _load_dataset(root)
    task_to_episodes = _tasks_to_episodes(dataset)
    episode_indices = task_to_episodes.get(task)
    if not episode_indices:
        print(f"SKIP (task not found or no kept episodes): {task!r}", flush=True)
        return

    slug = task_slug(task)
    out_path = out_dir / f"{slug}.pkl"
    print(f"Building pool for {task!r} ({len(episode_indices)} episodes) -> {out_path.name}", flush=True)
    value_estimator = None
    if annotate_value:
        from viktr.value.robometer import RobometerValueEstimator
        value_estimator = RobometerValueEstimator(image_key=PRIMARY_CAMERA)
    try:
        pool = _build_chunk_dictionary_streaming(
            dataset,
            episode_indices,
            task=task,
            chunk_size=CHUNK_SIZE,
            primary_camera=PRIMARY_CAMERA,
            embed_model=load_dinov2(),
            value_estimator=value_estimator,
        )
    except ValueError as e:
        print(f"SKIP (no chunks: {e}): {task!r}", flush=True)
        return
    save_pool(pool, out_path)
    has_values = pool.key_values is not None
    print(f"  -> {len(pool)} chunks, key_embeddings shape {pool.key_embeddings.shape}, "
          f"key_values={'yes, shape ' + str(pool.key_values.shape) if has_values else 'no'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lairlab/icl_ws/datasets/icl-demo-dataset")
    parser.add_argument("--out-dir", default="/home/lairlab/icl_ws/assets/victr_icl_pool_demo")
    parser.add_argument("--min-episodes", type=int, default=1, help="skip tasks with fewer usable episodes")
    parser.add_argument(
        "--single-task", default=None, help="internal: build only this one task, then exit (see main())"
    )
    parser.add_argument("--annotate-value", action="store_true",
                         help="also compute per-frame value estimates via RobometerValueEstimator "
                              "(lerobot/Robometer-4B) and attach them as pool.key_values -- see "
                              "_build_one_task's docstring")
    parser.add_argument("--overwrite", action="store_true",
                         help="rebuild a task's pool file even if one already exists (needed to add "
                              "value annotation to an existing vision-only pool in place)")
    parser.add_argument("--add-wrist-cameras", action="store_true",
                         help="add fish0/fish1 frames and per-camera keys to the existing pools in --out-dir "
                              "in place, keeping head frames and value annotation (see _add_wrist_cameras)")
    parser.add_argument("--single-pool", default=None, help="internal: --add-wrist-cameras for one pool file")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.add_wrist_cameras:
        if args.single_pool is not None:
            _add_wrist_cameras(root, pathlib.Path(args.single_pool))
            return
        # Smallest first: problems surface in minutes, and the largest file is written last
        # when the most is known about the real size ratio.
        done, failed = 0, []
        for pool_path in sorted(out_dir.glob("*.pkl"), key=lambda p: p.stat().st_size):
            cmd = [sys.executable, __file__, "--root", str(root), "--out-dir", str(out_dir),
                   "--add-wrist-cameras", "--single-pool", str(pool_path)]
            result = subprocess.run(cmd, check=False)
            if result.returncode == 3:
                print("Stopping: not enough disk space for the next pool.", flush=True)
                failed.append(pool_path.name)
                break
            if result.returncode != 0:
                print(f"FAILED (exit {result.returncode}): {pool_path.name}", flush=True)
                failed.append(pool_path.name)
                continue
            done += 1
        print(f"DONE. ok={done} failed={failed}", flush=True)
        return

    if args.single_task is not None:
        _build_one_task(root, out_dir, args.single_task, annotate_value=args.annotate_value)
        return

    print("Loading dataset metadata...", flush=True)
    dataset = _load_dataset(root)
    task_to_episodes = _tasks_to_episodes(dataset)
    print(f"{len(task_to_episodes)} tasks with kept episodes", flush=True)
    del dataset  # only needed metadata here; the subprocess below reloads it fresh per task

    built, skipped = 0, 0
    for task, episode_indices in sorted(task_to_episodes.items()):
        if len(episode_indices) < args.min_episodes:
            print(f"SKIP (too few episodes, {len(episode_indices)}): {task!r}", flush=True)
            skipped += 1
            continue
        slug = task_slug(task)
        out_path = out_dir / f"{slug}.pkl"
        if out_path.exists() and not args.overwrite:
            print(f"SKIP (already built): {task!r} -> {out_path.name}", flush=True)
            continue
        # One fresh subprocess per task -- see _build_one_task's docstring for why.
        cmd = [
            sys.executable,
            __file__,
            "--root",
            str(root),
            "--out-dir",
            str(out_dir),
            "--single-task",
            task,
        ]
        if args.annotate_value:
            cmd.append("--annotate-value")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"SKIP (subprocess exit {result.returncode}, likely OOM or crash): {task!r}", flush=True)
            skipped += 1
            continue
        if out_path.exists():
            built += 1
        else:
            skipped += 1

    print(f"DONE. built={built} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
