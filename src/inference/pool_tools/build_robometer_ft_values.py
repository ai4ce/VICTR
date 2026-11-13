"""Finetuned Robometer-4B progress values for the retrieval pools, one sidecar per pool:
<pool_dir>/<task_slug>.robometer_ft.npz, keyed (episode_index, start_frame) like the VFE sidecars. The
finetuned counterpart of the zero-shot Robometer key_values stored inside each pool, so value retrieval
with value_estimator="robometer_ft" compares a finetuned query with finetuned keys.

Inputs are exactly those the zero-shot key_values were computed from:
  - demo pools (build_victr_pool.py --annotate-value): head frames loaded from the v2.1 demo dataset
    and letterboxed to 224 by build_victr_pool._load_one_episode;
  - in-domain pools (build_indomain_pool.py --stage values): head frames decoded from the icl-dataset
    videos and letterboxed with cv2 (build_indomain_pool._letterbox_cv2).
Value at chunk start t = model progress for the K=4 window linspace(0, t, 4), batched exactly like
viktr RobometerValueEstimator.estimate_episode. Chunk keys come from the pool's .vfe.npz sidecar (one
entry per pool chunk). --verify-zero-shot scores with lerobot/Robometer-4B instead and compares with the
stored zero-shot values, which proves the inputs; it writes nothing.

Run in lerobot_env (resumable: pools with a sidecar are skipped unless --overwrite):
  lerobot_env/bin/python build_robometer_ft_values.py [--tasks ...] [--verify-zero-shot]
"""
import argparse, importlib.util, json, pathlib, pickle, sys, time
from types import SimpleNamespace
import numpy as np

W = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathlib.Path.home() / "viktr/src"))
sys.path.insert(0, str(W / "openpi/src/openpi/policies"))
import yor_value  # noqa: E402  (plain module: lerobot_env has no openpi package)

POOLS = {"demo": W / "assets/victr_icl_pool_demo", "indomain": W / "assets/victr_icl_pool_indomain"}
DEMO_ROOT = W / "datasets/icl-demo-dataset"
SUITE_FIRST = [  # the evaluation suite's pools are built before everything else
    "clean_the_plate", "hit_the_yellow_cube_with_the_mallet_using_the_left_arm",
    "pass_the_red_chilli_from_the_left_arm_to_the_right_arm", "pick_up_the_eggplant_and_place_it_on_the_plate_with_the_left_arm",
    "place_the_red_chilli_in_the_cup", "put_the_circle_on_the_peg", "put_the_eggplant_into_the_box_with_the_left_arm",
    "uncap_the_blue_marker", "stack_the_three_cubes_into_a_tower", "put_the_square_on_the_peg",
    "stack_the_cubes_and_hit_them_with_the_mallet", "hit_the_eggplant_with_the_mallet", "sort_the_items_into_their_containers",
    "pass_the_mustard_to_the_plate", "uncap_the_black_marker", "put_the_mustard_into_the_box",
]

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]; spec.loader.exec_module(mod); sys.argv = argv
    return mod

def demo_head_frames_av(bvp, dataset, ep_idx, expected):
    """Fallback for episodes whose lerobot/torchcodec read hits "Requested next frame while there are no
    more frames left to decode". The demo dataset is LeRobot v3: one video file holds many episodes, each
    at [from_timestamp, to_timestamp) inside it. Seek there with PyAV, keep frames from the episode's own
    span, and letterbox each with build_victr_pool._resize_with_pad (the pool's own math). If the stream
    ends early, the last decoded frame is repeated (a timestamp read clamps the same way). Only used after
    a parity check against the loader has passed on this dataset (see main)."""
    import av
    m = dataset.meta.episodes[ep_idx]
    key = bvp.PRIMARY_CAMERA
    path = pathlib.Path(dataset.root) / dataset.meta.get_video_file_path(ep_idx, key)
    t0 = float(m[f"videos/{key}/from_timestamp"])
    half = 0.5 / float(dataset.meta.fps)
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(t0 - 1.0, 0.0) / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        for f in container.decode(stream):
            if f.time is None or f.time < t0 - half:
                continue
            frames.append(bvp._resize_with_pad(f.to_ndarray(format="rgb24")))
            if len(frames) >= expected:
                break
    if not frames:
        raise RuntimeError(f"{path}: no frames decoded at {t0:.3f}s for episode {ep_idx}")
    short = expected - len(frames)
    frames.extend([frames[-1]] * max(0, short))
    return np.stack(frames[:expected]), short


def require_parity(bvp, dataset, episodes, tolerance_mean=1.0):
    """Prove the PyAV fallback reproduces the loader's frames on episodes that decode both ways, before
    any fallback frames are scored. Raises if they disagree, so no sidecar is ever built from wrong frames."""
    for e in episodes:
        m = dataset.meta.episodes[e]
        try:
            ep = bvp._load_one_episode(dataset, e, m["dataset_from_index"], m["dataset_to_index"], [bvp.PRIMARY_CAMERA])
        except RuntimeError:
            continue
        ref = np.asarray(ep["images"][bvp.PRIMARY_CAMERA])
        alt, short = demo_head_frames_av(bvp, dataset, e, len(ref))
        diff = np.abs(alt.astype(np.int16) - ref.astype(np.int16))
        print(f"PARITY episode {e}: PyAV fallback vs loader, {len(ref)} frames, mean|diff| {diff.mean():.3f}, "
              f"max {diff.max()}, short {short}", flush=True)
        if diff.mean() > tolerance_mean or short:
            raise RuntimeError(f"PyAV fallback does not reproduce the loader on episode {e}; refusing to use it")
        return
    raise RuntimeError("no episode decoded through the loader; cannot verify the fallback")


def slug(task):
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:80]

def score_starts(est, frames, starts, task, batch_size=32):
    import torch
    from lerobot.lerobot_types import TransitionKey
    out = np.zeros(len(starts), np.float32)
    with torch.no_grad():
        for b in range(0, len(starts), batch_size):
            ts = starts[b : b + batch_size]
            windows = [np.linspace(0, t, 4).round().astype(np.int64) for t in ts]
            stack = torch.from_numpy(np.stack([frames[w] for w in windows]))
            transition = {TransitionKey.OBSERVATION: {est.image_key: stack}, TransitionKey.COMPLEMENTARY_DATA: {"task": task}}
            obs = est.encoder(transition)[TransitionKey.OBSERVATION]
            batch = {k: (v.to(est.device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
            out[b : b + len(ts)] = est.model.compute_reward(batch).float().cpu().numpy()
    return out

class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "viktr.retrieval.chunk_dictionary" and name in ("Chunk", "ChunkDictionary"):
            return SimpleNamespace
        return super().find_class(module, name)

def stored_zero_shot(pool_path):
    """{(episode, start): key_value} from the pool pickle (only used by --verify-zero-shot)."""
    with open(pool_path, "rb") as f:
        raw = _Unpickler(f).load()
    fields = raw if isinstance(raw, dict) else vars(raw)
    chunks = [c if not isinstance(c, dict) else SimpleNamespace(**c) for c in fields["chunks"]]
    return {(int(c.episode_index), int(c.start_frame)): float(v) for c, v in zip(chunks, fields["key_values"])}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", help="pool slugs (default: every pool, evaluation suite first)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-zero-shot", action="store_true")
    args = ap.parse_args()

    model = "robometer" if args.verify_zero_shot else "robometer_ft"
    jobs = []
    for kind, d in POOLS.items():
        for p in sorted(d.glob("*.pkl")):
            if args.tasks and p.stem not in args.tasks:
                continue
            jobs.append((kind, p))
    jobs.sort(key=lambda j: (SUITE_FIRST.index(j[1].stem) if j[1].stem in SUITE_FIRST else len(SUITE_FIRST), j[0], j[1].stem))

    from viktr.value.robometer import RobometerValueEstimator
    t_load = time.time()
    est = RobometerValueEstimator(pretrained_path=yor_value.ROBOMETER_MODELS[model], image_key="observation.images.zed", device="cuda")
    print(f"{model}: loaded in {time.time()-t_load:.0f}s; {len(jobs)} pools", flush=True)

    bvp = bip = demo_ds = demo_tasks = indomain_tasks = None
    parity_done = False
    for kind, pool_path in jobs:
        out = pool_path.with_suffix(".robometer_ft.npz")
        if out.exists() and not args.overwrite and not args.verify_zero_shot:
            print(f"skip (exists) {out.name}", flush=True); continue
        with np.load(pool_path.with_suffix(".vfe.npz")) as z:
            keys = sorted(zip(z["episode_index"].astype(int).tolist(), z["start_frame"].astype(int).tolist()))
        by_ep = {}
        for e, s in keys:
            by_ep.setdefault(e, []).append(s)
        t0 = time.time(); vals = {}
        if kind == "demo":
            if bvp is None:
                bvp = load_module("bvp", W / "build_victr_pool.py")
                demo_ds = bvp._load_dataset(DEMO_ROOT)
                demo_tasks = {slug(t): t for t in bvp._tasks_to_episodes(demo_ds)}
            task = demo_tasks[pool_path.stem]
            for e, starts in by_ep.items():
                m = demo_ds.meta.episodes[e]
                n_ep = m["dataset_to_index"] - m["dataset_from_index"]
                try:
                    ep = bvp._load_one_episode(demo_ds, e, m["dataset_from_index"], m["dataset_to_index"], [bvp.PRIMARY_CAMERA])
                    frames = np.asarray(ep["images"][bvp.PRIMARY_CAMERA])
                except RuntimeError as exc:
                    if not parity_done:
                        require_parity(bvp, demo_ds, [x for x in by_ep if x != e])
                        parity_done = True
                    frames, short = demo_head_frames_av(bvp, demo_ds, e, n_ep)
                    print(f"FALLBACK episode {e} ({pool_path.stem}): loader failed ({exc}); PyAV decoded {n_ep - short}/{n_ep} frames"
                          + (f", last frame repeated {short}x" if short else ""), flush=True)
                for s, v in zip(starts, score_starts(est, frames, starts, task)):
                    vals[(e, s)] = float(v)
        else:
            if bip is None:
                bip = load_module("bip", W / "build_indomain_pool.py")
                indomain_tasks = {slug(t): t for t in bip.training_splits()}
                meta = bip.episode_meta()
            task = indomain_tasks[pool_path.stem]
            for e, starts in by_ep.items():
                frames = bip._decode(bip._video(e, bip.CAMERAS[0]), meta[e]["length"])
                frames = np.stack([bip._letterbox_cv2(f) for f in frames[: max(starts) + 1]])
                for s, v in zip(starts, score_starts(est, frames, starts, task)):
                    vals[(e, s)] = float(v)
        n = len(keys); dt = time.time() - t0
        ep_idx = np.array([k[0] for k in keys], np.int64); start = np.array([k[1] for k in keys], np.int64)
        value = np.array([vals[k] for k in keys], np.float32)
        if args.verify_zero_shot:
            if kind == "demo":
                ref = stored_zero_shot(pool_path)
            else:
                ref = {}
                for e, starts in by_ep.items():
                    cached = np.load(W / f"assets/victr_icl_pool_indomain/build/robometer_episode_{e:06d}.npy")
                    ref.update({(e, s): float(cached[s]) for s in starts})
            diff = np.array([abs(vals[k] - ref[k]) for k in keys])
            print(f"VERIFY {kind} {pool_path.stem}: {n} chunks, max|recomputed - stored zero-shot| = {diff.max():.2e}, "
                  f"mean {diff.mean():.2e}; {dt:.0f}s ({dt/n*1000:.0f} ms/chunk)", flush=True)
            continue
        if not (np.isfinite(value).all() and (value >= 0).all() and (value <= 1).all()):
            raise ValueError(f"{pool_path.stem}: values outside [0, 1]")
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, episode_index=ep_idx, start_frame=start, value=value, model=np.array(yor_value.ROBOMETER_MODELS[model]))
        tmp.replace(out)
        print(f"wrote {kind} {out.name}: {n} chunks in {dt:.0f}s ({dt/n*1000:.0f} ms/chunk)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
