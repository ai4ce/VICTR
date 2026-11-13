"""Canonical-space context fields for the demo retrieval pool, for the ctx3cam VICTR policies
(yor_icl_victr_{vision,value}_canonical_ctx3cam). Written as one small sidecar per task pool,
assets/victr_icl_pool_demo/<task_slug>.canonical.npz, instead of a second copy of the ~200 GB of
frames: openpi.policies.yor_retrieval attaches it to the pool's chunks on load.

Training (LucaMacesanu/openpi@5ea2659, viktr scripts/build_pool_canonical_3cam.py) reads two
per-chunk fields the demo pool does not have, both in the ICRA canonical space:
  - proprio: per arm [pos(3), rot6d(6)] = 18-D, from observation.left_ee/right_ee
    (yor_policy.YorInputsCanonical). Here from Hannibal52Barca/demo-end-effector-space, whose
    observation.*_ee is [r00, r01, r02, r10, r11, r12, x, y, z]: the first two rotation-matrix
    ROWS (the rot6d convention) plus position -- checked against yor_fk.fk of the joint state
    to 3e-8.
  - action_native30_truncated: the 30-step action window starting at the chunk's start frame,
    in canonical form (delta position from the window's first frame, rot6d, gripper), with DCT
    bins >= 20 of 30 zeroed. Chunks whose episode has fewer than 30 frames from the start were
    dropped from the training pool; here they are kept but flagged keep=False, and retrieval
    for these policies skips them.

Keyed by (episode_index, start_frame) for every stride-5 chunk start of every episode, so it
does not depend on the pool's chunk order; the loader checks each chunk's stored joint state
against `joints_start` bit-for-bit before attaching.

In-domain pools (build_indomain_pool.py) use Hannibal52Barca/icl-dataset-end-effector-space, the
dataset training's canonical pool read as icl-dataset-fixed-obs. Its observation.*_ee is
[qw, qx, qy, qz, x, y, z], converted exactly as the fork's YorInputsCanonical does
(position, quat_wxyz_to_rot6d); the layout is read from meta/info.json's feature names.

Run in the openpi venv (pandas, scipy):
  openpi/.venv/bin/python build_canonical_context.py [--pool-dir assets/victr_icl_pool_demo]
  openpi/.venv/bin/python build_canonical_context.py --dataset datasets/icl-dataset-end-effector-space \
      --pool-dir assets/victr_icl_pool_indomain
"""

import argparse
import glob
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.fft import dct, idct

ACTION_HORIZON = 30  # build_pool_canonical_3cam.py
KEEP_BINS = 20  # build_pool_canonical_3cam.py: notes/context_encoding_redesign.md Sec 7 cutoff
CHUNK_SIZE = 10
STRIDE = CHUNK_SIZE // 2  # viktr.retrieval.chunk_dictionary.chunk_starts
ACTION_COLUMNS = ("action.left_ee", "action.right_ee", "action.gripper", "action.base_vel", "action.lift_cmd")

_LEFT_QUAT, _LEFT_POS = slice(0, 4), slice(4, 7)
_RIGHT_QUAT, _RIGHT_POS = slice(7, 11), slice(11, 14)
_LEFT_GRIP, _RIGHT_GRIP = slice(14, 15), slice(15, 16)


def task_slug(task: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:80]


def quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """openpi yor_rotation.quat_wxyz_to_rot6d (fork 5ea2659): normalized quaternion -> first two
    rows of the rotation matrix, flattened."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
    ], axis=-1)


def canonical_action_window(action: np.ndarray) -> np.ndarray:
    """Verbatim math of build_pool_canonical_3cam.canonical_action_window."""
    action = action.astype(np.float64)
    left_delta_pos = action[:, _LEFT_POS] - action[0:1, _LEFT_POS]
    right_delta_pos = action[:, _RIGHT_POS] - action[0:1, _RIGHT_POS]
    left_rot6d = quat_wxyz_to_rot6d(action[:, _LEFT_QUAT])
    right_rot6d = quat_wxyz_to_rot6d(action[:, _RIGHT_QUAT])
    return np.concatenate(
        [left_delta_pos, left_rot6d, action[:, _LEFT_GRIP], right_delta_pos, right_rot6d, action[:, _RIGHT_GRIP]],
        axis=-1,
    ).astype(np.float32)


def truncate(canon30: np.ndarray) -> np.ndarray:
    """build_pool_canonical_3cam.truncate_and_tokenize's DCT step (the stored FAST tokens there
    are unused by training's context encoding, so they are not reproduced)."""
    coeff = dct(canon30[None], axis=1, norm="ortho")
    coeff[:, KEEP_BINS:, :] = 0
    return idct(coeff, axis=1, norm="ortho")[0].astype(np.float32)


EE_LAYOUTS = {
    ("r00", "r01", "r02", "r10", "r11", "r12", "x", "y", "z"): "rot6d_rows_xyz",  # demo-end-effector-space
    ("qw", "qx", "qy", "qz", "x", "y", "z"): "quat_wxyz_xyz",  # icl-dataset-end-effector-space
}


def canonical_state(left_ee: np.ndarray, right_ee: np.ndarray, layout: str = "rot6d_rows_xyz") -> np.ndarray:
    """observation.*_ee -> YorInputsCanonical state [pos(3), rot6d(6)] x 2."""
    if layout == "rot6d_rows_xyz":
        return np.concatenate([left_ee[:, 6:9], left_ee[:, 0:6], right_ee[:, 6:9], right_ee[:, 0:6]], axis=1).astype(np.float32)
    if layout == "quat_wxyz_xyz":  # YorInputsCanonical verbatim: pos, quat_wxyz_to_rot6d(quat)
        return np.concatenate([left_ee[:, 4:7], quat_wxyz_to_rot6d(left_ee[:, 0:4]),
                               right_ee[:, 4:7], quat_wxyz_to_rot6d(right_ee[:, 0:4])], axis=1).astype(np.float32)
    raise ValueError(f"Unknown end-effector layout {layout!r}")


def episode_fields(df: pd.DataFrame, layout: str = "rot6d_rows_xyz") -> dict:
    joints = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.concatenate([np.stack(df[c].to_numpy()).reshape(len(df), -1) for c in ACTION_COLUMNS], axis=1)
    state = canonical_state(np.stack(df["observation.left_ee"].to_numpy()), np.stack(df["observation.right_ee"].to_numpy()), layout)
    n = len(df)
    rows = []
    for s in range(0, n - CHUNK_SIZE + 1, STRIDE):
        keep = s + ACTION_HORIZON <= n
        rows.append({
            "start_frame": s,
            "keep": keep,
            "joints_start": joints[s],
            "proprio": state[s : s + CHUNK_SIZE],
            "action_native30_truncated": truncate(canonical_action_window(actions[s : s + ACTION_HORIZON]))
            if keep else np.zeros((ACTION_HORIZON, 20), np.float32),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="datasets/demo-end-effector-space")
    parser.add_argument("--pool-dir", default="assets/victr_icl_pool_demo")
    args = parser.parse_args()
    root, pool_dir = pathlib.Path(args.dataset), pathlib.Path(args.pool_dir)
    meta = {e["episode_index"]: e for e in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())}
    names = json.loads((root / "meta/info.json").read_text())["features"]["observation.left_ee"].get("names")
    layout = EE_LAYOUTS[tuple(names)]
    print(f"{root}: observation.*_ee layout {layout}")
    files = {int(pathlib.Path(f).stem.split("_")[1]): f for f in glob.glob(str(root / "data/*/*.parquet"))}
    by_task: dict[str, list] = {}
    for ep, path in sorted(files.items()):
        task = meta[ep]["tasks"][0]
        df = pd.read_parquet(path)
        for row in episode_fields(df, layout):
            by_task.setdefault(task, []).append((ep, row))
    for task, rows in sorted(by_task.items()):
        pool_path = pool_dir / f"{task_slug(task)}.pkl"
        if not pool_path.exists():
            print(f"skip (no pool): {task!r}")
            continue
        out = pool_dir / f"{task_slug(task)}.canonical.npz"
        np.savez(
            out,
            task=np.array(task),
            episode_index=np.array([ep for ep, _ in rows], np.int64),
            start_frame=np.array([r["start_frame"] for _, r in rows], np.int64),
            keep=np.array([r["keep"] for _, r in rows]),
            joints_start=np.stack([r["joints_start"] for _, r in rows]),
            proprio=np.stack([r["proprio"] for _, r in rows]),
            action_native30_truncated=np.stack([r["action_native30_truncated"] for _, r in rows]),
            source=np.array(f"{root} (ICRA canonical; ACTION_HORIZON={ACTION_HORIZON}, KEEP_BINS={KEEP_BINS})"),
        )
        keep = sum(r["keep"] for _, r in rows)
        print(f"{out.name}: {len(rows)} chunk starts, {keep} with a full {ACTION_HORIZON}-step window")


if __name__ == "__main__":
    main()
