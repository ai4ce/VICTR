"""
onion.py -- dataset "ghost" frames for the inference UI's onion-skin overlay.

Purpose: before evaluating a policy, the robot's base and arms should start from roughly the
pose a dataset episode of the same task started from. Otherwise a rollout is being judged from
an initial condition the policy never trained on, and a failure says nothing about the policy.
This module supplies the reference layer: a recorded episode's camera frames, composited in
exactly the live preview's layout so the UI can stack one on the other at low opacity and the
operator can physically nudge the robot until the two line up.

GEOMETRY -- the overlay works in POLICY INPUT space, not raw camera space. The ZED captures
1280x720 but every dataset episode stores it at 640x480, and inference_server_openpi.py's
match_dataset_zed_geometry() plain-resizes the live frame to 640x480 (a 16:9 -> 4:3 squash, full
field of view kept) before the policy ever sees it. Aligning in raw 1280x720 would therefore be
aligning in a frame the policy doesn't use. Both fisheyes are already 640x480 on both paths and
pass through untouched. compose() is the single layout definition, used for the live composite
and the ghost alike, so the two canvases cannot drift apart.

Frames come out of the dataset's mp4s with ffmpeg. The v3.0 layout packs many episodes into one
video file per camera, each episode a [from_timestamp, to_timestamp) slice, and the three cameras
land in different files at different offsets -- meta/episodes is the only thing that says where.
That mapping is flattened once into assets/onion_index.json, because reading it needs pandas and
the UI venv has only PIL/pyzmq/commlink.

Build the index (needs the openpi venv for pandas; the UI does this itself on first use):
  openpi/.venv/bin/python onion.py --build
"""

import io
import json
import logging
import pathlib
import shutil
import subprocess
import tempfile
import threading

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = pathlib.Path(
    "/home/lairlab/projects/ultra-tools/data/icl-dataset-fixed-action")
INDEX_PATH = PROJECT_ROOT / "assets" / "onion_index.json"

# (W, H), PIL order -- the size every dataset episode stores, and what
# inference_server_openpi.py resizes the live ZED to before inference.
DATASET_ZED_SIZE = (640, 480)
# Camera order is fixed everywhere in this module: index 0/1/2 == zed/fish0/fish1.
VIDEO_KEYS = ("observation.images.zed", "observation.images.fish0", "observation.images.fish1")
BAR_H = 32  # status strip above the top camera, part of the shared layout


# -- Layout, shared by the live composite and the ghost so they stay pixel-aligned --

def compose(zed: Image.Image, fish0: Image.Image | None, fish1: Image.Image | None,
            bar_text: str | None = None, labels: bool = True) -> Image.Image:
    """ZED on top, the two gripper fisheyes side by side beneath it, a status strip above.
    The ghost passes bar_text=None/labels=False so it contributes no text of its own -- it still
    reserves the same strip, which is what keeps the two layers registered."""
    top = zed
    half_w = top.width // 2
    left = fish0.resize((half_w, int(fish0.height * half_w / fish0.width)), Image.BILINEAR) \
        if fish0 is not None else None
    right = fish1.resize((top.width - half_w, int(fish1.height * (top.width - half_w) / fish1.width)),
                         Image.BILINEAR) if fish1 is not None else None

    heights = [im.height for im in (left, right) if im is not None]
    bottom_h = max(heights) if heights else 0
    bottom_row = Image.new("RGB", (top.width, bottom_h), (0, 0, 0)) if bottom_h else None
    if bottom_row is not None:
        bd = ImageDraw.Draw(bottom_row)
        if left is not None:
            bottom_row.paste(left, (0, 0))
            if labels:
                bd.text((6, 4), "left gripper", fill=(220, 220, 60))
        if right is not None:
            bottom_row.paste(right, (half_w, 0))
            if labels:
                bd.text((half_w + 6, 4), "right gripper", fill=(220, 220, 60))
        elif labels:
            bd.text((half_w + 6, 4), "no fish1 feed", fill=(200, 100, 100))

    bar = Image.new("RGB", (top.width, BAR_H), (20, 20, 20))
    if bar_text:
        ImageDraw.Draw(bar).text((8, 8), bar_text, fill=(220, 220, 60))

    combined = Image.new("RGB", (top.width, BAR_H + top.height + bottom_h))
    combined.paste(bar, (0, 0))
    combined.paste(top, (0, BAR_H))
    if bottom_row is not None:
        combined.paste(bottom_row, (0, BAR_H + top.height))
    return combined


def to_policy_view(zed: Image.Image) -> Image.Image:
    """The live ZED as the policy actually receives it -- see the module docstring.

    This mirrors inference_server_openpi.match_dataset_zed_geometry() rather than calling it:
    that module needs numpy/jax and the UI venv has neither. Keep the two in step."""
    if zed.size == DATASET_ZED_SIZE:
        return zed
    return zed.resize(DATASET_ZED_SIZE, Image.BILINEAR)


# -- Index (built under the openpi venv, read by the UI venv) --

def build_index(root: pathlib.Path, out_path: pathlib.Path) -> dict:
    """Flatten meta/episodes into the few fields the overlay needs. pandas-only, so this runs
    under openpi/.venv, never in the UI process itself."""
    import pandas as pd

    info = json.loads((root / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise SystemExit(f"{root} is layout {info.get('codebase_version')}; the overlay needs a "
                         f"v3.0 dataset (per-camera video files located through meta/episodes).")

    cols = ["episode_index", "tasks", "length", "success", "valid", "keep", "episode_uid"]
    for k in VIDEO_KEYS:
        cols += [f"videos/{k}/chunk_index", f"videos/{k}/file_index", f"videos/{k}/from_timestamp"]
    eps = pd.read_parquet(root / "meta" / "episodes", columns=cols)

    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_of_index = {int(v): str(k) for k, v in tasks["task_index"].items()}
    index_of_task = {name: i for i, name in task_of_index.items()}

    episodes = []
    for _, r in eps.iterrows():
        # 'tasks' is a list column; every episode in this dataset carries exactly one.
        name = str(r["tasks"][0]) if len(r["tasks"]) else ""
        episodes.append({
            "i": int(r["episode_index"]),
            "t": index_of_task.get(name, -1),
            "n": int(r["length"]),
            "s": bool(r["success"]), "v": bool(r["valid"]), "k": bool(r["keep"]),
            "uid": str(r["episode_uid"]),
            "c": [[int(r[f"videos/{k}/chunk_index"]), int(r[f"videos/{k}/file_index"]),
                   float(r[f"videos/{k}/from_timestamp"])] for k in VIDEO_KEYS],
        })
    episodes.sort(key=lambda e: e["i"])

    out = {
        "dataset_root": str(root),
        "fps": int(info["fps"]),
        "video_path": info["video_path"],
        "tasks": [task_of_index[i] for i in sorted(task_of_index)],
        "episodes": episodes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    return out


def ensure_index(root: pathlib.Path, builder_python: pathlib.Path) -> dict | None:
    """Load the index, building it first if it is missing or was built from another dataset.
    Returns None (and logs why) rather than raising, so a missing dataset only costs the UI its
    overlay rather than its camera preview."""
    try:
        if INDEX_PATH.exists():
            idx = json.loads(INDEX_PATH.read_text())
            if idx.get("dataset_root") == str(root):
                return idx
            log.info("Onion index was built from %s, rebuilding for %s",
                     idx.get("dataset_root"), root)
        if not (root / "meta" / "info.json").exists():
            log.warning("Onion overlay disabled: no dataset at %s", root)
            return None
        if not builder_python.exists():
            log.warning("Onion overlay disabled: %s not found (needed to read the dataset's "
                        "parquet metadata)", builder_python)
            return None
        log.info("Building onion index from %s ...", root)
        subprocess.run([str(builder_python), str(PROJECT_ROOT / "onion.py"),
                        "--build", "--dataset-root", str(root)], check=True)
        return json.loads(INDEX_PATH.read_text())
    except Exception as e:
        log.warning("Onion overlay disabled: %s", e)
        return None


# -- Episode frames --

class GhostEpisode:
    """One loaded episode's frames, held in memory as JPEG bytes per camera.

    A whole 23 s episode decodes in well under a second per camera (one ffmpeg seek plus a
    sequential decode, ~17 MB of JPEGs), so the overlay loads an episode in full rather than
    seeking per frame -- that is what makes scrubbing and playback instant. Only one episode is
    held at a time; selecting another drops the previous one."""

    def __init__(self, index: dict, episode: dict):
        self.index = index
        self.episode = episode
        self.n_frames = episode["n"]
        self.cams: list[list[bytes]] = [[], [], []]
        self.error: str | None = None
        self.loaded = False
        self._compose_cache: dict[int, bytes] = {}
        self._lock = threading.Lock()

    def _video_path(self, cam_i: int) -> pathlib.Path:
        chunk, file_i, _ = self.episode["c"][cam_i]
        return pathlib.Path(self.index["dataset_root"]) / self.index["video_path"].format(
            video_key=VIDEO_KEYS[cam_i], chunk_index=chunk, file_index=file_i)

    def load(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="onion_"))
        try:
            for cam_i in range(3):
                path = self._video_path(cam_i)
                if not path.exists():
                    # A partially downloaded dataset is normal here -- a missing fisheye should
                    # cost that panel, not the whole overlay.
                    log.warning("Onion: %s missing, skipping that camera", path)
                    continue
                start = self.episode["c"][cam_i][2]
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", str(path),
                     "-frames:v", str(self.n_frames), "-vsync", "0", "-q:v", "5",
                     "-f", "image2", "-y", str(tmp / f"c{cam_i}_%05d.jpg")],
                    check=True)
                self.cams[cam_i] = [p.read_bytes()
                                    for p in sorted(tmp.glob(f"c{cam_i}_*.jpg"))]
            if not self.cams[0]:
                self.error = "no ZED frames could be extracted for this episode"
        except Exception as e:
            self.error = str(e)
            log.warning("Onion: episode %d failed to load: %s", self.episode["i"], e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            self.loaded = True

    def frame_jpeg(self, frame: int) -> bytes | None:
        """Ghost composite for one frame, in the same canvas compose() gives the live view."""
        if not self.cams[0]:
            return None
        frame = max(0, min(frame, len(self.cams[0]) - 1))
        with self._lock:
            hit = self._compose_cache.get(frame)
        if hit is not None:
            return hit

        def _open(cam_i):
            frames = self.cams[cam_i]
            if not frames:
                return None
            return Image.open(io.BytesIO(frames[min(frame, len(frames) - 1)])).convert("RGB")

        img = compose(_open(0), _open(1), _open(2), bar_text=None, labels=False)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        with self._lock:
            # Bounded so a long scrub through several episodes can't grow without limit; the
            # source JPEGs are already in memory, so a miss only costs one recomposite.
            if len(self._compose_cache) > 400:
                self._compose_cache.clear()
            self._compose_cache[frame] = data
        return data


class Onion:
    """The overlay's whole state: the index, and whichever episode is currently selected."""

    def __init__(self, dataset_root: pathlib.Path, builder_python: pathlib.Path):
        self.dataset_root = dataset_root
        self.builder_python = builder_python
        self.index: dict | None = None
        self.current: GhostEpisode | None = None
        self._lock = threading.Lock()
        self._by_index: dict[int, dict] = {}

    def start(self):
        """Build/load the index off the main thread -- a first-time build reads a 3149-row
        parquet under another interpreter and shouldn't hold up the UI coming online."""
        def _work():
            idx = ensure_index(self.dataset_root, self.builder_python)
            if idx is None:
                return
            with self._lock:
                self.index = idx
                self._by_index = {e["i"]: e for e in idx["episodes"]}
            log.info("Onion overlay ready: %d episodes, %d tasks from %s",
                     len(idx["episodes"]), len(idx["tasks"]), self.dataset_root)
        threading.Thread(target=_work, daemon=True).start()

    def tasks(self) -> dict:
        with self._lock:
            idx = self.index
        if idx is None:
            return {"ready": False, "tasks": []}
        counts: dict[int, int] = {}
        for e in idx["episodes"]:
            if e["s"] and e["v"]:
                counts[e["t"]] = counts.get(e["t"], 0) + 1
        return {
            "ready": True, "fps": idx["fps"],
            "tasks": [{"task_index": i, "task": name, "n_episodes": counts.get(i, 0)}
                      for i, name in enumerate(idx["tasks"])],
        }

    def episodes(self, task_index: int) -> list[dict]:
        """Successful, valid episodes of one task, the ones in the training set first -- those
        are what the policy was actually fit to, so they are the right alignment reference."""
        with self._lock:
            idx = self.index
        if idx is None:
            return []
        out = [e for e in idx["episodes"] if e["t"] == task_index and e["s"] and e["v"]]
        out.sort(key=lambda e: (not e["k"], e["i"]))
        return [{"episode_index": e["i"], "length": e["n"], "uid": e["uid"], "in_training_set": e["k"]}
                for e in out]

    def select(self, episode_index: int) -> dict:
        with self._lock:
            meta = self._by_index.get(episode_index)
        if meta is None:
            raise ValueError(f"episode {episode_index} is not in the onion index")
        ghost = GhostEpisode(self.index, meta)
        ghost.load()
        with self._lock:
            self.current = ghost
        return self.status()

    def status(self) -> dict:
        with self._lock:
            g = self.current
            ready = self.index is not None
        if g is None:
            return {"ready": ready, "episode_index": None}
        return {
            "ready": ready,
            "episode_index": g.episode["i"],
            "uid": g.episode["uid"],
            "length": len(g.cams[0]) if g.cams[0] else 0,
            "cameras": [len(c) for c in g.cams],
            "error": g.error,
        }

    def frame_jpeg(self, frame: int) -> bytes | None:
        with self._lock:
            g = self.current
        return g.frame_jpeg(frame) if g is not None else None


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build the onion overlay's episode index.")
    p.add_argument("--build", action="store_true", required=True)
    p.add_argument("--dataset-root", type=pathlib.Path, default=DEFAULT_DATASET_ROOT)
    args = p.parse_args()
    idx = build_index(args.dataset_root.expanduser(), INDEX_PATH)
    log.info("Wrote %s: %d episodes, %d tasks (%.1f MB)", INDEX_PATH, len(idx["episodes"]),
             len(idx["tasks"]), INDEX_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
