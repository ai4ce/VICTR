"""Annotation loading and numerical helpers for paper prediction scoring."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.stats import kendalltau



def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)



def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    # Match vfe.evaluate.trajectory_metrics, including float32 MAE/MSE and
    # the zero-correlation convention for constant or single-frame sequences.
    error = prediction - target
    pearson = 0.0
    kendall = 0.0
    if len(target) >= 2 and np.std(target) != 0 and np.std(prediction) != 0:
        pearson = float(np.corrcoef(target, prediction)[0, 1])
    # Tau-b corrects for ties in both curves. As for Pearson, retain
    # degenerate episodes in the equal-query average with correlation zero.
    if len(target) >= 2 and np.any(target != target[0]) and np.any(prediction != prediction[0]):
        kendall = float(kendalltau(target, prediction, variant="b", nan_policy="raise").statistic)
    return {"mae": float(np.mean(np.abs(error))),
            "mse": float(np.mean(error**2)), "pearson": pearson, "kendall": kendall}



def annotation_records(root: Path, source: Path, episodes: dict):
    """Read only the selected episode annotations, without extracting ZIP files."""
    source_label = str(source.relative_to(root)) if source.is_relative_to(root) else str(source)
    if source.is_dir():
        for uid in sorted(episodes):
            path = source / (uid.replace(":", "__") + ".json")
            yield uid, f"{source_label}/{path.name}", path.read_bytes()
        return
    require(source.is_file() and zipfile.is_zipfile(source), f"Not an annotation directory or ZIP: {source}")
    with zipfile.ZipFile(source) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or not info.filename.endswith(".json"):
                continue
            name = Path(info.filename).name
            require(name not in members, f"Duplicate annotation basename in ZIP: {name}")
            members[name] = info.filename
        for uid in sorted(episodes):
            name = uid.replace(":", "__") + ".json"
            require(name in members, f"Missing annotation in ZIP: {name}")
            member = members[name]
            yield uid, f"{source_label}!/{member}", archive.read(member)



def load_annotations(root: Path, folder: Path, episodes: dict, fps: float) -> tuple[dict, list]:
    targets, manifest = {}, []
    for uid, path, raw in annotation_records(root, folder, episodes):
        episode = episodes[uid]
        annotation = json.loads(raw)
        n = episode["length"]
        expected = {
            "episode_uid": uid, "episode_index": episode["source_episode_index"],
            "num_frames": n, "task": episode["tasks"][0],
            "fps": fps, "success": "success",
        }
        for key, value in expected.items():
            require(annotation.get(key) == value, f"Annotation {key} mismatch: {path}")
        points = annotation["points"]
        require(len(points) == n, f"Annotation length mismatch: {path}")
        require([p["frame_index"] for p in points] == list(range(n)), f"Frame index mismatch: {path}")
        times = np.asarray([p["timestamp"] for p in points], dtype=np.float64)
        require(np.isfinite(times).all(), f"Nonfinite timestamp: {path}")
        require(np.allclose(times, np.arange(n) / fps, atol=5.1e-5, rtol=0), f"Timestamp mismatch: {path}")
        progress = np.asarray([p["progress"] for p in points], dtype=np.float32)
        require(np.isfinite(progress).all(), f"Nonfinite annotation: {path}")
        targets[uid] = (-1.0 + np.clip(progress, 0, 100) / 100.0).astype(np.float32)
        manifest.append({"path": path, "sha256": sha256(raw),
                         "episode_uid": uid, "num_frames": n,
                         "interpolation": annotation.get("interpolation"),
                         "steepness": annotation.get("steepness"),
                         "window_seconds": annotation.get("window_seconds")})
    return targets, manifest



def causal_ema(prediction: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    p = np.asarray(prediction, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError("prediction must be a nonempty finite 1D array")
    return lfilter([alpha], [1, -(1-alpha)], p, zi=[(1-alpha)*p[0]])[0]
