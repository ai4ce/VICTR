#!/usr/bin/env python
"""Score raw saved predictions with each paper row's recorded convention."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau, pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vfe.vfe_presets import load_vfe_preset
from vfe.scoring import annotation_records, causal_ema, load_annotations, metrics, require


def correlations(target: np.ndarray, prediction: np.ndarray, exclude: bool) -> dict:
    constant = len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0
    if not exclude:
        return metrics(target, prediction)
    return {
        "mae": float(np.mean(np.abs(target - prediction))),
        "mse": float(np.mean((target - prediction) ** 2)),
        "pearson": None if constant else float(pearsonr(target, prediction).statistic),
        "kendall": None if constant else float(kendalltau(target, prediction, variant="b").statistic),
    }


def mean_scores(rows: list[dict]) -> dict:
    result = {}
    for key in ("mae", "mse", "pearson", "kendall"):
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def score(config: dict, experiment: str, source: Path, root: Path = ROOT) -> dict:
    spec = config["experiments"][experiment]
    protocol = spec["scoring"]
    evaluation = config["evaluation"]
    dataset = root / evaluation["dataset_root"] / "meta"
    rows = [json.loads(line) for line in (dataset / "episodes.jsonl").read_text().splitlines() if line.strip()]
    episodes = {row["episode_uid"]: row for row in rows}
    require(len(rows) == len(episodes) == evaluation["expected_queries"], "Unexpected/duplicate evaluation episodes")
    require(all(row.get("keep") and row.get("success") for row in rows), "Expected successful keep-true episodes")
    fps = float(json.loads((dataset / "info.json").read_text())["fps"])
    human = root / evaluation["human_annotations"]
    targets, annotations = load_annotations(root, human, episodes, fps)
    if protocol["human_reference"] == "raw_float64_percent":
        targets = {uid: np.asarray([p["progress"] for p in json.loads(raw)["points"]], dtype=np.float64) / 100
                   for uid, _, raw in annotation_records(root, human, episodes)}
    exclude = protocol["constant_correlation"] == "exclude"
    model = load_vfe_preset(spec["vfe_preset"], root / config["vfe_config"])["model"]
    groups = defaultdict(list)
    contexts = defaultdict(set)
    splits = {}
    prediction_manifest = []
    checkpoints = set()
    files = sorted(source.rglob("*.npz"))
    require(bool(files), f"No prediction files at {source}")
    for path in files:
        with np.load(path, allow_pickle=False) as curve:
            meta = json.loads(str(curve["metadata_json"]))
            require("ema_alpha" not in meta and "raw_prediction" not in curve.files, "Supply raw predictions; scoring applies EMA exactly once")
            prediction = curve["prediction"].copy()
            ts = curve["timesteps"].copy()
            context_ids = curve["context_demo_ids"].copy()
            uid = meta["episode_uid"]
            context_uid = meta.get("context_episode_uid")
            require(uid in episodes, f"Unknown query: {uid}")
            episode = episodes[uid]
            require(prediction.shape == (episode["length"],) and np.isfinite(prediction).all(), f"Invalid prediction: {path}")
            require(np.array_equal(ts, np.arange(episode["length"])), f"Incomplete timestep coverage: {path}")
            require(meta["task_name"] == episode["tasks"][0], f"Task mismatch: {path}")
            require(meta["vfe_preset"] == spec["vfe_preset"], f"Wrong model preset: {path}")
            require(bool(meta.get("checkpoint")), f"Missing checkpoint identity: {path}")
            checkpoints.add(meta["checkpoint"])
            require(len(checkpoints) == 1, "Prediction directory mixes different checkpoints")
            require(context_uid not in contexts[uid], f"Duplicate query/context pair: {path}")
            if model["context_enabled"]:
                require(context_uid in episodes and context_uid != uid, f"Invalid/self context: {path}")
                require(episodes[context_uid]["tasks"] == episode["tasks"], f"Cross-task context: {path}")
                require(len(context_ids) == len(ts) and np.all(context_ids == meta["context_demo_id"]), f"Context changed within trajectory: {path}")
            else:
                require(context_uid is None and len(context_ids) == 0, f"Unexpected no-context input: {path}")
            split = meta["task_novelty"]
            require(split in {"seen", "unseen"} and splits.get(uid, split) == split, f"Inconsistent split: {path}")
            splits[uid] = split
            contexts[uid].add(context_uid)
            prediction_manifest.append({"path": str(path.relative_to(source)), "sha256": hashlib.sha256(prediction.tobytes()).hexdigest()})
            alpha = protocol["ema_alpha"]
            if alpha != 1:
                prediction = causal_ema(prediction, alpha)
            if exclude:
                # Original RECAP archives have float32 prediction_progress;
                # original EMA exports only have float64 prediction values.
                prediction = (np.asarray(curve["prediction_progress"], dtype=np.float64)
                              if alpha == 1 and "prediction_progress" in curve.files
                              else prediction.astype(np.float64) + 1.0)
            groups[uid].append(correlations(targets[uid], prediction, exclude))
    require(set(groups) == set(episodes), "Missing evaluation queries")
    per_query = []
    for uid, scores in sorted(groups.items()):
        expected = {other for other, row in episodes.items() if row["tasks"] == episodes[uid]["tasks"] and other != uid} if model["context_enabled"] else {None}
        require(contexts[uid] == expected, f"Incomplete leave-one-out context coverage: {uid}")
        per_query.append({"episode_uid": uid, "split": splits[uid], "metrics": mean_scores(scores)})
    aggregate = {}
    for split in ("seen", "unseen"):
        queries = [row for row in per_query if row["split"] == split]
        require(len(queries) == evaluation["expected_queries_per_split"], f"Wrong query count: {split}")
        included = [row for row in queries if row["metrics"]["pearson"] is not None]
        aggregate[split] = {"num_queries": len(queries), "num_scored_queries": len(included),
                            "metrics": mean_scores([row["metrics"] for row in included])}
    return {"experiment": experiment, "source": str(source), "checkpoint": next(iter(checkpoints)), "protocol": protocol,
            "aggregation": "mean over contexts within each query, then equal query weight",
            "aggregate": aggregate, "per_query": per_query,
            "annotations": annotations, "prediction_manifest": prediction_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vfe/paper_experiments.json")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are not overwritten")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.experiment not in config["experiments"]:
        parser.error("Unknown paper experiment")
    if args.output and args.output.exists():
        parser.error("Output already exists")
    source_path = args.source or config["experiments"][args.experiment].get("prediction_dir")
    if not source_path:
        parser.error("This experiment is pending training; supply --source from the new run")
    source = (ROOT / source_path).resolve()
    result = score(config, args.experiment, source)
    print(json.dumps(result["aggregate"], indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
