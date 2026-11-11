#!/usr/bin/env python
"""Print or run one of the four paper experiments (never submits cluster jobs)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vfe.vfe_presets import load_vfe_preset


def build_command(args: argparse.Namespace, config: dict) -> list[str]:
    experiment = config["experiments"][args.experiment]
    stage = args.stage
    data_preset = experiment["data_split_preset"]
    training = dict(experiment["training"])
    if stage in {"preprocess-pretrain", "pretrain"}:
        if args.experiment != "ic_vfe":
            raise ValueError("The linear-progress pretraining stage belongs to ic_vfe only")
        pre = config["pretraining"]
        data_preset = pre["data_split_preset"]
        training.update(max_steps=pre["max_steps"], epochs=pre["epochs"], requires_resume=False)
        training["distributed"] = {"world_size": pre["world_size"], "global_batch_size": pre["global_batch_size"]}
    data_flags = ["--data-split-config", config["data_split_config"], "--data-split-preset", data_preset]
    model_flags = ["--vfe-config", config["vfe_config"], "--vfe-preset", experiment["vfe_preset"]]
    model = load_vfe_preset(experiment["vfe_preset"], ROOT / config["vfe_config"])
    if stage in {"preprocess", "preprocess-pretrain"}:
        return [sys.executable, "scripts/preprocess_vfe.py", *data_flags]
    if stage == "features":
        if not model["context_features"]["enabled"]:
            raise ValueError("This experiment has no context feature cache")
        return [sys.executable, "scripts/precompute_context_native_features.py", *data_flags, *model_flags]
    if stage in {"train", "pretrain"}:
        if training["requires_resume"] and not args.resume:
            raise ValueError("ic_vfe step 10000 comes from continued training; supply --resume with the recorded step-6800 checkpoint. See README.md.")
        world_size = args.gpus if args.gpus is not None else training["distributed"]["world_size"]
        batch = training["distributed"]["global_batch_size"]
        if world_size <= 0 or batch % world_size:
            raise ValueError(f"GPU count must be positive and divide global batch size {batch}")
        name = args.exp_name or f"paper_{args.experiment}" + ("_pretrain" if stage == "pretrain" else "")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world_size}",
                   "scripts/train_vfe.py", "--exp-name", name, *data_flags, *model_flags,
                   "--tasks-per-batch", str(batch // world_size), "--samples-per-task", "1", "--device", "cuda"]
        for key in ("sampling_mode", "max_steps", "epochs", "num_workers", "prefetch_factor", "lr", "weight_decay", "grad_clip_norm", "seed", "checkpoint_every", "log_every"):
            command.extend(["--" + key.replace("_", "-"), str(training[key])])
        if training["pad_query_frame_once_last_batch"]:
            command.append("--pad-query-frame-once-last-batch")
        command.append("--error-on-nonfinite-grad")
        if args.resume:
            command.extend(["--resume", args.resume])
        if args.resume_model_only:
            command.append("--resume-model-only")
        if args.allow_inexact_resume:
            command.append("--allow-inexact-resume")
        return command
    if stage in {"evaluate", "validate-data"}:
        evaluation = config["evaluation"]
        checkpoint = args.checkpoint or experiment.get("checkpoint")
        if not checkpoint:
            raise ValueError("This experiment is pending training; supply --checkpoint from the new run")
        command = [sys.executable, "scripts/evaluate_vfe_icl.py", *model_flags,
                   "--checkpoint", checkpoint,
                   "--dataset-root", evaluation["dataset_root"], "--annotation-root", evaluation["annotation_root"],
                   "--output-dir", args.output or f"outputs/paper_{args.experiment}/eval",
                   "--seed", str(evaluation["seed"]), "--batch-size", str(args.batch_size),
                   "--context-selection", "round_robin", "--task-split", "all", "--no-plots"]
        if model["model"]["context_enabled"]:
            command.extend(["--all-contexts", "--cache-context-tokens"])
        if stage == "validate-data":
            command.extend(["--validate-only", "--device", "cpu"])
        else:
            command.extend(["--device", "cuda"])
        return command
    if stage == "score":
        source = args.source or experiment.get("prediction_dir")
        if not source:
            raise ValueError("This experiment is pending training; supply --source from the new run")
        command = [sys.executable, "scripts/score_paper.py", "--config", str(args.config), "--experiment", args.experiment,
                   "--source", source]
        if args.output:
            command.extend(["--output", args.output])
        return command
    raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["ic_vfe", "ic_vfe_no_context", "ic_vfe_no_block_embedding", "recap"])
    parser.add_argument("stage", choices=["preprocess", "preprocess-pretrain", "features", "pretrain", "train", "validate-data", "evaluate", "score"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_experiments.json")
    parser.add_argument("--execute", action="store_true", help="Execute the displayed command in the repository root")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--exp-name")
    parser.add_argument("--resume")
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--allow-inexact-resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--source", help="Raw prediction directory for score")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        if args.resume_model_only and not args.resume:
            raise ValueError("--resume-model-only requires --resume")
        command = build_command(args, config)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if args.execute:
        # No account or cloud logging is required for local reproduction.
        env = dict(os.environ)
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        if args.stage in {"evaluate", "validate-data"}:
            evaluation = config["evaluation"]
            info_path = ROOT / evaluation["dataset_root"] / "meta/info.json"
            info = json.loads(info_path.read_text())
            videos = [feature for feature in info["features"].values() if feature.get("dtype") == "video"]
            if not videos:
                parser.error("Evaluation dataset has no video features")
            for feature in videos:
                if feature["shape"] != [evaluation["expected_video_height"], evaluation["expected_video_width"], 3]:
                    parser.error("Evaluation requires the original 640x480 videos")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
