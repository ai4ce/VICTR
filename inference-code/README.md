# VICTR — Inference and Coding-Agent Experiments

This directory contains two things:

- **Inference code** for running VICTR policies on the YOR bimanual robot:
  - the openpi model and policy extensions;
  - live retrieval and the value-estimation workers;
  - the inference server and the operator UI;
  - the retrieval-pool builders;
  - the robot-side camera and teleoperation scripts.
- **Coding-agent experiments** (Codex Astra): the frozen executors, attempt
  prompts and labels from running a coding agent as the policy on the same
  robot and tasks.

## Installation

The pieces run in three Python environments:

| Environment | Used by | Setup |
| --- | --- | --- |
| openpi (JAX, `uv`) | inference server, retrieval | clone openpi at `15a9616`, then apply `openpi_overlay/` (below) |
| lerobot (PyTorch) | IC-VFE and Robometer workers, legacy lerobot server | `lerobot`, plus the `vfe` package from the `IC-VFE` branch for the IC-VFE worker |
| UI | `inference.py` | Python 3.10+, `Pillow` |

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi && git checkout 15a9616
git apply ../openpi_overlay/patches/openpi-15a9616-victr.patch
cp -r ../openpi_overlay/src/openpi/{models,policies} src/openpi/
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

The robot side needs `commlink`, `pyzmq` and `mujoco`/`mink` (Pi only). See `robot/`.

## Running a rollout

**1. Retrieval pools (offline, once per task).**

```bash
python pool_tools/build_victr_pool.py --root <icl-demo-dataset> --out-dir assets/victr_icl_pool_demo
python pool_tools/build_canonical_context.py      # ctx3cam context fields
lerobot_env/bin/python pool_tools/build_vfe_values.py --model vfe   # IC-VFE values (or vfe_recap), lerobot env
python pool_tools/build_pool_demo_lengths.py      # only needed with --pool-demos
```

**2. Robot side.** On the Orin, run `deploy_observer.py` and then
`camera_proxy.py`. On the Pi, run `joystick_policy_teleop.py`.

**3. Operator UI** (it starts and stops the inference server itself):

```bash
python inference.py --ui-port 8090 --cam-host <orin-ip>
```

In the UI, pick a policy and enter the task. Align the
robot to the onion-skin overlay, then press Start, and hold the joystick
interlock to execute. Each rollout is recorded to
`rollout_recordings/<policy>/<task>/<date>/` and annotated in the UI.

You can also run the server without the UI:

```bash
cd openpi && .venv/bin/python ../inference_server_openpi.py \
  --config yor_icl_victr_value_canonical_ctx3cam_serve --checkpoint <ckpt> \
  --retrieval-pool ../assets/victr_icl_pool_demo --value-estimator vfe \
  --task "put the square on the peg" --cam-host <orin-ip> --state-host <pi-ip> --publish
```

## Repository layout

### Workstation: inference

| Path | Contents |
| --- | --- |
| `inference.py` | Operator web UI. Starts and stops the server, shows the live camera preview with the onion-skin overlay, pushes task changes live, and logs outcome annotations per attempt |
| `inference_server_openpi.py` | Main inference server. Subscribes to cameras and state, builds the 15-D state and 3-camera observation, runs retrieval plus pi0.5/VICTR, and publishes action chunks at 30 Hz with replanning |
| `inference_server_lerobot.py` | Legacy PyTorch backend for native lerobot pi0.5 checkpoints, with Real-Time Chunking. Same wire protocol |
| `experiments.py` | Named deployment configurations shown in the UI (config, retrieval metric, context mode, demo count). Stdlib only |
| `onion.py` | Builds the onion-skin overlay of a dataset episode's first frame, so each rollout starts from a pose seen in training |
| `rollout_recorder.py` | Writes per-tick state, actions and camera frames for each rollout |
| `yor_fk.py` | Forward kinematics for the YOR arms (MuJoCo), matching the Pi's IK frame |

### openpi extensions (`openpi_overlay/`)

| Path | Contents |
| --- | --- |
| `models/pi0_victr.py` | `Pi0Victr`: pi0.5 flow-matching policy conditioned on retrieved context chunks |
| `models/pi0_fast_victr.py` | The same conditioning on pi0-FAST; native RICL-style baseline |
| `policies/yor_policy.py` | `YorInputs`/`YorOutputs`: YOR camera, state and action mapping for openpi |
| `policies/yor_retrieval.py` | Live retrieval: zscore(DINOv2 L2) + λ·zscore(\|Δvalue\|) over a task pool. Builds the context tokens |
| `policies/yor_vfe.py` | IC-VFE / RECAP progress worker (lerobot env, JSON-lines subprocess) |
| `policies/yor_value.py` | Robometer progress worker (lerobot env, JSON-lines subprocess) |
| `policies/yor_*_test.py` | CPU tests for pool compatibility, retrieval and deployment configurations |
| `training/config.py`, `models/model.py`, `transforms.py` | Full copies of the patched upstream files (VICTR configs, context fields) |
| `patches/openpi-15a9616-victr.patch` | The same changes as a patch against upstream openpi `15a9616` |

### Retrieval pools (`pool_tools/`)

| Path | Contents |
| --- | --- |
| `build_victr_pool.py` | Builds one retrieval pool (chunk dictionary pickle) per task from the demo dataset |
| `build_indomain_pool.py` | Pools for in-domain tasks, using the training pool's episodes |
| `build_canonical_context.py` | Canonical-space context sidecar (`<task>.canonical.npz`) for the ctx3cam policies |
| `build_vfe_values.py` | IC-VFE / RECAP progress sidecars (`<task>.<model>.npz`) |
| `build_robometer_ft_values.py` | Finetuned Robometer progress sidecars (`<task>.robometer_ft.npz`) |
| `build_pool_demo_lengths.py` | Per-demo length and success metadata (`<task>.demos.json`), used with `--pool-demos` |

### Robot side (`robot/`)

| Path | Runs on | Contents |
| --- | --- | --- |
| `deploy_observer.py` | Orin | Captures the ZED and fisheye cameras and publishes JPEG bundles on :5549 |
| `camera_proxy.py` | Orin | Relays :5549 → 0.0.0.0:5550 and stamps proxy timing |
| `joystick_policy_teleop.py` | Pi | Receives action topics on :5572 and gates them with the Xbox-controller interlock (Start/Back/A, X/B home), sends them to IK, publishes state on :5568 |
| `robot_arm/yor.py` | Pi | Bimanual YOR driver: EE targets → IK → joint commands, state readout |
| `robot_arm/arm.py`, `can_gripper.py` | Pi | Arm and CAN gripper drivers |
| `robot_arm/ik_solver.py`, `mdh_kinematics.py` | Pi | IK solver and modified-DH forward kinematics |
| `robot_arm/collect.py` | Pi | Teleop data collection in LeRobot v2.1 format |

### Other

| Path | Contents |
| --- | --- |
| `assets/yor_kinematics/` | Kinematics-only MJCF for `yor_fk.py` |
| `assets/onion_index.json` | Task → reference episode index for the onion-skin overlay |
| `assets/subtask_plans.json` | Per-task subtask lists used for partial-success scoring |
| `agent_baseline/` | Coding-agent experiments (below) |

## Coding-agent experiments (`agent_baseline/`)

We tested whether a coding agent (Codex Astra) can do robot in-context learning
on the same robot and tasks. Each scored attempt worked like this:

- the agent acted as the policy, in an asynchronous LLM-in-the-loop setup;
- it wrote action chunks into an inbox;
- a frozen executor, which the agent wrote and which was then locked, streamed
  those chunks to the robot, enforced the safety limits and logged everything.

There were two conditions per task: **ICL** (the task's demonstrations were
available) and **zero-shot** (no demonstrations and no task knowledge).

| Path | Contents |
| --- | --- |
| `conditions/<cond>/executor/` | Frozen executor: inbox, preemption, holds, safety limits, logging |
| `conditions/<cond>/attempt_prompt.md` | Prompt given to the agent for each attempt |
| `conditions/<cond>/FROZEN` | Executor commit and prompt sha256 at freeze time |
| `conditions/<cond>/notes/` | ICL only: the agent's own demo analysis |
| `conditions/<cond>/checks/` | Pre-freeze offline and live checks |
| `results/*_labels.csv`, `usage_summary.csv` | Per-attempt outcome, subtasks completed, duration and token usage |

| Condition | Attempts | Complete | Partial (credit-weighted) |
| --- | --- | --- | --- |
| Square on peg, ICL | 5 | 20% | 50% |
| Square on peg, zero-shot | 5 | 20% | 40% |
| Sort items, ICL | 5 | 0% | 4.4% |
| Sort items, zero-shot | 5 | 0% | 4.4% |

Notes: labels come from the agent's self-reports, with corrections recorded
per row in the CSVs. The sort ICL attempts ran within a single agent session
rather than in fresh sessions. Run logs and frames (~3 GB) are hosted
separately: **TODO** link.

## Tests

```bash
cd openpi && .venv/bin/python -m pytest src/openpi/policies/yor_retrieval_test.py src/openpi/policies/yor_experiments_test.py -q
```
