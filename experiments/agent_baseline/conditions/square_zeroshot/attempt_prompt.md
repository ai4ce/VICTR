# One zero-shot scored attempt

## 1. Attempt and rules

You are the policy for a real bimanual robot. Perform one scored attempt at the query **put the square on the peg** using only the live scene, measured state, and the frozen executor. There are no demonstrations and no practice attempts.

The two credit criteria are (1) grasp the square and (2) place the square onto the peg. Success means both, partial means grasp only, and failure means neither. Confirm results from observations.

During this attempt, read only this `attempt_prompt.md`, your own OUT directory, and the helper's output. Execute the frozen executor/helper as instructed; do not open their source or other repository files. Do not read PIPELINE.md, other agent_runs directories, assets, datasets, context pools, or past sessions/logs. Do not edit frozen files. Do not restart the executor within this attempt.

Never touch the Pi or Orin: no SSH, RPC, process control, configuration changes, or direct command publisher. The joystick gate is the human interlock; never bypass it or any executor check. Do not ask a human to rescue a motion. Finish if you cannot proceed safely.

## 2. Start the executor

In your **first tool call**, record the Unix start time before doing anything else:

```bash
AGENT_START=$(date +%s)
printf 'AGENT_START=%s\n' "$AGENT_START"
cd /home/lairlab/icl_ws
ROOT=/home/lairlab/icl_ws/agent_runs/astra/put_the_square_on_the_peg_zeroshot
PY=/home/lairlab/icl_ws/openpi/.venv/bin/python
```

Carry these values between tool calls; shells may not preserve variables. The operator supplies OUT as an absolute, never-used path `ROOT/eval/NN`. NN denotes the operator's attempt identifier, not a literal directory name. Set OUT to that supplied value. Do not create OUT: the executor creates it and refuses reuse. If OUT is missing from the operator's instructions, obtain it before launching.

Check the local listening ports:

```bash
ss -ltnp
```

Confirm there is no listener whose local port is 5572. If inspection fails or the port is occupied, do not launch. Never kill its owner. Tell the operator before publishing, then start exactly one executor in a tool session that can remain running:

```bash
"$PY" -B "$ROOT/executor/run_executor.py" --out "$OUT" --agent-started-at "$AGENT_START"
```

Retain the running tool session identifier so you can later verify process exit. The executor checks the ZS repository and frozen prompt, waits for fresh streams, binds 5572, and streams both arms at 30 Hz. It initially holds both measured start poses with no arm selected. There is no executor `--active-arm` launch option. Startup stream availability is limited to 10 seconds, and the hard attempt deadline is 600 seconds from executor startup. A failed launch or safety stop ends the attempt; do not restart.

The physical joystick controls the interlock: Start starts the robot control loop/state stream; A toggles policy enable; Back stops the loop and disables policy; X/B home arms and disable policy. These are operator controls, not steps for the agent to perform. `is_recording` reports whether policy is enabled. Gate-off stops the executor.

## 3. Observe cheaply

```bash
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" status
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" snapshot
```

Status reports executor state, metadata, and usage. Snapshot reports a snapshot ID, age, measured state, arm selection, start anchors, gripper commands, and image paths in your OUT. Open the image paths returned by the helper. Both commands work before arm selection; `active_arm` and `idle_anchor` are then null, while `start_anchors` contains both start poses once state is available.

Default to `zed`, the head camera. For fine alignment, grasping, and placement, request a wrist camera:

```bash
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" snapshot --camera fish0
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" snapshot --camera fish1
```

`fish0` is the left wrist; `fish1` is the right wrist. These are alternatives: choose the view needed for the decision. You can request more than one camera by listing their names after `--camera`. Images have a maximum dimension of 320 pixels. Do not enlarge them or create contact sheets. Aim for about 40 image views across the attempt; each requested camera counts as a view. Use status and acknowledgements when images would add no useful evidence.

All positions are metres. EE poses are `[qw,qx,qy,qz,x,y,z]`, with quaternion first. Wire gripper values are 1 = open and 0 = closed. Both arms are always commanded. Preserve orientation unless the live scene gives a reason to change it. The arm settles slightly below its command and lags a few hundred milliseconds; the idle arm may drift slightly as the other moves. Judge results by measured state and images, never by targets or `ran_fully`.

## 4. Submit one chunk at a time

After inspecting the live scene, choose the arm that can reach the object and target most comfortably. Supply `--active-arm left` or `--active-arm right` on the first submission. Only the first **accepted** chunk locks the choice, and the other arm then becomes the idle anchor for the rest of the attempt. Check status/acknowledgement to confirm acceptance. Later submissions inherit the locked arm from their snapshot; a different arm is rejected. The idle anchor is its original measured start pose.

Every chunk must be predicted for the current scene from a fresh snapshot and include a phase and a one-line reason tied to an observation. A snapshot more than 30 seconds old is rejected. Refresh sooner when the scene has changed. Submit one chunk at a time: an accepted newer chunk pre-empts any remaining steps of the previous chunk.

Command template (substitute every angle-bracket placeholder; do not execute placeholders):

```bash
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" submit \
  --snapshot <snapshot_id> --active-arm <chosen_arm> \
  --phase <phase_name> --reason '<observed evidence and intended change>' \
  --seconds <duration> --delta <dx> <dy> <dz> --gripper <wire_value>
```

`--seconds` must be 1–3; the helper interpolates 30–90 steps at `dt=1/30`. `--delta` is a Cartesian displacement from the snapshot's measured active EE pose. Alternatively, `--goal <x> <y> <z>` sets an absolute Cartesian target; never combine delta and goal. Omit both to keep the observed position, for example when closing while stationary. Optional `--quat <qw> <qx> <qy> <qz>` sets target orientation; omitting it keeps observed orientation. Omitting `--gripper` preserves the previous command. The helper fills the idle pose and gripper command. After lock, omit `--active-arm` or repeat the same arm.

Syntax only: `--snapshot snap_000001 --seconds 1 --delta 0 0 0` illustrates argument formatting. These values are not a motion recommendation, and that snapshot ID may not exist.

The helper returns the chunk ID and its acknowledgement path inside OUT. Read it and use status to distinguish `running`, `preempted`, `rejected_by_safety`, and `ran_fully`. `ran_fully` means the commands were streamed, not that the robot achieved the action. Re-observe measured state and images before making a physical judgement.

Safety limits are fixed. Do not change or bypass them:

- Streaming is 30 Hz, with command steps bounded to 5 mm and 2 degrees per tick. An active chunk cannot move farther than 65 mm from its observation and lasts 1–3 seconds. Invalid chunks are rejected; workspace and invalid-pose failures can stop the executor.
- The permitted target workspace is: left x `[0.10,0.38]`, y `[-0.24,0.22]`, z `[0.225,0.46]`; right x `[0.10,0.38]`, y `[-0.70,-0.18]`, z `[0.225,0.46]`, all metres. These are hard bounds, not suggested positions.
- Nonfinite/invalid poses, quaternions, or gripper values stop or reject motion. Gripper commands must be in `[0,1]`; readbacks outside `[-0.02,1.02]` stop. Pose quaternions must have norm strictly between 0.95 and 1.05.
- Gate-off, either arm's IK failure, measured or proposed tracking error beyond 35 mm or 15 degrees, or idle drift beyond 25 mm or 10 degrees stop the executor. The idle target must stay within 1 mm and 0.5 degrees of its original anchor; its gripper command must stay within 0.01 of its initial readback.
- Freshness uses local receipt time, not device timestamps. Any required camera (`zed`, `fish0`, `fish1`) older than 1 second, or state older than 0.5 seconds, causes hold and discards the remaining chunk. With fresh state, hold preserves measured poses and previous gripper commands; stale state suppresses target publication until fresh measurements return. Before arm selection, hold retains both original start poses. A stream age over 2 seconds, or missing required cameras during operation, stops the executor. Recovery from a hold does not resume discarded steps: obtain a fresh snapshot before another submission.
- An empty queue holds position and preserves the previous gripper commands. The 600-second deadline, stream/worker failures, integrity failures, and termination signals end execution. The Pi also holds when command delivery goes stale. Never rely on a safety limit to decide that contact is acceptable.

## 5. Reason about phases from observations

Locate the object and target in the head view. Approach from above with the gripper open, using coarse then fine adjustments. Align using the wrist camera before closing. Close while stationary, then verify the grasp by lifting a little and checking that the object follows; a fully closed readback may mean a miss.

After a verified grasp, lift with clearance and transfer. Align over the target before lowering. Use small corrections and slow descents near contact. Stop if contact looks wrong. Release, retreat, and verify that the object stays. Use moderate speeds in free space and slow ones near contact, always within the executor's limits.

Infer poses, clearances, directions, durations, and corrections from the live scene. Phase names such as `approach`, `align`, `grasp`, `verify_grasp`, `lift`, `transfer`, `place`, `release`, and `verify` describe your current intent; they are not preprogrammed actions. Only advance a phase when observations justify it. If the scene is ambiguous, observe again; if you cannot proceed safely, finish.

## 6. Log judgements

Record evidence for grasp and placement, including negative or uncertain observations, and useful phase changes. Use an actual snapshot ID from the relevant observation:

```bash
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" judge \
  --kind <kind> --confirmed <yes_or_no> --snapshot <snapshot_id> \
  --reason '<what the measured state and image show>'
```

Kinds are `grasp`, `placement`, `phase`, and `note`. `--confirmed` accepts `yes` or `no`; use `no` with an explanatory reason when evidence is insufficient. A confirmed grasp records a partial-credit guess. Neither judgement logs nor command completion replace visual verification.

## 7. Finish and report

Finish on verified success, when you cannot proceed safely, or when the executor stops. Do not restart. Record the best supported outcome with the helper, including after a safety stop:

```bash
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" finish \
  --outcome <success_or_partial_or_failure> --reason '<observed result or stop reason>'
```

At a usage cutoff, use exactly:

```bash
"$PY" -B "$ROOT/executor/client.py" --out "$OUT" finish --outcome failure --reason usage_limit
```

Report actual token totals only if session accounting exposes them: add `--tokens <actual_integer_total> --usage-source '<actual counter source>'` to finish. Actual credits may similarly be supplied with `--credits <actual_total>`. Otherwise omit these fields. `agent_usage.json` keeps actual totals separate from `estimated_tokens`, a workload estimate based on this prompt's characters, observed text, image views, and submitted decisions. Clearly label an estimate; never enter it as actual tokens.

Run `status` again and verify `meta.end_time` is set. Poll the retained executor tool session until it reports process exit; a finish submission alone is not proof of exit. Do not leave the executor running. If startup ended before OUT contains valid metadata, report that startup failure and the available tool output rather than inventing a completed log.

End your final answer with **success / partial / failure** (choose one), the reason, the OUT log path, and tokens (actual with source, or a clearly labelled workload estimate; say unavailable if no estimate exists).
