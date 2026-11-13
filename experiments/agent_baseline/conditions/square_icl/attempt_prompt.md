# One async scored attempt: put the square on the peg

You are the policy for ONE attempt on the real robot. The operator has reset the
scene and authorized this attempt. Use the frozen files in
`/home/lairlab/icl_ws/agent_runs/astra/put_the_square_on_the_peg/` (call this ROOT).
The operator supplies a new attempt output directory OUT, normally
`ROOT/eval/NN` or `ROOT/practice/async_main`. If OUT is missing, ask for it before
starting. Do not reuse or overwrite an existing attempt directory.

Your task is **put the square on the peg**, with partial credit for (1) grasping
the square, (2) placing it onto the peg. You may use only this task's frozen demo
notes and the current attempt's live snapshots/state. Read
`ROOT/notes/demo_analysis.md` once before moving. Its statistics come only from
episodes 241–251 of `assets/victr_icl_pool_demo/put_the_square_on_the_peg.pkl`.
Do not load the 6 GB pool during the attempt, inspect other attempts, browse other
task data, load pretrained models, call other models, or change code/notes/prompt.
You are the only model. Frozen helper code is allowed; no ad-hoc robot-control code.

Use `openpi/.venv/bin/python` from `/home/lairlab/icl_ws` (PY below). In your first
tool call obtain `date +%s` and retain that number as AGENT_START for wall-time
accounting. No human is needed after startup except the physical safety interlock.

## Start the executor

Tell the operator before publishing. Check that 5572 is free (`ss -ltnp`); never
kill its owner or start/modify anything on the Pi or Orin. The gate must be enabled.
Start this long-running command, retaining its process/session handle:

```sh
PY ROOT/executor/run_executor.py --task "put the square on the peg" \
  --out OUT --active-arm left --agent-started-at AGENT_START
```

Replace PY, ROOT, OUT and AGENT_START with their actual paths/value; these are
notation, not literal shell variables. Use normal tool escalation if network
sandboxing blocks the robot LAN. Do not consume the attempt by repeatedly trying
inside a network sandbox known to block these connections.

The established scene has the square on the left; the left arm is active and the
right arm holds its measured start anchor. If the scene cannot be handled safely
with that arm, finish as failure rather than restarting or switching arms. The
executor waits up to 10 seconds for streams, then streams at 30 Hz. Until an agent
chunk arrives, and between chunks, it holds a pose measured on entering hold.
It maintains the previous gripper commands, so queue-empty holds do not release
an object. It stops after at most 600 seconds. Do not leave it running at turn end.

Gate-off, IK failure, invalid values, tracking >35 mm, idle-arm drift >25 mm/10°,
workspace violations and the deadline stop execution. Camera arrival age >1 s or
state arrival age >0.5 s interrupts the current chunk and holds. Packet age >2 s
stops execution. A stale state produces no commands, relying on the Pi's hold.
Source clock differences and identical JPEG bytes do not affect freshness. Never
bypass these checks or request that a human rescue a motion mid-attempt.

## Observe cheaply while motion runs

Use this helper to select and record every frame you inspect:

```sh
PY ROOT/executor/client.py --out OUT snapshot --camera zed
```

It prints an immutable snapshot id, both measured EE poses, gripper readbacks,
gate/IK status, current chunk progress, and the image path. Use the image-viewing
tool on that path; it is already 320 pixels wide. Default to the head camera.
Use `--camera fish0` only for grasp or peg alignment/placement; use fish1 only if
the head view needs a confirming view of the peg. Do not enlarge images or make
contact sheets. Aim for no more than about **40 image views**: roughly 8 approach,
12 grasp/lift, 6 transfer, 10 insertion/release, 4 final checks. A new view is useful
when it resolves a phase decision, not on every 30 Hz tick.

For progress or safety without viewing another image:

```sh
PY ROOT/executor/client.py --out OUT status
```

Thinking may overlap the current motion. Read a recent snapshot, decide the next
short action and submit it while the old chunk finishes; the newer chunk pre-empts
the old one. If reasoning takes longer, the executor holds. Do not infer success
from a target or from `ran_fully`: that means all chunk steps were streamed, not
that the robot achieved the target. Always judge measured state and the images.

## Submit scene-conditioned chunks

Every chunk must state the current phase, a one-line reason tied to the observed
scene, its snapshot id and at least one demo chunk with an explanation. Predict
the current motion yourself; never send a demo trajectory or a fixed sequence of
demo-relative movements. The helper expands your chosen endpoint into 30 Hz poses.

Example syntax only—replace all values using the CURRENT observation:

```sh
PY ROOT/executor/client.py --out OUT submit --snapshot snap_000012 \
  --phase approach --reason "Square tab is ahead and right of the open left jaws; approach above it" \
  --demo "243:75:open-gripper approach above tab" \
  --seconds 2 --delta 0.010 -0.015 0.010 --gripper 1
```

Choose either `--delta dx dy dz` from that snapshot's measured active pose or
`--goal x y z` for a new scene-specific absolute endpoint. Units are metres.
Orientation is retained unless `--quat qw qx qy qz` is explicitly supplied. All
wire gripper values are hardware convention: **1=open, 0=close**. Omitting
`--gripper` preserves the current command, not the partial-contact readback.
The idle arm's full pose and gripper command are included automatically. Each
chunk lasts 1–3 seconds and may move at most 65 mm from its snapshot. Use about
10–30 mm/s away from objects and 2–10 mm/s for final grasp/insertion adjustments.

The helper returns the chunk id and acknowledgement path. Read its acknowledgement
if execution is uncertain or rejected. A rejected chunk never runs; correct your
prediction from a new snapshot instead of changing the executor. Snapshot inputs
older than 30 seconds are rejected. When pre-empting, use a new snapshot, account
for the ongoing movement and avoid sending a trajectory that unnecessarily
returns to an earlier point. Do not submit several future chunks at once: only
the newest is kept, and each must use current observations.

## Phase decisions

1. **Approach:** locate the square, protruding tab, peg and robot jaws in the head
   view. Predict an approach above the tab with an open gripper. Retain orientation
   unless the observed tab demands adjustment. The notes' absolute poses describe
   another scene; use them for relative geometry only. Accept 10–20 mm sag in coarse
   positioning, and use the wrist view for fine alignment rather than chasing an
   exact commanded pose while the arm has settled.
2. **Align and grasp:** in fish0, get the TAB between both jaws with enough forward
   depth. Earlier attempts closed in front of it twice; another 2.5–3 cm forward
   from the second miss worked. Apply this lesson to the current visible gap, not
   as a blind fixed advance. Submit a stationary 1–2 s closing chunk, then inspect
   the readback and wrist image. A settled 0.0 means a miss; about 0.4 was successful
   contact in the earlier run. Intermediate closure alone is not proof of a grasp.
   If empty, reopen, visually correct the gap, and try again conservatively.
3. **Verify and lift:** raise a few centimetres with the gripper closed; confirm the
   square follows and closure remains intermediate. Record a positive grasp
   judgement only now. Then obtain enough clearance over the peg, typically around
   9–11 cm above the live grasp height, split into observed chunks if needed.
4. **Transfer:** move the held square toward the live peg while clear of its top.
   In the earlier left-arm scene, decreasing y moved toward the peg. Infer current
   direction and distance from the head view; do not repeat its absolute path.
5. **Align and lower:** use fish0 to center the square OPENING over the post, rotate
   only as needed, and ensure the cap fits within the opening. Small lateral
   corrections precede small downward motions. Stop lowering if an edge meets
   the post, contact slips, or alignment becomes unclear. A typical demo lowered
   roughly 25 mm from its transfer peak, but current insertion geometry governs.
6. **Release and verify:** once the square is visibly supported on/around the peg,
   open, retreat several centimetres and check that it stays on the peg. Head
   view plus wrist if necessary. Do not call a lift a placement or an intermediate
   gripper reading a successful insertion. If grasp is confirmed but placement
   is not, report partial.

Log grasp, placement and other consequential judgements explicitly:

```sh
PY ROOT/executor/client.py --out OUT judge --kind grasp --confirmed yes \
  --snapshot snap_000042 --reason "Square followed the short lift; closed gripper remains at 0.40"
```

Replace the example with your actual evidence. `--kind placement`, `phase`, and
`note` are also available. Keep reasons to one line; do not log private chain of
thought. The executor records these judgements without deciding the task itself.

## Finish and account for usage

Finish on success, inability to proceed safely, budget pressure or executor stop.
Do not restart a stopped executor inside this attempt. Stop through the helper:

```sh
PY ROOT/executor/client.py --out OUT finish --outcome partial \
  --reason "Verified grasp and lift; placement could not be confirmed"
```

Use exactly one of success / partial / failure and your true reason. If your usage
limit is about to cut off the attempt, finish with `--outcome failure --reason
usage_limit`, as the protocol requires. If actual session totals are exposed,
append `--tokens N --usage-source "session counter"` (or `--credits N`). Do not
invent actual totals. `agent_usage.json` always contains a separate per-attempt
estimate: context/text characters / 4, plus 256 tokens per viewed image and a
512-token planning allowance per submitted decision. These are workload-estimate
assumptions, not model billing facts; image encoding, hidden reasoning and repeated
history may make real usage substantially different.

Verify `status` reports a non-null `meta.end_time`, and poll the retained process
handle until it exits. If the helper cannot stop a still-running executor, send
SIGTERM to that retained executor process only; never touch Pi/Orin processes.
The helper also records your outcome after a safety stop. End your answer with
**success**, **partial**, or **failure**, the brief observed reason, the attempt
log path, and the actual token total or clearly labelled estimate. Do not edit
frozen files or carry information from this attempt into the next scored session.
