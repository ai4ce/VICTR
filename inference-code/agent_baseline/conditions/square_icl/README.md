# Square on peg: async agent executor

The agent chooses scene-conditioned motions; the executor streams and checks them.
There is no autonomous task policy or model invocation in the executor. The previous
autonomous `pipeline/` and its practice logs are historical, excluded from this repo.

For a fresh attempt, give the agent **[attempt_prompt.md](attempt_prompt.md)** and a
new absolute output directory, e.g. `practice/async_main` under this directory.
The operator resets the scene and enables the joystick loop before the attempt.
The prompt contains the full startup, observation, submission and shutdown commands.
Use `/home/lairlab/icl_ws/openpi/.venv/bin/python` for all scripts. No installation
or robot-side change is needed.

## Components

- `executor/run_executor.py`: local subscribers, snapshot/inbox workers and 30 Hz
  command publisher on port 5572. Newer chunk ids pre-empt unfinished chunks.
- `executor/client.py`: select snapshots, submit 1–3 second chunks, record judgments,
  report usage and finish. The helper interpolates the agent's chosen endpoint;
  it does not select a task motion.
- `executor/safety.py`, `core.py`, `streams.py`: geometry, scheduling and transports.
  `vision.py` retains optional classical image-processing helpers from the earlier
  pipeline; it is not used by the default execution path.
- `notes/demo_analysis.md`: the 11 demos' numeric analysis, phase references and
  hand-guided grasp lessons. `demo_statistics.json` and `demo_index.json` preserve
  numeric evidence and valid references. `analyze_demos.py` documents its derivation
  from the previously extracted raw arrays; it is not run during attempts.

## Safety and execution

Freshness uses monotonic time since receipt on this PC. Source timestamps are only
metadata. Camera age >1 s or state age >0.5 s discards the unfinished chunk and
enters hold; packet age >2 s stops. With stale state, publishing pauses until fresh
state arrives. Startup waits up to 10 seconds for streams. Gate-off, failed IK,
invalid values, tracking >35 mm (or 15°), idle drift >25 mm (or 10°), workspace
violations, or the 600-second deadline stop execution. No automatic restart.

Each emitted step is bounded to 5 mm/2° from the previous command (initially the
measured pose), with the independent 35 mm measured tracking limit. A submitted
chunk is limited to 65 mm from its observation. The idle arm retains its measured
start anchor. Workspace bounds are in `safety.py`.

Empty queues hold the measured pose captured **once when entering hold**. Repeatedly
resetting the anchor to readback would follow gravity sag downward. Gripper commands
persist across holds: a contact reading of 0.4 does not replace the close command 0.
Hardware gripper convention is **1=open, 0=close**. The notes explain the raw-pool
convention discrepancy and the user's prior authorization to use the observed
majority convention.

There are no fixed phase distance thresholds: the agent judges live state and
images. The prompt explicitly accommodates sag and requires wrist confirmation
for grasp and placement. All frames presented to the agent are at most 320 px wide.

## Audit trail

The output directory must not exist. The executor creates:

| File | Meaning |
|---|---|
| `meta.json` | Task, commit, prompt SHA256, start/end, exit reason and outcome guess |
| `chunks.jsonl` | Full raw and parsed submissions followed by lifecycle status rows |
| `commands.jsonl` | Exact wire JSON and topic for every publish, chunk id and step |
| `state.jsonl` | Every received state packet and its local receipt time |
| `events.jsonl` | Gate, hold, phase, judgment, pre-emption and safety events |
| `frames/`, `snapshots/` | Retained downscaled frames and immutable observation JSON |
| `latest.json` | Atomically replaced newest snapshot |
| `inbox/`, `processed/`, `acks/` | Atomic submissions, preserved inputs and latest acknowledgments |
| `agent_usage.json` | Image/decision counts, wall time, token estimate and optional actual totals |

Join chunk lifecycle rows by `chunk_id`; a full submission survives even if no
steps execute. Final dispositions are `ran_fully`, `preempted` or
`rejected_by_safety`. Holds have null chunk id/step. `ran_fully` describes published
steps, not physical task success. Every helper-selected image is retained and
recorded as an `agent_view` event. `client.py finish` records the agent's outcome;
the agent then verifies `meta.json` has an end time and the executor has exited.

## Per-task attempt usage

The token estimate is a reproducible workload proxy:
`ceil((prompt + notes + observed text characters)/4 + 256*image views + 512*submitted decisions)`.
It is not a billing measurement; hidden reasoning, actual image encoding, and
repeated/cached history are unknown. Actual `tokens` and `credits` remain null
unless supplied from session accounting. No credit estimate is invented.

Aggregate only the relevant attempt cohort:

```sh
openpi/.venv/bin/python agent_runs/astra/put_the_square_on_the_peg/executor/summarize_usage.py \
  agent_runs/astra/put_the_square_on_the_peg/eval
```

This prints each attempt's estimate and counts, per-task totals and mean estimated
tokens per attempt. Actual aggregate totals remain null if any attempt lacks them.
Use `practice` instead of `eval` for practice accounting; do not mix cohorts.

## Validation and practice sequence

From `/home/lairlab/icl_ws`:

```sh
openpi/.venv/bin/python -m unittest discover \
  -s agent_runs/astra/put_the_square_on_the_peg/executor -p 'test_*.py' -v
```

Tests mock all robot connections and publishing. They exercise safety thresholds,
clock independence, sag, holds, pre-emption, file inbox/snapshots and exact logging.
They do not establish physical task success.

Build complete → operator runs main practice in a fresh session → one correction
pass → operator runs recovery → final commit and freeze. The old autonomous hashes
are void; there must be no active `FROZEN` marker during this build/practice stage.
After recovery, `FROZEN` will contain two lines: full commit hash, then the output
of `sha256sum attempt_prompt.md`. The executor checks both if the marker exists
and always requires a clean tracked repository. Scored attempts never edit code,
helpers, notes or the prompt.
