# Zero-shot sort harness — operator review

Prepared in its own new Git repository at
`/home/lairlab/icl_ws/agent_runs/astra/sort_the_items_into_their_containers_zeroshot/`.
The build is uncommitted and awaits operator review. No attempt, practice,
robot motion, command publisher, or gate enable was run. `FROZEN` will be
created after approval; no placeholder freeze marker is present.

## Baseline and executor diff

Baseline: the sort ICL repository's `FROZEN` line 1,
`331ea9ad3c4e2e10173179173db0241d372f65a8` (`331ea9a`). All eight executor files
were copied from that checkout. Before editing any copied executor file, both
the source and the copy were compared byte-for-byte with
`git show 331ea9a:executor/<file>`. All matched. The source checkout was not
modified.

Evidence:

- [Baseline byte verification](checks/baseline_verification.json) records each original hash and size.
- [Full executor diff](checks/executor_diff.patch) contains every changed executor line against `331ea9a`.
- [Per-file comparison](checks/executor_comparison.json) records original/current hashes, line counts, and unchanged components.

| File | Diff against `331ea9a` | Change |
| --- | --- | --- |
| `executor/client.py` | +4 / -13 | Remove `--demo`, its parser, the `demo_chunks` argument/field, and demo-reference text accounting. All other helper behavior is unchanged. |
| `executor/core.py` | +1 / -10 | Remove the demo index parameter and required reference validation. The entire `Scheduler` class is byte-identical. |
| `executor/run_executor.py` | +4 / -5 | Remove demo-index loading/argument plumbing and count only the local attempt prompt as initial context. |
| `executor/test_executor.py` | +58 / -53 | Remove demo fixtures and demo-reference tests; adapt callers and initial-context assertions. Add accepted no-demo CLI submissions with exact text-accounting checks and explicit preselection tracking coverage for both arms. Keep the integration rejection/pre-emption check using invalid required phase metadata. Assert chunk logs omit `demo_chunks`. |
| `executor/files.py` | Byte-identical | No change. |
| `executor/safety.py` | Byte-identical | No change. |
| `executor/streams.py` | Byte-identical | No change. |
| `executor/summarize_usage.py` | Byte-identical | No change. |

The baseline's sort task string, 3600-second deadline, 30 Hz streaming,
pre-emption, hold, freshness rules, workspace boxes, gripper retention, arm
switching, and switch-time idle re-anchoring remain unchanged. Measured and
proposed tracking remain 35 mm / 15 degrees for the active arm, 40 mm /
15 degrees for the idle arm, and 35 mm / 15 degrees for both arms before
selection. Idle anchor drift remains 40 mm / 15 degrees.

The worker/logging code and freeze-validation block are byte-identical. The
executor derives ROOT from its own file location, so repository cleanliness,
the code commit, and the prompt hash are checked against this zero-shot
repository and its local `attempt_prompt.md`. The existing two-commit freeze
validation is preserved and tested. Chunk submissions still contain phase,
reason, snapshot ID, and the complete predicted steps; the helper no longer
creates `demo_chunks`.

## Prompt isolation and allowed content

[attempt_prompt.md](attempt_prompt.md) uses the seven-section structure and
generic interface/manipulation guidance of the square zero-shot prompt at
`125ded1`. The sort ICL prompt is used only for arm-switching mechanics,
the associated enforced safety limits, and lenient final-containment scoring.

Content review confirms that the prompt contains no learned task knowledge:
no demo statistics or references, example trajectories or task poses, learned
heights, arm assignments, item order, container locations, prior-attempt
lessons, or hand-guided/session knowledge. Its numeric motion values are
executor/interface limits, not scene-derived targets. The task query and
requested scoring rule are the only task-specific instructions. Positions,
item identities/count, reachability, clearances, and action choices must come
from the live observations of a future fresh session.

The prompt states that every chunk names `--active-arm`, one arm moves at a
time, and switching anchors the newly idle arm at its acceptance-time measured
pose while retaining its last gripper command. It states all sort safety limits,
freshness rules, local ROOT, and operator-supplied `ROOT/eval/NN`. It reports
success only when every item is inside any container at the end, partial as
k/N, and excludes items subsequently knocked out. It also explains that the
inherited provisional grasp outcome is not final sort credit.

Attempts may read only this prompt, their own OUT, and helper output. The
prompt explicitly excludes PIPELINE.md, the ICL sort directory, both square
directories, `agent_runs/astra_2026-10-09/`, assets, datasets, pools, and past
sessions. The frozen executor/helper may be executed, but their sources and
these build/check documents are not attempt context. No `notes/`, demo index,
demo statistics, analysis, or keyframes were copied.

## No-motion checks

[checks/unit_tests.txt](checks/unit_tests.txt): **25 tests passed**, exit 0,
using `openpi/.venv/bin/python -B -m unittest discover -s <ROOT>/executor -p test_executor.py`.
The exact command and a fresh, nonexistent `SORT_OFFLINE_ROOT` under
`checks/offline/` are recorded at the top. The log constructor created each
mock output directory; no existing output directory was reused.

Coverage includes both arm choices and switching, acceptance-time idle
re-anchoring, idle gripper retention, 40/35 mm and 15-degree limits, 3600-second
deadline, stale-stream holds/stops, pre-emption, wire/file logging, prompt-only
context accounting, and two-commit freeze integrity. Integration streams and
publishers are in-memory mocks; their console messages about publishing and
safety aborts are expected test output, not live execution. Freeze fixture
commits exist only in temporary test repositories. An initial new-test fixture
omitted `end_time`; it was corrected before the passing run, and that first
output is retained in ignored `checks/offline/unit_tests_initial.txt`.

[checks/live_check.json](checks/live_check.json): **one read-only check passed**
at 2026-10-13T18:55:04 UTC. It received three state packets and two camera
packets; `zed`, `fish0`, and `fish1` decoded successfully. The gate was readable
and **false**. Port 5572 was free before and after. No bind, publisher, command,
or gate-change request occurred. No images or measured task poses were saved.

[checks/read_only_live.py](checks/read_only_live.py) is the inspected check
source. It permits only state SUB and camera read sockets, rejects binds and
publisher creation, and restricts sends to camera PULL requests. Commlink
pull mode avoids the internal inproc bind made by its buffered subscriber.
The sandbox blocked the initial port inspection; the port inspection and the
single live check then ran with approved access outside the sandbox. No
executor was launched against live inputs.

## Operator labels and scene parity

`eval/labels.csv` exists with the header:

```csv
attempt,annotation,subtasks_completed,duration_s,note
```

The ICL sort checkout has no existing `eval/labels.csv`; this is the shared
sort label schema specified by PIPELINE.md. No result rows were invented.
The `.gitignore` matches the ICL sort repository: `eval/` and offline mock
outputs are ignored. The labels file stays writable and outside the frozen
Git contents so operator annotations do not make the executor repository dirty.

For all five future zero-shot attempts, the operator must reuse the same item
set, containers, and layout as the five ICL sort attempts. Record item
identities, N, and a layout photo reference or description in each label's
note. This build neither resets nor moves the scene and does not infer the
inventory from prior attempts. The scene parity requirement must be checked
by the operator before scoring; it is not established by a connectivity check.
Use the lenient final k/N rule for comparison with the ICL sort runs.

## Freeze only after operator approval

No commit or chmod has been performed in this repository. After approval,
review the final files and use this sequence from ROOT:

```bash
git add .gitignore CHANGES.md attempt_prompt.md executor checks
git commit -m "Build zero-shot sort harness from frozen ICL executor"
git rev-parse HEAD > FROZEN
sha256sum attempt_prompt.md >> FROZEN
git add FROZEN
git commit -m "Record zero-shot sort freeze"
chmod -R a-w executor attempt_prompt.md FROZEN
```

Then verify that FROZEN line 1 equals `HEAD^`, its prompt hash matches the local
file, the second commit changes only FROZEN, the working tree is clean, and
the requested paths have no write bits. This procedure does not authorize
an attempt, practice, gate enable, or robot motion. Each future scored
attempt requires a fresh agent session and an operator-supplied OUT.
