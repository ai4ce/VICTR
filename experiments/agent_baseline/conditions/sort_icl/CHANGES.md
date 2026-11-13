# Sort ICL harness changes against 125ded1

Repository: `agent_runs/astra/sort_the_items_into_their_containers/`. The original build was committed as `7f1e4b1680aa0bce40fe5f4ab2516a61874a506c`, with freeze-marker commit `3a084315969d7aacab50290592489b368770c575`. The revision below is operator-authorized maintenance between attempts. No robot connection, command publication, or motion was used to validate this revision.

## Idle-limit revision — 2026-10-13

- `Safety.check` permits idle-arm drift up to 40 mm / 15° from its anchor. Its measured-versus-previous-command tracking check now uses 40 mm / 15° for the idle arm and retains 35 mm / 15° for the active arm.
- `Safety.command` uses the same per-arm position limits for proposed-command-versus-measured tracking; its angular limit remains 15° for both arms. Error messages report the applicable limit. Before arm selection, both arms retain 35 mm / 15° tracking.
- `attempt_prompt.md` describes both idle tracking checks and the anchor drift limit. Arm switching, re-anchoring, gripper retention, streaming, freshness, workspace, and other limits are unchanged.
- Tests update the idle drift boundaries and cover both arm choices: idle tracking passes at 39 mm and stops at 41 mm, active tracking stops at 36 mm, and both retain the 15° boundary. The switch test checks 39 mm directly from the new anchor.

**Evaluation history:** `eval/01`, `eval/01_retry1`, and `eval/02` ran under the original **25 mm / 10° idle drift** and **35 mm / 15° tracking for both arms**, pinned to `7f1e4b1` by the original FROZEN marker. `01` failed startup without motion. `01_retry1` and `02` aborted on idle drift at approximately **26 mm** and **25 mm**, respectively. Their existing logs and scores are unchanged; these runs must not be described as using the revised limits. `PIPELINE.md` records this distinction alongside the executor safety description.

**Validation:** all **25 tests passed in 6.050 seconds**, including the original 23 and two new tracking tests. The four mocked integration runs cover both arm orders, switching, holds and stale stops without network sockets or real publication. The fresh, initially nonexistent output root was `checks/offline/idle_limits_1791913880928/`; its `test_results.json` retains the exact command, result, and test output. Expected abort messages belong to integrity and stale-stream assertions.

Re-freeze procedure: commit this revision, write that commit and the current `attempt_prompt.md` SHA-256 to FROZEN, commit only FROZEN, then remove all write bits from `executor/`, `notes/`, `attempt_prompt.md`, and FROZEN. The baseline diff counts and build-time checks below describe the original `7f1e4b1` build, before this idle-limit revision.

Baseline: `125ded1787e2ce9bc87bf3d919c121f524929f6f` in the authorized square zero-shot repository. All eight `executor/*.py` files were compared byte-for-byte with `git show 125ded1:executor/<file>` before copying. The authorized prompt and CHANGES were read; no other task's eval/checks, excluded agent directory, other-task demos, past sessions, or pool pickle were read.

## Original executor diff

The full diff is `checks/executor_diff.patch`; per-file baseline/current SHA-256 hashes and counts are in `checks/executor_comparison.json`.

| File | Added / removed lines | Change |
| --- | ---: | --- |
| `executor/client.py` | +20 / −10 | Require an explicit active arm on every submission; permit either arm. Use the current idle anchor for same-arm chunks, snapshot measured pose for a proposed switch, and original anchor for initial selection. Require repeatable `--demo episode:frame:how`, parse references, include them in the submission and text accounting. |
| `executor/core.py` | +36 / −8 | Restore per-chunk demo validation against episodes 89–99 and the frozen length index. Accept arm switches and log `active_arm_switched` with old/new arm, new idle anchor, retained gripper command and chunk ID. Anchor the newly idle arm to the supplied current measured state. Copy the accepted chunk before filling its idle targets with that anchor and last gripper command; the logged submission remains exact. Preserve original start anchors and expose the current idle anchor in summaries. |
| `executor/run_executor.py` | +15 / −9 | Set the exact sort task string and 3600-second timeout, load `notes/demo_index.json`, pass the index through validation, retain the idle gripper's last command, and re-read/check current state immediately before accepting a switch. Initial context accounting includes the prompt and demo analysis. |
| `executor/safety.py` | +4 / −2 | Change the default timeout to 3600 seconds. Store a separate idle anchor; apply the unchanged drift check to the switch anchor, retaining initial-anchor behavior before the first switch. |
| `executor/test_executor.py` | +161 / −27 | Adapt the original suite to per-chunk selection, switching and required demo references. Add boundary/invalid demo cases, per-submission demo text accounting, the 3600-second boundary and runtime argument, measured re-anchoring after snapshot capture, gripper retention, unchanged start anchors and drift from the new anchor. Extend mocked integrations to actually switch arms. |
| `executor/files.py` | +0 / −0 | Byte-for-byte unchanged. |
| `executor/streams.py` | +0 / −0 | Byte-for-byte unchanged. |
| `executor/summarize_usage.py` | +0 / −0 | Byte-for-byte unchanged. |

Total: five files changed, **236 added / 56 removed lines**, including tests. Production code accounts for 75 added / 29 removed lines. The existing `Scheduler.tick`, `sent`, `discard`, and `status`, `Safety.command`, and every `Workers` method were additionally compared byte-for-byte with the baseline and are unchanged.

Only the requested arm choice, anchors/gripper retention, demo references/accounting, task and deadline behavior changed. The helper's snapshot/status/judge/finish/usage branches and the existing metadata/outcome-guess behavior are retained. In particular, a confirmed grasp still sets a provisional partial outcome guess; the prompt explicitly requires correcting the final outcome using sort containment scoring.

## Original preserved safety and execution

- 30 Hz streaming, 1–3 second chunks, 30–90 steps, 65 mm observation-relative displacement, and 5 mm / 2° bounded command steps.
- Workspace boxes, quaternion/value validation, 35 mm / 15° tracking checks, 25 mm / 10° idle drift checks, and the 1 mm / 0.5° idle-target and 0.01 idle-gripper tolerances.
- Local receipt freshness: camera >1 s or state >0.5 s causes hold/discard; stale state suppresses publication; receipt age >2 s stops. Required cameras and 30-second snapshot limit are unchanged.
- Gate-off, IK failure, worker failure, signals and integrity stops; newer accepted chunks pre-empt; duplicates do not; holds preserve measured pose and preceding gripper commands. Initial pre-selection hold still uses original start poses.
- Existing log files, exact submitted payload logging, exact published-wire logging, snapshot dimensions/cadence, usage-estimate formula, and two-commit freeze verification.

The executor never assigns simultaneous independent motions to both arms. A switch changes which arm receives the predicted motion; the other arm receives its measured switch anchor and retained gripper command, through the existing bounded command path. No task policy or demo replay is added to the executor.

## Demo context and attempt prompt

`notes/analyze_demos.py` uses `openpi/.venv/bin/python`, reads raw sort parquet episodes 89–99 one at a time, verifies flags/lengths/frame indices, computes command and measured-joint/FK statistics, and extracts downscaled video keyframes. It produces `demo_index.json`, `demo_statistics.json`, and `keyframes/`. The human-readable interpretation is `notes/demo_analysis.md`. All 94 explicit episode:frame references in the analysis were checked against the index.

The eleven episodes last 76.6–194.9 seconds and show nine task items each, with a front-center plate, left cup and right box. The analysis covers inventory, regional arm assignments, item ordering, relative pick/lift/transfer/drop geometry, durations, speeds, gripper ramps and retries. Episode 98 provides a per-item table for all nine items. All reported heights are relative to a pick or another phase, never live target poses. The JSON retains 123 candidate grip cycles, including failed/repeated adjustments; those are not 123 successful placements.

Evidence caveat: raw gripper transitions in episode 98 conflict with PIPELINE's stated dataset convention. The block-carry wrist sequence supports low-value carry intervals and rising-value release. The notes document concrete frames and raw thresholds; no raw values are replayed or inverted, and the live helper remains 1=open, 0=closed. This discrepancy is visible in the review rather than silently treated as a reliable label.

`attempt_prompt.md` adapts the square zero-shot prompt's seven-section structure. It includes the exact task query; any-container final k/N scoring; reading the analysis once before movement; per-chunk arm, snapshot, phase, reason and demo reference; one-item grasp/lift/place/verify loop; explicit per-item judgements; continuation after failures; final containment recount; deadline/stop rules; approximately 25 image views per item with head by default and wrist for grasp/drop; and isolation to this prompt, ROOT notes, own OUT and helper output. No other attempt or raw demo access is permitted during scoring.

## Original checks completed without motion

Saved command (run from `/home/lairlab/icl_ws`):

```bash
SORT_OFFLINE_ROOT=/home/lairlab/icl_ws/agent_runs/astra/sort_the_items_into_their_containers/checks/offline \
openpi/.venv/bin/python -B -m unittest discover \
  -s agent_runs/astra/sort_the_items_into_their_containers/executor \
  -p test_executor.py -v
```

**23 tests passed in 6.022 seconds.** All 18 original tests remain, with the arm-lock test adapted to the requested switching behavior, plus five additional tests. `checks/unit_tests.txt` contains the full result and `checks/results.json` the command and status. Expected abort messages come from stale-stream and freeze-integrity assertions.

Four retained offline runs under git-ignored `checks/offline/` exercise left→right and right→left switches, pre-emption, holds, gripper retention, both camera- and state-stale recovery/discard, and >2-second stale stops. Each uses the actual executor main loop, workers, logger and client, with **both Streams and the ZMQ context replaced by in-memory mocks**. No socket, command publisher or robot connection exists in these runs. The console's inherited “publishing at 30 Hz” message refers to that mocked publisher. The two normal runs finish; the two stale runs stop. Each logs exactly one switch; both stale runs log a safety stop.

One successful read-only live observation is saved in `checks/live_check.json`: state and zed/fish0/fish1 readable, gate readable and **disabled**, state receipt age 0.000014 s, camera response receipt age 0.0321 s, and port 5572 free before and after. No poses or camera images from the live scene are saved. Gate enable was neither requested nor changed.

Preflights: sandbox netlink inspection was denied; the authorized external retry confirmed the port was free but stopped at a deliberately strict SUB-only guard before camera access. Inspection of the installed commlink showed it uses a DEALER rather than plain SUB, and buffered mode binds an internal inproc control socket. The final check uses commlink's read-only PULL mode, permits only camera PULL messages and state SUB sockets, prohibits all binds and publisher socket types, and succeeded. The guard failure is retained in `checks/live_check_preflight.json`. No preflight or successful check bound a socket or published robot commands. The production `streams.py` was not changed.

## Original operator review and freeze

The original build stopped for review before attempts. After operator approval, the two commits named above recorded the reviewed harness and its FROZEN marker using this procedure:

1. Commit the reviewed ROOT files.
2. Write that commit's `git rev-parse HEAD` to line 1 of ROOT/FROZEN and `sha256sum attempt_prompt.md` to line 2.
3. Commit only FROZEN.
4. Apply `chmod -R a-w ROOT/executor ROOT/notes ROOT/attempt_prompt.md ROOT/FROZEN`.

Temporary fixture commits used by the freeze tests do not create commits in ROOT. The same two-commit procedure is used for the operator-authorized idle-limit revision.
