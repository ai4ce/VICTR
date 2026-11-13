# Zero-shot harness changes against ea4cd86

Review state: new, independent ZS git repository; no ZS commits, no ZS/FROZEN, and no read-only chmod applied. Approval is pending. No task attempt or robot motion was performed.

Baseline scope: copied the eight permitted `executor/*.py` files. Each source file was byte-for-byte checked against `git show ea4cd86:executor/<filename>` before modification. `vision.py` was neither read nor copied. No original prompt, demo index, notes, recordings, or other excluded content was inspected or copied. This document compares only the authorized executor files; it does not infer anything about excluded files at that commit.

## Every executor change

| File | Added / removed lines | Change |
| --- | ---: | --- |
| `executor/client.py` | +11 / -10 | Remove demo arguments, parsing, chunk references, and demo-character accounting. Add `submit --active-arm`, required before selection and otherwise inherited from the snapshot. Reject mismatches locally. Obtain the idle target from the selected arm's opposite start anchor. |
| `executor/core.py` | +23 / -13 | Remove the demo-index parameter and demo-reference checks. Keep required phase, reason, snapshot ID and all motion validation. Validate the proposed arm, allowing an initially unselected attempt. Use both recorded start anchors. Lock the arm inside accepted scheduling, emit `active_arm_selected` once, reject arm changes without pre-emption, and return acceptance status. Hold original start poses before selection, including after a transient stale hold. Make summaries safe before selection and expose `start_anchors`; `idle_anchor` stays null until selection. |
| `executor/run_executor.py` | +16 / -11 | Remove the demo-index load and executor launch `--active-arm`. Start with null arm metadata; persist selection after acceptance. Validate the prospective idle gripper before the first lock. Count only the local attempt prompt as initial context. Adjust freeze validation for the requested code commit followed by a FROZEN-only commit. |
| `executor/safety.py` | +6 / -5 | Default to no selected arm and no idle arm. Apply the existing idle-drift check once selection assigns the idle arm, retaining its original measured start anchor. All other checks and all numeric thresholds remain unchanged. |
| `executor/test_executor.py` | +266 / -47 | Remove demo fixtures and task-specific test descriptions; adapt helper/validator calls. Retain existing safety, hold, freshness, interpolation, logging, and usage tests. Add arm-selection/locking/rejection tests, original-anchor checks for both choices, invalid-first-chunk checks, the two-commit freeze workflow and integrity rejections, helper-driven mock runs for both arms, and camera/state stale-hold/recovery/stop runs. Optional `ZS_OFFLINE_ROOT` preserves synthetic logs. |
| `executor/files.py` | +0 / -0 | Byte-for-byte unchanged. |
| `executor/streams.py` | +0 / -0 | Byte-for-byte unchanged. |
| `executor/summarize_usage.py` | +0 / -0 | Byte-for-byte unchanged. |

Total permitted executor diff: five changed files, 322 inserted lines, 86 removed lines; three files unchanged. Per-file baseline/current SHA-256 values are in `checks/results.json`.

No demo index exists in ZS. No runtime demo-index dependency remains. Generated chunk submissions contain no `demo_chunks` field. The `demo_chunks` string in tests only asserts its absence.

## Repository, prompt, accounting, and freeze

- New `attempt_prompt.md`, written from scratch: the seven requested operational sections, task/credit rules, exact executor/helper interfaces, isolation, generic manipulation guidance, fixed safety limits, camera budget, judgements, finishing, and honest usage reporting. It contains no chosen arm, scene positions, task-specific grasp details, or learned motion parameters.
- New `.gitignore`: ignore scored output under `/eval/`, synthetic build logs under `/checks/offline/`, and Python bytecode. Attempt output does not dirty the frozen repository.
- The inherited `ROOT = Path(__file__).resolve().parents[1]` now resolves to ZS. Thus the existing git cleanliness check, prompt hash, and FROZEN lookup apply to ZS without any reference to the original repository. Initial context accounting reads only `ZS/attempt_prompt.md`.
- Freeze adaptation: writing HEAD into FROZEN and then committing FROZEN makes that recorded hash the parent of the new HEAD. Accept that form only when the recorded commit is exactly `HEAD^` and the full parent-to-HEAD diff changes only `FROZEN`. Continue checking prompt SHA-256 and repository cleanliness. The prior exact-HEAD form remains supported. Temporary test repositories verify acceptance of the prescribed two-commit form and rejection of dirty/untracked files, a bad prompt hash, extra changes in the marker commit, and subsequent commits. These fixture commits do not commit anything in ZS.
- New `checks/read_only_live.py`: one-purpose build check using only state SUB/connect and camera Subscriber; checks `ss -ltnp`, gate readability, state freshness, and all four camera images' decodability/freshness. It creates no publisher, binds no command port, saves no poses or camera images, and changes nothing on the Pi/Orin.
- New `checks/live_check.json`: successful read-only check metadata, including receipt ages and the free port result.
- New `checks/results.json`: test results, per-file diff counts/hashes, synthetic log inventory, live-check results, and review state.
- New `CHANGES.md`: this change inventory and validation record.
- Generated, git-ignored `checks/offline/`: four synthetic runs with meta, submitted/status chunks, exact would-be wire commands, synthetic state, events, 320-pixel frames, snapshots, acks, usage, final reports, and processed requests. No real stream recording is used.

## Preserved execution and safety

Production motion transport and scheduling cadence are unchanged: 30 Hz; newer accepted chunks pre-empt; empty queues hold with the preceding gripper commands. Both arms remain commanded. Snapshot images remain bounded to 320 pixels. All snapshot/status/submit/judge/finish commands and the usage command remain available.

Freshness still uses local receipt time: camera age over 1 second or state over 0.5 seconds causes hold/discard; state-stale hold suppresses target publication; any stream age over 2 seconds stops. The 600-second deadline, 1–3 second chunks, 65 mm observation-relative limit, 5 mm/2-degree command steps, 35 mm/15-degree tracking checks, 25 mm/10-degree idle-drift check, idle target/gripper tolerances, workspace bounds, gate-off, IK, and invalid-value stops are unchanged. Numeric constants in `core.py` and `safety.py` were compared with the baseline as an additional check. Existing logging channels are retained.

The only pre-selection safety distinction follows the requested arm choice: there is no designated idle arm until acceptance. Both start poses are held and ordinary tracking, gate, IK, value, workspace, freshness, and deadline checks apply; selection activates the unchanged drift check on the other arm.

## Validation completed without motion

Command from `/home/lairlab/icl_ws`:

```bash
openpi/.venv/bin/python -B -m unittest discover -s agent_runs/astra/put_the_square_on_the_peg_zeroshot/executor -p test_executor.py -v
```

Result: **18 tests passed in 5.979 seconds**. The saved run also set `ZS_OFFLINE_ROOT` to ZS/checks/offline. Leave this environment variable unset for disposable reruns, or supply a new empty path; existing outputs are never reused.

The integration tests run the real executor main loop, workers, file logger, and helper against synthetic state/blank images. `Streams` and the publisher are mocked: the apparent `bind`/`send` calls are in-memory methods and cannot bind 5572 or send to the robot. Checks cover pre-selection status/snapshots and both-arm holds; either arm's first acceptance; rejection of another arm without pre-emption; pre-emption by the next valid chunk; previous-gripper retention; camera and state stale holds, recovery without resuming discarded steps, and stops beyond 2 seconds; all log artifacts; prompt-only accounting; finish handling; and process return. The stale-stop messages in test output are expected assertions.

The read-only live check passed: gate readable and enabled, state receipt age about 0.000004 seconds, all four cameras about 0.048 seconds old, and port 5572 free. The first sandboxed check could not access the network/netlink inspection; the explicitly permitted retry outside the sandbox succeeded. There was one successful live check, no live publisher, no bind to 5572, no task execution, and no practice.

The source and new helper `--help` interfaces were checked. No excluded files, other pipeline sections, datasets/context pools, past sessions, or excluded git history were read.

## Pending operator review

Do not run a scored attempt yet. After approval and any requested edits: commit the reviewed ZS files; write that commit's `git rev-parse HEAD` to line 1 of ZS/FROZEN and `sha256sum attempt_prompt.md` to line 2; commit only FROZEN; then apply `chmod -R a-w` to ZS/executor, ZS/attempt_prompt.md, and ZS/FROZEN. None of these ZS commit/freeze/permission steps has been performed.
