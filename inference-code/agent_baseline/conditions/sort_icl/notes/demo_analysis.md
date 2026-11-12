# Sort demonstrations: build-time context for one fresh ICL attempt

Read this once before moving. These are observations from **episodes 89–99 only**, not a trajectory or a list of live targets. Predict each action from the current scene. Reference examples as `--demo '98:219:lift a grasped block clear before transferring'`; use the example's relative geometry and intent, never its absolute pose. Frames are zero-based at 30 Hz.

## Evidence and a gripper-label discrepancy

`analyze_demos.py` reads one authorized parquet at a time, verifies episode ID, contiguous frame indices, length, keep/valid flags and finite arrays, computes action and measured-joint statistics, and converts joints to measured EE estimates with the workspace's `yor_fk.fk_states`. It reads no pool pickle, policy, other task, or past attempt. `demo_index.json` maps each allowed episode to its exclusive frame bound. `demo_statistics.json` contains all per-arm statistics and candidate pick/release intervals. The script also extracts six 320×240 head keyframes per episode (`keyframes/<episode>_overview.jpg`); episode 98 has additional head endpoints and eight wrist views for the detailed timing audit. No image is enlarged. These images were inspected at build time; they need not be opened during an attempt.

**Raw gripper values conflict with the dataset convention stated in PIPELINE.** PIPELINE describes dataset 0=open, 1=closed. In these raw sort files, the inspected video shows objects carried during the low-value interval: right gripper drops below 0.5 at **98:219**, the orange block is carried at **98:250**, the value rises to 0.5 over the box at **98:290**, and the block has left the gripper by **98:330**. The yellow block repeats this pattern at **98:446, 98:500, 98:587, 98:620**. These references are also in the right-wrist audit sheet. Therefore the analysis calls a high→low crossing a visually supported pick/close candidate, and the following low→high crossing a release candidate; the JSON keys `close` and `release` use that operational definition. This is an evidence-based interpretation of these files, not a correction to the frozen transport or an instruction to convert raw values.

**Live commands always use the existing helper convention: 1=open, 0=closed. Do not invert or replay demo values.** Use fresh imagery to verify the actual result. Threshold times are approximate phases of gradual commands, not contact timestamps. Typical full gripper ramps in episode 98 take about 20 frames (0.67 s); threshold crossing is around their midpoint. There is no object detector, contact sensor analysis, or per-frame human success annotation here. A candidate cycle can be a miss or recovery.

## Scene layout and inventory

All eleven inspected episodes use the same visible **nine task items**: orange block, translucent yellow-green block, purple eggplant-shaped item, red chili-shaped item, green pepper-shaped item, yellow elongated item, dark gray elongated item, and two white-bodied markers with red and blue/purple ends. Names describe appearance; the dark gray item's exact identity is uncertain. The count comes from the initial/final head views together, not the number of gripper cycles. Some items are occluded near the image bottom or by an arm at individual frames.

The three repeatedly used destinations are a white oval plate near the front center, a patterned cup to its left, and an open cardboard box to its right. The plate is a valid destination in this task. Demonstrated grouping is blocks→box, elongated/produce-like pieces→plate, markers→left cup. A second patterned cup farther right and background tools, bottles, batteries, plant, white fixtures and red plastic prop are visible, but the sampled demos do not establish them as task items. The current attempt's item set and N must be established from its own initial scene and operator setup.

Relative geometry: the plate is between the arms, the box is lateral to the right-side items, and the left cup has a tall, narrow rim relative to the plate. Blocks need rim clearance; the long markers need more vertical clearance and orientation control over the cup. Avoid treating a camera pixel direction as a calibrated Cartesian axis. Match each item's and container's current location with the head and relevant wrist view. Some demo items begin on the plate (notably the blocks at **99:0**); do not remove items already sorted under the current rule just to reproduce that demo.

The attempt scoring is deliberately more lenient than the demonstrations: **any container counts**. Prefer a safe reachable container with room. Recheck that every counted item stays inside at the end.

## Episode summary, arm use and order

Lengths are exact parquet/video frame counts. Paths are sums of successive commanded EE translation distances, including approach, retreat, corrections and idle jitter. They are descriptive, not motion budgets. Low-value cycle counts include retries. Every row has nine scene items.

| Episode | Valid frames | Duration s | Command path L / R, m | Pick candidates L / R | Observed order and useful head references |
| --- | --- | ---: | --- | --- | --- |
| 89 | 0–2549 | 85.0 | 3.54 / 3.01 | 5 / 4 | Right-side work first; left-side plate items and markers later. `89:510`, `89:1020`, `89:1529`, `89:2039`, `89:2549`. |
| 90 | 0–2629 | 87.7 | 3.02 / 2.75 | 4 / 6 | Right-side box work, then left cup/plate work, then right returns to remaining plate-side items. `90:526`, `90:1052`, `90:1577`, `90:2103`. |
| 91 | 0–2904 | 96.8 | 3.68 / 2.43 | 5 / 4 | Right-side items then left-side items; markers finish late. `91:581`, `91:1162`, `91:1742`, `91:2323`, `91:2904`. |
| 92 | 0–3588 | 119.6 | 4.97 / 3.59 | 6 / 5 | Right-side box/plate work precedes extended left cup and remaining plate work. `92:718`, `92:1435`, `92:2153`, `92:2870`. |
| 93 | 0–2541 | 84.7 | 3.36 / 2.90 | 4 / 5 | Right-side items first; left markers then remaining left-side plate items. `93:508`, `93:1016`, `93:1525`, `93:2033`. |
| 94 | 0–2750 | 91.7 | 4.35 / 2.69 | 6 / 4 | Right-side items first; long left marker work and short extra left regrasp near the plate. `94:550`, `94:1100`, `94:1650`, `94:2200`. |
| 95 | 0–5845 | 194.9 | 7.76 / 10.97 | 10 / 9 | Items spread across both sides; early left and right activity overlaps, followed by multiple corrections and late marker work. `95:1169`, `95:2338`, `95:3507`, `95:4676`. |
| 96 | 0–4875 | 162.5 | 6.75 / 10.84 | 7 / 10 | Right fills the plate early; marker and displaced box/block work continue much later. `96:975`, `96:1950`, `96:2925`, `96:3900`. |
| 97 | 0–2800 | 93.4 | 2.91 / 4.20 | 3 / 8 | Right box/plate work includes short repeated grasps, then left plate item and markers. `97:560`, `97:1120`, `97:1680`, `97:2240`. |
| 98 | 0–2296 | 76.6 | 2.82 / 3.48 | 3 / 6 | Orange block, yellow-green block, purple, green, gray, red with right; yellow elongated item and both markers with left. Detailed table below. |
| 99 | 0–2505 | 83.5 | 2.92 / 4.49 | 3 / 6 | Right moves blocks off the plate to the box, then other right-side pieces; left finishes markers and yellow item. `99:0`, `99:501`, `99:1002`, `99:1503`, `99:2004`. |

The demonstrator generally groups work by reachable region, rather than alternating arms for every item or imposing a universal color order. The right arm handles the right box and central/right plate approaches; the left handles the left cup and left plate approaches. Which arm handles a particular item changes with layout. Episode 95 shows overlapping bimanual activity in the recording; **the executor permits only one active arm at a time**. Learn the regional division without reproducing simultaneous motion. Switch only when the current scene supports it; remember an idle gripper can still hold an item.

## A concrete pick → lift → transfer → drop sequence: episode 98

Each row identifies a different item. Close/release are the raw-value threshold phases defined above. Heights are **relative to that row's EE height at the pick crossing**, not table height or live targets. Lift is the maximum command-height increase before release; FK lift is the corresponding increase estimated from measured joints. XY is the horizontal endpoint displacement magnitude, not total path length. Speed is the median commanded translation speed during the low-value carry interval.

| Item / arm | Pick → release refs | Carry s | Peak lift cmd / FK mm | Release above pick mm | XY transfer mm | Median speed m/s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Orange block / R | `98:219` → `98:290` | 2.37 | 73 / 75 | 25 | 84 | 0.064 |
| Yellow-green block / R | `98:446` → `98:587` | 4.70 | 107 / 108 | 65 | 91 | 0.063 |
| Purple item / R | `98:746` → `98:827` | 2.70 | 57 / 52 | 44 | 49 | 0.064 |
| Green item / R | `98:926` → `98:980` | 1.80 | 75 / 75 | 47 | 134 | 0.137 |
| Dark gray item / R | `98:1095` → `98:1175` | 2.67 | 52 / 46 | 16 | 113 | 0.042 |
| Red item / R | `98:1268` → `98:1325` | 1.90 | 71 / 77 | 38 | 150 | 0.104 |
| Yellow elongated item / L | `98:1488` → `98:1558` | 2.33 | 49 / 38 | −2 | 87 | 0.073 |
| Red-ended marker / L | `98:1723` → `98:1921` | 6.60 | 167 / 157 | 167 | 158 | 0.051 |
| Blue-ended marker / L | `98:2091` → `98:2244` | 5.10 | 219 / 204 | 202 | 251 | 0.089 |

The item assignments use the episode 98 head sequence and endpoints; the block carrying/release is additionally checked with the wrist frames. The first block approach starts with gripper opening and alignment, then the block travels with the wrist, clears the rim, lowers and releases over the box. **98:219→243→290** illustrates 73 mm rise, then about 48 mm descent from that peak. **98:446→491→587** similarly rises 107 mm then descends about 42 mm. These are short lateral transfers to a nearby box, not a move across the whole table.

The four right-side plate placements use roughly 52–75 mm command lift and 49–150 mm horizontal transfer, with release 16–47 mm above the pick EE height. Their carry intervals are about 1.8–2.7 s. The left yellow item has a low plate placement, roughly level with its pick height. Markers instead rise about 167–219 mm relative to their low tabletop grasp and are released high over the cup. Such clearances may need several live chunks, and the fixed workspace may prevent a demo-like path in a new scene; use a different safe container when appropriate.

Open/approach time between a previous release and the next pick includes retreat, reorientation and alignment: in episode 98 it is 5.20 s before the second block, 5.30 before purple, 3.30 before green, 3.83 before gray, 3.10 before red, and 5.50/5.67 before the markers. The left arm's first pick is at 49.6 s because the right arm worked first; that interval is not 49.6 s of approach. This distinction applies to every first-arm cycle in the JSON.

Across **all 123 candidate cycles**, including misses and recoveries, the 10th/50th/90th percentiles are: carry duration **1.54 / 2.73 / 8.22 s**; command lift **9 / 73 / 176 mm**; FK lift **8 / 64 / 166 mm**; XY endpoint transfer **23 / 101 / 231 mm**. For cycles that rise at least 30 mm, time from the pick crossing to that rise is **0.37 / 0.75 / 1.35 s**. These are descriptive distributions with mixed item types, not successful-grasp thresholds.

Per-arm full-episode command-speed 90th percentiles span about **0.082–0.168 m/s**; brief peaks are larger (for example 0.514 m/s in the last marker interval of episode 98). The frozen executor's 5 mm/tick and 65 mm/chunk limits take precedence. Command-versus-FK median position differences are approximately **28–60 mm** across episode/arm pairs, which is another reason not to replay commanded absolute poses or equate a target with achieved contact. Measured-path totals range about **2.15–10.14 m** across arms/episodes. Per-joint measured ranges and all FK checks are retained in the JSON.

## Recoveries and useful decision references

- **Repeated grasp adjustment:** `94:2167→2234`, `94:2280→2308`, `94:2386→2447` are successive left low-value intervals near the late plate work; the middle one lasts only 0.93 s. `97:1277→1302` and `97:1373→1400` are brief right intervals before a longer `97:1451→1520`. More cycles than items supports retry/regrasp activity; it does not certify the exact reason for each miss. Use a fresh wrist view, reopen if appropriate, realign and verify with a lift.
- **Long or awkward item handling:** `95:158→602` and `95:1358→1714` last 14.8 and 11.9 s on the left. `95:3951→4748` lasts 26.6 s on the right. Head views `95:1169`, `95:3507`, `95:4676` show the scene changing and late remaining work. A long command interval is not proof of a secure grasp; keep verifying.
- **Tiny grip toggles are not placements:** `95:2292→2295` lasts 0.10 s; `96:2291→2306` lasts 0.50 s. Treat these as adjustment candidates and avoid counting them as sorted items.
- **Scene changes after contact:** compare the box orientation/position in `96:0`, `96:2925`, and `96:3900`. Re-localize the container after it moves. The demonstrator continues through this difficult episode; a live agent should continue with reachable items when safe, and record failures explicitly rather than ending at the first miss.
- **Final verification:** use the completed-layout examples `89:2549`, `91:2904`, `98:2296`, and `99:2505` to understand containment and occlusion. An item on a rim or hidden behind an arm needs another view; success metadata alone is not visual proof that every individual item stayed inside. The live final count must come from the live scene.

For a new item, the reusable cycle is: choose a reachable item and container, approach open, align, close, make a small verification lift, gain visible obstacle clearance, transfer, align over the opening, release, retreat and judge containment. Cite a relevant phase above on every chunk, but choose the actual arm, displacement, duration and gripper command afresh.
