# Square on peg: frozen context notes

Context is only `assets/victr_icl_pool_demo/put_the_square_on_the_peg.pkl`, episodes
241–251, plus the expressly authorized earlier hand-guided run. No other tasks,
embeddings, progress values, learned features or model checkpoints are used.

`analyze_demos.py` computes the accompanying `demo_statistics.json` and
`demo_index.json` from the earlier run's `ep241.npz`–`ep251.npz`. Its `extract.py`
copied the pool's raw actions/proprio without transforming them. Source extraction
hashes are in the statistics. This avoids loading the 6 GB pool again; if a fresh
load is ever necessary, run `free -g` first. The full pool was already inspected
in the preceding build. Its SHA256 is
`e44ea7fa4a8aa96745221af9db4b12c2abc028b8c2d5372b5a32580604ebf567`.

The extraction takes the first five frames from each overlapping 10-frame chunk.
Each episode's final five-frame tail is absent, so durations below are sampled
coverage, about 0.17 s short of the complete final window. Do not infer timing
from the document's shorthand “8–15 seconds”: episodes 242, 245 and 246 contain
much longer manipulation sequences.

## Which arm and which phases

All 11 episodes have a right-arm grasp interval. Eight contain only right-arm
grasp intervals; 242, 245 and 251 also contain a left-arm interval. Gripper
transitions alone do not prove those extra intervals are successful placements.
The prior live scene put the square on the left, so the successful hand-guided
grasp used the left arm. Choose motions from the current scene, not demo arm bias.

There is a documented convention discrepancy. All eleven raw action sequences
start `[1,1]`; grasp/lift intervals are near zero and releases return to one.
The user explicitly authorized the majority convention after this was checked.
For analysis here, 1 means open and 0 means closed. All executor commands are
hand-generated hardware values: **1=open, 0=close**. Do not invert them again.

Grasp means the first action gripper sample below 0.2; release is the end of that
interval. Peak is the highest commanded EE z while closed. The following are
right-arm frame indices, not commands to replay. Round a frame DOWN to a multiple
of five to reference its source chunk (e.g. grasp frame 97 → chunk 95).

| Episode | Coverage s | Grasp arms | Close begins | Grasp | Peak z frame | Release |
|---|---:|---|---:|---:|---:|---:|
| 241 | 13.33 | R | 99 | 111 | 149 | 331 |
| 242 | 46.00 | L, R | 225 | 238 | 277 | 416 |
| 243 | 9.17 | R | 84 | 97 | 135 | 232 |
| 244 | 12.33 | R | 195 | 208 | 247 | 316 |
| 245 | 57.50 | L, R | 1352 | 1364 | 1412 | 1641 |
| 246 | 22.67 | R | 174 | 186 | 292 | 612 |
| 247 | 10.33 | R | 102 | 115 | 194 | 250 |
| 248 | 12.83 | R | 131 | 144 | 183 | 340 |
| 249 | 10.33 | R | 131 | 143 | 182 | 270 |
| 250 | 14.00 | R | 130 | 143 | 283 | 373 |
| 251 | 14.50 | L, R | 246 | 258 | 300 | 374 |

The useful phase structure is approach open → align jaws with tab → close →
verify contact → lift clear → transfer while closed → align opening over post →
lower → release → retreat and verify. Numeric median timing across the 11 right
intervals: closure 0.43 s; grasp to +40 mm clearance 0.53 s; grasp to peak z 1.30 s;
total closed interval 5.93 s; peak to release 3.23 s. That last interval includes
transfer and alignment, so it is not a pure descent duration. Agent observation
latency makes elapsed demo time an unsuitable phase trigger.

| Episode | Close s | Held s | Lift +40 mm s | Peak-to-release s |
|---|---:|---:|---:|---:|
| 241 | .40 | 7.33 | .60 | 6.07 |
| 242 | .43 | 5.93 | .73 | 4.63 |
| 243 | .43 | 4.50 | .43 | 3.23 |
| 244 | .43 | 3.60 | .33 | 2.30 |
| 245 | .40 | 9.23 | .70 | 7.63 |
| 246 | .40 | 14.20 | .53 | 10.67 |
| 247 | .43 | 4.50 | .43 | 1.87 |
| 248 | .43 | 6.53 | .47 | 5.23 |
| 249 | .40 | 4.23 | .57 | 2.93 |
| 250 | .43 | 7.67 | .57 | 3.00 |
| 251 | .40 | 3.87 | .43 | 2.47 |

## Grasp the tab, not the opening

The square is a pale ring with a protruding solid tab. The peg is a separate pale
base with an upright post. In the wrist view, distinguish the ring opening, tab,
and two jaws. Put the tab between the jaws with enough insertion depth; merely
centering the ring in the head image does not establish a grasp.

The approved hand-guided run closed in front of the tab twice. Those readbacks
were about 0.001. Advancing roughly 2.5–3 cm in +x from the second miss allowed
the tab to enter the jaws; closure then stopped around 0.41. A zero readback
after settling means an empty close, while an intermediate reading means contact.
Contact alone is insufficient: verify that the square follows a short lift.

Historical measured left EE positions (metres, evidence only):

| Observation | x | y | z | Gripper |
|---|---:|---:|---:|---:|
| First missed close | .1932 | .0206 | .2709 | .001 |
| Second missed close | .2127 | .0226 | .2724 | .001 |
| Successful close | .2364 | .0237 | .2609 | .410 |
| Verified lift | .2363 | .0249 | .2965 | .404 |
| Cleared peg height | .2360 | .0273 | .3407 | .403 |
| Later transfer | .2516 | −.1102 | .3442 | .400 |

The successful measured grasp quaternion was approximately
`[.303,.330,.672,-.590]` in wxyz order. It is a useful orientation reference only
when current square/tab alignment agrees; do not copy its absolute pose blindly.
In that scene +x advanced into the tab; decreasing y moved the left-held square
toward the peg to its right. The earlier run confirmed grasp and lift, not final
placement. Preserve that distinction when judging this attempt.

## Heights, transfer and placement

These are commanded **right** EE z values in the demo wire frame, not object
heights. Approach z is the median during the second before closure starts.
Measured poses can settle 10–15 mm below/beside targets. Use relative clearance
and live images; do not demand a 10 mm Cartesian residual before inspecting.

| Episode | Approach z | Grasp z | Peak/transfer z | Release z |
|---|---:|---:|---:|---:|
| 241 | .299 | .296 | .384 | .346 |
| 242 | .302 | .292 | .389 | .352 |
| 243 | .307 | .303 | .371 | .346 |
| 244 | .274 | .264 | .374 | .347 |
| 245 | .350 | .338 | .432 | .375 |
| 246 | .301 | .302 | .430 | .407 |
| 247 | .307 | .290 | .417 | .390 |
| 248 | .300 | .295 | .403 | .389 |
| 249 | .293 | .289 | .395 | .372 |
| 250 | .287 | .288 | .409 | .385 |
| 251 | .315 | .305 | .392 | .371 |

Median grasp-to-peak lift is 107 mm; grasp-to-release lateral displacement is
118 mm; peak-to-release descent is 25 mm. Median grasp-to-release orientation
change is 27.1 degrees. These summarize geometry, not a fixed transport path.
The robot rotates the held square during alignment; observe whether the opening
is level with and centered around the post, then lower incrementally. The cap
must fit inside the opening, not just share its head-image center. Avoid pushing
an edge into the post. Keep the gripper closed until visual insertion is clear.

Release and retreat are separate judgements. Open only after the square is
supported by/around the peg. Then retreat several centimetres and confirm the
square stays on the peg, rather than following the gripper or falling beside it.
If placement cannot be verified, report partial after a verified grasp, not success.

## Speeds and cheap chunk references

Moving-frame translation speed excludes samples below 2 mm/s. Right-arm per-demo
median speeds range 13–76 mm/s; 90th percentiles range 97–278 mm/s. These are
action derivatives, including fast transitions, not safe defaults for this loop.
Use approximately 10–30 mm/s in free space and 2–10 mm/s close to the tab/post,
with 1–3 s chunks. The helper/executor enforces at most 65 mm per predicted chunk,
5 mm and 2 degrees per streamed tick, and the 35 mm tracking stop.

Useful references (all valid chunk starts in `demo_index.json`):

- Approach: episode 243, frame 75; episode 247, frame 85.
- Close on tab: 243/95, 247/115; cite the earlier-run offset lesson in the reason.
- Lift: 243/115 or 243/135; 249/180 for the relative clearance.
- Transfer/alignment: 243/150, 243/190; 247/195.
- Lower and release: 243/230, 249/270, 250/370.
- Retreat: 243/250, 250/390.

Each submitted chunk must state how a reference informed the current prediction.
Never submit a stored demo action or a prewritten sequence of these references.

## Known hardware lessons

Use this PC's monotonic packet arrival times; source clocks are unrelated.
The unused arm can couple by 10–17 mm through IK; its drift stop is 25 mm / 10°.
Capture a hold pose once when entering hold. Repeatedly replacing it with the
latest sagged readback makes the target drift downward. The autonomous approach
also stalled far from its long-term goal because it repeatedly generated tiny
measured-relative horizons; use a persistent 1–3 s scene-conditioned target.
Let wrist observations decide final alignment rather than chasing exact Cartesian
equality. Gate-off, IK failure, invalid values and tracking >35 mm remain stops.
