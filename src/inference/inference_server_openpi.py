"""
pi0.5 YOR bimanual inference server -- openpi/JAX venv. Rebuilt from scratch 2026-08-07 to rule
out the previous server's accumulated gripper-filtering logic as a source of the gripper
instability seen in live testing. See PIPELINE_AUDIT.md for every fact this script relies on and
its confirmation source -- do not re-derive those from scratch, they're already verified.

Summary of the wire contract (see PIPELINE_AUDIT.md for details):
  - Camera: commlink.Subscriber (NOT plain zmq) on camera_proxy.py's port, topic "cameras".
  - State: plain zmq.SUB on the Pi's teleop/episode port, topic "teleop/episode". build_state()
    always builds the full 15D [left_joints(7), right_joints(7), lift(1)] (added 2026-08-14 to
    support a second checkpoint family; identical to inference_server_lerobot.py's build_state()).
    yor_policy.YorInputs(state_dim=...) slices this to 14 for the older yor_bimanual_pi05/h30
    checkpoints, or uses it unsliced for the newer 15D-state checkpoints -- see its docstring.
  - Model output: 16 real dims [left_ee(7), right_ee(7), gripper(2)] per chunk step (some newer
    checkpoints predict 20 -- + base_vel(3) + lift_cmd(1) -- but YorOutputs always slices back to
    the first 16, which is all this script publishes either way), EE-space in mink's wxyz_xyz
    convention, gripper in DATASET convention (0=open, 1=closed).
  - Publish: inference/bimanual_ee_action, JSON payload with left_ee/right_ee/left_gripper/
    right_gripper/infer_ms/timestamp. Gripper values are inverted (1.0 - value) before
    publishing to match the Pi hardware's convention (0=closed, 1=open). No other transform is
    applied to the gripper -- this rebuild deliberately publishes the model's raw prediction
    (after only the convention fix) with zero smoothing/hysteresis/debounce, so we can see what
    the model actually does before deciding whether any filtering is still needed.
    (A sigmoid that closed the gripper earlier and released it later was added and then disabled
    on 2026-09-11 -- see the commented-out shape_gripper block below to bring it back.)
  - This script does not run IK. joystick_policy_teleop.py on the Pi converts the published
    EE poses to mink.SE3 and calls yor.py's set_bimanual_ee_target(), which runs the robot's own
    IK. The Pi's physical joystick interlock (control loop + policy rollout must be enabled) is
    the actual safety boundary.

Usage:
  cd /home/lairlab/icl_ws/openpi && source .venv/bin/activate
  python /home/lairlab/icl_ws/inference_server_openpi.py \\
      --config yor_bimanual_pi05 --checkpoint /path/to/checkpoint/step_dir [--publish]
"""

import argparse
import io
import json
import logging
# import math  # DISABLED 2026-09-11: only shape_gripper() below used it
import os
import pathlib
import signal
import threading
import time

# Must be set before anything imports jax (load_policy()'s deferred imports do this later, but
# set it here at module load to be safe regardless of import order). JAX's default is to eagerly
# preallocate 75% of total GPU memory on init, regardless of what the model actually needs --
# measured empirically 2026-08-11: our checkpoints peak at ~7-9GB (JAX's own peak_bytes_in_use,
# cross-checked against real nvidia-smi deltas), against the ~73GB the 75% default grabs on this
# machine's 96GB card. That's a real problem on a GPU shared with other people's jobs (caused a
# genuine OOM collision earlier), not just wasted headroom. 0.115 of this card's ~97887MiB total
# is ~11GiB, comfortable headroom over the measured peak. setdefault so an operator can still
# override via the environment without editing this file.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.115")

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CAM_TOPIC = "cameras"
STATE_TOPIC = "teleop/episode"
TASK_TOPIC = b"task"
ACTION_TOPIC = b"inference/bimanual_ee_action"
# For absolute-joint-space checkpoints (yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k and any
# future family with the same 15-dim [left_joints(7), right_joints(7), lift(1)] output) -- matches
# the topic joystick_policy_teleop.py's _action_worker now subscribes to on the Pi (see that
# script's module docstring). NOT the same as the older left-arm-only "inference/left_joints_action"
# topic, which this codebase never publishes to.
BIMANUAL_JOINT_ACTION_TOPIC = b"inference/bimanual_joints_action"
STALE_TIMEOUT = 2.0


def decode_jpeg(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)))


# The ZED's raw HD720 frame is 1280x720, but every icl-dataset episode stores
# observation.images.zed at 640x480 (verified uniform across the dataset). Both
# reference_scripts/robot_arm/collect.py and reference_scripts/deploy_observer.py open the camera
# the same way -- sl.RESOLUTION.HD720, retrieve_image(sl.VIEW.LEFT) -- so the raw capture is
# identical on both paths and the downscale happened in a later dataset-conversion step whose code
# isn't in this repo. Recovered empirically instead (2026-09-10, see
# diagnostics/rollout_2026-09-10_201537/zed_transform_hypotheses.png): it is a PLAIN full-frame
# resize that keeps the whole field of view and accepts the 16:9 -> 4:3 squash, NOT a centre crop.
# Static landmarks settle it -- the squash reproduces the metal frame post at the left edge, both
# grippers in the bottom corners and the table/curtain line at the dataset frame's own positions;
# a 4:3 centre crop cuts off that left post, which the dataset frames plainly show.
#
# This matters because ResizeImages(224,224) -> resize_with_pad is aspect-preserving: an
# un-resized 1280x720 frame lands as 224x126 content with 49 black rows a side, while training saw
# 224x168 with 28. Resizing here restores the training geometry exactly.
#
# Only the base camera needs this. Both wrist fisheyes are already captured at 640x480 and match
# the dataset natively, so they are passed through untouched.
#
# NOTE this is a genuine input-geometry fix, but it is NOT a fix for the chunk-shape problem in
# diagnostics/POLICY_JERKINESS_SUMMARY.md. Measured on four saved live observations at H30, it
# moves predictions about as much as redrawing the diffusion noise (0.0010-0.0046 rad vs
# 0.0012-0.0031 rad) and leaves the frozen-then-sprint profile intact.
# Changing this transform means changing onion.to_policy_view() to match: the inference UI's
# onion overlay reproduces it independently (the UI venv can't import this module -- no
# numpy/jax), and if the two drift, the overlay silently aligns the robot in a frame the
# policy no longer sees.
DATASET_ZED_SIZE = (640, 480)  # (W, H), PIL order


# DISABLED 2026-09-11 on request. The publish block again sends the model's raw gripper
# prediction. To restore: uncomment this block, the math import, the two
# --gripper-sigmoid-* arguments, the startup log, and the publish lines marked the same.
# # -- Gripper shaping ---------------------------------------------------------------------------
# # The model's gripper output is continuous in the dataset convention (0=open, 1=closed) and
# # spends a real share of every rollout in the middle of the range. Measured over 49k predicted
# # steps from yor_icl_pi05_canonical_sanity15k rollouts, the active arm sat below 0.1 for 52% of
# # steps, above 0.9 for 23%, and in the mushy 0.1-0.9 band for the remaining 25%.
# #
# # Pushing that through a sigmoid makes the command decisive instead of drifting through
# # half-closed, and putting the sigmoid's CENTRE BELOW 0.5 moves the commit point: the arm reaches
# # a closed command at a lower predicted value, so it closes earlier on the approach. The same
# # threshold governs the way back out, so it also keeps holding until the prediction falls further,
# # i.e. it releases later. One knob buys both, which is what was asked for -- but note they are not
# # independently tunable with a single monotonic curve. Decoupling them would need two thresholds
# # (hysteresis), which is exactly the stateful filtering the 2026-08-07 rebuild tore out as a
# # suspected cause of gripper instability, so it is not done here.
# #
# # Deliberately stateless: one value in, one value out, no memory between steps.
# #
# # gain <= 0 disables the transform and publishes the raw prediction unchanged.
# def shape_gripper(g: float, gain: float, center: float) -> float:
#     if gain <= 0.0:
#         return g
#     # Predictions occasionally land slightly outside [0, 1] (observed -0.011 to 1.015), which is
#     # harmless here; the clamp is only to keep math.exp away from overflow at high gain.
#     x = max(-40.0, min(40.0, gain * (g - center)))
#     return 1.0 / (1.0 + math.exp(-x))

# NOT part of the gripper-shaping block commented out above: this got swallowed by that
# comment-out on 2026-09-11 while its call site in the main loop stayed live, so every rollout
# died on its first camera frame with NameError. Restored 2026-09-11; the gripper shaping stays
# disabled as intended.
def match_dataset_zed_geometry(img: np.ndarray) -> np.ndarray:
    if img.shape[:2] == (DATASET_ZED_SIZE[1], DATASET_ZED_SIZE[0]):
        return img
    return np.asarray(Image.fromarray(img).resize(DATASET_ZED_SIZE, Image.Resampling.BILINEAR))

def build_state(pkt: dict) -> np.ndarray:
    """15D bimanual joint state: [left_joints(7), right_joints(7), lift(1)].

    Always builds the full 15D array (identical to inference_server_lerobot.py's build_state(), by
    design -- see 2026-08-14 comparison work) regardless of which checkpoint family is loaded.
    YorInputs(state_dim=...) slices this down to 14 for the older yor_bimanual_pi05/h30 checkpoints;
    the newer state_dim=15 checkpoints use it unsliced. See yor_policy.py's YorInputs docstring for
    the confirmation sources on both state layouts.
    """
    left_joints = np.asarray(pkt["left_joints"], dtype=np.float32)
    right_joints = np.asarray(pkt["right_joints"], dtype=np.float32)
    lift = np.asarray([pkt["lift"]], dtype=np.float32)
    return np.concatenate([left_joints, right_joints, lift]).astype(np.float32)


def build_current_ee_pose(pkt: dict) -> tuple[np.ndarray, np.ndarray]:
    """Current (left, right) EE pose, each a 7-float [qw, qx, qy, qz, x, y, z] array (mink's
    wxyz_xyz convention -- confirmed from reference_scripts/joystick_policy_teleop.py, which
    both consumes this same "teleop/episode" packet's left_ee/right_ee fields as
    mink.SE3(wxyz_xyz=...) and, symmetrically, publishes its own curr_pose that way).

    Needed only by pi05_extended_deltarot6d_full-family checkpoints (yor_icl_pi05_deltarot6d_
    expanded), to integrate their predicted delta actions into an absolute target pose -- see
    _integrate_delta_rot6d_arm. Absolute-pose checkpoints (YorOutputs) don't need this; their
    raw output already IS the absolute target.
    """
    return (
        np.asarray(pkt["left_ee"], dtype=np.float64),
        np.asarray(pkt["right_ee"], dtype=np.float64),
    )


_ee_hemisphere_state: dict[str, np.ndarray | None] = {"left": None, "right": None}


def reset_ee_hemisphere_state() -> None:
    """Call once at the start of each rollout -- continuity is only meaningful within one
    continuous pose stream, not carried over from whatever the arm was doing last time this
    process ran inference (or a previous rollout entirely)."""
    _ee_hemisphere_state["left"] = None
    _ee_hemisphere_state["right"] = None


def _hemisphere_correct(ee: np.ndarray, key: str) -> np.ndarray:
    """Flips [qw,qx,qy,qz] sign so it dot-products non-negative against the PREVIOUS call's
    (already-corrected) quaternion for this arm -- keeps the frame-to-frame quaternion stream on
    a consistent hemisphere, exactly matching Hannibal52Barca/icl-dataset-fixed-obs's own
    documented preprocessing: "Each arm's quaternion stream is post-processed frame-to-frame to
    stay on a consistent hemisphere (sign flipped whenever it would otherwise dot-product
    negative against the previous frame)."

    q and -q represent the IDENTICAL rotation, but yor_icl_pi05_aligned_deltarot6d_sanity15k was
    trained exclusively on hemisphere-continuous sequences -- it has never seen an arbitrary sign
    flip. mink.SE3's raw FK output (what ArmNode.forward_kinematics()/get_ee_pose() on the Pi
    actually returns, unmodified, into the teleop packet's left_ee/right_ee fields) has no such
    continuity guarantee frame-to-frame. Without this correction, a live sign flip is a genuine
    out-of-distribution jump to the model -- x/y/z untouched, but the quaternion looks like a
    ~180 degree orientation change even though physically nothing moved. This is the confirmed
    root cause of the 2026-09-10 "exploded" rollout: position stayed calm the whole time
    (delta_xyz near zero every tick) while the reconstructed orientation swung wildly -- exactly
    where an uncorrected hemisphere flip would show up and nowhere else. Sequential, not a fixed
    per-frame rule (e.g. "always qw >= 0") -- must track state across ticks, hence the module
    dict + reset_ee_hemisphere_state() rather than a stateless function.
    """
    prev = _ee_hemisphere_state[key]
    ee = np.asarray(ee, dtype=np.float64).copy()
    if prev is not None and np.dot(ee[:4], prev[:4]) < 0:
        ee[:4] = -ee[:4]
    _ee_hemisphere_state[key] = ee
    return ee


def build_ee_state(pkt: dict) -> np.ndarray:
    """14D bimanual EE-pose state: [left_ee(7), right_ee(7)], each wxyz_xyz -- the MODEL INPUT
    for yor_icl_pi05_aligned_deltarot6d_sanity15k, whose proprioception is genuinely EE-space
    (FK'd from the real joint encoders in Hannibal52Barca/icl-dataset-fixed-obs), unlike every
    other checkpoint in this file which takes joint-space observation.state. See that
    checkpoint's own README: the older pi05_extended_deltarot6d_full conditioned this same
    EE-space delta/rot6d action on joint-space state (a cross-modality mismatch) -- this one
    fixes that by using EE-space state to match.

    build_current_ee_pose()'s two arrays, each hemisphere-corrected (see _hemisphere_correct)
    before concatenating -- NOT used raw, unlike build_current_ee_pose()'s other caller (the
    delta-integration step in the main loop), which needs the robot's true, uncorrected current
    pose to integrate a physically-real delta against, not a hemisphere-flipped stand-in.
    """
    left_ee, right_ee = build_current_ee_pose(pkt)
    left_ee = _hemisphere_correct(left_ee, "left")
    right_ee = _hemisphere_correct(right_ee, "right")
    return np.concatenate([left_ee, right_ee]).astype(np.float32)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 6D -> rotation-matrix, after Zhou et al. 2019 ("On the Continuity of
    Rotation Representations in Neural Networks"). rot6d: (6,) = [a1(3), a2(3)], the first
    two ROWS of the target rotation matrix.

    ROWS, not the COLUMNS of Zhou et al.'s original parameterization, which this function
    assumed until 2026-09-10. Stacking on axis=1 returned the TRANSPOSE (i.e. the INVERSE)
    of every orientation the policy predicted. Transposing a rotation matrix flips only the
    quaternion scalar, so the published target stayed a plausible unit quaternion roughly
    100 deg from the intended one -- no NaN, no norm violation, nothing downstream to catch
    it. Confirmed root cause of the 2026-09-10 "wrists snap to a wrong rotation" rollout
    (diagnostics/delta_rollout_2026-09-10_201722): its first published left target was
    [-0.433 0.376 0.660 -0.485] where rows give [+0.433 0.376 0.660 -0.485], 103 deg apart;
    the joint state then jumped up to 1.55 rad (left j5/j6, right j6) on the very next tick
    and froze there for the rest of the run. Mismatch over that whole rollout averaged
    61 deg on the left arm and 130 deg on the right.

    Evidence for rows: re-deriving the 6D from icl-dataset's own commanded quaternions
    reproduces this checkpoint's assets/icl-dataset/norm_stats.json action mean to a max
    deviation of 0.016 with rows, against 1.392 with columns -- and that same fit recovers
    the README's 4-task left-arm pnp subset (task_index 12/14/18/20) on its own. Measured
    against the left-arm orientations actually demonstrated for that rollout's task, the rows
    decode sits inside the distribution (median 15.7 deg, min 0.3 deg) while the columns
    decode never comes within 56.8 deg of anything a human commanded. policy-inference-spec's
    replay_lerobot.py reached the same conclusion independently and transposes on the way
    back -- which is exactly why offline replay looked correct while the robot did not.
    Replay has its own decoder and never calls this one.
    """
    a1, a2 = rot6d[:3], rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def _quat_wxyz_to_rot6d(wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) -> rot6d: the ENCODING-direction inverse of _rot6d_to_matrix() above.
    Takes the first two ROWS of the rotation matrix, flattened -- matching that function's own
    empirically-verified row convention (see its docstring on the 2026-09-10 rows-vs-columns
    fix), not independently re-verified for this direction. A round-trip self-consistency check
    (encode then decode reproduces the input to float precision) has been run and passes, but
    that only catches an implementation bug, not a wrong axis convention -- Gram-Schmidt
    round-trips correctly either way when the input is already an exact orthonormal matrix, so
    it can't distinguish rows from columns on its own. Re-verify against real training data
    (Hannibal52Barca/icl-dataset-fixed-obs's observation.left_ee/right_ee vs.
    yor_icl_pi05_canonical_sanity15k's own assets/icl-dataset/norm_stats.json state
    distribution) before trusting this on hardware -- the exact same caution that fix's own
    empirical check was worth applying, given how quietly a transpose error hid the first time.
    """
    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
    return np.concatenate([matrix[0, :], matrix[1, :]])


def build_rot6d_state(pkt: dict) -> np.ndarray:
    """18D bimanual pos+rot6d state: per arm [pos(3), rot6d(6)] -- MODEL INPUT for
    yor_icl_pi05_canonical_sanity15k. Hemisphere-invariant BY CONSTRUCTION (R(q) == R(-q) for
    any rotation-matrix-based encoding, so both signs of a quaternion decode to the identical
    rot6d) -- this checkpoint's entire reason for existing is to eliminate the quaternion-
    hemisphere instability build_ee_state()'s _hemisphere_correct() exists to patch around for
    the older aligned_deltarot6d checkpoint. See that checkpoint's own README: qw is negative in
    42% of frames dataset-wide and sign-flips mid-episode in 16% of episodes, so a live
    rollout's fresh-per-frame FK (no cross-episode memory of which hemisphere the training data
    settled on) fed the model an effectively out-of-distribution state at random roughly half
    the time -- manifesting as wildly incorrect rotation commands from the first control step of
    every rollout. No hemisphere correction is needed or applied here -- the representation
    itself structurally cannot carry that ambiguity, so there is nothing to correct.
    """
    left_ee, right_ee = build_current_ee_pose(pkt)
    left_state = np.concatenate([left_ee[4:], _quat_wxyz_to_rot6d(left_ee[:4])])
    right_state = np.concatenate([right_ee[4:], _quat_wxyz_to_rot6d(right_ee[:4])])
    return np.concatenate([left_state, right_state]).astype(np.float32)


def _integrate_delta_rot6d_arm(current_wxyz_xyz: np.ndarray, delta_xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    """One arm's predicted [delta_xyz(3), rot6d(6)] + its current [qw,qx,qy,qz,x,y,z] pose
    -> an absolute target in that same 7-float wxyz_xyz layout (what the Pi's action
    listener expects, unchanged from the absolute-pose checkpoints -- see module docstring).

    Position: target = current + delta (plain world-frame addition -- matches openpi's own
    DeltaActions/AbsoluteActions transform convention, openpi/src/openpi/transforms.py's
    DeltaActions, which does exactly this elementwise for masked dims).

    Rotation: rot6d IS the ABSOLUTE target orientation, NOT a delta to compose with the
    current one -- confirmed 2026-09-06, correcting this function's original body-frame-
    composition implementation, which was wrong. Evidence: (1) a single step's raw rot6d,
    decoded relative to identity with zero composition, is ~120-150 deg -- utterly
    inconsistent with "a small per-tick incremental rotation" (which composition would only
    make sense for); (2) across a real 10-step predicted chunk, consecutive steps' rot6d
    values are nearly IDENTICAL (~0.001-0.005 change per component) while delta_xyz grows
    smoothly and monotonically (0.0003 -> 0.0043) -- exactly the signature of "each step h
    predicts [position delta from the SAME chunk-start current pose, absolute target
    orientation for that future timestep]", not a per-step delta meant to be compounded.
    Composing (current * delta) as this function originally did produced ~900 deg/sec
    implied wrist velocity in a dry-run sanity check -- physically impossible; using rot6d
    directly as the absolute target instead gives ~0.6 deg/sec on the same real data, which
    is what a careful reach phase should look like. See this project's history for the full
    empirical comparison. Position is NOT compounded step-to-step either, for the same
    reason (see the main loop below, which now passes the SAME real current pose to every
    step in a chunk instead of updating current_left/current_right after each one).

    Still not independently confirmed against this checkpoint's own training-side
    preprocessing script (no such source is available in this repo) -- this conclusion
    rests on internal consistency of the model's own real outputs, which is strong but not
    the same as ground truth. Re-validate with a fresh dry run (no --publish) before
    trusting this on hardware -- see this function's own recommendation in the TrainConfig
    comment for yor_icl_pi05_deltarot6d_expanded.
    """
    from scipy.spatial.transform import Rotation

    current_xyz = current_wxyz_xyz[4:]

    target_quat_wxyz = Rotation.from_matrix(_rot6d_to_matrix(rot6d)).as_quat(scalar_first=True)
    target_xyz = current_xyz + delta_xyz
    return np.concatenate([target_quat_wxyz, target_xyz])


def select_delta_anchor(measured_wxyz_xyz: np.ndarray, last_cmd_xyz: np.ndarray | None,
                        max_drift_m: float) -> tuple[np.ndarray, bool]:
    """One arm's ANCHOR: the pose this tick's predicted position deltas are added to.
    Returns (anchor as [qw,qx,qy,qz,x,y,z], whether the drift clamp engaged).

    This checkpoint family's position label is delta(t,h) = C(t+h) - C(t), where C is the
    *commanded* EE target logged during collection -- stated outright in the canonical
    dataset's own README ("target_pos[t] - target_pos[window_start]"). The matching
    deployment anchor is this server's own last commanded pose. Anchoring on the MEASURED
    pose instead (what this loop did until 2026-09-11) mixes two different reference frames,
    because the arm trails its own EE target by about 200ms:

        collection, best-fit delay over all 196 canonical episodes   200 ms
        live rollout 2026-09-11_181314, 40 s of commands vs poses    210 ms
        live rollout 2026-09-11_181047, 116 s                        230 ms

    The measured pose at time t is therefore where the command stream was at t-200ms. Writing
    the arm's realized pose as its command delayed by tau plus a standing offset, the height
    of the deployed hand relative to the demonstration hand works out to

        lag lead + k'   where   lag lead = C_z(t0 - tau) - C_z(t0),  k' = this system's
                                                                    own standing offset

    The COLLECTION-time standing offset cancels: the measured anchor carries it and the
    demonstration trajectory carries it equally. The deployment-time one does not.

    NOT the default, on measurement rather than on principle. The Pi's IK pinned `lift_joint`
    at 15:26 on 2026-09-11 (see ik_solver.SingleArmIK's own comment there, which records the
    phantom lift it removed: +1.07 cm left, +9.60 cm right), so every rollout after that time
    already runs without it. k' is nevertheless still about -5 mm on those post-pin runs,
    measured two independent ways that agree:

        settled-pose samples, no lag model at all   -5.3 / -5.2 / -0.8 mm
        best-fit delay + offset over a whole run    -6.6 / -5.3 / -0.7 mm

    Against a lag lead of about +6.3 mm on the eggplant approach, that residual very nearly
    cancels. So the MEASURED anchor currently lands within roughly a millimetre, while this
    function's commanded anchor would remove the cancellation and leave the hand about 5 mm
    LOW. Revisit if that standing offset is ever driven to zero, at which point the measured
    anchor becomes the ~6 mm-high one and this becomes correct.

    Replanning more often does NOT substitute for this: the offset is injected fresh at each
    new anchor rather than accumulated, so a shorter --replan-steps prefix re-applies it
    sooner rather than shrinking it.

    Only POSITION is anchored. Orientation is absolute in this representation (the predicted
    rot6d IS the target orientation -- see _integrate_delta_rot6d_arm, which reads only [4:]
    of this pose), so the returned pose carries the measured quaternion through untouched.

    Safety: with no previous command (first chunk of a rollout, or after a stream gap reset)
    this falls back to the measured pose, which is the only honest anchor when nothing has
    been commanded yet. Otherwise the anchor is clamped to within max_drift_m of the measured
    pose, so a command stream the arm is not actually following can never run away unbounded.
    """
    measured_wxyz_xyz = np.asarray(measured_wxyz_xyz, dtype=np.float64)
    if last_cmd_xyz is None:
        return measured_wxyz_xyz.copy(), False
    measured_xyz = measured_wxyz_xyz[4:]
    offset = np.asarray(last_cmd_xyz, dtype=np.float64) - measured_xyz
    drift = float(np.linalg.norm(offset))
    clamped = drift > max_drift_m
    if clamped:
        offset = offset * (max_drift_m / drift)
    return np.concatenate([measured_wxyz_xyz[:4], measured_xyz + offset]), clamped


def integrate_delta_rot6d_action(current_left: np.ndarray, current_right: np.ndarray,
                                  raw_action: np.ndarray) -> np.ndarray:
    """raw_action: this checkpoint family's real 20-dim output, PER-ARM
    [delta_xyz(3), rot6d(6), gripper(1)] x 2 (see yor_policy.YorDeltaRot6dOutputs). Returns a
    16-dim [left_ee(7), right_ee(7), gripper(2)] array in the SAME layout/convention
    YorOutputs' absolute-pose checkpoints already produce, so the existing publish code below
    (gripper inversion, JSON packaging) needs no changes to handle either checkpoint family."""
    left_target = _integrate_delta_rot6d_arm(current_left, raw_action[0:3], raw_action[3:9])
    left_gripper = raw_action[9]
    right_target = _integrate_delta_rot6d_arm(current_right, raw_action[10:13], raw_action[13:19])
    right_gripper = raw_action[19]
    # [left_ee(7), right_ee(7), gripper(2)] -- matches YorOutputs' absolute-pose layout exactly
    # (grippers both at the end, indices 14/15), so the existing publish code's action[0:7]/
    # action[7:14]/action[14]/action[15] indexing works unchanged for either checkpoint family.
    return np.concatenate([left_target, right_target, [left_gripper, right_gripper]])


def _value_estimator_label(estimator):
    if estimator == "robometer":
        return "lerobot/Robometer-4B (zero-shot)"
    from openpi.policies.yor_vfe import VFE_MODELS

    if estimator == "robometer_ft":
        return "Robometer-4B icl-finetuned (Hannibal52Barca/robometer-4b-icl-finetuned, merged LoRA)"
    return VFE_MODELS[estimator]["label"]


def load_policy(config_name: str, checkpoint_path: pathlib.Path, action_horizon_override: int | None = None,
                retrieval_overrides: dict | None = None):
    import dataclasses
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    log.info("Loading openpi policy: config=%s checkpoint=%s ...", config_name, checkpoint_path)
    train_config = _config.get_config(config_name)
    if retrieval_overrides:
        if not isinstance(train_config.data, _config.YorVictrDataConfig):
            raise ValueError("Retrieval options require a VICTR policy config")
        if "lamda" in retrieval_overrides and not getattr(train_config.data, "use_continuous_action_interpolation", False):
            raise ValueError(f"--interp-lamda needs an action-interpolation (RICL) config; {config_name} has none")
        train_config = dataclasses.replace(
            train_config, data=dataclasses.replace(train_config.data, **retrieval_overrides))
    if isinstance(train_config.data, _config.YorVictrDataConfig):
        data = train_config.data
        interp = bool(getattr(data, "use_continuous_action_interpolation", False))
        retrieval_metadata = {
            "metric": data.retrieval_metric or train_config.model.retrieval_metric,
            "trained_metric": train_config.model.retrieval_metric,
            "context_mode": data.context_mode,
            # EXP E: only the N shortest keep=true demos of each task pool are candidates (None: all).
            "pool_demos": data.pool_demos,
            "key_cameras": list(data.retrieval_key_cameras),
            "fusion_lambda": data.fusion_lambda, "pool_dir": data.pool_dir,
            "query_image_mode": data.query_image_mode, "missing_pool": data.missing_pool,
            "value_estimator": data.value_estimator,
            "value_estimator_label": _value_estimator_label(data.value_estimator),
            "prompt_to_task_overrides": data.prompt_to_task_overrides,
            "num_context_chunks": data.num_context_chunks,
            "context_frames_per_chunk": data.context_frames_per_chunk,
            "context_text_max_length": data.context_text_max_length,
            "context_encoding": "ctx3cam" if getattr(data, "context_camera_keys", None) else "head_text64",
            "context_cameras": list(getattr(data, "context_camera_keys", None) or ("observation.images.zed",)),
            "context_fast_tokenizer_path": getattr(data, "context_fast_tokenizer_path", None),
            "context_norm_stats_path": getattr(data, "context_norm_stats_path", None),
            # RICL-pi0.5: the lamda that sets the live blend weight (the data transform's; the model
            # config's own lamda is only read in training) next to the value the checkpoint trained with.
            "action_interpolation": interp,
            "interp_lamda": data.lamda if interp else None,
            "trained_interp_lamda": getattr(train_config.model, "lamda", None) if interp else None,
        }
        log.info("Retrieval configuration: %s", retrieval_metadata)
        train_config = dataclasses.replace(
            train_config, policy_metadata={**(train_config.policy_metadata or {}), "retrieval": retrieval_metadata})
    if action_horizon_override is not None and action_horizon_override != train_config.model.action_horizon:
        # EXPERIMENTAL: action_horizon is just a query-shape spec for the flow-matching sampling
        # loop (empirically confirmed 2026-08-10 -- action_in_proj/action_out_proj weight shapes
        # don't depend on it), so a checkpoint CAN be queried for a different horizon than the one
        # baked into its TrainConfig without erroring. That does NOT mean the model was trained to
        # produce good outputs at that length -- quality beyond what it actually saw in training
        # is unverified. Use with that caveat in mind.
        log.warning("Overriding action_horizon: config default=%d -> requested=%d (unverified past "
                    "what this checkpoint was actually trained with)",
                    train_config.model.action_horizon, action_horizon_override)
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, action_horizon=action_horizon_override))
    policy = _policy_config.create_trained_policy(train_config, str(checkpoint_path))
    log.info("Policy loaded.")
    return policy


def run_inference(policy, zed_img, fish0_img, fish1_img, state15: np.ndarray, task: str,
                  retrieval_episode_id: str | None = None, joint_state: np.ndarray | None = None):
    """One inference call -> (action_chunk[action_horizon, 16], infer_ms).

    Keys match yor_policy.make_yor_example()'s shape exactly (nested "images" dict), which is
    what YorInputs expects. joint_state: the 15-D joint state, for VFE value retrieval when the
    model state is something else (ctx3cam's canonical EE state); only read by retrieval.
    """
    obs = {
        "images": {"zed": zed_img, "fish0": fish0_img, "fish1": fish1_img},
        "state": state15,
        "prompt": task,
        "retrieval_episode_id": retrieval_episode_id,
    }
    if joint_state is not None:
        obs["retrieval_joint_state"] = joint_state
    result = policy.infer(obs)
    action_chunk = np.asarray(result["actions"])
    infer_ms = result.get("policy_timing", {}).get("infer_ms", 0.0)
    return action_chunk, infer_ms


# -- Live task-instruction control channel (so inference.py's UI can push a new task without
# restarting this process -- JAX JIT warmup is slow enough that a restart-per-edit workflow would
# be painful) --

_task_state = {"task": None, "attempt_id": None, "attempt_index": 0,
               "lock": threading.Lock()}
# Set by main() once the recorder exists, so a live task change can be written into the recording
# as an attempt boundary. Stays None when recording is disabled.
_task_recorder = {"rec": None}


def get_current_task() -> str:
    with _task_state["lock"]:
        return _task_state["task"]


def get_current_attempt() -> tuple[str | None, int]:
    """(attempt_id, attempt_index) of the evaluation attempt currently in progress.

    inference.py's UI opens a new attempt every time the language query changes or an outcome is
    annotated, so one loaded policy spans many attempts. Recorded on every tick, which is what
    makes a multi-query recording separable after the fact -- meta.json's single startup "task"
    cannot express it."""
    with _task_state["lock"]:
        return _task_state["attempt_id"], _task_state["attempt_index"]


def set_task_recorder(rec) -> None:
    _task_recorder["rec"] = rec


def _parse_task_payload(payload: bytes) -> tuple[str, str | None, int | None]:
    """-> (task, attempt_id, attempt_index).

    inference.py sends a JSON object {"task", "attempt_id", "attempt_index"}. Older senders (and
    anything hand-rolled with zmq for a quick test) send the bare instruction string, so a
    non-JSON or non-object payload is treated as just the task with no attempt info -- the live
    task swap keeps working either way."""
    raw = payload.decode()
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw, None, None
    if not isinstance(obj, dict) or "task" not in obj:
        return raw, None, None
    return str(obj["task"]), obj.get("attempt_id"), obj.get("attempt_index")


def start_task_control_listener(port: int | None, initial_task: str,
                                initial_attempt_id: str | None = None,
                                initial_attempt_index: int = 0):
    _task_state["task"] = initial_task
    _task_state["attempt_id"] = initial_attempt_id
    _task_state["attempt_index"] = initial_attempt_index
    if not port:
        return

    def _listener():
        import zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, TASK_TOPIC)
        sock.connect(f"tcp://127.0.0.1:{port}")
        log.info("Task-control listener connected to tcp://127.0.0.1:%d (topic=%r)", port, TASK_TOPIC)
        while True:
            try:
                _, payload = sock.recv_multipart()
                new_task, attempt_id, attempt_index = _parse_task_payload(payload)
            except Exception as e:
                log.warning("Task-control listener error: %s", e)
                continue
            with _task_state["lock"]:
                prev_task = _task_state["task"]
                prev_attempt = _task_state["attempt_id"]
                task_changed = new_task != prev_task
                attempt_changed = attempt_id is not None and attempt_id != prev_attempt
                if task_changed:
                    _task_state["task"] = new_task
                if attempt_changed:
                    _task_state["attempt_id"] = attempt_id
                    if attempt_index is not None:
                        _task_state["attempt_index"] = int(attempt_index)
            if task_changed:
                log.info("Task instruction updated: %r -> %r", prev_task, new_task)
            if attempt_changed:
                log.info("Attempt %s started (index %s)", attempt_id, attempt_index)
            if task_changed or attempt_changed:
                rec = _task_recorder["rec"]
                if rec is not None:
                    rec.record_event(
                        "attempt_boundary", task=new_task, previous_task=prev_task,
                        attempt_id=attempt_id, attempt_index=attempt_index,
                        previous_attempt_id=prev_attempt,
                    )

    threading.Thread(target=_listener, daemon=True).start()


# -- Debug frame viewer: composite JPEG of the latest camera frame + predicted action, served over
# HTTP so it's visible without reading the log --

_debug = {"frame": 0, "state": None, "zed": None, "action": None, "infer_ms": None,
          "lock": threading.Lock()}

# -- Retrieval monitor: one compact summary per tick of what RetrievalContextInputs did, for the UI's
# "Retrieval check" card (/retrieval.json) and for the recording (inferences.jsonl "retrieval"). --
import collections  # noqa: E402

_retrieval_monitor = {"config": None, "experiment": None, "history": collections.deque(maxlen=300),
                      "lock": threading.Lock()}


def summarize_retrieval(tick: int, infer_ms: float) -> dict | None:
    """Compact, JSON-safe summary of the latest retrieval snapshot for this tick (None when the
    loaded policy does no retrieval). Appended to the monitor history."""
    from openpi.policies import yor_retrieval

    snap = yor_retrieval.get_last_retrieval()
    if snap is None:
        return None
    chunk = (snap.get("chunks") or [None])[0]
    summary = {
        "tick": tick, "time": snap.get("time"), "infer_ms": infer_ms,
        "hit": bool(snap["hit"]), "prompt": snap.get("prompt"), "pool_task": snap.get("task"),
        "pool_path": snap.get("pool_path"), "context_mode": snap.get("context_mode", "retrieved"),
        "missing_pool": snap.get("missing_pool"),
        "metric": snap.get("metric"), "fusion_lambda": snap.get("fusion_lambda"),
        "key_cameras": snap.get("key_cameras"), "context_encoding": snap.get("context_encoding"),
        "num_candidates": snap.get("num_candidates"), "query_value": snap.get("query_value"),
        "pool_demos": snap.get("pool_demos"),
        "query_image_mode": snap.get("query_image_mode"), "pool_key_image_mode": snap.get("pool_key_image_mode"),
        "value_estimator": snap.get("value_estimator"), "value_context_episode": snap.get("value_context_episode"),
        "value_ms": snap.get("value_ms"),
        "retrieval_ms": snap.get("retrieval_ms"),
        # RICL-pi0.5: blend weight, distances and neighbor (None for every other policy).
        "interp": snap.get("interp"),
        "context_tokens_used": (snap.get("context_tokens_used") or [None])[0],
        "context_text_max_length": snap.get("context_text_max_length"),
        "episode": None if chunk is None else int(chunk["episode_index"]),
        "start_frame": None if chunk is None else int(chunk["start_frame"]),
        "end_frame": None if chunk is None else int(chunk["end_frame"]),
        "chunk_task": None if chunk is None else chunk["task"],
        "score": None if chunk is None or chunk["score"] is None else float(chunk["score"]),
    }
    if summary["query_value"] is not None:
        summary["query_value"] = float(summary["query_value"])
    with _retrieval_monitor["lock"]:
        _retrieval_monitor["history"].append(summary)
    return summary


def retrieval_monitor_json() -> bytes:
    with _retrieval_monitor["lock"]:
        history = list(_retrieval_monitor["history"])
    with _debug["lock"]:
        frame = _debug["frame"]
    return json.dumps({
        "config": _retrieval_monitor["config"], "experiment": _retrieval_monitor["experiment"],
        "task": get_current_task(), "server_tick": frame, "now": time.time(),
        "latest": history[-1] if history else None, "history": history[-120:],
    }).encode()


def update_debug_frame(frame_n, state, cam_pkt, action, infer_ms):
    try:
        with _debug["lock"]:
            _debug["frame"] = frame_n
            _debug["state"] = state.copy()
            _debug["zed"] = cam_pkt["cameras"]["zed"]
            _debug["action"] = None if action is None else action.copy()
            _debug["infer_ms"] = infer_ms
    except Exception as e:
        log.warning("update_debug_frame error: %s", e)


def _make_composite() -> bytes:
    from PIL import ImageDraw
    with _debug["lock"]:
        frame, state, zed, action, infer_ms = (
            _debug["frame"], _debug["state"], _debug["zed"], _debug["action"], _debug["infer_ms"])
    if zed is None:
        img = Image.new("RGB", (640, 100), (30, 30, 30))
        ImageDraw.Draw(img).text((10, 40), "Waiting for first frame...", fill=(200, 200, 200))
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    top = Image.open(io.BytesIO(zed)).convert("RGB")
    bar_h = 40
    bar = Image.new("RGB", (top.width, bar_h), (20, 20, 20))
    draw = ImageDraw.Draw(bar)
    line = f"Frame #{frame}"
    if state is not None:
        line += f"  state={np.round(state, 2).tolist()}"
    draw.text((8, 4), line, fill=(220, 220, 60))
    if action is not None:
        line2 = f"pred action[0]={np.round(action[0], 3).tolist()}"
        if infer_ms is not None:
            line2 += f"  infer={infer_ms:.0f}ms"
        draw.text((8, 20), line2, fill=(160, 200, 255))
    combined = Image.new("RGB", (top.width, bar_h + top.height))
    combined.paste(bar, (0, 0))
    combined.paste(top, (0, bar_h))
    buf = io.BytesIO()
    combined.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_context_composite() -> bytes:
    """Composite JPEG of the most recently retrieved VICTR context chunk(s) -- one
    thumbnail + task label + retrieval score per chunk, side by side. Served at
    /context.jpg by the same debug server as /frame.jpg, for any checkpoint that runs
    openpi.policies.yor_retrieval.RetrievalContextInputs (see that module's own
    get_last_retrieval docstring for the snapshot's exact shape). Harmless to call for
    a non-VICTR checkpoint -- get_last_retrieval() just returns None forever, rendered
    as a plain placeholder, since that transform never runs at all in that case."""
    from PIL import ImageDraw
    from openpi.policies import yor_retrieval

    snapshot = yor_retrieval.get_last_retrieval()
    W, H = 640, 260

    if snapshot is None:
        img = Image.new("RGB", (W, 100), (30, 30, 30))
        ImageDraw.Draw(img).text(
            (10, 40), "No retrieval yet (not a VICTR checkpoint, or waiting for first tick)",
            fill=(200, 200, 200),
        )
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    if snapshot.get("context_mode") == "none":
        img = Image.new("RGB", (W, 100), (40, 40, 60))
        ImageDraw.Draw(img).text(
            (10, 30), "Context disabled for this run (--context-mode none):\n"
                      "the policy receives no retrieved demonstration.",
            fill=(210, 210, 255),
        )
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    if not snapshot["hit"]:
        img = Image.new("RGB", (W, 100), (60, 30, 30))
        ImageDraw.Draw(img).text(
            (10, 30), f"No context pool for task:\n{snapshot['prompt']!r}\n"
                      f"(missing-pool policy: {snapshot.get('missing_pool', 'error')})",
            fill=(255, 200, 200),
        )
        buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

    # Left: the live head frame the query was retrieved from. Right: exactly the frames the model is
    # given for each retrieved chunk (frame 0 of every context camera), with a one-line label.
    from openpi.policies.yor_retrieval import CAMERA_KEYS

    cam_label = {key: name for name, (key, _) in CAMERA_KEYS.items()}
    cam_label.update({"observation.images.zed": "head", "observation.images.fish0": "left wrist",
                      "observation.images.fish1": "right wrist"})
    size, pad, label_h = 200, 8, 34
    with _debug["lock"]:
        query_jpeg = _debug["zed"]
    chunks = snapshot["chunks"]
    frames_per_chunk = max(len(c.get("context_frames") or {"head": None}) for c in chunks)
    width = pad + (size + pad) * (1 + frames_per_chunk)
    combined = Image.new("RGB", (width, pad + (size + label_h + pad) * len(chunks)), (20, 20, 20))
    draw = ImageDraw.Draw(combined)
    for row, chunk in enumerate(chunks):
        y = pad + row * (size + label_h + pad)
        if row == 0 and query_jpeg is not None:
            query = decode_jpeg(query_jpeg)
            combined.paste(Image.fromarray(query).resize((size, int(size * query.shape[0] / query.shape[1]))), (pad, y))
            draw.text((pad, y + size + 4), "live query (head)", fill=(230, 230, 230))
        frames = chunk.get("context_frames") or {"observation.images.zed": chunk["thumbnail"]}
        for k, (cam, frame) in enumerate(frames.items()):
            x = pad + (k + 1) * (size + pad)
            combined.paste(Image.fromarray(frame).resize((size, size)), (x, y))
            draw.text((x, y + size + 4), f"retrieved · {cam_label.get(cam, cam)}", fill=(160, 220, 190))
        draw.text((pad + size + pad, y + size + 18),
                  f"episode {chunk['episode_index']}  frames {chunk['start_frame']}-{chunk['end_frame']}  "
                  + ("random draw" if chunk["score"] is None else f"score {chunk['score']:.3f}"), fill=(200, 220, 255))
    buf = io.BytesIO()
    combined.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def start_debug_server(port: int):
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/frame.jpg"):
                data = _make_composite()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/retrieval.json"):
                data = retrieval_monitor_json()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                # Read cross-origin by inference.py's UI page (served from a different port).
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/context.jpg"):
                data = _make_context_composite()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/":
                html = (b"<html><body style='background:#111'>"
                        b"<img src='/frame.jpg' id='f' style='max-width:100%'>"
                        b"<img src='/context.jpg' id='c' style='max-width:100%; margin-top:8px;'>"
                        b"<script>"
                        b"setInterval(()=>{document.getElementById('f').src='/frame.jpg?t='+Date.now()},800);"
                        b"setInterval(()=>{document.getElementById('c').src='/context.jpg?t='+Date.now()},800);"
                        b"</script></body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self.send_response(404); self.end_headers()

        def log_message(self, *_): pass

    try:
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log.info("Frame viewer -> http://localhost:%d", port)
    except OSError as e:
        log.warning("Could not start frame viewer on port %d: %s", port, e)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="openpi TrainConfig name, e.g. yor_bimanual_pi05.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--retrieval-pool", help="Directory, or os.pathsep-separated directories, of per-task VICTR pools")
    parser.add_argument("--retrieval-lambda", type=float, help="Fixed fusion value weight; default 1 matches training")
    parser.add_argument("--retrieval-key-cameras", type=lambda s: tuple(c.strip() for c in s.split(",") if c.strip()),
                        help="Comma-separated cameras whose DINOv2 keys enter the vision distance, e.g. "
                             "zed,fish0,fish1 (default zed, as trained). Wrist keys need a pool built with "
                             "build_victr_pool.py --add-wrist-cameras")
    parser.add_argument("--retrieval-metric", choices=("vision", "value", "vision_value"),
                        help="Explicit experimental override of the checkpoint's retrieval metric")
    parser.add_argument("--context-mode", choices=("retrieved", "random", "none"),
                        help="VICTR configs only. 'none' gives the policy no retrieved context at all "
                             "(deployment-time ablation, experiments.py EXP B1); 'random' gives it a chunk drawn "
                             "uniformly at random from the task's pool every tick instead of the nearest one "
                             "(EXP B3; no DINOv2 or progress estimator runs); default 'retrieved'")
    parser.add_argument("--pool-demos", type=int,
                        help="VICTR configs only. Keep just the N shortest keep=true demonstrations of each task "
                             "pool as retrieval candidates (experiments.py EXP E); default: every demonstration. "
                             "Needs the pools' build_pool_demo_lengths.py sidecars")
    parser.add_argument("--experiment-arm",
                        help="experiments.py arm id (e.g. A1, B2c). Checks that --config and the "
                             "retrieval flags match that arm, and records the arm in every rollout")
    parser.add_argument("--retrieval-query-image-mode", choices=("auto", "raw", "letterbox"),
                        help="Query preprocessing for DINOv2. Default auto: whatever each pool's keys were made with "
                             "(raw for the in-domain pools, letterbox for the demo pools)")
    parser.add_argument("--retrieval-task-map", type=pathlib.Path,
                        help="JSON mapping display prompts to exact pool task strings")
    parser.add_argument("--robometer-python", help="Python executable in the existing Robometer environment")
    parser.add_argument("--robometer-device", choices=("cpu", "cuda"))
    parser.add_argument("--value-estimator", choices=("robometer", "robometer_ft", "vfe", "vfe_recap"),
                        help="Progress estimator for value / vision_value retrieval (query and pool): "
                             "robometer (default, zero-shot Robometer-4B), robometer_ft (icl-finetuned Robometer-4B), vfe (lair IC-VFE step 30000) or "
                             "vfe_recap (RECAP-style no-context VFE step 20000); the VFEs need "
                             "build_vfe_values.py --model <name> pool sidecars")
    parser.add_argument("--interp-lamda", type=float,
                        help="RICL configs only (use_continuous_action_interpolation): override lamda in the "
                             "blend weight w = exp(-lamda * top1_distance / pool_max_distance). Only w changes: "
                             "same weights, same retrieved neighbor and context. The checkpoint trained with 3.0. "
                             "1000 gives w = 0 (no blend at sampling), 0 gives w = 1 (the neighbor's action is "
                             "executed verbatim; dry-run first)")
    parser.add_argument("--vfe-python", help="Python executable with the VFE dependencies (default lerobot_env)")
    parser.add_argument("--vfe-device", choices=("cpu", "cuda"))
    parser.add_argument("--task", default="pick up the cube with the left hand and place it on the plate")
    parser.add_argument("--cam-host", default="192.168.123.116")
    parser.add_argument("--cam-port", type=int, default=5550)
    parser.add_argument("--state-host", default="192.168.123.55")
    parser.add_argument("--state-port", type=int, default=5568)
    parser.add_argument("--action-host", default="0.0.0.0")
    parser.add_argument("--action-port", type=int, default=5572)
    parser.add_argument("--publish-hz", type=float, default=30.0)
    parser.add_argument("--replan-steps", type=int, default=5,
                         help="Publish only the first N steps of each freshly-predicted action "
                              "chunk, then immediately replan with fresh camera/state instead of "
                              "continuing to the end of the chunk (receding-horizon execution). "
                              "E.g. with a 10-step chunk and --replan-steps 5, only steps 0-4 are "
                              "ever published; the model never gets to execute steps 5-9 of any "
                              "single prediction. Clamped to the chunk length if larger.")
    parser.add_argument("--delta-anchor", choices=("command", "measured"), default="measured",
                         help="Delta/rot6D checkpoints only: which pose the predicted position "
                              "deltas are added to. 'measured' (default) uses the arm's measured "
                              "pose. 'command' uses this server's own last commanded pose, which "
                              "is what the training label is actually relative to (delta = "
                              "commanded(t+h) - commanded(t)). Despite that, MEASURED is the "
                              "default on measurement, not on principle: with the Pi's lift now "
                              "pinned, the arm still settles ~5mm below its commanded Cartesian "
                              "pose, and that almost exactly cancels the ~+6mm lead the ~200ms "
                              "tracking lag puts into the measured anchor. Switching to 'command' "
                              "removes the cancellation and leaves the hand ~5mm LOW. Revisit "
                              "if that residual offset is ever driven to zero. See "
                              "select_delta_anchor() for the numbers.")
    parser.add_argument("--anchor-max-drift", type=float, default=0.05, metavar="METRES",
                         help="Safety clamp for --delta-anchor command: the command anchor is "
                              "never allowed further than this from the measured pose, so a "
                              "command stream the arm is not following cannot run away. Normal "
                              "drift is the ~200ms lag lead, order 10-15mm during fast motion, so "
                              "the default 0.05 leaves ~3x headroom before it engages.")
    parser.add_argument("--action-horizon", type=int, default=None,
                         help="EXPERIMENTAL: override the loaded checkpoint's action_horizon "
                              "(chunk length) at inference time, independent of the --config "
                              "TrainConfig's own default. Technically works because horizon is "
                              "just a query-shape spec, not baked into checkpoint weights, but "
                              "quality beyond what the checkpoint was actually trained with is "
                              "unverified. Default: use whatever --config specifies.")
    parser.add_argument("--task-control-port", type=int, default=5573,
                         help="ZMQ port for live task-instruction updates. 0 disables.")
    parser.add_argument("--viewer-port", type=int, default=8082,
                         help="Debug HTTP server port (serves /frame.jpg and /context.jpg -- see "
                              "start_debug_server). NOT 8081: confirmed 2026-08-30 that port is "
                              "permanently occupied by an unrelated pre-existing service on this "
                              "machine (an app called 'COBALT'), so the debug server silently failed "
                              "to bind there every single run all session -- the main UI's "
                              "retrieved-context panel was hitting that other app's own HTML fallback "
                              "instead of ever reaching this server, which is why it looked empty.")
    parser.add_argument("--publish", action="store_true",
                         help="Publish predicted actions to the Pi. Without this flag, actions are "
                              "only logged. The Pi's joystick interlock (control loop + policy "
                              "rollout must be physically enabled) is the actual safety boundary.")
    parser.add_argument("--attempt-id", default=None,
                         help="Id of the first evaluation attempt, minted by inference.py before "
                              "this process is spawned. Passed on the command line rather than "
                              "over the task-control channel because a just-bound zmq PUB drops "
                              "messages sent before its subscribers have propagated, which would "
                              "leave every tick of attempt 1 recorded under the default index 0 "
                              "and unseparable from attempt 2.")
    parser.add_argument("--attempt-index", type=int, default=0,
                         help="1-based index of the first attempt (see --attempt-id).")
    parser.add_argument("--rollout-id", default=None,
                         help="inference.py's Backend.start() passes this -- recorded in "
                              "RolloutRecorder's meta.json for correlation with rollout_logs/"
                              "rollouts.jsonl and /annotate, but no longer used to name the "
                              "recording directory itself (see --recording-dir).")
    parser.add_argument("--recording-dir", default=None,
                         help="Directory RolloutRecorder persists full-precision state/action/"
                              "camera-JPEG data per tick into -- see rollout_recorder.py. "
                              "inference.py's Backend.start() computes this as "
                              "rollout_recordings/<policy>/<task>/<date_ran>/. Omit for a manual/ "
                              "non-UI invocation to disable recording entirely.")
    # parser.add_argument("--gripper-sigmoid-gain", type=float, default=12.0,
    #                     help="Steepness of the sigmoid applied to the predicted gripper before "
    #                          "publishing. Higher is more decisive/binary. 0 disables the "
    #                          "transform and publishes the raw prediction.")
    # parser.add_argument("--gripper-sigmoid-center", type=float, default=0.4,
    #                     help="Predicted gripper value (dataset convention, 0=open 1=closed) that "
    #                          "maps to half-closed. Below 0.5 makes the gripper close earlier and "
    #                          "release later; 0.5 is a pure sharpener with no timing shift.")
    args = parser.parse_args()
    if args.interp_lamda is not None and not (np.isfinite(args.interp_lamda) and args.interp_lamda >= 0):
        parser.error("--interp-lamda must be finite and >= 0")
    if args.pool_demos is not None and args.pool_demos < 1:
        parser.error("--pool-demos must be >= 1 (use --context-mode none for no context)")

    # if args.gripper_sigmoid_gain > 0:
    #     log.info("Gripper sigmoid ON: gain=%.1f centre=%.2f -- predicted %.2f/%.2f/%.2f map to "
    #              "%.2f/%.2f/%.2f closed (dataset convention). Centre below 0.50 closes earlier "
    #              "and releases later.",
    #              args.gripper_sigmoid_gain, args.gripper_sigmoid_center,
    #              0.25, args.gripper_sigmoid_center, 0.75,
    #              shape_gripper(0.25, args.gripper_sigmoid_gain, args.gripper_sigmoid_center),
    #              shape_gripper(args.gripper_sigmoid_center, args.gripper_sigmoid_gain, args.gripper_sigmoid_center),
    #              shape_gripper(0.75, args.gripper_sigmoid_gain, args.gripper_sigmoid_center))
    # else:
    #     log.info("Gripper sigmoid OFF -- publishing the model's raw gripper prediction.")
    experiment = None
    if args.experiment_arm:
        import experiments as _experiments

        _, _, arm = _experiments.find_arm(args.experiment_arm)
        mismatches = [f"{name}: arm needs {want!r}, got {got!r}" for name, want, got in (
            ("--config", arm["config"], args.config),
            ("--retrieval-metric", arm["retrieval_metric"], args.retrieval_metric),
            ("--context-mode", arm["context_mode"], args.context_mode),
            ("--pool-demos", arm.get("pool_demos"), args.pool_demos),
        ) if want != got]
        if pathlib.Path(arm["checkpoint"]).name != pathlib.Path(args.checkpoint).name:
            mismatches.append(f"--checkpoint: arm needs {arm['checkpoint']!r}, got {str(args.checkpoint)!r}")
        # Only arms that fix a progress estimator constrain it (e.g. EXP A2: IC-VFE only).
        if arm.get("value_estimator") is not None and arm["value_estimator"] != args.value_estimator:
            mismatches.append(f"--value-estimator: arm needs {arm['value_estimator']!r}, got {args.value_estimator!r}")
        if mismatches:
            parser.error(f"experiment arm {args.experiment_arm} does not match the launch flags: " + "; ".join(mismatches))
        experiment = _experiments.arm_record(args.experiment_arm)
        log.info("Experiment: %s", _experiments.arm_label(args.experiment_arm))
    retrieval_overrides = {key: value for key, value in {
        "pool_dir": args.retrieval_pool, "fusion_lambda": args.retrieval_lambda,
        "retrieval_metric": args.retrieval_metric, "context_mode": args.context_mode,
        "pool_demos": args.pool_demos,
        "retrieval_key_cameras": args.retrieval_key_cameras,
        "query_image_mode": args.retrieval_query_image_mode,
        "robometer_python": args.robometer_python, "robometer_device": args.robometer_device,
        "value_estimator": args.value_estimator, "vfe_python": args.vfe_python, "vfe_device": args.vfe_device,
        "lamda": args.interp_lamda,
    }.items() if value is not None}
    if args.retrieval_task_map:
        retrieval_overrides["prompt_to_task_overrides"] = json.loads(args.retrieval_task_map.read_text())
    policy = load_policy(args.config, args.checkpoint, action_horizon_override=args.action_horizon,
                         retrieval_overrides=retrieval_overrides)
    # See yor_policy.YorDeltaRot6dOutputs / integrate_delta_rot6d_action above: this config's
    # real output is per-arm [delta_xyz(3), rot6d(6), gripper(1)] x 2, not an absolute pose,
    # so it needs the extra current-EE-pose read + delta integration step below that no other
    # config needs.
    # Both configs' real output is per-arm [delta_xyz(3), rot6d(6), gripper(1)] x 2 (20-dim), so
    # both need the extra current-EE-pose read + delta integration step below. Only the newer
    # aligned_ checkpoint ALSO takes EE-space proprioception as its model input (see
    # uses_ee_state_input below) -- the older deltarot6d_expanded checkpoint still takes
    # joint-space observation.state as input despite predicting this same EE-space action; that
    # mismatch is exactly what the aligned_ checkpoint's README says it fixes.
    # yor_icl_pi05_canonical_sanity15k's 20-dim action is the same per-arm
    # [delta_xyz(3), rot6d(6), gripper(1)] x 2 layout and the same "delta relative to the
    # query window's own first frame, rot6d as absolute orientation" semantics as the other two
    # -- see its own README -- so it reuses YorDeltaRot6dOutputs/integrate_delta_rot6d_action
    # unchanged. Only its STATE input differs (see uses_rot6d_state_input below).
    _DELTA_ROT6D_CONFIGS = {
        "yor_icl_pi05_deltarot6d_expanded",
        "yor_icl_pi05_aligned_deltarot6d_sanity15k",
        "yor_icl_pi05_canonical_sanity15k",
        # Same canonical space as canonical_sanity15k, trained on the extended dataset (step
        # 20000). Identical 18-dim state / 20-dim action layout, verified by diffing the two
        # checkpoints' own assets/icl-dataset/norm_stats.json. MUST also be listed in
        # _ROT6D_STATE_CONFIGS below -- see the warning there.
        "yor_icl_pi05_canonical_extended",
        "yor_icl_pi05_canonical_warmstart_baseline",
        # EXP B2: the warmstart baseline loaded into the VICTR class to take injected context.
        "yor_icl_pi05_canonical_warmstart_baseline_with_context",
        # RICL-pi0.5: vision ctx3cam + action interpolation toward the top-1 neighbor.
        "yor_icl_victr_vision_interp_canonical_ctx3cam_serve",
        # ctx3cam VICTR arms (canonical space, warm-started from canonical_extended step 40000);
        # also in _ROT6D_STATE_CONFIGS.
        "yor_icl_victr_vision_canonical_ctx3cam_serve",
        "yor_icl_victr_value_canonical_ctx3cam_serve",
        "yor_icl_victr_vision_value_canonical_ctx3cam_serve",
    }
    is_delta_rot6d = args.config in _DELTA_ROT6D_CONFIGS
    # See yor_policy.YorAbsoluteJointOutputs / YorAlignedJointOutputs above: both configs' real
    # output IS joint angles (not an EE pose) -- publish straight to the Pi's joint-space
    # listener (no IK). yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k's 15 real dims are
    # [left_joints(7), right_joints(7), lift(1)] (a pseudo-action -- icl-dataset never recorded
    # a real joint-space action, see that checkpoint's own README); yor_icl_pi05_aligned_q_
    # sanity15k's 14 real dims are [left_joints(7), right_joints(7)] with NO lift at all -- a
    # genuine recorded action.q_target column, not a pseudo-action (see its own README). The
    # publish block below only ever reads action[0:7]/action[7:14], so both shapes work
    # unmodified through the same code path.
    _JOINT_SPACE_CONFIGS = {
        "yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k",
        "yor_icl_pi05_aligned_q_sanity15k",
    }
    is_absolute_joint = args.config in _JOINT_SPACE_CONFIGS
    # Only yor_icl_pi05_aligned_deltarot6d_sanity15k's proprioception is EE-space (14D
    # [left_ee(7), right_ee(7)]) -- see build_ee_state()'s docstring.
    uses_ee_state_input = args.config == "yor_icl_pi05_aligned_deltarot6d_sanity15k"
    # yor_icl_pi05_canonical_sanity15k's proprioception is 18D [pos(3), rot6d(6)] per arm --
    # see build_rot6d_state()'s docstring. A third, distinct state representation from the
    # other two deltarot6d-family configs (joint-space for the original, raw EE quat+xyz for
    # aligned_), so it gets its own flag rather than overloading uses_ee_state_input.
    # NOTE: a canonical-family config MUST be added to BOTH this set and _DELTA_ROT6D_CONFIGS
    # above. Registering a new TrainConfig in openpi/training/config.py is NOT enough: these
    # allowlists are matched on the config NAME, so an unlisted name silently falls through to
    # the joint-state input path AND the absolute-EE output path. That combination is not a
    # crash -- the model gets a 15-dim joint vector where it expects 18-dim [pos, rot6d], and
    # its 20-dim delta output is then read as an absolute EE pose. It produces unreachable IK
    # targets and violent arm motion. This happened live on 2026-09-11 with
    # yor_icl_pi05_canonical_extended.
    _ROT6D_STATE_CONFIGS = {
        "yor_icl_pi05_canonical_sanity15k",
        "yor_icl_pi05_canonical_extended",
        "yor_icl_pi05_canonical_warmstart_baseline",
        # EXP B2: the warmstart baseline loaded into the VICTR class to take injected context.
        "yor_icl_pi05_canonical_warmstart_baseline_with_context",
        # RICL-pi0.5: vision ctx3cam + action interpolation toward the top-1 neighbor.
        "yor_icl_victr_vision_interp_canonical_ctx3cam_serve",
        "yor_icl_victr_vision_canonical_ctx3cam_serve",
        "yor_icl_victr_value_canonical_ctx3cam_serve",
        "yor_icl_victr_vision_value_canonical_ctx3cam_serve",
    }
    uses_rot6d_state_input = args.config in _ROT6D_STATE_CONFIGS

    # Fail-fast dim check. The allowlists above are matched on config NAME, so a newly registered
    # config that nobody remembered to list here does not crash -- it silently feeds the model the
    # wrong state representation and misreads its output. Refuse to start instead: the checkpoint's
    # own norm_stats.json records the exact state width the model was trained on, so compare it to
    # the width the flags above actually selected. This is a startup-time check on purpose -- by
    # the time a bad action reaches the Pi the arm is already moving.
    _EXPECTED_STATE_DIM = {"rot6d": 18, "ee": 14, "joint": 15}
    _selected = "rot6d" if uses_rot6d_state_input else ("ee" if uses_ee_state_input else "joint")
    _ns_files = sorted(pathlib.Path(args.checkpoint).glob("assets/*/norm_stats.json"))
    if _ns_files:
        _ns = json.loads(_ns_files[0].read_text())
        _ns = _ns.get("norm_stats", _ns)
        _trained_dim = len(_ns["state"]["mean"])
        _selected_dim = _EXPECTED_STATE_DIM[_selected]
        if _trained_dim != _selected_dim:
            raise SystemExit(
                f"State representation mismatch for config {args.config!r}: this checkpoint was "
                f"trained on a {_trained_dim}-dim state ({_ns_files[0]}), but the config-name "
                f"allowlists in this file selected the {_selected!r} representation, which is "
                f"{_selected_dim}-dim. Add {args.config!r} to the right allowlist "
                f"(_ROT6D_STATE_CONFIGS / uses_ee_state_input / _DELTA_ROT6D_CONFIGS / "
                f"_JOINT_SPACE_CONFIGS) near this line. Refusing to start: running with the wrong "
                f"state representation produces unreachable IK targets and violent arm motion."
            )
        log.info("State representation: %s (%d-dim), matches checkpoint norm stats.",
                 _selected, _trained_dim)
    else:
        log.warning("No norm_stats.json under %s/assets -- cannot verify state representation.",
                    args.checkpoint)

    from rollout_recorder import RolloutRecorder
    recorder = RolloutRecorder(args.recording_dir, rollout_id=args.rollout_id, meta={
        "family": "openpi", "config": args.config, "checkpoint": str(args.checkpoint),
        "task": args.task, "publish": args.publish, "publish_hz": args.publish_hz,
        "replan_steps": args.replan_steps, "requested_action_horizon": args.action_horizon,
        "retrieval": (policy.metadata or {}).get("retrieval"),
        "experiment": experiment,
    })
    _retrieval_monitor["config"] = (policy.metadata or {}).get("retrieval")
    if _retrieval_monitor["config"] is not None:
        _retrieval_monitor["config"] = {**_retrieval_monitor["config"], "config": args.config,
                                        "checkpoint": str(args.checkpoint)}
    _retrieval_monitor["experiment"] = experiment

    def _stop_for_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_for_signal)

    start_debug_server(args.viewer_port)
    # Before the listener starts, so the very first live task change is recorded too.
    set_task_recorder(recorder)
    start_task_control_listener(args.task_control_port or None, args.task,
                                args.attempt_id, args.attempt_index)

    log.info("Running warmup inference (JAX JIT compile, ~30-60s the first time%s)...",
             "; plus the VFE worker's startup/compile, seconds to ~6 min"
             if retrieval_overrides.get("value_estimator") in ("vfe", "vfe_recap") else "")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Same width as the live state for this config, so warmup exercises the real input shapes.
    dummy_state = np.zeros(_EXPECTED_STATE_DIM[_selected], dtype=np.float32)
    t0 = time.monotonic()
    run_inference(policy, dummy_img, dummy_img, dummy_img, dummy_state, args.task,
                  retrieval_episode_id="__warmup__", joint_state=np.zeros(15, dtype=np.float32))
    log.info("Warmup done in %.1fs.", time.monotonic() - t0)

    import zmq

    latest_cam_pkt = {"pkt": None, "t": 0.0, "receipt_timestamp": None, "recording": None}
    cam_lock = threading.Lock()

    def _cam_reader():
        from commlink import Subscriber
        sub = Subscriber(args.cam_host, port=args.cam_port, topics=[CAM_TOPIC], buffer=True, queue_size=2)
        log.info("Camera SUB -> tcp://%s:%d topic=%r", args.cam_host, args.cam_port, CAM_TOPIC)
        while True:
            pkt = sub.get(CAM_TOPIC)  # blocks until the next frame
            receipt_timestamp = time.time()
            recording = recorder.record_camera(pkt, receipt_timestamp)
            with cam_lock:
                latest_cam_pkt["pkt"] = pkt
                latest_cam_pkt["t"] = time.monotonic()
                latest_cam_pkt["receipt_timestamp"] = receipt_timestamp
                latest_cam_pkt["recording"] = recording

    def _cam_watchdog():
        while True:
            time.sleep(2)
            with cam_lock:
                t = latest_cam_pkt["t"]
            age = None if t == 0.0 else time.monotonic() - t
            if age is None or age > 2.0:
                log.warning("Camera: no frames in >%.0fs -- is camera_proxy.py running on %s:%d?",
                            age or 0.0, args.cam_host, args.cam_port)

    threading.Thread(target=_cam_reader, daemon=True).start()
    threading.Thread(target=_cam_watchdog, daemon=True).start()

    latest_state_pkt = {"pkt": None, "t": 0.0, "receipt_timestamp": None, "event_id": None}
    state_lock = threading.Lock()

    def _state_reader():
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, STATE_TOPIC)
        sock.connect(f"tcp://{args.state_host}:{args.state_port}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        log.info("State SUB -> tcp://%s:%d topic=%r", args.state_host, args.state_port, STATE_TOPIC)
        while True:
            ready = dict(poller.poll(timeout=2000))
            if sock not in ready:
                continue
            received = []
            try:
                received.append(sock.recv_multipart(flags=zmq.NOBLOCK))
                while True:  # drain to the latest
                    try:
                        received.append(sock.recv_multipart(flags=zmq.NOBLOCK))
                    except zmq.Again:
                        break
            except zmq.Again:
                continue
            for frames in received:
                try:
                    if len(frames) < 2 or frames[0] != STATE_TOPIC.encode():
                        continue
                    pkt = json.loads(frames[1])
                    receipt_timestamp = time.time()
                    event_id = recorder.record_observation(pkt, receipt_timestamp)
                    with state_lock:
                        latest_state_pkt["pkt"] = pkt
                        latest_state_pkt["t"] = time.monotonic()
                        latest_state_pkt["receipt_timestamp"] = receipt_timestamp
                        latest_state_pkt["event_id"] = event_id
                except Exception as e:
                    log.warning("State parse error: %s", e)

    threading.Thread(target=_state_reader, daemon=True).start()

    pub_sock = None
    if args.publish:
        pub_sock = zmq.Context.instance().socket(zmq.PUB)
        pub_sock.bind(f"tcp://{args.action_host}:{args.action_port}")
        time.sleep(0.1)
        if is_absolute_joint:
            log.info("PUBLISHING: predicted joint-space actions -> inference/bimanual_joints_action "
                      "(%s:%d). No IK, no gripper, no lift (see this loop's publish block for why).",
                      args.action_host, args.action_port)
        else:
            log.info("PUBLISHING: predicted EE-space actions -> inference/bimanual_ee_action (%s:%d). "
                      "Gripper is inverted to hardware convention (0=closed/1=open); no other "
                      "filtering is applied.", args.action_host, args.action_port)
    else:
        log.info("DRY RUN: predicted actions will be logged only. Pass --publish to drive the robot.")

    dt = 1.0 / args.publish_hz
    frame_count = 0
    infer_times: list[float] = []
    # Command-anchor carry for delta/rot6D checkpoints -- see select_delta_anchor(). None until
    # the first chunk is produced, and cleared again on any stream gap below, so a rollout that
    # resumes after a pause never integrates onto a pose the arm has long since left. Holds the
    # last commanded xyz per arm; orientation is absolute and is never anchored.
    last_cmd_xyz: dict[str, np.ndarray | None] = {"left": None, "right": None}
    anchor_clamp_hits = 0
    if is_delta_rot6d:
        log.info("Delta anchor: %s (drift clamp %.0f mm). %s", args.delta_anchor,
                  args.anchor_max_drift * 1000,
                  "Matches the training label's anchor, but leaves the hand ~5mm LOW while the "
                  "arm's standing offset is nonzero -- see select_delta_anchor()."
                  if args.delta_anchor == "command" else
                  "Measured anchor: its ~200ms lag lead currently cancels the arm's ~5mm "
                  "standing offset.")
    # Harmless no-op for every other checkpoint; only build_ee_state()'s hemisphere correction
    # reads this. Reset here (not at import time) so it never carries state across a Stop/Start
    # cycle within one long-lived process, or leaks from whatever the arm was doing before this
    # rollout began.
    reset_ee_hemisphere_state()

    try:
        while True:
            with cam_lock:
                cam_pkt, cam_t = latest_cam_pkt["pkt"], latest_cam_pkt["t"]
                cam_receipt_timestamp = latest_cam_pkt["receipt_timestamp"]
                cam_recording = latest_cam_pkt["recording"]
            with state_lock:
                state_pkt, state_t = latest_state_pkt["pkt"], latest_state_pkt["t"]
                state_receipt_timestamp = latest_state_pkt["receipt_timestamp"]
                state_event_id = latest_state_pkt["event_id"]

            if cam_pkt is None or state_pkt is None:
                log.warning("Waiting for camera/state data...")
                # Nothing is being commanded through this gap, and whatever was commanded before
                # it is no longer safe to integrate onto: the arm has had an unbounded amount of
                # time to move. The next chunk re-anchors on the measured pose.
                last_cmd_xyz["left"] = last_cmd_xyz["right"] = None
                time.sleep(0.1)
                continue

            now = time.monotonic()
            if (now - cam_t) > STALE_TIMEOUT or (now - state_t) > STALE_TIMEOUT:
                log.warning("Stream stale (cam=%.1fs ago, state=%.1fs ago) -- pausing.",
                            now - cam_t, now - state_t)
                last_cmd_xyz["left"] = last_cmd_xyz["right"] = None
                time.sleep(0.1)
                continue

            # Surfaced by joystick_policy_teleop.py's own episode packet (2026-09-10) -- see
            # yor.py's get_ik_status()/ArmNode.last_ik_solved. Only meaningful on ticks where the
            # Pi actually ran an EE-target IK solve (i.e. EE-space checkpoints); None on ticks
            # where it didn't (joint-space checkpoints never call IK at all, or no action has
            # reached the Pi yet). Logged here so a failure is visible in this process's own log
            # without needing a separate SSH capture session on the Pi every time.
            left_ik_solved = state_pkt.get("left_ik_solved")
            right_ik_solved = state_pkt.get("right_ik_solved")
            if left_ik_solved is False or right_ik_solved is False:
                log.warning("Pi IK not converged -- left_solved=%s right_solved=%s",
                            left_ik_solved, right_ik_solved)

            # Always built and always what gets recorded/logged as "state" below -- keeps every
            # rollout's recording schema consistent (real joint state) regardless of which array
            # actually feeds the model this tick. See model_state below for the model's own input.
            state15 = build_state(state_pkt)
            model_state = state15
            if is_delta_rot6d:
                # measured_* is what the arm actually reports, and is what gets recorded as this
                # tick's current pose. current_* is the ANCHOR the predicted deltas are added to,
                # which by default is this server's own last commanded pose instead. The two
                # differ by the arm's ~200ms tracking lag. See select_delta_anchor().
                measured_left, measured_right = build_current_ee_pose(state_pkt)
                if args.delta_anchor == "command":
                    current_left, clamped_left = select_delta_anchor(
                        measured_left, last_cmd_xyz["left"], args.anchor_max_drift)
                    current_right, clamped_right = select_delta_anchor(
                        measured_right, last_cmd_xyz["right"], args.anchor_max_drift)
                    if clamped_left or clamped_right:
                        anchor_clamp_hits += 1
                        # Rate-limited: expected to fire continuously in a dry run, where the arm
                        # never moves so the command anchor legitimately walks away from it, and
                        # expected never to fire in a healthy --publish rollout.
                        if anchor_clamp_hits <= 3 or anchor_clamp_hits % 50 == 0:
                            log.warning("Anchor drift clamp engaged (left=%s right=%s, limit "
                                        "%.0fmm, %d hits) -- the arm is not tracking the command "
                                        "stream. Expected in a dry run; investigate if publishing.",
                                        clamped_left, clamped_right,
                                        args.anchor_max_drift * 1000, anchor_clamp_hits)
                else:
                    current_left, current_right = measured_left, measured_right
                if uses_ee_state_input:
                    model_state = build_ee_state(state_pkt)
                elif uses_rot6d_state_input:
                    model_state = build_rot6d_state(state_pkt)

            cameras = cam_pkt["cameras"]
            zed_img = match_dataset_zed_geometry(decode_jpeg(cameras["zed"]))
            fish0_img = decode_jpeg(cameras["fish0"])
            fish1_img = decode_jpeg(cameras["fish1"])

            # One snapshot per tick, so the instruction fed to the model and the instruction
            # recorded against the tick can never disagree -- the UI can change it at any moment.
            _tick_task = get_current_task()
            _tick_attempt_id, _tick_attempt_index = get_current_attempt()

            t0 = time.monotonic()
            action_chunk, model_infer_ms = run_inference(
                policy, zed_img, fish0_img, fish1_img, model_state, _tick_task,
                retrieval_episode_id=_tick_attempt_id, joint_state=state15)
            infer_ms = (time.monotonic() - t0) * 1000
            infer_times.append(infer_ms)
            retrieval_summary = summarize_retrieval(frame_count, infer_ms) if _retrieval_monitor["config"] else None

            # Receding-horizon execution: only ever publish the first --replan-steps steps of a
            # freshly-predicted chunk, then immediately replan with fresh camera/state instead of
            # continuing to the end of the chunk. Clamped to the chunk length if replan_steps is
            # larger (e.g. a 10-step checkpoint with --replan-steps 30 just executes all 10).
            steps_to_run = action_chunk[: args.replan_steps]

            update_debug_frame(frame_count, state15, cam_pkt, action_chunk, infer_ms)
            log.info("[%5d] infer=%.0fms (model=%.0fms)  state=%s", frame_count, infer_ms,
                      model_infer_ms, np.round(state15, 3))
            log.info("         pred action[0]=%s  action[-1]=%s  (running %d/%d steps)",
                      np.round(action_chunk[0], 3), np.round(action_chunk[-1], 3),
                      len(steps_to_run), len(action_chunk))
            if is_delta_rot6d:
                # Computed unconditionally (dry run or --publish) so a sanity-check dry run
                # actually exercises the real integration path and logs the RECONSTRUCTED
                # absolute poses -- not just the raw, hard-to-interpret deltas -- which is the
                # whole point of running without --publish first.
                #
                # Every step in this chunk integrates against the SAME real current_left/
                # current_right captured once at the top of this tick -- NOT the previous
                # step's own reconstructed target. Confirmed 2026-09-06 (see
                # _integrate_delta_rot6d_arm's docstring): each predicted step already
                # represents [position delta from THIS tick's real current pose, absolute
                # target orientation], not a per-step delta meant to compound. Compounding
                # here (as an earlier version of this code did) double-counts motion that's
                # already accounted for in each step's own prediction.
                # Reconstruct the FULL prediction for recording. Only the requested prefix is
                # published, but retaining the full commanded-pose chunk makes later plots and
                # boundary analysis possible without re-running inference or relying on the
                # rounded text log.
                converted_chunk = np.stack([
                    integrate_delta_rot6d_action(current_left, current_right, action)
                    for action in action_chunk
                ])
                converted_steps = converted_chunk[:len(steps_to_run)]
                log.info("         [deltarot6d] reconstructed left_ee[0]=%s  left_ee[-1]=%s",
                          np.round(converted_steps[0][0:7], 3), np.round(converted_steps[-1][0:7], 3))
                log.info("         [deltarot6d] reconstructed right_ee[0]=%s  right_ee[-1]=%s",
                          np.round(converted_steps[0][7:14], 3), np.round(converted_steps[-1][7:14], 3))
                steps_to_publish = converted_steps
                # Carry the last pose of the prefix we are about to publish into the next tick's
                # anchor (see select_delta_anchor()). Taken from converted_steps, not the full
                # chunk: only this prefix is ever sent, so the tail is a pose the arm was never
                # asked to reach. Advanced in dry runs too, so a run without --publish really
                # exercises this path instead of silently falling back to measured anchoring --
                # the drift clamp will engage there, which is expected and logged.
                last_cmd_xyz["left"] = converted_steps[-1][4:7].copy()
                last_cmd_xyz["right"] = converted_steps[-1][11:14].copy()
            else:
                steps_to_publish = steps_to_run

            # Full-precision state/action + the actual camera JPEGs for this tick. Delta/rot6D
            # policies also record the measured current EE pose and full reconstructed absolute
            # EE command chunk; the raw delta/rot6D action remains in action_chunk.
            recorder.record_tick(
                frame_count, state15, action_chunk, infer_ms,
                None if cam_recording else cameras,
                current_left_ee=measured_left if is_delta_rot6d else None,
                current_right_ee=measured_right if is_delta_rot6d else None,
                anchor_left_ee=current_left if is_delta_rot6d else None,
                anchor_right_ee=current_right if is_delta_rot6d else None,
                commanded_left_ee_chunk=converted_chunk[:, 0:7] if is_delta_rot6d else None,
                commanded_right_ee_chunk=converted_chunk[:, 7:14] if is_delta_rot6d else None,
                executed_steps=len(steps_to_publish), state_event_id=state_event_id,
                camera_event_id=cam_recording["event_id"] if cam_recording else None,
                frame_names=cam_recording["frames"] if cam_recording else None,
                state_receipt_timestamp=state_receipt_timestamp,
                retrieval=retrieval_summary,
                camera_receipt_timestamp=cam_receipt_timestamp,
                # Per-tick, not per-run: the UI can swap the language query and open a new
                # attempt mid-run without restarting this process.
                task=_tick_task, attempt_id=_tick_attempt_id, attempt_index=_tick_attempt_index,
            )

            if pub_sock is not None and is_absolute_joint:
                # 15-dim absolute joint output: [left_joints(7), right_joints(7), lift(1)] -- no
                # IK, straight to the Pi's joint-space listener. Deliberately NO gripper and NO
                # lift in this payload (per user direction 2026-09-08): this checkpoint never
                # predicted gripper at all (YorAbsoluteJointOutputs' docstring), and the Pi's
                # set_lift_position() is a BLOCKING call (waits up to 30s for the lift to
                # physically converge) -- calling it every tick from this loop would stall the
                # joystick control loop's own e-stop (Back-button) read on the Pi. Omitting
                # left_gripper/right_gripper here makes joystick_policy_teleop.py pass
                # gripper_target=None to set_left/right_joint_target, which is that function's
                # documented "leave gripper alone" no-op -- not a missing feature.
                t_chunk = time.monotonic()
                for i, action in enumerate(steps_to_publish):
                    # Built as a dict and serialized only for the wire: record_command() takes the
                    # dict (see its signature), so it lands in the recording as a nested object
                    # rather than an escaped JSON string.
                    payload = {
                        "timestamp": time.time(),
                        "left_joints": action[0:7].tolist(),
                        "right_joints": action[7:14].tolist(),
                        "infer_ms": infer_ms,
                    }
                    pub_sock.send_multipart([BIMANUAL_JOINT_ACTION_TOPIC, json.dumps(payload).encode()])
                    published_wire = np.concatenate([action[0:7], action[7:14]])
                    recorder.record_command(frame_count, i, steps_to_run[i], published_wire, payload,
                                            BIMANUAL_JOINT_ACTION_TOPIC.decode(), time.time())
                    next_t = t_chunk + (i + 1) * dt
                    slack = next_t - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
            elif pub_sock is not None:
                t_chunk = time.monotonic()
                for i, action in enumerate(steps_to_publish):
                    # Only transform applied: invert gripper to hardware convention
                    # (0=closed, 1=open -- see PIPELINE_AUDIT.md). Deliberately no
                    # smoothing/hysteresis/debounce in this rebuild.
                    payload = {
                        "timestamp": time.time(),
                        "left_ee": action[0:7].tolist(),
                        "right_ee": action[7:14].tolist(),
                        "left_gripper": 1.0 - float(action[14]),
                        "right_gripper": 1.0 - float(action[15]),
                        # DISABLED 2026-09-11 -- the shaped version of the two lines above:
                        # "left_gripper": 1.0 - shape_gripper(
                        #     float(action[14]), args.gripper_sigmoid_gain, args.gripper_sigmoid_center),
                        # "right_gripper": 1.0 - shape_gripper(
                        #     float(action[15]), args.gripper_sigmoid_gain, args.gripper_sigmoid_center),
                        "infer_ms": infer_ms,
                    }
                    pub_sock.send_multipart([ACTION_TOPIC, json.dumps(payload).encode()])
                    published_wire = np.concatenate([
                        action[0:14], [payload["left_gripper"], payload["right_gripper"]]
                    ])
                    recorder.record_command(frame_count, i, steps_to_run[i], published_wire, payload,
                                            ACTION_TOPIC.decode(), time.time())
                    next_t = t_chunk + (i + 1) * dt
                    slack = next_t - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
            else:
                time.sleep(0.03)

            frame_count += 1

    except KeyboardInterrupt:
        if infer_times:
            log.info("Stopped. %d inferences, mean=%.0fms, min=%.0fms, max=%.0fms",
                      len(infer_times), np.mean(infer_times), np.min(infer_times), np.max(infer_times))
        else:
            log.info("Stopped after %d frames (no inference ran).", frame_count)
    finally:
        # The UI stops this subprocess with SIGTERM. The handler above routes it here so the
        # background recorder drains queued observations/commands and writes recorder_stats.json.
        recorder.close()


if __name__ == "__main__":
    main()
