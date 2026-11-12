"""
yor_fk.py -- forward kinematics for the YOR arms, matching the Pi's IK frame.

Uses assets/yor_kinematics/nero_kinematics_only.mjcf: a kinematics-only copy of the Pi's
robot/yor-description/nero-welded-base-and-lift.mjcf (the model SingleArmIK loads), with meshes,
geoms and actuators stripped so it loads with this machine's older mujoco. Bodies, joints and
sites are untouched, so FK is identical.

lift_joint is always 0: the hardware has no lift feedback, and since 2026-09-11 the Pi's
SingleArmIK freezes it at 0 too (it used to be a free IK variable -- see
diagnostics/z_offset_evidence_ik_virtual_lift_2026-09-11.png). Validated 2026-09-11 against
yor.py's own get_bimanual_state() on the live robot: identical to float precision.

Returned poses are wxyz_xyz (mink's convention, same as the teleop packet and action.*_ee).
"""

import pathlib

import mujoco
import numpy as np

_MJCF = pathlib.Path(__file__).resolve().parent / "assets" / "yor_kinematics" / "nero_kinematics_only.mjcf"

_model = mujoco.MjModel.from_xml_path(str(_MJCF))
_data = mujoco.MjData(_model)
_LEFT = [_model.jnt_qposadr[_model.joint(f"left_arm_joint{i}").id] for i in range(1, 8)]
_RIGHT = [_model.jnt_qposadr[_model.joint(f"right_arm_joint{i}").id] for i in range(1, 8)]
_LIFT = _model.jnt_qposadr[_model.joint("lift_joint").id]
# lift_joint's (min, max) travel in metres -- also the most a phantom lift could ever have been.
LIFT_RANGE = tuple(float(x) for x in _model.jnt_range[_model.joint("lift_joint").id])


def fk(left_joints, right_joints) -> tuple[np.ndarray, np.ndarray]:
    """7 left + 7 right joint angles -> (left_ee, right_ee), each [qw, qx, qy, qz, x, y, z]."""
    _data.qpos[:] = _model.qpos0
    _data.qpos[_LEFT] = left_joints
    _data.qpos[_RIGHT] = right_joints
    _data.qpos[_LIFT] = 0.0
    mujoco.mj_kinematics(_model, _data)
    poses = []
    for site in ("left_arm_ee", "right_arm_ee"):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, _data.site(site).xmat)
        poses.append(np.r_[quat, _data.site(site).xpos])
    return poses[0], poses[1]


def fk_states(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, >=14) joint states ([left(7), right(7), ...], e.g. observation.state) -> (T,7), (T,7)."""
    out = [fk(s[:7], s[7:14]) for s in np.asarray(states, dtype=np.float64)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])
