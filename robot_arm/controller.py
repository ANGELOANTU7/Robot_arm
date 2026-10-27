"""Cartesian control of the Panda: TCP position targets -> joint position targets via differential IK."""

import mujoco
import numpy as np

from robot_arm.scene import SceneIds

# World orientation of the hand at the home keyframe: gripper pointing straight down,
# fingers opening along world x.
TARGET_HAND_MAT = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0]])
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0


def _orientation_error(current_mat: np.ndarray, target_mat: np.ndarray) -> np.ndarray:
    """Rotation vector taking current -> target, in the world frame."""
    err_mat = target_mat @ current_mat.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, err_mat.ravel())
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


class IKController:
    """Damped least-squares IK on the 7 arm joints, holding the gripper top-down.

    Each call solves for joint targets that move the TCP to `target_pos` and writes them to
    `data.ctrl` (the Panda's position actuators then track them).
    """

    def __init__(self, model: mujoco.MjModel, ids: SceneIds, damping: float = 1e-2,
                 iters: int = 3, ori_weight: float = 0.5, posture_gain: float = 0.05):
        self.model, self.ids = model, ids
        self.damping, self.iters = damping, iters
        self.ori_weight, self.posture_gain = ori_weight, posture_gain
        self.home = model.key_qpos[ids.home_key][ids.arm_joints].copy()
        self.lo = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 0] + 0.02
        self.hi = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 1] - 0.02
        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(self, data: mujoco.MjData, target_pos: np.ndarray, q_start: np.ndarray | None = None) -> np.ndarray:
        """Joint targets reaching target_pos, starting from the current (or given) arm configuration."""
        ids, scratch = self.ids, self._scratch
        scratch.qpos[:] = data.qpos
        if q_start is not None:
            scratch.qpos[ids.arm_joints] = q_start
        q = scratch.qpos[ids.arm_joints].copy()
        for _ in range(self.iters):
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            pos_err = target_pos - scratch.site_xpos[ids.tcp_site]
            ori_err = _orientation_error(scratch.xmat[ids.hand_body].reshape(3, 3), TARGET_HAND_MAT)
            mujoco.mj_jacSite(self.model, scratch, self._jacp, self._jacr, ids.tcp_site)
            jac = np.vstack([self._jacp[:, ids.arm_dofs], self.ori_weight * self._jacr[:, ids.arm_dofs]])
            err = np.concatenate([pos_err, self.ori_weight * ori_err])
            jjt = jac @ jac.T + self.damping ** 2 * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)
            # Null-space pull towards the home posture keeps the elbow well-behaved.
            null = np.eye(7) - jac.T @ np.linalg.solve(jjt, jac)
            dq += null @ (self.posture_gain * (self.home - q))
            q = np.clip(q + dq, self.lo, self.hi)
            scratch.qpos[ids.arm_joints] = q
        return q

    def apply(self, data: mujoco.MjData, target_pos: np.ndarray, grip_closed: bool) -> None:
        # Start from the previous joint target, not the measured pose, so gravity sag
        # does not accumulate into the command.
        q = self.solve(data, target_pos, q_start=data.ctrl[:7].copy())
        data.ctrl[:7] = q
        data.ctrl[7] = GRIP_CLOSED if grip_closed else GRIP_OPEN
