"""Right-hand control for the G1: 6D grasp-site targets -> waist + arm joint targets, and a hand synergy."""

import mujoco
import numpy as np

from robot_arm.humanoid.scene import GRASP_SITE_POS, RIGHT_HAND, G1Ids

# Dex3 key poses (thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1) for the right hand.
HAND_OPEN = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])  # thumb swung back for a wide opening
HAND_CLOSED = np.array([0.0, -0.9, -1.2, 1.2, 1.3, 1.2, 1.3])


# Claw pinch for the bottle cap lying on the table: palm facing down, the index finger hooked down at its base
# joint and the thumb meeting its tip; the middle finger is curled out of the way. Found by searching joint
# poses for fingertips that are the lowest points of the hand (5 mm below everything else), so they reach the
# rim of a 12 mm-tall cap without the knuckles touching the table, then by pick trials in simulation.
# Open: tips ~4 cm apart; closed: squeezed together.
PINCH_OPEN = np.array([-0.51, -0.34, -0.44, 1.33, 0.00, 1.57, 1.75])
PINCH_CLOSED = np.array([-0.454, -0.644, -0.116, 1.436, 0.059, 1.570, 1.750])
# In the right_wrist_yaw_link frame: where the cap sits between the tips, the direction that must point down,
# and the thumb-to-index axis (kept horizontal).
PINCH_POINT = np.array([0.1275, 0.0942, 0.0258])
PINCH_DOWN = np.array([0.283, 0.937, 0.204])
PINCH_AXIS = np.array([-0.939, 0.242, -0.246])


def pinch_frame(throw_dir: np.ndarray) -> np.ndarray:
    """Hand orientation with the claw pinch pointing down and the thumb-index axis horizontal, across the
    horizontal throw direction (so the opening fingers move sideways to the cap's flight)."""
    g = PINCH_DOWN / np.linalg.norm(PINCH_DOWN)
    a = PINCH_AXIS - (PINCH_AXIS @ g) * g
    a /= np.linalg.norm(a)
    H = np.column_stack([g, a, np.cross(g, a)])
    n = np.cross([0.0, 0.0, 1.0], throw_dir)
    n /= np.linalg.norm(n)
    W = np.column_stack([[0.0, 0.0, -1.0], n, np.cross([0.0, 0.0, -1.0], n)])
    return W @ H.T


def site_for_pinch(pinch_world: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Grasp-site position that puts the hand's pinch point at `pinch_world` with hand orientation R."""
    return pinch_world + R @ (np.asarray(GRASP_SITE_POS) - PINCH_POINT)


def hand_targets(grip: float, pinch: bool = False) -> np.ndarray:
    """Grasp synergy: grip in [-1, 1] (-1 open, 1 closed) -> 7 finger joint targets. `pinch` selects the
    fingertip pinch (for the cap) instead of the whole-hand wrap (for cups)."""
    s = (np.clip(grip, -1.0, 1.0) + 1.0) / 2.0
    lo, hi = (PINCH_OPEN, PINCH_CLOSED) if pinch else (HAND_OPEN, HAND_CLOSED)
    return lo + s * (hi - lo)


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking `current` to `target` (3x3 matrices)."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (target @ current.T).ravel())
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    mujoco.mju_axisAngle2Quat(quat, v / angle, angle)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


class G1HandController:
    """Damped least-squares IK on waist (3) + right arm (7) for the grasp site's position and the hand's
    orientation, with a null-space pull towards a posture (the ready posture, or the pinch posture)."""

    def __init__(self, model: mujoco.MjModel, ids: G1Ids, damping: float = 0.02, iters: int = 8,
                 rot_weight: float = 0.3, posture_gain: float = 0.02, waist_weight: float = 3.0):
        self.model, self.ids = model, ids
        self.damping, self.iters, self.rot_weight, self.posture_gain = damping, iters, rot_weight, posture_gain
        self.home = ids.ready_qpos[ids.arm_qpos].copy()
        self.posture = self.home  # null-space target; the env switches it for the pinch
        self.lo = ids.arm_range[:, 0] + 0.02
        self.hi = ids.arm_range[:, 1] - 0.02
        # Moving the waist is "expensive": the arm should do most of the work.
        self.joint_scale = np.ones(len(ids.arm_qpos))
        self.joint_scale[:3] = 1.0 / waist_weight
        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        # Position servos lag a moving target by (kv/kp) * velocity; commanding q + lag * qdot cancels that.
        self.lag = -model.actuator_biasprm[ids.arm_ctrl, 2] / model.actuator_gainprm[ids.arm_ctrl, 0]
        self.q_prev = None

    def solve(self, qpos: np.ndarray, target_pos: np.ndarray, target_mat: np.ndarray,
              q_start: np.ndarray | None = None, iters: int | None = None) -> np.ndarray:
        ids, s = self.ids, self._scratch
        s.qpos[:] = qpos
        if q_start is not None:
            s.qpos[ids.arm_qpos] = q_start
        q = s.qpos[ids.arm_qpos].copy()
        W = np.diag(self.joint_scale)
        for _ in range(iters or self.iters):
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)
            pos_err = target_pos - s.site_xpos[ids.grasp_site]
            rot_err = rotation_error(s.xmat[ids.hand_body].reshape(3, 3), target_mat)
            mujoco.mj_jacSite(self.model, s, self._jacp, self._jacr, ids.grasp_site)
            J = np.vstack([self._jacp[:, ids.arm_dofs], self.rot_weight * self._jacr[:, ids.arm_dofs]]) @ W
            err = np.concatenate([pos_err, self.rot_weight * rot_err])
            JJt = J @ J.T + self.damping ** 2 * np.eye(6)
            dq = W @ (J.T @ np.linalg.solve(JJt, err))
            null = np.eye(len(q)) - W @ J.T @ np.linalg.solve(JJt, J)
            dq += null @ (self.posture_gain * (self.posture - q))
            q = np.clip(q + dq, self.lo, self.hi)
            s.qpos[ids.arm_qpos] = q
        return q

    def fk(self, qpos: np.ndarray, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self._scratch
        s.qpos[:] = qpos
        s.qpos[self.ids.arm_qpos] = q_arm
        mujoco.mj_kinematics(self.model, s)
        return s.site_xpos[self.ids.grasp_site].copy(), s.xmat[self.ids.hand_body].reshape(3, 3).copy()

    def reset(self) -> None:
        self.q_prev = None

    def apply(self, data: mujoco.MjData, target_pos: np.ndarray, target_mat: np.ndarray, grip: float,
              dt: float, pinch: bool = False) -> None:
        """Solve IK for the target and command it with velocity feed-forward (dt: time since the last call)."""
        start = self.q_prev if self.q_prev is not None else data.qpos[self.ids.arm_qpos].copy()
        q = self.solve(data.qpos, target_pos, target_mat, q_start=start)
        qdot = (q - start) / dt if self.q_prev is not None else 0.0
        self.q_prev = q
        data.ctrl[self.ids.arm_ctrl] = q + self.lag * qdot
        data.ctrl[self.ids.hand_ctrl] = hand_targets(grip, pinch)


assert len(HAND_OPEN) == len(RIGHT_HAND)
