"""Right-hand control for the G1: 6D grasp-site targets -> waist + arm joint targets, and a hand synergy."""

import mujoco
import numpy as np

from robot_arm.humanoid.hands import Hand
from robot_arm.humanoid.scene import G1Ids


def pinch_frame(throw_dir: np.ndarray, hand: Hand) -> np.ndarray:
    """Hand orientation with the pinch pointing down and the thumb-finger axis horizontal, across the
    horizontal throw direction (so the opening fingers move sideways to the cap's flight)."""
    g = hand.pinch_down / np.linalg.norm(hand.pinch_down)
    a = hand.pinch_axis - (hand.pinch_axis @ g) * g
    a /= np.linalg.norm(a)
    H = np.column_stack([g, a, np.cross(g, a)])
    n = np.cross([0.0, 0.0, 1.0], throw_dir)
    n /= np.linalg.norm(n)
    W = np.column_stack([[0.0, 0.0, -1.0], n, np.cross([0.0, 0.0, -1.0], n)])
    return W @ H.T


def site_for_pinch(pinch_world: np.ndarray, R: np.ndarray, hand: Hand) -> np.ndarray:
    """Grasp-site position that puts the hand's pinch point at `pinch_world` with hand orientation R."""
    return pinch_world + R @ (np.asarray(hand.grasp_site) - hand.pinch_point)


def hand_targets(grip: float, shape: str, hand: Hand) -> np.ndarray:
    """Hand synergy: grip in [-1, 1] (-1 open, 1 closed) -> the hand's actuator targets, blended along the
    shape's key poses (open, ..., closed). `shape` is what closing makes: the whole-hand wrap (cups), the
    fingertip pinch (the cap) or a thumbs-up."""
    if shape not in hand.poses:
        raise NotImplementedError(f"The {hand.name} hand has no '{shape}' pose yet (see hands.py)")
    keys = hand.poses[shape]
    s = (np.clip(grip, -1.0, 1.0) + 1.0) / 2.0 * (len(keys) - 1)
    i = min(int(s), len(keys) - 2)
    return keys[i] + (s - i) * (keys[i + 1] - keys[i])


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
                 rot_weight: float = 0.3, posture_gain: float = 0.02, waist_weight: float = 3.0,
                 elbow_clearance: float = 0.22, elbow_gain: float = 5.0):
        self.model, self.ids = model, ids
        self.elbow_clearance, self.elbow_gain = elbow_clearance, elbow_gain
        self.torso_body, self.elbow_body = model.body("torso_link").id, model.body("right_elbow_link").id
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
            dq += null @ (self.posture_gain * (self.posture - q) + self._elbow_push())
            q = np.clip(q + dq, self.lo, self.hi)
            s.qpos[ids.arm_qpos] = q
        return q

    def _elbow_push(self) -> np.ndarray:
        """Joint motion taking the elbow back out to the robot's right. The torso and upper arm have no collision
        geometry, so nothing else stops the arm folding across the chest when the hand works close in front of it."""
        ids, s = self.ids, self._scratch
        side = s.xmat[self.torso_body].reshape(3, 3)[:, 1]  # the robot's left
        inside = (s.xpos[self.elbow_body] - s.xpos[self.torso_body]) @ side + self.elbow_clearance
        if inside <= 0:
            return np.zeros(len(ids.arm_qpos))
        mujoco.mj_jacBody(self.model, s, self._jacp, None, self.elbow_body)
        push = -self.elbow_gain * inside * (side @ self._jacp[:, ids.arm_dofs])
        push[:3] = 0.0  # the waist carries the torso with the elbow, so it would only twist the robot
        return push

    def fk(self, qpos: np.ndarray, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self._scratch
        s.qpos[:] = qpos
        s.qpos[self.ids.arm_qpos] = q_arm
        mujoco.mj_kinematics(self.model, s)
        return s.site_xpos[self.ids.grasp_site].copy(), s.xmat[self.ids.hand_body].reshape(3, 3).copy()

    def reset(self) -> None:
        self.q_prev = None

    def apply(self, data: mujoco.MjData, target_pos: np.ndarray, target_mat: np.ndarray, grip: float,
              dt: float, shape: str = "wrap") -> None:
        """Solve IK for the target and command it with velocity feed-forward (dt: time since the last call)."""
        start = self.q_prev if self.q_prev is not None else data.qpos[self.ids.arm_qpos].copy()
        q = self.solve(data.qpos, target_pos, target_mat, q_start=start)
        qdot = (q - start) / dt if self.q_prev is not None else 0.0
        self.q_prev = q
        data.ctrl[self.ids.arm_ctrl] = q + self.lag * qdot
        data.ctrl[self.ids.hand_ctrl] = hand_targets(grip, shape, self.ids.hand)
