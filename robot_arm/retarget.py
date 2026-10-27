"""Retarget human demonstrations onto the Panda and turn them into robot training data.

A human demo is a pinch-point path in the table frame plus when the hand held the ball. For any robot layout
(ball and cup positions in the robot's workspace) we:

1. map the path with the 2D similarity transform that sends the human's ball to the robot's ball and the human's
   cup to the robot's cup, so the approach is relative to the ball and the drop is relative to the cup;
2. shift heights so the grasp happens at the simulated ball's centre, and keep the carried ball above the rim;
3. slow the motion down to what the arm can follow and bend the start of the path so it begins at the gripper;
4. track it with a closed-loop expert in the simulator, while noise is injected into the executed actions
   (DART) so the recorded labels include corrections, and keep the episodes that drop the ball in the cup.
"""

from dataclasses import dataclass

import numpy as np

from robot_arm.config import CFG, TaskConfig
from robot_arm.env import OBS_BALL, OBS_TCP, PandaBallCupEnv, sample_layout
from robot_arm.synthetic import min_jerk
from robot_arm.video.reconstruct import HumanDemo

ROBOT_CENTER = np.array([0.5, 0.0])
# The person reaches in from the table frame's +y side; the robot reaches in from its -x side.
HUMAN_TO_ROBOT_ROT = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass
class RobotPlan:
    waypoints: np.ndarray  # (N, 3) TCP targets, one per control step
    closed: np.ndarray  # (N,) gripper closed
    grasp_idx: int
    release_idx: int
    scale: float  # similarity scale human -> robot


def similarity_2d(src_a, src_b, dst_a, dst_b):
    """(scale * rotation) matrix M and offset c with M @ src_a + c = dst_a and M @ src_b + c = dst_b."""
    s = complex(*(np.subtract(src_b, src_a)))
    d = complex(*(np.subtract(dst_b, dst_a)))
    a = d / s
    M = np.array([[a.real, -a.imag], [a.imag, a.real]])
    return M, np.asarray(dst_a) - M @ np.asarray(src_a)


def canonical_layout(demo: HumanDemo, cfg: TaskConfig = CFG) -> tuple[np.ndarray, np.ndarray]:
    """The human's own layout, rotated into the robot's view and centred in its workspace."""
    mid = (demo.ball[:2] + demo.cup[:2]) / 2
    ball = ROBOT_CENTER + HUMAN_TO_ROBOT_ROT @ (demo.ball[:2] - mid)
    cup = ROBOT_CENTER + HUMAN_TO_ROBOT_ROT @ (demo.cup[:2] - mid)
    lo = np.array([cfg.workspace_x[0], cfg.workspace_y[0]])
    hi = np.array([cfg.workspace_x[1], cfg.workspace_y[1]])
    return np.clip(ball, lo, hi), np.clip(cup, lo, hi)


def retarget(demo: HumanDemo, ball_xy, cup_xy, start_tcp: np.ndarray, cfg: TaskConfig = CFG,
             max_speed: float = 0.22, max_duration: float | None = None) -> RobotPlan:
    M, c = similarity_2d(demo.ball[:2], demo.cup[:2], ball_xy, cup_xy)
    scale = float(np.sqrt(abs(np.linalg.det(M))))
    xy = demo.pinch[:, :2] @ M.T + c
    z = demo.pinch[:, 2] - demo.ball[2] + cfg.ball_radius
    path = np.column_stack([xy, z])

    # Over the cup with the ball, stay high enough for the ball to clear the rim.
    clearance = cfg.cup_height + cfg.ball_radius + 0.03
    near_cup = np.linalg.norm(xy - np.asarray(cup_xy), axis=1) < cfg.cup_inner_radius + 0.08
    after_grasp = demo.t >= demo.t_grasp + 0.2
    lift_zone = demo.closed & after_grasp & near_cup
    path[lift_zone, 2] = np.maximum(path[lift_zone, 2], clearance)

    # Bend the start of the path so it begins at the robot's gripper instead of where the human's hand was,
    # fading out 0.5 s before the grasp.
    blend_end = max(2, int(np.searchsorted(demo.t, demo.t_grasp - 0.5)))
    w = np.ones(len(path))
    w[:blend_end] = min_jerk(np.zeros(1), np.ones(1), blend_end)[:, 0]
    path = path + (1 - w)[:, None] * (np.asarray(start_tcp) - path[0])
    path[:, 2] = np.clip(path[:, 2], cfg.tcp_z_min, cfg.tcp_z_max)

    # Re-time: keep the human's timing where the hand is slow, stretch only the parts faster than the arm.
    dist = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t_robot = np.concatenate([[0.0], np.cumsum(np.maximum(np.diff(demo.t), dist / max_speed))])
    steps = np.arange(0.0, t_robot[-1], cfg.control_dt)
    if max_duration is not None:
        steps = steps[steps < max_duration]
    wp = np.column_stack([np.interp(steps, t_robot, path[:, k]) for k in range(3)])
    closed = np.interp(steps, t_robot, demo.closed.astype(float)) > 0.5
    grasp_idx = int(np.argmax(closed))
    release_idx = int(grasp_idx + np.argmax(~closed[grasp_idx:]))
    return RobotPlan(wp, closed, grasp_idx, release_idx, scale)


class TrackingExpert:
    """Follows a RobotPlan in closed loop.

    The plan's clock only advances while the gripper keeps up with it, the approach follows the ball if it has
    been nudged, and the gripper only closes once it is actually around the ball.
    """

    def __init__(self, plan: RobotPlan, cfg: TaskConfig = CFG, lag_tol: float = 0.03, grasp_tol: float = 0.012,
                 close_steps: int = 8):
        self.plan, self.cfg = plan, cfg
        self.lag_tol, self.grasp_tol, self.close_steps = lag_tol, grasp_tol, close_steps
        self.reset()

    def reset(self):
        self.idx, self.ball0, self.closing = 0, None, 0

    @property
    def grasping(self) -> bool:
        """Settling on the ball or closing the fingers (no exploration noise here)."""
        return self.idx == self.plan.grasp_idx and self.closing < self.close_steps

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        plan = self.plan
        tcp, ball = obs[OBS_TCP], obs[OBS_BALL]
        if self.ball0 is None:
            self.ball0 = ball.copy()
        wp = plan.waypoints[self.idx].copy()
        closed = bool(plan.closed[self.idx])
        if self.idx < plan.grasp_idx:
            wp += ball - self.ball0  # follow the ball if it moved

        if self.idx == plan.grasp_idx and self.closing < self.close_steps:
            # Settle on the ball, close, and wait for the fingers before moving on.
            wp = ball.copy()
            if np.linalg.norm(tcp - ball) < self.grasp_tol or self.closing > 0:
                self.closing += 1
            closed = self.closing > 0
        elif np.linalg.norm(wp - tcp) < self.lag_tol or self.idx >= plan.release_idx:
            self.idx = min(self.idx + 1, len(plan.waypoints) - 1)

        delta = np.clip((wp - tcp) / self.cfg.max_step, -1.0, 1.0)
        return np.concatenate([delta, [1.0 if closed else -1.0]]).astype(np.float32)


def sample_robot_layout(demo: HumanDemo, rng: np.random.Generator, cfg: TaskConfig = CFG,
                        scale_range=(0.6, 1.6), tries: int = 200):
    """A random layout whose ball-cup distance is not too different from the human's."""
    d_h = np.linalg.norm(demo.cup[:2] - demo.ball[:2])
    for _ in range(tries):
        ball, cup = sample_layout(rng, cfg)
        if scale_range[0] <= np.linalg.norm(cup - ball) / d_h <= scale_range[1]:
            return ball, cup
    return sample_layout(rng, cfg)


def rollout(env: PandaBallCupEnv, demo: HumanDemo, ball_xy, cup_xy, rng: np.random.Generator,
            noise: float = 0.0, seed: int | None = None, after_success: int = 10, render: bool = False):
    """Run the tracking expert on one layout. Returns (obs, labels, success, frames)."""
    obs, _ = env.reset(seed=seed, options={"ball_xy": ball_xy, "cup_xy": cup_xy})
    plan = retarget(demo, ball_xy, cup_xy, env.tcp_pos, env.cfg,
                    max_duration=env.cfg.episode_seconds - 1.0)
    expert = TrackingExpert(plan, env.cfg)
    observations, labels, frames = [], [], []
    success_steps = 0
    for _ in range(env.cfg.max_episode_steps):
        action = expert(obs)
        executed = action.copy()
        if noise > 0 and not expert.grasping:
            executed[:3] = np.clip(executed[:3] + rng.normal(0, noise, 3), -1, 1)
        observations.append(obs)
        labels.append(action)
        obs, _, terminated, truncated, info = env.step(executed)
        if render:
            frames.append(env.render())
        success_steps = success_steps + 1 if info["is_success"] else 0
        if success_steps > after_success or terminated or truncated:
            break
    return np.array(observations), np.array(labels), bool(info["is_success"]), frames
