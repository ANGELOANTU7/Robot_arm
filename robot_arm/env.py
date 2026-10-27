"""Gymnasium environment: Panda picks up a ball and drops it into a cup."""

import gymnasium as gym
import mujoco
import numpy as np

from robot_arm.config import CFG, TaskConfig
from robot_arm.controller import IKController
from robot_arm.scene import SceneIds, build_model

OBS_DIM = 34
ACT_DIM = 4  # [dx, dy, dz, grip]; grip > 0 closes the gripper
# Slices of the observation vector.
OBS_TCP = slice(15, 18)
OBS_BALL = slice(18, 21)
OBS_CUP = slice(24, 27)
OBS_GRIP = 33


def sample_layout(rng: np.random.Generator, cfg: TaskConfig = CFG) -> tuple[np.ndarray, np.ndarray]:
    """Random (ball_xy, cup_xy) inside the workspace with a sensible distance between them."""
    while True:
        ball = rng.uniform([cfg.workspace_x[0], cfg.workspace_y[0]], [cfg.workspace_x[1], cfg.workspace_y[1]])
        cup = rng.uniform([cfg.workspace_x[0], cfg.workspace_y[0]], [cfg.workspace_x[1], cfg.workspace_y[1]])
        if cfg.min_ball_cup_dist <= np.linalg.norm(ball - cup) <= cfg.max_ball_cup_dist:
            return ball, cup


class PandaBallCupEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CFG.control_dt)}

    def __init__(self, cfg: TaskConfig = CFG, render_mode: str | None = None,
                 camera: str = "front", width: int = 640, height: int = 480):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.ids = SceneIds(self.model)
        self.controller = IKController(self.model, self.ids)
        self.render_mode, self.camera = render_mode, camera
        self._renderer = mujoco.Renderer(self.model, height, width) if render_mode else None

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.ws_lo = np.array([cfg.workspace_x[0] - 0.15, cfg.workspace_y[0] - 0.15, cfg.tcp_z_min])
        self.ws_hi = np.array([cfg.workspace_x[1] + 0.15, cfg.workspace_y[1] + 0.15, cfg.tcp_z_max])
        self.target = np.zeros(3)
        self.grip_closed = False
        self.steps = 0

    # ------------------------------------------------------------------ state accessors
    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ids.tcp_site].copy()

    @property
    def ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ids.ball_body].copy()

    @property
    def cup_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self.ids.cup_mocap].copy()

    def finger_width(self) -> float:
        return float(self.data.qpos[self.ids.finger_qpos].sum())

    def ball_contacts(self) -> tuple[bool, bool]:
        """Is the ball touching the (left, right) finger?"""
        left = right = False
        for c in self.data.contact[: self.data.ncon]:
            pair = (c.geom1, c.geom2)
            if self.ids.ball_geom not in pair:
                continue
            other = c.geom2 if c.geom1 == self.ids.ball_geom else c.geom1
            left |= other in self.ids.left_finger_geoms
            right |= other in self.ids.right_finger_geoms
        return left, right

    def is_grasped(self) -> bool:
        left, right = self.ball_contacts()
        return left and right and self.grip_closed

    def is_success(self) -> bool:
        ball, cup = self.ball_pos, self.cup_pos
        inside = np.linalg.norm(ball[:2] - cup[:2]) < self.cfg.cup_inner_radius
        below_rim = ball[2] < self.cfg.cup_height
        return bool(inside and below_rim and not any(self.ball_contacts()))

    # ------------------------------------------------------------------ gym API
    def reset_to(self, ball_xy, cup_xy, arm_noise: np.ndarray | None = None):
        """Reset to a specific layout (used by retargeting and evaluation)."""
        m, d, ids = self.model, self.data, self.ids
        mujoco.mj_resetDataKeyframe(m, d, ids.home_key)
        if arm_noise is not None:
            d.qpos[ids.arm_joints] += arm_noise
            d.ctrl[:7] = d.qpos[ids.arm_joints]
        d.qpos[ids.ball_qpos:ids.ball_qpos + 3] = [ball_xy[0], ball_xy[1], self.cfg.ball_radius]
        d.qpos[ids.ball_qpos + 3:ids.ball_qpos + 7] = [1, 0, 0, 0]
        d.mocap_pos[ids.cup_mocap] = [cup_xy[0], cup_xy[1], 0.0]
        mujoco.mj_forward(m, d)
        self.target = self.tcp_pos
        self.grip_closed = False
        self.steps = 0
        return self._obs(), self._info()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "ball_xy" in options:
            ball_xy, cup_xy = options["ball_xy"], options["cup_xy"]
        else:
            ball_xy, cup_xy = sample_layout(self.np_random, self.cfg)
        noise = self.np_random.normal(0, 0.03, 7)
        return self.reset_to(ball_xy, cup_xy, arm_noise=noise)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.apply_action(action)
        self.steps += 1
        reward, success = self._reward()
        ball = self.ball_pos
        dropped = ball[2] < -0.05 or not (0.1 < ball[0] < 0.95 and abs(ball[1]) < 0.6)
        truncated = self.steps >= self.cfg.max_episode_steps
        return self._obs(), reward, bool(dropped), truncated, self._info(success)

    def apply_action(self, action: np.ndarray) -> None:
        """Move the TCP by up to action[:3] * max_step, set the gripper, and simulate one control step.

        The target is relative to the measured TCP, so the policy never has to track a hidden target.
        """
        lead = self.cfg.max_step * self.cfg.target_lead
        self.target = np.clip(self.tcp_pos + action[:3] * lead, self.ws_lo, self.ws_hi)
        self.grip_closed = bool(action[3] > 0)
        self.controller.apply(self.data, self.target, self.grip_closed)
        mujoco.mj_step(self.model, self.data, nstep=self.cfg.frame_skip)

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        d, ids = self.data, self.ids
        tcp, ball, cup = self.tcp_pos, self.ball_pos, self.cup_pos
        ball_vel = d.qvel[ids.ball_dof:ids.ball_dof + 3]
        return np.concatenate([
            d.qpos[ids.arm_joints], d.qvel[ids.arm_dofs] * 0.1, [self.finger_width() * 10],
            tcp, ball, ball_vel, cup, ball - tcp, cup - ball, [1.0 if self.grip_closed else -1.0],
        ]).astype(np.float32)

    def _reward(self) -> tuple[float, bool]:
        """Staged dense reward in [0, 1] for progress; 2 per step once the ball is in the cup."""
        cfg = self.cfg
        if self.is_success():
            return 2.0, True
        tcp, ball, cup = self.tcp_pos, self.ball_pos, self.cup_pos
        reward = 0.1 * (1 - np.tanh(10 * np.linalg.norm(ball - tcp)))
        if self.is_grasped():
            lift = np.clip((ball[2] - cfg.ball_radius) / (cfg.cup_height + 0.04), 0, 1)
            reward = 0.25 + 0.25 * lift
            if ball[2] > cfg.cup_height + cfg.ball_radius:
                reward += 0.5 * (1 - np.tanh(10 * np.linalg.norm(ball[:2] - cup[:2])))
        return float(reward), False

    def _info(self, success: bool | None = None) -> dict:
        return {"is_success": self.is_success() if success is None else success}

    def render(self):
        if self._renderer is None:
            raise RuntimeError("Create the env with render_mode='rgb_array' to render.")
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register("PandaBallCup-v0", entry_point=PandaBallCupEnv)
