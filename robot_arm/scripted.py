"""Hand-written waypoint expert. It checks that the task is solvable in the simulator and is a baseline."""

import numpy as np

from robot_arm.config import CFG
from robot_arm.env import OBS_BALL, OBS_CUP, OBS_TCP


class ScriptedPolicy:
    """approach -> descend -> close -> lift -> carry over cup -> open -> retreat."""

    def __init__(self, cfg=CFG, carry_height: float | None = None):
        self.cfg = cfg
        self.carry_z = carry_height or cfg.cup_height + cfg.ball_radius + 0.06
        self.reset()

    def reset(self):
        self.phase, self.timer = "approach", 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        tcp, ball, cup = obs[OBS_TCP], obs[OBS_BALL], obs[OBS_CUP]
        r = self.cfg.ball_radius
        grip = -1.0
        self.timer += 1

        if self.phase == "approach":
            goal = ball + [0, 0, 0.08]
            if np.linalg.norm(goal - tcp) < 0.01:
                self.phase = "descend"
        if self.phase == "descend":
            goal = np.array([ball[0], ball[1], r])
            if np.linalg.norm(goal - tcp) < 0.006:
                self.phase, self.timer = "close", 0
        if self.phase == "close":
            goal, grip = tcp, 1.0
            if self.timer > 8:
                self.phase = "lift"
        if self.phase == "lift":
            goal, grip = np.array([tcp[0], tcp[1], self.carry_z]), 1.0
            if tcp[2] > self.carry_z - 0.01:
                self.phase = "carry"
        if self.phase == "carry":
            goal, grip = np.array([cup[0], cup[1], self.carry_z]), 1.0
            if np.linalg.norm(goal - tcp) < 0.008:
                self.phase, self.timer = "release", 0
        if self.phase == "release":
            goal = tcp
            if self.timer > 8:
                self.phase = "retreat"
        if self.phase == "retreat":
            goal = np.array([tcp[0], tcp[1], self.carry_z + 0.1])

        delta = np.clip((goal - tcp) / self.cfg.max_step, -1, 1)
        return np.concatenate([delta, [grip]]).astype(np.float32)
