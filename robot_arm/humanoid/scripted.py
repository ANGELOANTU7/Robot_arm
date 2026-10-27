"""Hand-written experts for the G1 pyramid: they check the physics is solvable and serve as baselines."""

import numpy as np

from robot_arm.humanoid.config import G1_CFG, G1TaskConfig
from robot_arm.humanoid.controller import rotation_error, rotvec_to_mat
from robot_arm.humanoid.envs import (OBS_BALL, OBS_CMD_POS, OBS_CMD_ROT, OBS_HAND_ROT, OBS_SITE, OBS_STAGE,
                                     OBS_TARGET, obs_cup)
from robot_arm.humanoid.objects import ORIENTATIONS, UP, cup_grasp_point

GRAVITY = 9.81


def hand_frame(yaw: float, palm_dir: np.ndarray | None = None) -> np.ndarray:
    """Hand orientation with z up. By default the hand points along `yaw`; with `palm_dir` the palm (hand y)
    faces that horizontal direction instead."""
    z = np.array([0.0, 0.0, 1.0])
    if palm_dir is not None:
        y = np.array([palm_dir[0], palm_dir[1], 0.0])
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
    else:
        x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y = np.cross(z, x)
    return np.column_stack([x, y, z])


def roll_x(angle: float) -> np.ndarray:
    """Rotation about the hand's own x axis (right-multiply onto a hand frame)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def obs_hand_mat(obs: np.ndarray) -> np.ndarray:
    r = obs[OBS_HAND_ROT]
    x, y = r[:3], r[3:6]
    return np.column_stack([x, y, np.cross(x, y)])


def to_action(obs, goal_pos, goal_mat, grip, max_step, cfg: G1TaskConfig = G1_CFG,
              speed: float | None = None) -> np.ndarray:
    """Action moving the commanded grasp-site pose towards goal_pos (at most `speed` m per step) and goal_mat.
    `max_step` is the env's translation scale for the current stage."""
    cmd_pos = obs[OBS_SITE] + obs[OBS_CMD_POS]
    cmd_mat = rotvec_to_mat(obs[OBS_CMD_ROT]) @ obs_hand_mat(obs)
    delta = goal_pos - cmd_pos
    if speed is not None:
        delta = np.clip(delta, -speed, speed)
    dp = np.clip(delta / max_step, -1, 1)
    dr = np.clip(rotation_error(cmd_mat, goal_mat) / cfg.max_rot_step, -1, 1)
    return np.concatenate([dp, dr, [grip]]).astype(np.float32)


class PlaceExpert:
    """rise -> over the pre-grasp point -> down -> slide in -> close -> lift -> carry -> lower -> open -> back off."""

    def __init__(self, cfg: G1TaskConfig = G1_CFG, yaw: float = 0.0, pre_dist: float = 0.16):
        self.cfg, self.yaw, self.pre_dist = cfg, yaw, pre_dist
        self.reset()

    def reset(self):
        self.phase, self.timer, self.stage = "rise", 0, None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        stage = int(np.argmax(obs[OBS_STAGE]))
        if stage != self.stage:
            self.reset()
            self.stage = stage
        k = stage
        T, h = cfg.table_height, cfg.cup.height
        R = hand_frame(self.yaw)
        xh = R[:, 0]
        site = obs[OBS_SITE]
        cup_pos, cup_axis = obs_cup(obs, k)
        frac = cfg.top_grasp_height_frac if k == 2 else cfg.grasp_height_frac
        grasp = cup_grasp_point(cup_pos, cup_axis, frac, cfg)
        target = obs[OBS_TARGET]
        final_axis = np.array([0, 0, 1.0]) if ORIENTATIONS[k] is UP else np.array([0, 0, -1.0])
        place = cup_grasp_point(target, final_axis, frac, cfg) + [0, 0, 0.003]
        safe_z = T + 0.30
        carry_z = T + 1.6 * h + 0.05 + (h if k == 2 else 0.0)
        pre = grasp - self.pre_dist * xh
        self.timer += 1
        grip = -1.0
        near = lambda p, tol: np.linalg.norm(p - site) < tol  # noqa: E731

        if self.phase == "rise":
            goal = np.array([site[0], site[1], safe_z])
            if near(goal, 0.03):
                self.phase = "over_pre"
        if self.phase == "over_pre":
            goal = np.array([pre[0], pre[1], safe_z])
            if near(goal, 0.02):
                self.phase = "down_pre"
        if self.phase == "down_pre":
            goal = pre
            if near(goal, 0.01):
                self.phase, self.timer = "approach", 0
        if self.phase == "approach":
            goal = grasp
            if near(goal, 0.006) or (near(goal, 0.012) and self.timer > 25):
                self.phase, self.timer = "close", 0
        if self.phase == "close":
            goal, grip = grasp, 1.0
            if self.timer > 8:
                self.phase = "lift"
        if self.phase == "lift":
            goal, grip = np.array([site[0], site[1], carry_z]), 1.0
            if near(goal, 0.015):
                self.phase = "carry"
        # From here on, servo the held cup (not the hand) onto the target, so slip inside the grasp is corrected.
        cup_offset = site - cup_pos
        if self.phase == "carry":
            goal, grip = np.array([target[0], target[1], site[2]]) + [*cup_offset[:2], 0], 1.0
            goal[2] = carry_z
            if np.linalg.norm(cup_pos[:2] - target[:2]) < 0.006:
                self.phase = "lower"
        if self.phase == "lower":
            goal, grip = target + cup_offset + [0, 0, 0.002], 1.0
            goal[2] = max(goal[2], site[2] - cfg.max_step)  # come down gently
            if cup_pos[2] - target[2] < 0.004:
                self.phase, self.timer = "open", 0
        if self.phase == "open":
            goal = site
            if self.timer > 12:
                self.phase = "up"
        if self.phase == "up":  # straight up: the open hand slides off the cup without dragging it
            goal = np.array([site[0], site[1], safe_z])
        return to_action(obs, goal, R, grip, cfg.max_step, cfg)


class TossExpert:
    """Underhand scoop toss: slide the palm (facing up) under the ball, lift it off the stand and close the hand
    over it, turn towards the cup while winding up, swing along the ballistic release velocity, open just before
    the release point and stop, so the ball flies off the palm."""

    # Defaults found by a parameter search plus aim calibration over random layouts (scripts in the README).
    def __init__(self, cfg: G1TaskConfig = G1_CFG, flight_time: float = 0.268, release_dist: float = 0.30,
                 release_height: float = 0.163, release_bearing: float = -2.377, windup: float = 0.177,
                 speed_gain: float = 0.8, aim_offset=(0.0, 0.04, 0.0), open_lead: float = 0.03,
                 follow: float = 0.05, ball_depth: float = 0.02, palm_gap: float = 0.012,
                 open_before_swing: bool = True, palm_tilt: float = 0.099, ballistic: bool = True,
                 carry_time: float = 0.068):
        self.cfg = cfg
        self.flight_time, self.release_dist, self.release_height = flight_time, release_dist, release_height
        self.release_bearing, self.windup, self.speed_gain = release_bearing, windup, speed_gain
        self.aim_offset = np.asarray(aim_offset, float)
        self.open_lead, self.follow, self.ball_depth, self.palm_gap = open_lead, follow, ball_depth, palm_gap
        self.open_before_swing, self.palm_tilt = open_before_swing, palm_tilt
        self.ballistic, self.carry_time = ballistic, carry_time
        self.reset()

    def reset(self):
        self.phase, self.timer = "rise", 0
        self.plan = None

    def release_plan(self, mouth: np.ndarray):
        """Release point and velocity so a free ball passes through the mouth centre after flight_time."""
        b = self.release_bearing
        R = mouth + np.array([self.release_dist * np.cos(b), self.release_dist * np.sin(b), self.release_height])
        t = self.flight_time
        v = (mouth + self.aim_offset - R + np.array([0, 0, 0.5 * GRAVITY * t * t])) / t
        return R, v

    @staticmethod
    def palm_up(direction: np.ndarray, tilt: float = 0.0) -> np.ndarray:
        """Palm facing up, fingers along `direction`; `tilt` raises the fingers so the ball settles at the
        base of the palm (the same spot every throw)."""
        x = np.array([direction[0], direction[1], 0.0])
        x /= np.linalg.norm(x)
        y = np.array([0.0, 0.0, 1.0])
        x, y = np.cos(tilt) * x + np.sin(tilt) * y, -np.sin(tilt) * x + np.cos(tilt) * y
        return np.column_stack([x, y, np.cross(x, y)])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T = cfg.table_height
        site, ball = obs[OBS_SITE], obs[OBS_BALL]
        mouth = obs[OBS_TARGET]
        if self.plan is None:
            self.plan = self.release_plan(mouth)
        R_pt, v = self.plan
        vhat = v / np.linalg.norm(v)
        pick_frame = self.palm_up(np.array([1.0, 0.0]))
        throw_frame = self.palm_up(v, self.palm_tilt)
        xh = pick_frame[:, 0]
        safe_z = T + 0.30
        # Site (4.4 cm above the palm) placed so the palm passes just under the ball, ball over the palm centre.
        under = ball + self.ball_depth * xh - [0, 0, self.palm_gap]
        pre = under - 0.16 * xh
        windup_pt = R_pt - vhat * self.windup
        self.timer += 1
        grip, frame, speed = -1.0, pick_frame, cfg.max_step
        near = lambda p, tol: np.linalg.norm(p - site) < tol  # noqa: E731

        if self.phase == "rise":
            goal = np.array([site[0], site[1], safe_z])
            if near(goal, 0.03):
                self.phase = "over_pre"
        if self.phase == "over_pre":
            goal = np.array([pre[0], pre[1], safe_z])
            if near(goal, 0.02):
                self.phase = "down_pre"
        if self.phase == "down_pre":
            goal = pre
            if near(goal, 0.01) and np.linalg.norm(rotation_error(obs_hand_mat(obs), pick_frame)) < 0.08:
                self.phase, self.timer = "slide", 0
        if self.phase == "slide":  # slide the palm under the ball
            goal = under
            if near(goal, 0.006) or (near(goal, 0.012) and self.timer > 25):
                self.phase, self.timer = "scoop", 0
                self.scoop_goal = under + [0, 0, 0.05]
        if self.phase == "scoop":  # lift the ball off the stand and close the hand over it
            goal, grip = self.scoop_goal, 1.0
            if near(goal, 0.015) and self.timer > 8:
                self.phase = "windup"
        if self.phase == "windup":  # turn towards the cup while moving to the wind-up point
            goal, grip, frame = windup_pt, 1.0, throw_frame
            if near(goal, 0.012) and np.linalg.norm(rotation_error(obs_hand_mat(obs), throw_frame)) < 0.08:
                self.phase, self.timer = ("settle" if self.open_before_swing else "swing"), 0
        if self.phase == "settle":  # open the hand while still: the ball rests on the flat palm
            goal, frame = windup_pt, throw_frame
            if self.timer > 6:
                self.phase, self.timer = "swing", 0
        if self.phase == "swing" and self.ballistic:
            # Time-parameterised path: accelerate from the wind-up point to reach velocity v at R, then carry the
            # ball along its own free-flight parabola (it floats on the palm) and drop the hand away underneath.
            dt, tau = cfg.control_dt, self.timer * cfg.control_dt
            v_cmd = self.speed_gain * v
            t_acc = 2 * self.windup / np.linalg.norm(v_cmd)
            if tau <= t_acc:
                s_ = tau / t_acc  # constant acceleration from rest
                goal = windup_pt + (R_pt - windup_pt) * s_ ** 2
            else:
                tb = tau - t_acc
                goal = R_pt + v_cmd * tb - np.array([0, 0, 0.5 * GRAVITY * tb ** 2])
                if tb > self.carry_time:
                    self.phase, self.drop_from = "release", goal
            frame, speed = throw_frame, None
        elif self.phase == "swing":
            along = np.dot(site - R_pt, vhat)
            if not self.open_before_swing:
                grip = 1.0 if along < -self.open_lead else -1.0  # open while still moving
            goal, frame, speed = site + self.speed_gain * v * cfg.control_dt, throw_frame, None
            if along > self.follow:
                self.phase, self.timer = "release", 0
        if self.phase == "release":  # stop (or drop away under the ball); the ball flies on
            goal = self.drop_from - [0, 0, 0.10] if self.ballistic else site
            frame, speed = throw_frame, None
        return to_action(obs, goal, frame, grip, cfg.toss_max_step, cfg, speed=speed)


def expert_for(obs: np.ndarray, place: PlaceExpert, toss: TossExpert):
    return toss if int(np.argmax(obs[OBS_STAGE])) == 3 else place
