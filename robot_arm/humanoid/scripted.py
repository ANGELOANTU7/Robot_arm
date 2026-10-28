"""Hand-written experts for the G1 pyramid: they check the physics is solvable and serve as baselines."""

import numpy as np

from robot_arm.humanoid.config import G1_CFG, G1TaskConfig
from robot_arm.humanoid.controller import pinch_frame, rotation_error, rotvec_to_mat, site_for_pinch
from robot_arm.humanoid.envs import (OBS_CAP, OBS_CMD_POS, OBS_CMD_ROT, OBS_HAND_ROT, OBS_SITE, OBS_STAGE,
                                     OBS_TARGET, obs_cup)
from robot_arm.humanoid.objects import ORIENTATIONS, UP, cup_grasp_point, cup_lowest_z, cup_table_point

GRAVITY = 9.81
OPEN_STEPS = 14
BACK_OFF = 0.16  # how far the open hand slides back out of a released top cup (the fingertips reach ~11 cm past it)
FLIP_ROLL = np.pi  # turning a held cup over: roll the hand about its fingers' axis (forearm rotation)
FLIP_RELEASE = -0.2  # grip while letting go of a cup with the hand turned over: loosened, not wide open


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


def grasp_frame(yaw: float, mouth_up: bool, cfg: G1TaskConfig = G1_CFG) -> np.ndarray:
    """Hand orientation for a side grasp. A mouth-up cup widens upwards, so the hand rolls by the wall's slope
    to keep the flat palm parallel to the wall (a vertical palm would press the upper rim and tip the cup)."""
    taper = np.arctan2(cfg.cup.mouth_radius - cfg.cup.bottom_radius, cfg.cup.height)
    return hand_frame(yaw) @ roll_x(taper if mouth_up else 0.0)


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
    """rise -> over the pre-grasp point -> down -> slide in -> close -> lift -> (flip) -> carry -> lower -> open ->
    leave.

    The top cup starts mouth-down like the others: it is gripped where it will be held once it is the right way
    up, lifted, carried over the pyramid, turned over in the air by rolling the hand about its fingers' axis, and
    placed mouth-up. It is left by sliding the loosened hand back out along the fingers before rising: rising
    straight up would drag it by its widening walls."""

    def __init__(self, cfg: G1TaskConfig = G1_CFG, yaw: float = 0.0, pre_dist: float = 0.16):
        self.cfg, self.yaw, self.pre_dist = cfg, yaw, pre_dist
        self.reset()

    def reset(self):
        self.phase, self.timer, self.stage = "rise", 0, None
        self.flip, self.flipped = None, False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        stage = int(np.argmax(obs[OBS_STAGE]))
        if stage != self.stage:
            self.reset()
            self.stage = stage
        k = stage
        T, h = cfg.table_height, cfg.cup.height
        xh = hand_frame(self.yaw)[:, 0]
        site = obs[OBS_SITE]
        cup_pos, cup_axis = obs_cup(obs, k)
        final_up = ORIENTATIONS[k] is UP
        if self.flip is None:
            self.flip = bool(final_up and cup_axis[2] < 0)
        R_grasp = grasp_frame(self.yaw, final_up and not self.flip, cfg)
        R_place = R_grasp @ roll_x(FLIP_ROLL) if self.flip else R_grasp
        R = R_place if self.flipped else R_grasp
        final_frac = cfg.top_grasp_height_frac if k == 2 else cfg.grasp_height_frac
        # Grip height from the cup's table side as it stands now: the same spot on the cup before and after a flip.
        frac = final_frac if (cup_axis[2] > 0) == final_up else 1 - final_frac
        grasp = cup_grasp_point(cup_pos, cup_axis, frac, cfg)
        target = obs[OBS_TARGET]
        final_axis = np.array([0, 0, 1.0]) if ORIENTATIONS[k] is UP else np.array([0, 0, -1.0])
        target_tp = cup_table_point(target, final_axis, cfg)  # where the cup's table-side end must go
        safe_z = T + 0.30
        # High enough that the held cup (hanging frac * h below the hand) clears cups standing on the table.
        carry_z = T + h + final_frac * h + 0.04
        flip_z = T + h + 0.16  # turning over, the cup sweeps about 8 cm around the grip
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
            goal, grip = np.array([site[0], site[1], flip_z if self.flip else carry_z]), 1.0
            if near(goal, 0.015):
                self.phase = "to_flip" if self.flip else "carry"
        if self.phase == "to_flip":  # carried upright to above the pyramid, where the arm can roll the hand over
            goal, grip = np.array([target[0] - 0.03, target[1], flip_z]), 1.0
            if near(goal, 0.02):
                self.phase = "flip"
        if self.phase == "flip":
            goal, grip, R = np.array([target[0] - 0.03, target[1], flip_z]), 1.0, R_place
            if np.linalg.norm(rotation_error(obs_hand_mat(obs), R_place)) < 0.08:
                self.phase, self.flipped = "carry", True
        # From here on, servo the held cup's table-side end (not the hand) onto the target, so slip and tilt in
        # the grasp are corrected.
        cup_tp = cup_table_point(cup_pos, cup_axis, cfg)
        cup_offset = site - cup_tp
        if self.phase == "carry":
            goal, grip = np.array([*(target_tp[:2] + cup_offset[:2]), carry_z]), 1.0
            if np.linalg.norm(cup_tp[:2] - target_tp[:2]) < 0.006:
                self.phase = "lower"
        if self.phase == "lower":
            goal, grip = target_tp + cup_offset + [0, 0, 0.002], 1.0
            goal[2] = max(goal[2], site[2] - cfg.max_step)  # come down gently
            if cup_lowest_z(cup_pos, cup_axis, cfg) - target_tp[2] < 0.003:  # touched down
                self.phase, self.timer = "open", 0
        # A hand turned over has its thumb underneath: opening it all the way would swing the thumb into a base
        # cup, so it only loosens its grip until it is out of the cup.
        loosen = FLIP_RELEASE if self.flipped else -1.0
        if self.phase == "open":
            goal, grip = site, loosen
            if self.timer > OPEN_STEPS:  # fully open (see release_rate)
                self.phase, self.back_from = ("back" if k == 2 else "up"), site.copy()
        if self.phase == "back":
            goal, grip = self.back_from - BACK_OFF * xh, loosen
            if near(goal, 0.01):
                self.phase = "up"
        if self.phase == "up":
            goal = np.array([site[0], site[1], safe_z])
        return to_action(obs, goal, R, grip, cfg.max_step, cfg)


class TossExpert:
    """Pinch the cap off the table and throw it sideways into the top cup.

    - Pick: hand pointing down over the cap; lower until the pinch point is level with the cap; pinch.
    - Throw: lift, then wind up on cup B's side of the pyramid. Swing along the line of the two base cups, so
      the cap's push on landing goes the way the top cup is held by both base cups (thrown front-to-back it
      would tip the top cup). Release at a ballistic velocity that drops the cap into the mouth: the hand
      follows the cap's own flight path while the fingers open, then stops and pulls up.

    `release_bearing` turns the release point away from the base cups' axis (0: straight along it, from B's
    side); `release_dist` and `release_height` place the cap at release relative to the mouth.
    """

    # Defaults from a parameter sweep with the aim calibrated over random layouts (see the README).
    def __init__(self, cfg: G1TaskConfig = G1_CFG, flight_time: float = 0.30, release_dist: float = 0.12,
                 release_height: float = 0.08, release_bearing: float = 0.05, windup: float = 0.18,
                 speed_gain: float = 1.0, aim_offset=(0.0, 0.0, 0.0), carry_time: float = 0.05,
                 close_steps: int = 6, pinch_depth: float = 0.008):
        self.cfg = cfg
        self.flight_time, self.release_dist, self.release_height = flight_time, release_dist, release_height
        self.release_bearing, self.windup, self.speed_gain = release_bearing, windup, speed_gain
        self.aim_offset = np.asarray(aim_offset, float)
        self.carry_time, self.close_steps, self.pinch_depth = carry_time, close_steps, pinch_depth
        self.reset()

    def reset(self):
        self.phase, self.timer = "rise", 0
        self.plan, self.aimed, self.retries = None, False, 0

    @staticmethod
    def base_axis(obs: np.ndarray) -> np.ndarray:
        """Horizontal unit vector from base cup A to base cup B."""
        a, b = obs_cup(obs, 0)[0], obs_cup(obs, 1)[0]
        u = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        return u / max(np.linalg.norm(u), 1e-6)

    def release_plan(self, mouth: np.ndarray, axis: np.ndarray):
        """Release point (of the cap) and velocity so a free cap passes through the mouth after flight_time."""
        c, s = np.cos(self.release_bearing), np.sin(self.release_bearing)
        side = np.array([c * axis[0] - s * axis[1], s * axis[0] + c * axis[1], 0.0])
        R = mouth + self.release_dist * side + [0, 0, self.release_height]
        t = self.flight_time
        v = (mouth + self.aim_offset - R + np.array([0, 0, 0.5 * GRAVITY * t * t])) / t
        return R, v

    pinch_frame = staticmethod(pinch_frame)
    site_for = staticmethod(site_for_pinch)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T = cfg.table_height
        site, cap = obs[OBS_SITE], obs[OBS_CAP]
        mouth = obs[OBS_TARGET]
        if self.plan is None:
            self.plan = self.release_plan(mouth, self.base_axis(obs))
        R_pt, v = self.plan
        throw_dir = np.array([v[0], v[1], 0.0]) / np.linalg.norm(v[:2])
        frame = self.pinch_frame(throw_dir)
        hand = obs_hand_mat(obs)
        safe_z = T + 2 * cfg.cup.height + 0.12  # the hand stays well above the pyramid while it turns
        # The fingertips skim the table as they close, so they meet the cap low on its sides (from pick trials).
        pinch_at_cap = cap - [0, 0, self.pinch_depth]
        vhat = v / np.linalg.norm(v)
        windup_pt = R_pt - vhat * self.windup
        self.timer += 1
        grip, speed = -1.0, cfg.max_step
        near = lambda p, tol: np.linalg.norm(p - site) < tol  # noqa: E731
        rot_err = np.linalg.norm(rotation_error(hand, frame))
        # Aligned, or as close as the joint limits allow (a few degrees off): don't wait for it forever.
        aligned = rot_err < 0.07 or (rot_err < 0.12 and self.timer > 20)

        # Straight up, across to above the cap, and only then turn the fingers down: turning over the pyramid
        # would sweep the fingers into the top cup.
        above_cap = self.site_for(pinch_at_cap, frame)
        above_cap[2] = max(safe_z, site[2]) if self.phase in ("rise", "traverse", "turn") else safe_z
        if self.phase == "rise" and self.timer == 1 and site[2] < safe_z - 0.05:
            # Right after letting go of the top cup the hand is still around it: slide out along the fingers first.
            self.phase, self.back_from = "back", site.copy()
        if self.phase == "back":
            goal, frame_cmd = self.back_from - BACK_OFF * hand[:, 0], hand
            if near(goal, 0.01) or self.timer > 15:
                self.phase = "rise"
        if self.phase == "rise":
            goal, frame_cmd = np.array([site[0], site[1], max(safe_z, site[2])]), hand
            if site[2] > safe_z - 0.03:
                self.phase = "traverse"
        if self.phase == "traverse":
            goal, frame_cmd = np.array([above_cap[0], above_cap[1], max(safe_z, site[2])]), hand
            if np.linalg.norm(goal[:2] - site[:2]) < 0.03:
                self.phase = "turn"
        if self.phase == "turn":
            goal, frame_cmd = np.array([above_cap[0], above_cap[1], max(safe_z, site[2])]), frame
            if aligned:
                self.phase, self.timer = "over", 0
        if self.phase == "over":
            goal, frame_cmd = self.site_for(pinch_at_cap + [0, 0, 0.06], frame), frame
            if near(goal, 0.01) and aligned:
                self.phase = "down"
        if self.phase == "down":
            goal, frame_cmd, speed = self.site_for(pinch_at_cap, frame), frame, 0.5 * cfg.max_step
            if near(goal, 0.004) or (near(goal, 0.008) and self.timer > 20):
                self.phase, self.timer = "pinch", 0
        if self.phase == "pinch":  # close progressively, so the cap is not flicked away
            goal, frame_cmd = self.site_for(pinch_at_cap, frame), frame
            grip = min(1.0, -1.0 + 2.0 * self.timer / self.close_steps)
            if self.timer > self.close_steps + 2:
                self.phase = "lift"
                self.lift_goal = self.site_for(cap + [0, 0, 0.10], frame)
        if self.phase == "lift":
            goal, frame_cmd, grip = self.lift_goal, frame, 1.0
            if site[2] > self.lift_goal[2] - 0.06 and cap[2] < T + 2 * cfg.cap_half_height + 0.004 and \
                    self.retries < 2:
                self.phase, self.timer, self.retries = "over", 0, self.retries + 1  # the cap stayed behind: retry
            elif near(goal, 0.015) or site[2] > self.lift_goal[2] - 0.03:
                self.phase, self.timer = "windup", 0
        if self.phase == "windup":
            if not self.aimed:  # aim at the top cup as it is now: it may have settled since the plan was made
                self.plan, self.aimed = self.release_plan(mouth, self.base_axis(obs)), True
                R_pt, v = self.plan
                vhat = v / np.linalg.norm(v)
                windup_pt = R_pt - vhat * self.windup
            goal, frame_cmd, grip = self.site_for(windup_pt, frame), frame, 1.0
            if near(goal, 0.01) and aligned:
                self.phase, self.timer = "settle", 0
        if self.phase == "settle":
            goal, frame_cmd, grip = self.site_for(windup_pt, frame), frame, 1.0
            if self.timer > 3:
                self.phase, self.timer = "swing", 0
        if self.phase == "swing":
            # Constant acceleration from rest at the wind-up point, reaching speed_gain * v at R; then the hand
            # carries the cap along its own free-flight parabola while the fingers open, so they do not push it.
            tau = self.timer * cfg.control_dt
            v_cmd = self.speed_gain * v
            t_acc = 2 * self.windup / np.linalg.norm(v_cmd)
            grip = 1.0
            if tau <= t_acc:
                p = windup_pt + (R_pt - windup_pt) * (tau / t_acc) ** 2
            else:
                tb = tau - t_acc
                p = R_pt + v_cmd * tb - np.array([0, 0, 0.5 * GRAVITY * tb ** 2])
                grip = -1.0
                if tb > self.carry_time:
                    self.phase, self.drop_from = "release", p
            goal, frame_cmd, speed = self.site_for(p, frame), frame, None
        if self.phase == "release":  # stop, and pull the hand up and back out of the cap's way
            goal = self.site_for(self.drop_from - 0.08 * throw_dir + [0, 0, 0.08], frame)
            frame_cmd, speed = frame, None
        return to_action(obs, goal, frame_cmd, grip, cfg.toss_max_step, cfg, speed=speed)


def expert_for(obs: np.ndarray, place: PlaceExpert, toss: TossExpert):
    return toss if int(np.argmax(obs[OBS_STAGE])) == 3 else place
