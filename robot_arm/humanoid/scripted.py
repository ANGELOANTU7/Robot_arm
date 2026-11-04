"""Hand-written experts for the G1 pyramid: they check the physics is solvable and serve as baselines."""

import numpy as np

from robot_arm.humanoid.config import G1_CFG, G1TaskConfig
from robot_arm.humanoid.controller import pinch_frame, rotation_error, rotvec_to_mat, site_for_pinch
from robot_arm.humanoid.hands import get_hand
from robot_arm.humanoid.envs import (OBS_CAP, OBS_CAP_VEL, OBS_CMD_POS, OBS_CMD_ROT, OBS_HAND_ROT, OBS_SITE,
                                     OBS_STAGE, OBS_TARGET, THUMBS_UP, obs_cup)
from robot_arm.humanoid.objects import (ORIENTATIONS, UP, cup_grasp_point, cup_lowest_z, cup_table_point,
                                        reach_heading)

GRAVITY = 9.81
OPEN_STEPS = 3  # control steps to open the hand fully (see release_rate)
BACK_OFF = 0.16  # how far the open hand slides back out of a released top cup (the fingertips reach ~11 cm past it)
FLIP_ROLL = np.pi  # turning a held cup over: roll the hand about its fingers' axis (forearm rotation)
FLIP_RATE = 0.1  # rad per control step when the roll is made a step at a time (PlaceExpert's flip_dir)
FLIP_RELEASE = -1.0  # grip while letting go of a cup with the hand turned over
THUMBS_UP_YAW = -0.3  # heading of the fist while showing the thumbs-up
NEUTRAL_XY = (0.30, -0.22)  # in front of the right shoulder: the hand passes here, upright, before reaching for the cap


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


def taper_angle(cfg: G1TaskConfig = G1_CFG) -> float:
    """Slope of a cup's wall from the vertical."""
    return float(np.arctan2(cfg.cup.mouth_radius - cfg.cup.bottom_radius, cfg.cup.height))


def grasp_frame(yaw: float, mouth_up: bool, cfg: G1TaskConfig = G1_CFG) -> np.ndarray:
    """Hand orientation for a side grasp. A mouth-up cup widens upwards, so the hand rolls by the wall's slope
    to keep the flat palm parallel to the wall (a vertical palm would press the upper rim and tip the cup)."""
    return hand_frame(yaw) @ roll_x(taper_angle(cfg) if mouth_up else 0.0)


def upright_hand(R_nominal: np.ndarray, hand: np.ndarray, cup_axis: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Hand orientation near R_nominal that turns a held cup's axis to `want`, given the cup's axis now and the
    hand's orientation now (the cup is assumed to move with the hand)."""
    a = R_nominal @ (hand.T @ cup_axis)  # where the cup's axis would point with the hand at R_nominal
    c = np.cross(a, want)
    s, cos = np.linalg.norm(c), float(np.dot(a, want))
    if s < 1e-9:
        return R_nominal
    return rotvec_to_mat(c / s * np.arctan2(s, cos)) @ R_nominal


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

    def __init__(self, cfg: G1TaskConfig = G1_CFG, yaw: float | None = None, place_yaw: float = 0.0,
                 top_place_yaw: float = 0.0, pre_dist: float = 0.16, drop: float = 0.014,
                 flip_taper_sign: float = -1.0, flip_wait: int = 0, flip_exit_rise: float = 0.0,
                 flip_exit_speed: float = 1.0, flip_offset: float = 0.012, flip_exit_out: float | None = 0.5,
                 align_tol: float = 0.006, top_yaw_rel: float | None = None, flip_dir: float | None = None):
        """`yaw`: hand heading for the grasp (None: away from the shoulder, towards the cup); `place_yaw` and
        `top_place_yaw`: heading while putting a base cup / the top cup down (the hand turns while carrying).
        `flip_dir`: turn the top cup over a step at a time, rolling the hand this way round (+1 / -1) about its
        fingers' axis. None: aim straight for the turned-over orientation; half a turn away, which way round the
        hand goes is then down to small differences (one way can end at the arm's joint limits).
        Settings the hand needs (`Hand.expert`) replace the defaults unless they are given here."""
        given = {k: v for k, v in locals().items() if k not in ("self", "cfg")}
        defaults = PlaceExpert.__init__.__kwdefaults__ or dict(zip(PlaceExpert.__init__.__code__.co_varnames[2:],
                                                                    PlaceExpert.__init__.__defaults__[1:]))
        for k, v in get_hand(cfg.hand).expert.items():
            if given[k] == defaults[k]:
                given[k] = v
        yaw, place_yaw, top_place_yaw, pre_dist, drop = (given[k] for k in ("yaw", "place_yaw", "top_place_yaw",
                                                                            "pre_dist", "drop"))
        flip_taper_sign, flip_wait, flip_exit_rise = given["flip_taper_sign"], given["flip_wait"], given["flip_exit_rise"]
        flip_exit_speed, flip_offset, flip_exit_out = given["flip_exit_speed"], given["flip_offset"], given["flip_exit_out"]
        self.align_tol, self.top_yaw_rel, self.flip_dir = given["align_tol"], given["top_yaw_rel"], given["flip_dir"]
        self.cfg, self.yaw, self.pre_dist, self.drop = cfg, yaw, pre_dist, drop
        self.place_yaw, self.top_place_yaw = place_yaw, top_place_yaw
        if cfg.real:
            flip_wait = 12  # the heavier hand needs a moment to settle after letting go
            self.top_yaw_rel, self.top_place_yaw = None, 0.0  # the same heading whatever the base cups did
        self.flip_taper_sign, self.flip_wait = flip_taper_sign, flip_wait
        self.flip_exit_rise, self.flip_exit_speed = flip_exit_rise, flip_exit_speed
        self.flip_offset, self.flip_exit_out = flip_offset, flip_exit_out
        self.reset()

    def reset(self):
        self.phase, self.timer, self.stage = "rise", 0, None
        self.open_at = None
        self.flip, self.flipped, self.R_hold = None, False, None
        self.yaw_g = self.yaw_p = None
        self.z_hist = []

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        stage = int(np.argmax(obs[OBS_STAGE]))
        if stage != self.stage:
            self.reset()
            self.stage = stage
        k = stage
        T, h = cfg.table_height, cfg.cup.height
        site = obs[OBS_SITE]
        cup_pos, cup_axis = obs_cup(obs, k)
        final_up = ORIENTATIONS[k] is UP
        if self.flip is None:
            self.flip = bool(final_up and cup_axis[2] < 0)
            self.yaw_g = reach_heading(cup_pos, cfg) if self.yaw is None else self.yaw
            self.yaw_p = self.top_place_yaw if k == 2 else self.place_yaw
            if k == 2 and self.top_yaw_rel is not None:  # relative to the line from base cup A to base cup B
                a, b = obs_cup(obs, 0)[0], obs_cup(obs, 1)[0]
                yaw = np.arctan2(b[1] - a[1], b[0] - a[0]) + self.top_yaw_rel
                self.yaw_p = float((yaw + np.pi) % (2 * np.pi) - np.pi)
        xh_p = hand_frame(self.yaw_p)[:, 0]
        if self.flip:
            # The palm lies along the cup's sloping wall as it will once the cup is turned over (mirror of a
            # mouth-up grasp), then the hand rolls half a turn. A cup is round, so turning the hand about the
            # vertical in between (to the placing heading) doesn't change how it ends up.
            taper = roll_x(self.flip_taper_sign * taper_angle(cfg))
            R_grasp = hand_frame(self.yaw_g) @ taper
            R_place = hand_frame(self.yaw_p) @ taper @ roll_x(FLIP_ROLL)
        else:
            R_grasp = grasp_frame(self.yaw_g, final_up, cfg)
            R_place = grasp_frame(self.yaw_p, final_up, cfg)
        R = R_place if self.flipped else R_grasp
        approach = hand_frame(self.yaw_g) @ np.asarray(get_hand(cfg.hand).approach)  # the hand's way onto the cup
        final_frac = cfg.top_grasp_height_frac if k == 2 else cfg.grasp_height_frac
        # Grip height from the cup's table side as it stands now: the same spot on the cup before and after a flip.
        frac = final_frac if (cup_axis[2] > 0) == final_up else 1 - final_frac
        grasp = cup_grasp_point(cup_pos, cup_axis, frac, cfg)
        target = obs[OBS_TARGET]
        final_axis = np.array([0, 0, 1.0]) if ORIENTATIONS[k] is UP else np.array([0, 0, -1.0])
        target_tp = cup_table_point(target, final_axis, cfg)  # where the cup's table-side end must go
        if self.flip:  # let go of it a little towards the palm: the opening hand pushes it away from the palm
            palm = R_place[:, 1] * np.array([1.0, 1.0, 0.0])
            target_tp = target_tp - self.flip_offset * palm / max(np.linalg.norm(palm), 1e-6)
        safe_z = T + 0.30
        # High enough that the held cup (hanging frac * h below the hand) clears cups standing on the table.
        carry_z = T + h + final_frac * h + 0.04
        flip_z = T + h + 0.16  # turning over, the cup sweeps about 8 cm around the grip
        pre = grasp - self.pre_dist * approach
        self.timer += 1
        grip = -1.0
        near = lambda p, tol: np.linalg.norm(p - site) < tol  # noqa: E731

        if self.phase == "rise":
            goal = np.array([site[0], site[1], safe_z])
            if cfg.real and site[2] < T + h + 0.12:  # the fingers may still be around the last cup: don't turn yet
                R = obs_hand_mat(obs)
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
                self.phase, self.timer = ("to_flip" if self.flip else "carry"), 0
        flip_at = np.array([*(target[:2] - 0.03 * xh_p[:2]), flip_z])
        if self.phase == "to_flip":  # carried mouth-down to above the pyramid, turning to the placing heading
            goal, grip, R = flip_at, 1.0, hand_frame(self.yaw_p) @ taper
            if (near(goal, 0.02) and np.linalg.norm(rotation_error(obs_hand_mat(obs), R)) < 0.1) or \
                    (near(goal, 0.04) and self.timer > 40):
                self.phase, self.timer = "flip", 0
        if self.phase == "flip":  # then roll the hand over about its fingers' axis
            goal, grip, R = flip_at, 1.0, R_place
            if self.flip_dir is not None:
                R = hand_frame(self.yaw_p) @ taper @ roll_x(self.flip_dir * min(FLIP_ROLL, FLIP_RATE * self.timer))
            err = np.linalg.norm(rotation_error(obs_hand_mat(obs), R_place))
            if err < 0.08 or (self.timer > 25 and err < 0.2) or (self.timer > 50 and err < 0.4):
                self.phase, self.flipped, self.timer = "carry", True, 0  # or as close as the joint limits allow
        # From here on, servo the held cup's table-side end (not the hand) onto the target, so slip and tilt in
        # the grasp are corrected: the hand turns so the cup's axis is upright whatever its tilt in the grip.
        if self.phase in ("carry", "lower") and (self.flipped or not self.flip):
            R = self.R_hold = upright_hand(R_place, obs_hand_mat(obs), cup_axis, final_axis)
        elif self.phase in ("open", "back", "up") and self.R_hold is not None:
            R = self.R_hold  # let go without turning the hand
        cup_tp = cup_table_point(cup_pos, cup_axis, cfg)
        cup_offset = site - cup_tp
        if self.phase == "carry":
            goal, grip = np.array([*(target_tp[:2] + cup_offset[:2]), carry_z]), 1.0
            off = np.linalg.norm(cup_tp[:2] - target_tp[:2])
            # (The real robot's delay makes this settle slower. Coming down early brings a wide rim onto a standing cup.)
            patience = 3 if cfg.real else 1
            if off < self.align_tol or (self.timer > 20 * patience and off < 2 * self.align_tol) or \
                    (self.timer > 40 * patience and off < 0.012):
                self.phase = "lower"
        if self.phase == "lower":
            goal, grip = target_tp + cup_offset + [0, 0, 0.002], 1.0
            goal[2] = max(goal[2], site[2] - cfg.max_step)  # come down gently
            # Touched down. A cup held with the hand turned over is let go as soon as it stops sinking: a finger
            # may rest on a base cup's top before the cup is fully down, and the cup drops the rest of the way.
            self.z_hist.append(cup_pos[2])
            stalled = len(self.z_hist) > 3 and self.z_hist[-4] - self.z_hist[-1] < 0.002
            above = cup_lowest_z(cup_pos, cup_axis, cfg) - target_tp[2]
            if (self.flipped and ((above < 0.025 and stalled) or above < self.drop)) or \
                    (not self.flipped and (above < 0.003 or (above < 0.012 and stalled))):
                self.phase, self.timer = "open", 0
        # A hand turned over has its thumb underneath: opening it all the way would swing the thumb into a base
        # cup, so it only loosens its grip until it is out of the cup.
        loosen = FLIP_RELEASE if self.flipped else -1.0
        if self.phase == "open":
            if self.open_at is None:
                self.open_at = site.copy()
            # Held where it let go: "stay where you are" lets a heavy hand on soft servos creep down onto the cup.
            goal, grip = (self.open_at if cfg.real else site), loosen
            if self.timer > OPEN_STEPS + (self.flip_wait if self.flipped else 0):  # fully open (see release_rate)
                self.phase, self.back_from = ("back" if k == 2 else "up"), site.copy()
                self.timer = 0
        if self.phase == "back":
            exit_dir = -xh_p + (np.array([0, 0, self.flip_exit_rise]) if self.flipped else 0.0)
            if self.flipped and self.flip_exit_out is not None:  # up, and out along the cup's sloping wall
                palm = R_place[:, 1] * np.array([1.0, 1.0, 0.0])
                exit_dir = np.array([0, 0, 1.0]) - self.flip_exit_out * palm / max(np.linalg.norm(palm), 1e-6)
            goal, grip = self.back_from + BACK_OFF * exit_dir / np.linalg.norm(exit_dir), loosen
            if self.flipped:
                speed_back = self.flip_exit_speed * cfg.max_step
                goal = site + np.clip(goal - site, -speed_back, speed_back)
            if near(self.back_from + BACK_OFF * exit_dir / np.linalg.norm(exit_dir), 0.01) or (cfg.real and self.timer > 40):
                self.phase = "up"
        if self.phase == "up":
            goal = np.array([site[0], site[1], safe_z])
            if cfg.real and self.flipped:  # a hand turned over leaves slowly, and away from the cup it has just left
                away = site[:2] - cup_pos[:2]
                goal[:2] += 0.04 * away / max(np.linalg.norm(away), 1e-6)
                goal = site + np.clip(goal - site, -0.4 * cfg.max_step, 0.4 * cfg.max_step)
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

    A miss: once the cap lies still, it is picked up where it landed and thrown again, aimed off by however far
    the last throw passed from the mouth (`correction` of it). Once it is in: a thumbs-up beside the shoulder.
    """

    # Defaults from a parameter sweep with the aim calibrated over random layouts (see the README).
    def __init__(self, cfg: G1TaskConfig = G1_CFG, flight_time: float = 0.30, release_dist: float = 0.12,
                 release_height: float = 0.08, release_bearing: float = 0.05, windup: float = 0.18,
                 speed_gain: float = 1.0, aim_offset=(0.0, 0.0, 0.0), carry_time: float = 0.05,
                 close_steps: int = 6, pinch_depth: float = 0.008, correction: float = 0.8):
        self.cfg, self.correction, self.hand = cfg, correction, get_hand(cfg.hand)
        if cfg.real:  # calibrated on the real arm's lag: a longer swing, released by position (see "swing")
            windup, speed_gain, aim_offset = 0.26, 0.85, (-0.02, 0.03, 0.0)
        self.flight_time, self.release_dist, self.release_height = flight_time, release_dist, release_height
        self.release_bearing, self.windup, self.speed_gain = release_bearing, windup, speed_gain
        self.aim_offset = np.asarray(aim_offset, float)
        self.carry_time, self.close_steps, self.pinch_depth = carry_time, close_steps, pinch_depth
        self.reset()

    def reset(self):
        self.phase, self.timer = "rise", 0
        self.plan, self.aimed, self.retries = None, False, 0
        self.attempt, self.aim = 0, np.zeros(3)
        self.miss, self.still, self.resets = None, 0, 0
        self.release_t = None

    def next_attempt(self):
        """After a miss: aim off by the miss, and go and pick the cap up again."""
        if self.miss is not None:
            self.aim = self.aim - self.correction * np.array([*self.miss, 0.0])
        attempt, aim = self.attempt + 1, self.aim
        self.reset()
        self.attempt, self.aim = attempt, aim

    @staticmethod
    def base_axis(obs: np.ndarray) -> np.ndarray:
        """Horizontal unit vector from base cup A to base cup B."""
        a, b = obs_cup(obs, 0)[0], obs_cup(obs, 1)[0]
        u = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        return u / max(np.linalg.norm(u), 1e-6)

    def release_plan(self, mouth: np.ndarray, axis: np.ndarray):
        """Release point (of the cap) and velocity so a free cap passes through the mouth after flight_time."""
        c, s = np.cos(self.release_bearing), np.sin(self.release_bearing)
        side = self.hand.throw_side * np.array([c * axis[0] - s * axis[1], s * axis[0] + c * axis[1], 0.0])
        R = mouth + self.release_dist * side + [0, 0, self.release_height]
        t = self.flight_time
        along = mouth[:2] - R[:2]
        long = self.hand.throw_long * np.array([*along / max(np.linalg.norm(along), 1e-6), 0.0])
        v = (mouth + self.aim_offset + self.aim + long - R + np.array([0, 0, 0.5 * GRAVITY * t * t])) / t
        return R, v

    def pinch_frame(self, throw_dir: np.ndarray) -> np.ndarray:
        return pinch_frame(throw_dir, self.hand)

    def site_for(self, pinch_world: np.ndarray, R: np.ndarray) -> np.ndarray:
        return site_for_pinch(pinch_world, R, self.hand)

    def thumbs_up(self, obs: np.ndarray) -> np.ndarray:
        """Over to the spot beside the shoulder, turning the hand thumb-side up, then close the fist."""
        cfg = self.cfg
        site, spot = obs[OBS_SITE], obs[OBS_TARGET]
        R = hand_frame(THUMBS_UP_YAW) @ roll_x(self.hand.thumbs_up_roll)
        goal = np.array([spot[0], spot[1], max(spot[2], site[2])]) if site[2] < spot[2] - 0.02 else spot
        err = np.linalg.norm(rotation_error(obs_hand_mat(obs), R))
        if np.linalg.norm(spot - site) < 0.03 and err < 0.2:  # (as close as this arm gets at the spot)
            self.phase = "thumbs_up"
        grip = 1.0 if self.phase == "thumbs_up" else -1.0
        return to_action(obs, goal, R, grip, cfg.toss_max_step, cfg, speed=0.03)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T = cfg.table_height
        site, cap = obs[OBS_SITE], obs[OBS_CAP]
        mouth = obs[OBS_TARGET]
        if int(np.argmax(obs[OBS_STAGE])) == THUMBS_UP:
            return self.thumbs_up(obs)
        if self.phase == "release":
            # Where the cap passes the mouth's height on its way down (for the next aim), and whether it lies still.
            vel = obs[OBS_CAP_VEL]
            if self.miss is None and vel[2] < 0 and cap[2] < mouth[2]:
                self.miss = cap[:2] - mouth[:2]
            self.still = self.still + 1 if np.linalg.norm(vel) < 0.03 else 0
            if self.still > 4:
                self.next_attempt()
        if self.plan is None:
            self.plan = self.release_plan(mouth, self.base_axis(obs))
        R_pt, v = self.plan
        throw_dir = np.array([v[0], v[1], 0.0]) / np.linalg.norm(v[:2])
        frame = self.pinch_frame(throw_dir)
        if self.phase in ("rise", "neutral", "traverse", "turn", "over", "down", "pinch", "lift"):
            c, s = np.cos(self.hand.pick_turn), np.sin(self.hand.pick_turn)
            frame = self.pinch_frame(np.array([c * throw_dir[0] - s * throw_dir[1], s * throw_dir[0] + c * throw_dir[1], 0.0]))
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
        aligned = rot_err < 0.07 or (rot_err < 0.12 and self.timer > 20) or (rot_err < 0.3 and self.timer > 40)

        # Straight up, across to above the cap, and only then turn the fingers down: turning over the pyramid
        # would sweep the fingers into the top cup.
        above_cap = self.site_for(pinch_at_cap, frame)
        above_cap[2] = max(safe_z, site[2]) if self.phase in ("rise", "traverse", "turn") else safe_z
        if self.phase == "rise":
            # The place stage ends with the hand just clear of the top cup: up, slowly and away from the cup.
            away = site[:2] - mouth[:2]
            away = away / max(np.linalg.norm(away), 1e-6)
            goal, frame_cmd = np.array([*(site[:2] + 0.02 * away), max(safe_z, site[2])]), hand
            speed = 0.02
            if site[2] > safe_z - 0.03:
                self.phase, self.timer = "neutral", 0
        if self.phase == "neutral":
            # Upright in front of the shoulder first: after turning the top cup over the hand is upside down, and
            # turning from there straight to the pinch runs the wrist into its limits.
            # Turning the hand over near the pyramid would sweep the fingers through the top cup: away first.
            clear = np.linalg.norm(site[:2] - mouth[:2]) > 0.15
            goal, frame_cmd = np.array([*NEUTRAL_XY, safe_z]), hand_frame(0.0) if clear else hand
            if (near(goal, 0.03) and np.linalg.norm(rotation_error(hand, hand_frame(0.0))) < 0.2) or self.timer > 60:
                self.phase, self.timer = "traverse", 0
        if self.phase == "traverse":
            goal, frame_cmd = np.array([above_cap[0], above_cap[1], max(safe_z, site[2])]), hand_frame(0.0)
            if np.linalg.norm(goal[:2] - site[:2]) < 0.03 or self.timer > 40:
                self.phase, self.timer = "turn", 0
        if self.phase == "turn":
            goal, frame_cmd = np.array([above_cap[0], above_cap[1], max(safe_z, site[2])]), frame
            if aligned:
                self.phase, self.timer = "over", 0
            elif self.timer > 60 and self.resets < 2:  # stuck at a joint limit: start the turn again from upright
                self.phase, self.timer, self.resets = "neutral", 0, self.resets + 1
        if self.phase == "over":
            goal, frame_cmd = self.site_for(pinch_at_cap + [0, 0, 0.06], frame), frame
            # (or as close as the arm gets: not every spot on the table is reached with the pinch exactly)
            if (near(goal, 0.01) and aligned) or (near(goal, 0.03) and self.timer > 30) or self.timer > 60:
                self.phase, self.timer = "down", 0
        if self.phase == "down":
            goal, frame_cmd, speed = self.site_for(pinch_at_cap, frame), frame, 0.5 * cfg.max_step
            if near(goal, 0.004) or (near(goal, 0.008) and self.timer > 20) or self.timer > 40:
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
                    self.retries < 5:
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
                # A lagging arm is not there yet when the clock says so: let go when the cap has reached the release
                # point, or it leaves short and slow.
                there = (cap - R_pt) @ vhat > 0 or tau > 2.5 * t_acc
                if not cfg.real or there:
                    grip = -1.0
                    if cfg.real:
                        self.release_t = self.release_t or tau
                    if not cfg.real and tb > self.carry_time or cfg.real and tau - self.release_t > self.carry_time:
                        self.phase, self.drop_from = "release", p
            goal, frame_cmd, speed = self.site_for(p, frame), frame, None
        if self.phase == "release":  # stop, and pull the hand up and back out of the cap's way
            goal = self.site_for(self.drop_from - 0.08 * throw_dir + [0, 0, 0.08], frame)
            frame_cmd, speed = frame, None
        return to_action(obs, goal, frame_cmd, grip, cfg.toss_max_step, cfg, speed=speed)
