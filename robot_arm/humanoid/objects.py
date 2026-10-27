"""Cup pyramid geometry and the success checks for building it and getting the ball in."""

from dataclasses import dataclass

import mujoco
import numpy as np

from robot_arm.humanoid.config import G1_CFG, G1TaskConfig

UP = np.array([1.0, 0.0, 0.0, 0.0])  # cup mouth up
DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # cup mouth down (180 degrees about x)


@dataclass
class PyramidLayout:
    """Target poses of the three cups. Positions are each cup's body origin (centre of its bottom disc)."""

    base_a: np.ndarray
    base_b: np.ndarray
    top: np.ndarray

    @classmethod
    def make(cls, center_xy, yaw: float, cfg: G1TaskConfig = G1_CFG) -> "PyramidLayout":
        cup = cfg.cup
        spacing = 2 * (cup.mouth_radius + cup.wall) + cfg.base_gap
        u = np.array([np.cos(yaw), np.sin(yaw)])
        c = np.asarray(center_xy, float)
        z = cfg.table_height + cup.height  # inverted cups: origin on top; top cup: origin on the base tops
        return cls(np.array([*(c - u * spacing / 2), z]), np.array([*(c + u * spacing / 2), z]),
                   np.array([*c, z]))

    def targets(self) -> list[np.ndarray]:
        return [self.base_a, self.base_b, self.top]

    @property
    def mouth_center(self) -> np.ndarray:
        """Centre of the top cup's mouth: where the ball must go."""
        return self.top + [0, 0, G1_CFG.cup.height]


ORIENTATIONS = [DOWN, DOWN, UP]
TOP_TILT_DEG = 20.0  # how far the top cup may lean and still count as standing  # final orientation of cup0, cup1, cup2 (also how they are given)


def cup_axis(data: mujoco.MjData, body: int) -> np.ndarray:
    """Unit vector out of the cup's mouth."""
    return data.xmat[body].reshape(3, 3)[:, 2].copy()


def cup_grasp_point(pos: np.ndarray, axis: np.ndarray, frac: float, cfg: G1TaskConfig = G1_CFG) -> np.ndarray:
    """Point on the cup's axis at `frac` of its height measured from the table side of the cup."""
    h = cfg.cup.height
    if axis[2] < 0:  # mouth down: origin is at the top, the table side is the mouth
        return pos + axis * h * (1 - frac)
    return pos + axis * h * frac


def cup_in_place(data, body: int, target: np.ndarray, mouth_up: bool, pos_tol: float = 0.012,
                 tilt_tol_deg: float = 12.0) -> bool:
    axis = cup_axis(data, body)
    ok_axis = axis[2] > np.cos(np.radians(tilt_tol_deg)) if mouth_up else axis[2] < -np.cos(np.radians(tilt_tol_deg))
    return bool(ok_axis and np.linalg.norm(data.xpos[body] - target) < pos_tol)


def base_spacing(cfg: G1TaskConfig = G1_CFG) -> float:
    """Centre distance of the two base cups."""
    return 2 * (cfg.cup.mouth_radius + cfg.cup.wall) + cfg.base_gap


def pyramid_standing(data, cup_bodies, cfg: G1TaskConfig = G1_CFG, spacing_tol=(0.012, 0.008),
                     top_tol: float = 0.012) -> bool:
    """Judged on the cups' relative placement, since that is what holds a pyramid up: both base cups
    mouth-down on the table at a distance the top cup can bridge, top cup mouth-up resting on both."""
    a, b, t = cup_bodies
    T, h = cfg.table_height, cfg.cup.height
    for body in (a, b):
        if cup_axis(data, body)[2] > -np.cos(np.radians(12)) or abs(data.xpos[body][2] - (T + h)) > 0.006:
            return False
    dist = np.linalg.norm(data.xpos[a][:2] - data.xpos[b][:2])
    s = base_spacing(cfg)
    if not s - spacing_tol[0] < dist < s + spacing_tol[1]:
        return False
    mid = np.array([*(data.xpos[a][:2] + data.xpos[b][:2]) / 2, T + h])
    # The top cup bridges a gap, so it may lean a little into it and still stand.
    return cup_in_place(data, t, mid, True, top_tol, tilt_tol_deg=TOP_TILT_DEG)


def ball_in_cup(data, ball_body: int, cup_body: int, ball_radius: float, cfg: G1TaskConfig = G1_CFG) -> bool:
    """Ball centre inside the (mouth-up) cup, below the rim."""
    R = data.xmat[cup_body].reshape(3, 3)
    local = R.T @ (data.xpos[ball_body] - data.xpos[cup_body])
    cup = cfg.cup
    if not (0.0 < local[2] < cup.height):
        return False
    radius_here = cup.bottom_radius + (cup.mouth_radius - cup.bottom_radius) * local[2] / cup.height
    return bool(np.hypot(local[0], local[1]) < radius_here)
