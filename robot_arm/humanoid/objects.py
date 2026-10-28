"""Cup pyramid geometry and the success checks for building it and getting the cap in."""

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
        """Centre of the top cup's mouth: where the cap must go."""
        return self.top + [0, 0, G1_CFG.cup.height]


ORIENTATIONS = [DOWN, DOWN, UP]  # how each cup ends up in the pyramid
START_ORIENTATIONS = [DOWN, DOWN, DOWN]  # all cups start mouth-down: the top cup is turned over on the way
TOP_TILT_DEG = 20.0  # how far the top cup may lean and still count as standing


def cup_axis(data: mujoco.MjData, body: int) -> np.ndarray:
    """Unit vector out of the cup's mouth."""
    return data.xmat[body].reshape(3, 3)[:, 2].copy()


def cup_grasp_point(pos: np.ndarray, axis: np.ndarray, frac: float, cfg: G1TaskConfig = G1_CFG) -> np.ndarray:
    """Point on the cup's axis at `frac` of its height measured from the table side of the cup."""
    h = cfg.cup.height
    if axis[2] < 0:  # mouth down: origin is at the top, the table side is the mouth
        return pos + axis * h * (1 - frac)
    return pos + axis * h * frac


def cup_table_point(pos: np.ndarray, axis: np.ndarray, cfg: G1TaskConfig = G1_CFG) -> np.ndarray:
    """Centre of the end of the cup that stands on the table (the mouth of a mouth-down cup, else the bottom).
    Placing aligns this point, so a cup held slightly tilted still lands where it should."""
    return pos + axis * cfg.cup.height if axis[2] < 0 else pos


def cup_lowest_z(pos: np.ndarray, axis: np.ndarray, cfg: G1TaskConfig = G1_CFG) -> float:
    """Height of the lowest point of the cup's table-side rim (a tilted cup touches down there first)."""
    r = (cfg.cup.mouth_radius if axis[2] < 0 else cfg.cup.bottom_radius) + cfg.cup.wall
    return float(cup_table_point(pos, axis, cfg)[2] - r * np.sqrt(max(0.0, 1 - axis[2] ** 2)))


def cup_in_place(data, body: int, target: np.ndarray, mouth_up: bool, pos_tol: float = 0.012,
                 tilt_tol_deg: float = 12.0) -> bool:
    axis = cup_axis(data, body)
    ok_axis = axis[2] > np.cos(np.radians(tilt_tol_deg)) if mouth_up else axis[2] < -np.cos(np.radians(tilt_tol_deg))
    return bool(ok_axis and np.linalg.norm(data.xpos[body] - target) < pos_tol)


def base_spacing(cfg: G1TaskConfig = G1_CFG) -> float:
    """Centre distance of the two base cups."""
    return 2 * (cfg.cup.mouth_radius + cfg.cup.wall) + cfg.base_gap


def base_pair_ok(data, cup_bodies, cfg: G1TaskConfig = G1_CFG) -> bool:
    """Both base cups mouth-down on the table, at a distance the top cup can rest across."""
    a, b, _ = cup_bodies
    T, h = cfg.table_height, cfg.cup.height
    for body in (a, b):
        if cup_axis(data, body)[2] > -np.cos(np.radians(12)) or abs(data.xpos[body][2] - (T + h)) > 0.006:
            return False
    s = base_spacing(cfg)
    return bool(s - cfg.spacing_tol[0] < np.linalg.norm(data.xpos[a][:2] - data.xpos[b][:2]) < s + cfg.spacing_tol[1])


def top_cup_resting(data, cup_bodies, cfg: G1TaskConfig = G1_CFG) -> bool:
    """The top cup mouth-up on the base cups: near their midpoint, and at the base cups' height or wedged up to
    `top_sink` lower between them (it sinks in when they stand a little apart, and still stands)."""
    a, b, t = cup_bodies
    T, h = cfg.table_height, cfg.cup.height
    if cup_axis(data, t)[2] < np.cos(np.radians(TOP_TILT_DEG)):
        return False
    mid = (data.xpos[a][:2] + data.xpos[b][:2]) / 2
    z = data.xpos[t][2] - (T + h)
    return bool(np.linalg.norm(data.xpos[t][:2] - mid) < cfg.top_tol and -cfg.top_sink < z < 0.01)


def pyramid_standing(data, cup_bodies, cfg: G1TaskConfig = G1_CFG) -> bool:
    """Judged on the cups' relative placement, since that is what holds a pyramid up."""
    return base_pair_ok(data, cup_bodies, cfg) and top_cup_resting(data, cup_bodies, cfg)


def cap_in_cup(data, cap_body: int, cup_body: int, cap_radius: float, cfg: G1TaskConfig = G1_CFG) -> bool:
    """Ball centre inside the (mouth-up) cup, below the rim."""
    R = data.xmat[cup_body].reshape(3, 3)
    local = R.T @ (data.xpos[cap_body] - data.xpos[cup_body])
    cup = cfg.cup
    if not (0.0 < local[2] < cup.height):
        return False
    radius_here = cup.bottom_radius + (cup.mouth_radius - cup.bottom_radius) * local[2] / cup.height
    return bool(np.hypot(local[0], local[1]) < radius_here)
