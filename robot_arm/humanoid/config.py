"""Constants for the G1 cup-pyramid task. Distances in metres, world frame: x forward, y left, z up."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CupSpec:
    """A tapered plastic cup. Measure yours and change these to match."""

    bottom_radius: float = 0.027
    mouth_radius: float = 0.028
    height: float = 0.075
    wall: float = 0.003
    mass: float = 0.012
    segments: int = 16


@dataclass(frozen=True)
class G1TaskConfig:
    cup: CupSpec = CupSpec()
    ball_radius: float = 0.018  # 3.6 cm ball; it sits on the tilted palm, away from the gap between the fingers
    ball_mass: float = 0.003  # ping-pong-ball light: a heavier ball knocks the 12 g top cup off when it hits

    table_height: float = 0.76
    table_center: tuple[float, float] = (0.40, -0.10)
    table_half: tuple[float, float] = (0.20, 0.34)

    # Nominal layout; reset() jitters these. Cup A goes front-left, cup B back-right of it: with the side grasp
    # used here (hand pointing forward, thumb on the cup's left, palm on its right) neither the thumb nor the
    # palm/fingers come near cup A while cup B is put down.
    pyramid_center: tuple[float, float] = (0.40, -0.08)
    pyramid_yaw: float = -2.356  # direction from base cup A to base cup B
    base_gap: float = 0.035  # space between the base cups' rims: room for the fingertips; top cup still overlaps 1.4 cm
    supply_cups: tuple[tuple[float, float], ...] = ((0.36, -0.24), (0.36, -0.36), (0.46, -0.30))  # A, B, top
    ball_pedestal: tuple[float, float] = (0.32, 0.12)
    pedestal_height: float = 0.07
    layout_jitter: float = 0.015

    # Side power grasp: the hand points along its x axis, palm facing +y, hand z up. Grasp height as a
    # fraction of the cup's height (measured from the table side of the cup).
    grasp_height_frac: float = 0.62  # high enough that the lower finger clears the table
    top_grasp_height_frac: float = 0.7

    # Throw: the ball must leave the hand at least this far (horizontally) from the top cup.
    min_throw_distance: float = 0.20

    sim_dt: float = 0.002
    control_dt: float = 0.05
    max_step: float = 0.015  # palm translation per control step (m)
    toss_max_step: float = 0.08  # the toss needs fast swings (up to 1.6 m/s)
    max_rot_step: float = 0.12  # palm rotation per control step (rad)
    grip_rate: float = 0.25  # synergy change per control step when closing (open -> closed in 0.4 s)
    release_rate: float = 0.15  # when opening around a cup: slower, so the fingers don't push the light cup away
    toss_grip_rate: float = 1.0  # releasing a throw: open in 0.1 s
    leash: float = 0.015  # max distance of the commanded grasp-site position from the actual one (m)
    rot_leash: float = 0.3  # same for the hand orientation (rad)
    ik_substeps: int = 5  # IK solves per control step (the command is interpolated between them)

    place_seconds: float = 10.0
    toss_seconds: float = 8.0

    @property
    def frame_skip(self) -> int:
        return int(round(self.control_dt / self.sim_dt))

    @property
    def table_top(self) -> float:
        return self.table_height


G1_CFG = G1TaskConfig()
