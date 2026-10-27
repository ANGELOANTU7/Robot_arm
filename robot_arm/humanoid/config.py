"""Constants for the G1 cup-pyramid task. Distances in metres, world frame: x forward, y left, z up."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CupSpec:
    """A tapered plastic party cup, like the red ones in your video (12 cm tall, 5 cm base, 9 cm mouth).
    Measure yours and change these to match."""

    bottom_radius: float = 0.025
    mouth_radius: float = 0.045
    height: float = 0.12
    wall: float = 0.002  # real cups are ~0.5 mm; much thinner walls make contacts unreliable in simulation
    mass: float = 0.012
    segments: int = 24


@dataclass(frozen=True)
class G1TaskConfig:
    cup: CupSpec = CupSpec()
    # The thrown object: a plastic bottle cap (a short cylinder), 2.8 cm across.
    cap_radius: float = 0.014
    cap_half_height: float = 0.006
    cap_mass: float = 0.002

    table_height: float = 0.72
    table_center: tuple[float, float] = (0.40, -0.10)
    table_half: tuple[float, float] = (0.22, 0.36)

    # Nominal layout; reset() jitters these. Cup A goes front-left, cup B back-right of it: with the side grasp
    # used here (hand pointing forward, thumb on the cup's left, palm on its right) neither the thumb nor the
    # palm/fingers come near cup A while cup B is put down.
    pyramid_center: tuple[float, float] = (0.42, -0.06)
    pyramid_yaw: float = -2.356  # direction from base cup A to base cup B
    base_gap: float = 0.004  # the base cups' mouths almost touch, so the top cup's base rests on both
    # Supply spots: each cup is approached from behind (-x) and the fingertips reach ~11 cm past it, so the top
    # cup waits further forward, clear of the corridors used for A and B.
    supply_cups: tuple[tuple[float, float], ...] = ((0.33, -0.20), (0.31, -0.37), (0.53, -0.28))  # A, B, top
    cap_spot: tuple[float, float] = (0.28, 0.08)  # the cap lies flat on the table here
    layout_jitter: float = 0.015
    # How precisely the pyramid must be built (from settling tests: with the base cups up to 2 cm further apart
    # than nominal the top cup still stands, wedged up to 2.5 cm lower between them).
    spacing_tol: tuple[float, float] = (0.012, 0.016)  # base cups' centre distance vs nominal: below, above
    top_tol: float = 0.02  # top cup's horizontal distance from the base cups' midpoint
    top_sink: float = 0.03  # how far the top cup may sit below the base cups' tops

    # Side power grasp: the hand points along its x axis, palm facing +y, hand z up. Grasp height as a
    # fraction of the cup's height (measured from the table side of the cup). The cups taper, so the base cups
    # (mouth down) are held where they are narrower than at the table, which leaves room for the fingers.
    grasp_height_frac: float = 0.55
    top_grasp_height_frac: float = 0.45

    # Throw: the cap must leave the hand at least this far (horizontally) from the top cup's centre.
    min_throw_distance: float = 0.05

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
