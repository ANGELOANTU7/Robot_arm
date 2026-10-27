"""Human-like synthetic demonstrations, for trying the pipeline before recording and for tests."""

import numpy as np

from robot_arm.video.reconstruct import HumanDemo


def min_jerk(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    s = np.linspace(0, 1, n)[:, None]
    return p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def synthetic_demo(rng: np.random.Generator, ball_xy=None, cup_xy=None, fps: float = 30.0,
                   ball_radius: float = 0.02, cup_height: float = 0.10, noise: float = 0.002) -> HumanDemo:
    """A pick -> carry -> drop demonstration in the table frame with natural variability.

    The layout matches the recording guide: the marker sheet (table origin) lies between the camera (-y side)
    and the objects, and the person reaches in from the far (+y) side.
    """
    if ball_xy is None:
        ball_xy = rng.uniform([-0.20, 0.15], [-0.05, 0.30])
        cup_xy = rng.uniform([0.06, 0.15], [0.22, 0.30])
    ball = np.array([*ball_xy, ball_radius])
    cup = np.array([*cup_xy, 0.0])
    carry_z = cup_height + rng.uniform(0.06, 0.12)

    rest = ball + [rng.uniform(-0.05, 0.05), rng.uniform(0.18, 0.25), rng.uniform(0.12, 0.2)]
    pre = ball + [rng.uniform(-0.02, 0.02), rng.uniform(0.0, 0.03), rng.uniform(0.05, 0.09)]
    lift = ball + [0, 0, carry_z - ball_radius + rng.uniform(0.0, 0.04)]
    mid = (lift + np.array([*cup_xy, carry_z])) / 2 + [0, 0, rng.uniform(0.0, 0.05)]
    over = np.array([cup_xy[0], cup_xy[1], carry_z])
    away = over + [rng.uniform(-0.05, 0.05), rng.uniform(0.12, 0.2), rng.uniform(0.03, 0.08)]

    # (target, seconds, aperture at the end of the segment)
    open_ap, hold_ap = rng.uniform(0.08, 0.11), rng.uniform(0.042, 0.05)
    segments = [
        (rest, rng.uniform(0.4, 0.7), 0.05),          # still at the start
        (pre, rng.uniform(0.6, 0.9), open_ap),        # reach, opening the hand
        (ball, rng.uniform(0.35, 0.5), open_ap),      # descend onto the ball
        (ball, rng.uniform(0.2, 0.35), hold_ap),      # close on the ball
        (lift, rng.uniform(0.4, 0.6), hold_ap),       # lift
        (mid, rng.uniform(0.3, 0.5), hold_ap),        # carry
        (over, rng.uniform(0.3, 0.5), hold_ap),       # arrive over the cup
        (over, rng.uniform(0.15, 0.3), hold_ap),      # settle
        (over, rng.uniform(0.2, 0.3), open_ap),       # open: ball drops
        (away, rng.uniform(0.5, 0.8), 0.06),          # withdraw
        (away, 0.4, 0.06),
    ]
    pos, ap = [rest[None]], [np.array([0.05])]
    for target, secs, a_end in segments:
        n = max(2, int(round(secs * fps)))
        pos.append(min_jerk(pos[-1][-1], np.asarray(target, float), n)[1:])
        ap.append(min_jerk(np.array([ap[-1][-1]]), np.array([a_end]), n)[1:, 0])
    pinch = np.concatenate(pos)
    aperture = np.concatenate(ap)
    T = len(pinch)
    t = np.arange(T) / fps
    # Correlated hand tremor.
    tremor = np.cumsum(rng.normal(0, noise, (T, 3)), axis=0)
    tremor -= np.linspace(0, 1, T)[:, None] * tremor[-1]
    pinch = pinch + 0.3 * tremor

    seg_frames = np.cumsum([max(2, int(round(s * fps))) - 1 for _, s, _ in segments])
    grasp = seg_frames[2] + 1  # closing starts once the hand is at the ball
    release = seg_frames[7] + 1
    pinch[grasp] = ball
    closed = np.zeros(T, bool)
    closed[grasp:release] = True
    return HumanDemo(t, pinch, aperture, closed, ball, cup, float(t[grasp]), float(t[release]),
                     {"synthetic": 1.0, "fps": fps})
