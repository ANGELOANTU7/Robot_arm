"""Per-skill clips of the cup-pyramid task -> metric human demonstrations in the table frame.

Clip types (see docs/RECORDING_PYRAMID.md):
- "place_base": move one mouth-down cup from the supply area onto the table next to (or as) the first base cup;
- "place_top": put the mouth-up cup onto the two base cups;
- "toss": pick up the ball and toss it into the top cup.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from robot_arm.video.calibrate import Camera
from robot_arm.video.hand_pose import closure, palm_center_cam, palm_frame
from robot_arm.video.reconstruct import _fill_gaps, _smooth

GRAVITY = 9.81


@dataclass
class SkillTrack:
    """Per-frame detections for a skill clip. Missing values are NaN."""

    t: np.ndarray  # (T,)
    hand_px: np.ndarray  # (T, 21, 2)
    hand_world: np.ndarray  # (T, 21, 3)
    objects_px: np.ndarray  # (T, N, 2) bottom pixel of each tracked cup (N cups) or ball centre
    objects_r_px: np.ndarray  # (T, N) blob radius in pixels


@dataclass
class SkillDemo:
    """A human skill demonstration in the table frame."""

    skill: str
    t: np.ndarray  # (T,)
    palm: np.ndarray  # (T, 3) palm centre
    palm_rot: np.ndarray  # (T, 3, 3) palm frame (x fingers, y thumb side, z out of the palm)
    closure: np.ndarray  # (T,) 0 open .. 1 closed
    obj_start: np.ndarray  # (3,) object position at the start (cup: centre of its table-side rim; ball: centre)
    obj_end: np.ndarray  # (3,) where the object ended up (place) / the cup mouth it was thrown into (toss)
    t_grasp: float
    t_release: float
    release_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # toss: ball velocity after release
    release_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # toss: ball centre at release
    info: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        info = d.pop("info")
        np.savez(path, **d, info_keys=np.array(list(info.keys())), info_values=np.array(list(info.values()), float))

    @classmethod
    def load(cls, path: str | Path) -> "SkillDemo":
        z = np.load(path)
        info = dict(zip(z["info_keys"].tolist(), z["info_values"].tolist()))
        return cls(str(z["skill"]), z["t"], z["palm"], z["palm_rot"], z["closure"], z["obj_start"], z["obj_end"],
                   float(z["t_grasp"]), float(z["t_release"]), z["release_vel"], z["release_pos"], info)


def smooth_rotations(R: np.ndarray, fps: float, seconds: float = 0.2) -> np.ndarray:
    """Smooth a rotation sequence by filtering its rotation vectors relative to the first frame."""
    from scipy.spatial.transform import Rotation

    rel = Rotation.from_matrix(R[0].T @ R)
    vec = np.unwrap(rel.as_rotvec(), axis=0)
    vec = _smooth(vec, fps, seconds)
    return R[0] @ Rotation.from_rotvec(vec).as_matrix()


def moved_object(objects_px: np.ndarray) -> int:
    """Index of the tracked object that moved the most (in pixels) between its first and last sighting."""
    best, idx = -1.0, 0
    for k in range(objects_px.shape[1]):
        valid = ~np.isnan(objects_px[:, k, 0])
        if valid.sum() < 2:
            continue
        p = objects_px[valid, k]
        d = np.linalg.norm(p - p[0], axis=1).max()
        if d > best:
            best, idx = d, k
    return idx


def _motion_window(px: np.ndarray, r_px: float, fps: float) -> tuple[int, int]:
    """First frame where the object has left its start position and the frame from which it stays at its final
    position (each sustained for 0.15 s). `px` must not contain gaps."""
    thr = max(0.8 * r_px, 5.0)
    moved = np.linalg.norm(px - px[0], axis=1) > thr
    k = max(2, int(0.15 * fps))
    sustained = np.array([moved[i:i + k].all() for i in range(len(moved) - k + 1)])
    if not sustained.any():
        raise RuntimeError("The object never visibly moves; check the color setting.")
    first = int(np.argmax(sustained))
    away = np.nonzero(np.linalg.norm(px - px[-1], axis=1) > thr)[0]
    settle = int(away[-1]) + 1 if len(away) else first
    return first, max(first, min(settle, len(px) - 1))


def _fit_ballistic(t: np.ndarray, p: np.ndarray, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares p(t) = p0 + v (t - t0) + g/2 (t - t0)^2 with known gravity: returns p0, v."""
    dt = t - t0
    A = np.column_stack([np.ones_like(dt), dt])
    y = p - 0.5 * np.array([0, 0, -GRAVITY]) * dt[:, None] ** 2
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return sol[0], sol[1]


def reconstruct_skill(track: SkillTrack, camera: Camera, skill: str, cup_height: float = 0.075,
                      cup_radius: float = 0.031, ball_radius: float = 0.02, palm_offset: float = 0.05,
                      grasp_frac: float = 0.6, ball_rest_height: float | None = None,
                      object_index: int | None = None) -> SkillDemo:
    """Turn detections of one skill clip into a SkillDemo.

    The depth scale (MediaPipe's hand-size guess) is fixed with the grasp anchor: while the hand closes on the
    object, the object is in front of the palm. For a cup the palm centre is `palm_offset` from its axis along
    the palm normal (side grasp); for the ball the palm is `ball_radius + 8 mm` from its centre. The ball starts
    at rest with its centre `ball_rest_height` above the table (default: lying on the table).
    """
    T = len(track.t)
    fps = 1 / np.median(np.diff(track.t))
    ok = ~np.isnan(track.hand_px).any(axis=(1, 2))
    if ok.sum() < max(10, 0.5 * fps):
        raise RuntimeError(f"Hand detected in only {ok.sum()}/{T} frames.")
    hand_px = _fill_gaps(track.hand_px, ok)
    hand_world = _fill_gaps(track.hand_world, ok)
    toss = skill == "toss"

    k = moved_object(track.objects_px) if object_index is None else object_index
    obj_seen = ~np.isnan(track.objects_px[:, k, 0])
    if obj_seen.sum() < 2:
        raise RuntimeError("The manipulated object was not found; check the color setting.")
    obj_px = _fill_gaps(track.objects_px[:, k], obj_seen)
    obj_r = float(np.nanmedian(track.objects_r_px[:, k]))

    # Palm path in the camera frame (up to MediaPipe's depth scale), orientation and closure.
    palm_cam = np.array([palm_center_cam(hand_px[i], hand_world[i], camera.K) for i in range(T)])
    depth = _smooth(palm_cam[:, 2], fps, 0.15 if toss else 0.4)
    uv = (palm_cam / palm_cam[:, 2:3]) @ camera.K.T
    uv = _smooth(uv[:, :2], fps, 0.08 if toss else 0.15)
    rays = np.linalg.solve(camera.K, np.column_stack([uv, np.ones(T)]).T).T
    palm_cam = rays * depth[:, None]
    rot_cam = np.array([palm_frame(w) for w in hand_world])
    rot_table = smooth_rotations(np.einsum("ij,tjk->tik", camera.R.T, rot_cam), fps, 0.1 if toss else 0.2)
    clos = _smooth(np.array([closure(w) for w in hand_world]), fps, 0.2)

    first_move, settle = _motion_window(obj_px, obj_r, fps)
    # Grasp: the last time the closure rises through its midpoint before the object moves.
    lo, hi = np.percentile(clos, 10), np.percentile(clos, 90)
    mid = (lo + hi) / 2
    before = np.nonzero(clos[:first_move + 1] < mid)[0]
    grasp = min(int(before[-1]) + 1 if len(before) else first_move, first_move)

    if toss:
        # The ball starts at rest at a known height (on the table: its radius): back-project onto that plane.
        # That also calibrates the apparent radius (blob edges add a roughly constant few tenths of a pixel),
        # which gives the ball's depth in flight.
        rest = obj_seen.copy()
        rest[max(1, grasp - int(0.3 * fps)):] = False
        rest = rest if rest.any() else obj_seen
        obj_start = camera.backproject_to_plane(np.nanmedian(track.objects_px[rest, k], axis=0),
                                                z=ball_radius if ball_rest_height is None else ball_rest_height)
        pc = camera.to_cam(obj_start[None])[0]
        dist = np.linalg.norm(pc)
        r_bias = float(np.nanmedian(track.objects_r_px[rest, k]) - camera.K[0, 0] * ball_radius * dist / pc[2] ** 2)
        ball_tab = camera.to_table(ball_positions(track.objects_px[:, k], track.objects_r_px[:, k] - r_bias, camera,
                                                  ball_radius))
        obj_end = mouth_from_top_cup(track, camera, cup_height, cup_radius, until=max(grasp, 1))
        anchor, offset = obj_start, ball_radius + 0.008
    else:
        # Cups: the moved cup's table-side rim centre at the start and at the end.
        obj_start = cup_base_from_px(obj_px[0], camera, 0.0, cup_radius)
        obj_end = cup_base_from_px(obj_px[-1], camera, cup_height if skill == "place_top" else 0.0, cup_radius)
        anchor, offset = obj_start + [0, 0, grasp_frac * cup_height], palm_offset

    # Depth scale: the palm centre plus `offset` along the palm normal hits the anchor during the grasp.
    hold = slice(grasp, max(grasp, first_move) + 1)
    p_cam, normal = palm_cam[hold].mean(axis=0), rot_table[hold, :, 2].mean(axis=0)
    weight = np.array([1.0, 1.0, 1.0 if toss else 0.1])  # a cup can be held at any height
    scales = np.linspace(0.5, 2.0, 301)
    pts = camera.to_table(scales[:, None] * p_cam) + offset * normal
    err = (((pts - anchor) ** 2) * weight).sum(axis=1)
    scale = float(scales[np.argmin(err)])
    palm = camera.to_table(scale * palm_cam)
    info = {"scale": scale, "fps": fps, "grasp_anchor_err": float(np.sqrt(err.min())), "object_index": float(k)}

    release_vel, release_pos = np.zeros(3), np.zeros(3)
    if toss:
        release, release_vel, release_pos = _toss_release(track, camera, k, ball_tab, hand_px, obj_r, first_move, fps)
        info["ball_radius_bias_px"] = r_bias
        info["release_fit_points"] = float(np.isfinite(ball_tab[release:release + int(0.25 * fps), 0]).sum())
    else:
        # Release: the hand opens after the cup has come to rest at its new place.
        after = np.nonzero(clos[settle:] < mid)[0]
        release = settle + int(after[0]) if len(after) else settle
    return SkillDemo(skill, track.t - track.t[0], palm, rot_table, clos, obj_start, obj_end,
                     float(track.t[grasp] - track.t[0]), float(track.t[release] - track.t[0]),
                     release_vel, release_pos, info)


def _toss_release(track: SkillTrack, camera: Camera, k: int, ball_tab: np.ndarray, hand_px: np.ndarray,
                  r_px: float, first_move: int, fps: float) -> tuple[int, np.ndarray, np.ndarray]:
    """Release frame, ball velocity and position at release.

    The ball clearly leaves the hand a few frames after the release; the ballistic fit of its flight from there
    is then followed back in the image to the last frame where the ball still lay on that parabola."""
    t = track.t
    ball_px = track.objects_px[:, k]
    palm_px = hand_px[:, [0, 5, 9, 13, 17]].mean(axis=1)
    with np.errstate(invalid="ignore"):
        sep = np.linalg.norm(ball_px - palm_px, axis=1)
    speed = np.linalg.norm(np.gradient(palm_px, axis=0), axis=1)
    peak = first_move + int(np.argmax(speed[first_move:]))
    baseline = np.nanmedian(sep[first_move:peak + 1])
    with np.errstate(invalid="ignore"):
        apart = sep > baseline + 1.0 * r_px
    run = 3
    cand = [i for i in range(first_move, len(t) - run) if apart[i:i + run].all()]
    if not cand:
        raise RuntimeError("Could not see the ball leave the hand.")
    sep_idx = cand[0]
    flight = np.arange(sep_idx, min(len(t), sep_idx + int(0.25 * fps)))
    flight = flight[np.isfinite(ball_tab[flight, 0])]
    if len(flight) < 3:
        raise RuntimeError("Too few frames of the ball in flight; record at 60 fps and keep the ball in view.")
    p0, v0 = _fit_ballistic(t[flight], ball_tab[flight], t[sep_idx])

    def on_parabola(i):
        dt = t[i] - t[sep_idx]
        p = p0 + v0 * dt + 0.5 * np.array([0, 0, -GRAVITY]) * dt ** 2
        return np.linalg.norm(camera.project(p)[0] - ball_px[i]) < max(0.35 * r_px, 4.0)

    release = sep_idx
    while release - 1 > first_move and t[sep_idx] - t[release - 1] < 0.15 and on_parabola(release - 1):
        release -= 1
    dt = t[release] - t[sep_idx]
    g = np.array([0, 0, -GRAVITY])
    return release, v0 + g * dt, p0 + v0 * dt + 0.5 * g * dt ** 2


def cup_base_from_px(bottom_px, camera: Camera, z: float, cup_radius: float) -> np.ndarray:
    """Centre of a cup's table-side rim from the lowest pixel of its blob (the rim edge nearest the camera)."""
    front = camera.backproject_to_plane(bottom_px, z=z)
    view = front - camera.center_table
    view[2] = 0
    return front + cup_radius * view / np.linalg.norm(view)


def ball_positions(px: np.ndarray, r_px: np.ndarray, camera: Camera, radius: float) -> np.ndarray:
    """Metric ball centres in the camera frame from pixel position and apparent radius.

    A sphere at distance d seen at angle theta off the optical axis projects to an ellipse whose long semi-axis
    (what the enclosing circle measures) is about f R / (d cos^2 theta)."""
    f = camera.K[0, 0]
    rays = np.linalg.solve(camera.K, np.column_stack([px, np.ones(len(px))]).T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    d = f * radius / (r_px * rays[:, 2] ** 2)
    return rays * d[:, None]


def mouth_from_top_cup(track: SkillTrack, camera: Camera, cup_height: float, cup_radius: float,
                       until: int | None = None) -> np.ndarray:
    """The top cup is the last tracked object column; its blob bottom sits on the base cups (z = cup height).

    Uses the frames before `until` (before the ball arrives, in case the throw knocks the pyramid over)."""
    px = track.objects_px[:until, -1]
    if np.isnan(px[:, 0]).all():
        px = track.objects_px[:, -1]
    if np.isnan(px[:, 0]).all():
        raise RuntimeError("The top cup was not found; check its color setting.")
    base = cup_base_from_px(np.nanmedian(px, axis=0), camera, cup_height, cup_radius)
    return base + [0, 0, cup_height]
