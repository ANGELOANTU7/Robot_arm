"""Per-skill clips of the cup-pyramid task -> metric human demonstrations in the table frame.

Clip types (see docs/RECORDING_PYRAMID.md):
- "place_base": move one mouth-down cup from the supply area onto its pyramid spot (cup A alone, cup B next to A);
- "place_top": put the mouth-up cup onto the two base cups;
- "toss": pick the bottle cap off the table and throw it sideways (along the base cups) into the top cup.

All cups have the same color, so a cup is located from the pixels that change when it moves, and from the shape
of the red blobs in the still frames at the start and end of a clip (see skill_process.track_skill):
- a cup's lowest pixel is the point of its table-side rim nearest the camera (for a mouth-down cup, just above the
  white rolled lip);
- a mouth-up cup's red wall is widest just under its white lip, level with the centre of its mouth.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from robot_arm.video.calibrate import Camera
from robot_arm.video.hand_pose import PALM_POINTS, closure, palm_center_cam, palm_frame
from robot_arm.video.reconstruct import _fill_gaps, _smooth

GRAVITY = 9.81
PINCH_TIPS = (4, 8)  # thumb tip and index fingertip: the cap is held between them


@dataclass(frozen=True)
class Props:
    """The person's cups (tapered party cups) and the thrown bottle cap. Metres."""

    cup_height: float = 0.12
    mouth_radius: float = 0.045
    bottom_radius: float = 0.025
    lip: float = 0.005  # the white rolled rim at the mouth: the red wall stops this far from the rim
    cap_radius: float = 0.014
    cap_height: float = 0.012

    def radius_at(self, z: float, mouth_up: bool) -> float:
        """Outer radius of a cup standing on the table, at height z."""
        s = np.clip(z / self.cup_height, 0, 1)
        if not mouth_up:
            s = 1 - s
        return float(self.bottom_radius + s * (self.mouth_radius - self.bottom_radius))


@dataclass
class SkillTrack:
    """Detections for a skill clip. Missing values are NaN.

    obj_px: per frame, the moved cup's lowest pixel (place clips) or the cap's centre (toss).
    marks: pixel measurements in the still frames (each optional):
      "start" / "end": the moved cup's lowest pixel at the start / end;
      "rim": the top cup's mouth (centre of its widest red row) once the pyramid stands;
      "base_a" / "base_b": the two base cups' lowest pixels."""

    t: np.ndarray  # (T,)
    hand_px: np.ndarray  # (T, 21, 2)
    hand_world: np.ndarray  # (T, 21, 3)
    obj_px: np.ndarray  # (T, 2)
    obj_r_px: np.ndarray  # (T,) blob radius in pixels
    marks: dict = field(default_factory=dict)


@dataclass
class SkillDemo:
    """A human skill demonstration in the table frame."""

    skill: str
    t: np.ndarray  # (T,)
    palm: np.ndarray  # (T, 3) palm centre
    palm_rot: np.ndarray  # (T, 3, 3) palm frame (x fingers, y thumb side, z out of the palm)
    closure: np.ndarray  # (T,) 0 open .. 1 closed
    obj_start: np.ndarray  # (3,) object at the start (cup: centre of its table-side rim; cap: centre)
    obj_end: np.ndarray  # (3,) where the cup ended up / the centre of the top cup's mouth (toss)
    t_grasp: float
    t_release: float
    release_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # toss: cap velocity after release
    release_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # toss: cap centre at release
    info: dict = field(default_factory=dict)  # numbers only; toss: "axis_x", "axis_y" = base cup A -> B

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

    @property
    def base_axis(self) -> np.ndarray | None:
        """Toss clips: horizontal unit vector between the base cups, if they were found."""
        return np.array([self.info["axis_x"], self.info["axis_y"], 0.0]) if "axis_x" in self.info else None


def smooth_rotations(R: np.ndarray, fps: float, seconds: float = 0.2) -> np.ndarray:
    """Smooth a rotation sequence by filtering its rotation vectors relative to the first frame."""
    from scipy.spatial.transform import Rotation

    rel = Rotation.from_matrix(R[0].T @ R)
    vec = np.unwrap(rel.as_rotvec(), axis=0)
    vec = _smooth(vec, fps, seconds)
    return R[0] @ Rotation.from_rotvec(vec).as_matrix()


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


def cup_base_from_px(bottom_px, camera: Camera, z: float, radius: float) -> np.ndarray:
    """Centre of a cup's circle at height z from the image of its point nearest the camera (the lowest pixel of
    the cup's blob)."""
    front = camera.backproject_to_plane(bottom_px, z=z)
    view = front - camera.center_table
    view[2] = 0
    return front + radius * view / np.linalg.norm(view)


def cup_on_table(bottom_px, camera: Camera, props: Props, mouth_up: bool) -> np.ndarray:
    """Centre of the table-side rim of a cup standing on the table, from its lowest red pixel. A mouth-down cup's
    red wall starts above its white lip."""
    z = 0.0 if mouth_up else props.lip
    p = cup_base_from_px(bottom_px, camera, z, props.radius_at(z, mouth_up))
    p[2] = 0.0
    return p


def top_cup_mouth(rim_px, camera: Camera, props: Props, base_mid: np.ndarray | None = None) -> np.ndarray:
    """Centre of the top cup's mouth from the centre of its widest red row (just under the lip).

    With the base cups' midpoint known, the mouth is taken at the same distance from the camera (the top cup
    sits centred on them, give or take a centimetre), which also gives its height: the top cup wedges between
    the base cups' walls by an amount we don't know in advance. Otherwise it is assumed to sit on top of them."""
    if base_mid is None:
        p = camera.backproject_to_plane(rim_px, z=2 * props.cup_height - props.lip)
    else:
        o, d = camera.center_table, camera.ray(rim_px)
        fwd = np.array([d[0], d[1], 0.0]) / np.linalg.norm(d[:2])
        p = o + d * (np.dot(base_mid - o, fwd) / np.dot(d, fwd))
        p[2] = np.clip(p[2], 1.6 * props.cup_height, 2.05 * props.cup_height)
    return p + [0, 0, props.lip]


def base_pair(marks: dict, camera: Camera, props: Props) -> tuple[np.ndarray, np.ndarray] | None:
    """Table-side rim centres of the base cups A and B (mouth down), if both were seen."""
    if "base_a" not in marks or "base_b" not in marks:
        return None
    return (cup_on_table(marks["base_a"], camera, props, mouth_up=False),
            cup_on_table(marks["base_b"], camera, props, mouth_up=False))


def _fit_ballistic(t: np.ndarray, p: np.ndarray, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares p(t) = p0 + v (t - t0) + g/2 (t - t0)^2 with known gravity: returns p0, v."""
    dt = t - t0
    A = np.column_stack([np.ones_like(dt), dt])
    y = p - 0.5 * np.array([0, 0, -GRAVITY]) * dt[:, None] ** 2
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return sol[0], sol[1]


def fit_flight_2d(t: np.ndarray, px: np.ndarray, camera: Camera, t0: float, p_init: np.ndarray,
                  v_init: np.ndarray, prior_px: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Ballistic flight p(t) = p0 + v (t - t0) + g/2 (t - t0)^2 fitted to its image positions.

    The cap tumbles, so its size in the image says little about its depth. Its pixel path does: gravity is known,
    so the path's curvature in the image fixes the distance. A weak pull towards `p_init` (in pixels per
    centimetre of deviation, scaled by `prior_px`) keeps a short flight from wandering along the view ray."""
    from scipy.optimize import least_squares

    g = np.array([0, 0, -GRAVITY])
    dt = (t - t0)[:, None]
    depth = float(np.linalg.norm(camera.to_cam(p_init[None])[0]))
    w = prior_px * camera.K[0, 0] / depth  # pixels per metre at that distance, scaled

    def residuals(x):
        p = x[:3] + x[3:] * dt + 0.5 * g * dt ** 2
        return np.concatenate([(camera.project(p) - px).ravel(), w * (x[:3] - p_init)])

    sol = least_squares(residuals, np.concatenate([p_init, v_init]), x_scale=[0.05] * 3 + [0.5] * 3)
    return sol.x[:3], sol.x[3:]


def reconstruct_skill(track: SkillTrack, camera: Camera, skill: str, props: Props = Props(),
                      palm_offset: float = 0.05, grasp_frac: float = 0.5) -> SkillDemo:
    """Turn detections of one skill clip into a SkillDemo.

    The depth scale (MediaPipe's hand-size guess) is fixed with the grasp anchor: while the hand closes on the
    object, the object is in the hand. For a cup the palm centre is `palm_offset` from its axis along the palm
    normal (side grasp, about `grasp_frac` up the cup); for the cap, the point between the thumb tip and the
    index fingertip is at its centre. The cap starts lying flat on the table.
    """
    T = len(track.t)
    fps = 1 / np.median(np.diff(track.t))
    ok = ~np.isnan(track.hand_px).any(axis=(1, 2))
    if ok.sum() < max(10, 0.5 * fps):
        raise RuntimeError(f"Hand detected in only {ok.sum()}/{T} frames.")
    hand_px = _fill_gaps(track.hand_px, ok)
    hand_world = _fill_gaps(track.hand_world, ok)
    toss = skill == "toss"

    obj_seen = ~np.isnan(track.obj_px[:, 0])
    if obj_seen.sum() < 2:
        raise RuntimeError("The manipulated object was not found; check the color setting.")
    obj_px = _fill_gaps(track.obj_px, obj_seen)
    obj_r = float(np.nanmedian(track.obj_r_px))

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
    hold = slice(grasp, max(grasp, first_move) + 1)
    info = {"fps": fps}

    pair = base_pair(track.marks, camera, props)
    base_mid = None if pair is None else (pair[0] + pair[1]) / 2
    if toss:
        # The cap lies flat on the table at the start: back-project its centre onto that plane.
        rest = obj_seen.copy()
        rest[max(1, grasp - int(0.3 * fps)):] = False
        rest = rest if rest.any() else obj_seen
        obj_start = camera.backproject_to_plane(np.nanmedian(track.obj_px[rest], axis=0), z=props.cap_height / 2)
        if "rim" not in track.marks:
            raise RuntimeError("The top cup was not found; check the cup color and that the pyramid is in view.")
        obj_end = top_cup_mouth(track.marks["rim"], camera, props, base_mid)
        if pair is not None:
            axis = pair[1] - pair[0]
            info["axis_x"], info["axis_y"] = (axis[:2] / np.linalg.norm(axis[:2])).tolist()
        # The cap is held between the thumb tip and the index fingertip: that point relative to the palm centre
        # (MediaPipe's world landmarks are camera-aligned), so it shares the palm's smoothing and depth scale.
        tips = hand_world[:, list(PINCH_TIPS)].mean(axis=1) - hand_world[:, PALM_POINTS].mean(axis=1)
        pinch_cam = palm_cam + _smooth(tips, fps, 0.05)
        point_cam, anchor, offset, normal = pinch_cam[hold].mean(axis=0), obj_start, 0.0, np.zeros(3)
        weight = np.ones(3)
    else:
        mouth_up = skill == "place_top"
        if "start" not in track.marks:
            raise RuntimeError("Could not tell where the moved cup started; keep your hand out of view at first.")
        obj_start = cup_on_table(track.marks["start"], camera, props, mouth_up)
        if skill == "place_top" and "rim" in track.marks:
            obj_end = top_cup_mouth(track.marks["rim"], camera, props, base_mid) - [0, 0, props.cup_height]
        elif "end" in track.marks:
            obj_end = cup_on_table(track.marks["end"], camera, props, mouth_up)
        else:
            raise RuntimeError("Could not tell where the moved cup ended; pull your hand out of view at the end.")
        point_cam, anchor = palm_cam[hold].mean(axis=0), obj_start + [0, 0, grasp_frac * props.cup_height]
        offset, normal = palm_offset, rot_table[hold, :, 2].mean(axis=0)
        weight = np.array([1.0, 1.0, 0.1])  # a cup can be held at any height

    # Depth scale: the hand's grasp point lands on the anchor during the grasp.
    scales = np.linspace(0.5, 2.0, 301)
    pts = camera.to_table(scales[:, None] * point_cam) + offset * normal
    err = (((pts - anchor) ** 2) * weight).sum(axis=1)
    scale = float(scales[np.argmin(err)])
    palm = camera.to_table(scale * palm_cam)
    info.update({"scale": scale, "grasp_anchor_err": float(np.sqrt(err.min()))})

    release_vel, release_pos = np.zeros(3), np.zeros(3)
    if toss:
        pinch = camera.to_table(scale * pinch_cam)
        release, release_vel, release_pos, n_fit = _toss_release(track, camera, pinch, hand_px, obj_r, first_move,
                                                                 obj_end, fps)
        info["release_fit_points"] = float(n_fit)
    else:
        # Release: the hand opens after the cup has come to rest at its new place.
        after = np.nonzero(clos[settle:] < mid)[0]
        release = settle + int(after[0]) if len(after) else settle
    return SkillDemo(skill, track.t - track.t[0], palm, rot_table, clos, obj_start, obj_end,
                     float(track.t[grasp] - track.t[0]), float(track.t[release] - track.t[0]),
                     release_vel, release_pos, info)


def _toss_release(track: SkillTrack, camera: Camera, pinch: np.ndarray, hand_px: np.ndarray, r_px: float,
                  first_move: int, mouth: np.ndarray, fps: float) -> tuple[int, np.ndarray, np.ndarray, int]:
    """Release frame, cap velocity and position at release, and the number of flight frames used.

    The cap clearly leaves the fingertips a few frames after the release. Its flight from there is fitted in the
    image (gravity known) and followed back to the earliest frame where the cap still lay on that parabola: the
    release."""
    t = track.t
    cap_px = track.obj_px
    tips_px = hand_px[:, list(PINCH_TIPS)].mean(axis=1)
    with np.errstate(invalid="ignore"):
        sep = np.linalg.norm(cap_px - tips_px, axis=1)
    speed = np.linalg.norm(np.gradient(tips_px, axis=0), axis=1)
    peak = first_move + int(np.argmax(speed[first_move:]))
    baseline = np.nanmedian(sep[first_move:peak + 1])
    with np.errstate(invalid="ignore"):
        apart = sep > baseline + 1.0 * r_px
    run = 3
    cand = [i for i in range(first_move, len(t) - run) if apart[i:i + run].all()]
    if not cand:
        raise RuntimeError("Could not see the cap leave the hand.")
    sep_idx = cand[0]
    flight = np.arange(sep_idx, min(len(t), sep_idx + int(0.4 * fps)))
    seen = np.isfinite(cap_px[flight, 0])
    if not seen.all():  # stop where the cap is lost (inside the cup, or out of view)
        flight = flight[: int(np.argmin(seen))]
    if len(flight) < 4:
        raise RuntimeError("Too few frames of the cap in flight; record at 60 fps and keep the throw in view.")
    g = np.array([0, 0, -GRAVITY])
    tf = max(t[flight[-1]] - t[sep_idx] + 1 / fps, 0.05)

    def fit(i0, frames, prior_px):
        """Flight with its reference time at frame i0, pulled towards the pinch point there."""
        v_init = (mouth - pinch[i0] + np.array([0, 0, 0.5 * GRAVITY * tf * tf])) / tf
        return fit_flight_2d(t[frames], cap_px[frames], camera, t[i0], pinch[i0], v_init, prior_px)

    def on_parabola(i, i0, p0, v0):
        dt = t[i] - t[i0]
        p = p0 + v0 * dt + 0.5 * g * dt ** 2
        return np.linalg.norm(camera.project(p)[0] - cap_px[i]) < max(0.35 * r_px, 3.0)

    # Once apart, the opening fingers have moved away from the cap, so the first fit is only loosely tied to
    # them; it finds the release: the earliest frame on the parabola. Until then the cap was in the fingers, so
    # the refit is tied to the pinch point at the release.
    p0, v0 = fit(sep_idx, flight, 0.03)
    release = sep_idx
    while release - 1 > first_move and t[sep_idx] - t[release - 1] < 0.15 and on_parabola(release - 1, sep_idx,
                                                                                            p0, v0):
        release -= 1
    frames = np.concatenate([np.arange(release, sep_idx)[np.isfinite(cap_px[release:sep_idx, 0])], flight])
    p0, v0 = fit(release, frames, 0.1)
    return release, v0, p0, len(frames)
