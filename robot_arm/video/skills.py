"""Per-skill clips of the cup-pyramid task -> metric human demonstrations in the table frame.

Clip types (see docs/RECORDING_PYRAMID.md):
- "place_base": move one mouth-down cup onto its pyramid spot (cup A alone, cup B next to A);
- "place_top": take the third cup (mouth-down like the others), turn it over in the air and put it mouth-up onto
  the two base cups;
- "toss": pick the bottle cap off the table and throw it into the top cup; after a miss, pick it up again and throw
  again, until it goes in; then a thumbs-up.

The cups' positions come from the image side (skill_process.track_skill) as table-frame points; this module turns
the hand landmarks and the object tracks into the demonstration.
"""

from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from robot_arm.video.calibrate import Camera
from robot_arm.video.hand_pose import MIDDLE_MCP, PALM_POINTS, WRIST, closure, finger_curl, palm_center_cam, palm_frame
from robot_arm.video.reconstruct import _fill_gaps, _smooth

GRAVITY = 9.81
PINCH_TIPS = (4, 8)  # thumb tip and index fingertip: the cap is held between them
GRIP_HEIGHTS = (0.3, 0.95)  # where a mouth-down cup can be held, as fractions of its height
MIN_PALM_HEIGHT = 0.03  # the hand's size is noisy: the palm stays above the table
RELEASE_GAP = 1.2  # the cap has left the fingertips once it is this many of its radii from them (in the image)


@dataclass(frozen=True)
class Props:
    """The person's cups (tapered party cups) and the thrown bottle cap. Metres."""

    cup_height: float = 0.12
    mouth_radius: float = 0.045
    bottom_radius: float = 0.025
    lip: float = 0.005  # the white rolled rim at the mouth: the red wall stops this far from the rim
    cap_radius: float = 0.014
    cap_height: float = 0.012
    top_sink: float = 0.01  # how far the top cup sits below the base cups' tops (it wedges between them)

    def radius_at(self, z: float, mouth_up: bool) -> float:
        """Outer radius of a cup standing on the table, at height z."""
        s = np.clip(z / self.cup_height, 0, 1)
        if not mouth_up:
            s = 1 - s
        return float(self.bottom_radius + s * (self.mouth_radius - self.bottom_radius))

    @property
    def top_bottom_z(self) -> float:
        """Height of the top cup's base in a built pyramid."""
        return self.cup_height - self.top_sink


@dataclass
class SkillTrack:
    """Detections for a skill clip. Missing values are NaN.

    obj_px: per frame, the moved cup's blob centre (place clips) or the cap's centre (toss).
    marks: table-frame points measured in the still frames (each optional):
      "start" / "end": centre of the moved cup's table-side rim at the start / end;
      "base_a" / "base_b": the base cups' table-side rim centres; "mouth": the top cup's mouth centre;
      "axis": unit vector along the base cups.
    in_cup_px: (M, 2) image polygon around the pyramid: a cap that comes to rest inside it is in the top cup."""

    t: np.ndarray  # (T,)
    hand_px: np.ndarray  # (T, 21, 2)
    hand_world: np.ndarray  # (T, 21, 3)
    obj_px: np.ndarray  # (T, 2)
    obj_r_px: np.ndarray  # (T,) blob radius in pixels
    marks: dict = field(default_factory=dict)
    in_cup_px: np.ndarray | None = None
    info: dict = field(default_factory=dict)  # copied into the demo's info


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
    t_grasp: float  # the (last) grasp
    t_release: float  # the (last) release: for a toss, of the throw that went in
    release_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # toss: cap velocity after release
    release_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # toss: cap centre at release
    # Numbers only. Toss: "axis_x", "axis_y" (base cups' line), "throws" (attempts), "throw{k}_grasp" /
    # "throw{k}_release" (times of each attempt), "thumbs_up_start" / "thumbs_up_end" (if shown).
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
    away = np.nonzero(np.linalg.norm(px - px[-1], axis=1) > max(0.2 * r_px, 5.0))[0]  # at rest, not just close by
    settle = int(away[-1]) + 1 if len(away) else first
    return first, max(first, min(settle, len(px) - 1))


def thumbs_up(world_m: np.ndarray, R_cam: np.ndarray) -> bool:
    """A fist with the thumb stuck out pointing up (table frame). MediaPipe's world landmarks are camera-aligned."""
    if not np.isfinite(world_m).all():
        return False
    size = np.linalg.norm(world_m[MIDDLE_MCP] - world_m[WRIST])
    thumb = world_m[4] - world_m[2]  # thumb MCP -> tip
    up = (R_cam.T @ thumb)[2] / max(np.linalg.norm(thumb), 1e-9)
    return bool(finger_curl(world_m) > 0.3 and np.linalg.norm(thumb) > 0.55 * size and up > 0.5)


def fit_flight(t: np.ndarray, px: np.ndarray, camera: Camera, t0: float, p0_prior: np.ndarray, v_init: np.ndarray,
               end: tuple[float, np.ndarray] | None = None, prior_px: float = 0.1,
               end_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Ballistic flight p(t) = p0 + v (t - t0) + g/2 (t - t0)^2 fitted to its image positions.

    The cap tumbles, so its size in the image says little about its depth; its pixel path and gravity do (the
    path's curvature in the image fixes the distance for a side view). Seen from above the path is nearly straight
    and the height must come from its ends: a pull towards `p0_prior` at t0 (where the fingers let go) and, if
    given, towards the landing point `end` = (time, point). The pulls are in pixels per centimetre of deviation,
    scaled by `prior_px` and `end_px`."""
    from scipy.optimize import least_squares

    g = np.array([0, 0, -GRAVITY])
    dt = (t - t0)[:, None]
    depth = float(np.linalg.norm(camera.to_cam(p0_prior[None])[0]))
    w0 = prior_px * camera.K[0, 0] / depth  # pixels per metre at that distance, scaled
    w1 = end_px * camera.K[0, 0] / depth

    def residuals(x):
        p = x[:3] + x[3:] * dt + 0.5 * g * dt ** 2
        r = [(camera.project(p) - px).ravel(), w0 * (x[:3] - p0_prior)]
        if end is not None:
            te = end[0] - t0
            r.append(w1 * (x[:3] + x[3:] * te + 0.5 * g * te ** 2 - end[1]))
        return np.concatenate(r)

    sol = least_squares(residuals, np.concatenate([p0_prior, v_init]), x_scale=[0.05] * 3 + [0.5] * 3)
    return sol.x[:3], sol.x[3:]


def _inside(poly: np.ndarray | None, p: np.ndarray) -> bool:
    import cv2

    if poly is None or not np.isfinite(p).all():
        return False
    return cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), False) >= 0


def depth_at_height(camera: Camera, ray: np.ndarray, z: float) -> float:
    """Depth at which a camera ray (with third component 1) crosses the horizontal plane at height z."""
    a, b = camera.to_table(np.zeros(3))[2], (ray @ camera.R)[2]
    return float((z - a) / b)


def reconstruct_skill(track: SkillTrack, camera: Camera, skill: str, props: Props = Props()) -> SkillDemo:
    """Turn detections of one skill clip into a SkillDemo.

    Objects are held between the thumb and the index finger, so the point midway between their tips is on the cap
    or on the cup's axis. That anchors the hand's depth, see below.
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

    # Palm image path, orientation and closure. Depth: MediaPipe's own is off by a factor of about 1.7 and jumps from
    # frame to frame; the hand's size in the image is steady, and is tied to metres below.
    palm_cam = np.array([palm_center_cam(hand_px[i], hand_world[i], camera.K) for i in range(T)])
    uv = (palm_cam / palm_cam[:, 2:3]) @ camera.K.T
    uv = _smooth(uv[:, :2], fps, 0.08 if toss else 0.15)
    rays = np.linalg.solve(camera.K, np.column_stack([uv, np.ones(T)]).T).T
    size = sum(np.linalg.norm(hand_px[:, a] - hand_px[:, b], axis=1) for a, b in combinations(PALM_POINTS, 2))
    inv_size = 1 / _smooth(size, fps, 0.15 if toss else 0.25)
    rot_cam = np.array([palm_frame(w) for w in hand_world])
    rot_table = smooth_rotations(np.einsum("ij,tjk->tik", camera.R.T, rot_cam), fps, 0.1 if toss else 0.2)
    clos = _smooth(np.array([closure(w) for w in hand_world]), fps, 0.2)
    lo, hi = np.percentile(clos, 10), np.percentile(clos, 90)
    mid = (lo + hi) / 2
    # The point midway between the thumb tip and the index fingertip: on the cap, or on a held cup's axis.
    tips_uv = _smooth(hand_px[:, list(PINCH_TIPS)].mean(axis=1), fps, 0.05)
    tip_rays = np.linalg.solve(camera.K, np.column_stack([tips_uv, np.ones(T)]).T).T
    info = {"fps": fps, "hand_seen": float(ok.mean())}
    marks = track.marks

    if toss:
        throws, rests = _toss_segments(track, camera, props, hand_px, fps)
        if not throws:
            raise RuntimeError("Could not see the cap being thrown.")
        obj_start = rests[0][2]
        final = throws[-1]
        in_cup = final["in_cup"]
        if in_cup:
            mouth = final["land"] + [0, 0, props.cup_height - props.cap_height / 2 - 0.002]
        elif "mouth" in marks:
            mouth = marks["mouth"]
        else:
            raise RuntimeError("The cap never came to rest in the top cup, and the top cup was not found.")
        obj_end = mouth
        if "axis" in marks:
            info["axis_x"], info["axis_y"] = (marks["axis"][:2] / np.linalg.norm(marks["axis"][:2])).tolist()
        grasp, first_move = final["grasp"], final["lift"]
        anchor_frames = slice(throws[0]["grasp"], throws[0]["lift"] + 1)  # the first pick: the cap's known start
        anchors = [(anchor_frames, obj_start, np.array([props.cap_height / 2]))]
    else:
        first_move, settle = _motion_window(obj_px, obj_r, fps)
        # Grasp: the last time the closure rises through its midpoint before the object moves.
        before = np.nonzero(clos[:first_move + 1] < mid)[0]
        grasp = min(int(before[-1]) + 1 if len(before) else first_move, first_move)
        if "start" not in marks or "end" not in marks:
            raise RuntimeError("Could not tell where the moved cup started or ended; keep your hand out of view "
                               "for the first and last second.")
        obj_start, obj_end = np.asarray(marks["start"]), np.asarray(marks["end"])
        # Release: the hand opens after the cup has come to rest at its new place.
        after = np.nonzero(clos[settle:] < mid)[0]
        release = settle + int(after[0]) if len(after) else settle
        grips = np.linspace(GRIP_HEIGHTS[0], GRIP_HEIGHTS[1], 14) * props.cup_height  # above the mouth
        put = obj_end[2] + props.cup_height - grips if skill == "place_top" else grips
        anchors = [(slice(grasp, max(grasp, first_move) + 1), obj_start, grips),
                   (slice(max(0, release - int(0.1 * fps)), release + 1), obj_end, put)]

    # Depth scale: at the grasp (and the release) the fingertips' ray meets the object, at a height the object
    # allows. That gives the depth there; how far the ray passes from the object is the check on it.
    ks, errs, frames = [], [], []
    for span, target, heights in anchors:
        ray = tip_rays[span].mean(axis=0)
        depths = np.array([depth_at_height(camera, ray, z) for z in heights])
        miss = np.linalg.norm(camera.to_table(depths[:, None] * ray)[:, :2] - target[:2], axis=1)
        best = int(np.argmin(miss))
        ks.append(depths[best] / inv_size[span].mean())
        errs.append(float(miss[best]))
        frames.append(int(np.mean(np.arange(T)[span])))
    k = np.exp(np.interp(np.arange(T), frames, np.log(ks))) if len(ks) > 1 else np.full(T, ks[0])
    depth = np.minimum(k * inv_size, [depth_at_height(camera, r, MIN_PALM_HEIGHT) for r in rays])
    palm = camera.to_table(rays * depth[:, None])
    pinch_cam = tip_rays * depth[:, None]
    info.update({"scale": float(ks[0] / np.median(k)), "grasp_anchor_err": errs[0], "release_anchor_err": errs[-1]})

    release_vel, release_pos = np.zeros(3), np.zeros(3)
    if toss:
        pinch = camera.to_table(pinch_cam)
        for k, th in enumerate(throws):
            p0, v0 = _fit_throw(track, camera, pinch, th, fps, props.top_bottom_z + props.cup_height,
                                looking_down(camera))
            th["p0"], th["v0"] = p0, v0
            info[f"throw{k}_grasp"] = float(track.t[th["grasp"]] - track.t[0])
            info[f"throw{k}_release"] = float(track.t[th["release"]] - track.t[0])
            info[f"throw{k}_in"] = float(th["in_cup"])
        info["throws"] = float(len(throws))
        release, release_pos, release_vel = final["release"], final["p0"], final["v0"]
        # Thumbs-up after the cap has gone in.
        after = np.arange(final["land_frame"], T)
        up = np.array([ok[i] and thumbs_up(track.hand_world[i], camera.R) for i in after])
        run = _longest_run(up)
        if run is not None and (run[1] - run[0]) >= 0.2 * fps:
            info["thumbs_up_start"] = float(track.t[after[run[0]]] - track.t[0])
            info["thumbs_up_end"] = float(track.t[after[run[1] - 1]] - track.t[0])
    return SkillDemo(skill, track.t - track.t[0], palm, rot_table, clos, obj_start, obj_end,
                     float(track.t[grasp] - track.t[0]), float(track.t[release] - track.t[0]),
                     release_vel, release_pos, info)


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    best, start = None, None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _toss_segments(track: SkillTrack, camera: Camera, props: Props, hand_px: np.ndarray, fps: float):
    """Split a toss clip into its throws.

    A throw is where the cap leaves the fingertips fast and stays away from them. It was picked up at the end of
    the cap's last rest before that (lying still, away from the fingertips for 0.25 s), and it lands where it next
    comes to rest, if that is seen before the next throw (a missed cap can fly out of view). Every clip ends with
    the cap in the top cup, so the last throw went in; it is checked against the cup's opening when that was
    found, and a cap last seen inside the opening may have dropped out of sight behind the cup's wall.
    Returns the throws (dicts of frame indices, landing point or None, whether it went in) and the rests
    (start, end, table point)."""
    T = len(track.t)
    cap = track.obj_px
    seen = np.isfinite(cap[:, 0])
    filled = _fill_gaps(cap, seen) if seen.sum() >= 2 else cap
    tips = hand_px[:, list(PINCH_TIPS)].mean(axis=1)
    r_px = float(np.nanmedian(track.obj_r_px))
    speed = np.linalg.norm(np.gradient(_smooth(filled, fps, 0.05), axis=0), axis=1) * fps  # px/s
    dist = np.linalg.norm(filled - tips, axis=1)
    held = dist < 2.5 * r_px
    # At rest: still, and not between the fingertips (a hand hovering above it can look near it in the image).
    slow = np.linalg.norm(np.gradient(_smooth(filled, fps, 0.15), axis=0), axis=1) * fps < 3 * r_px
    still = slow & (dist > 1.2 * r_px) & seen
    k = max(3, int(0.25 * fps))
    rests, i = [], 0
    while i < T:
        if still[i]:
            j = i
            while j < T and still[j]:
                j += 1
            if j - i >= k:
                rests.append((i, j))
            i = j
        else:
            i += 1
    if not rests:
        raise RuntimeError("The cap is never seen at rest; check the cap color and that it is in view.")
    in_cup_z = props.top_bottom_z + props.cap_height / 2 + 0.002

    def point(a, b, in_cup):
        px = np.nanmedian(cap[a:b], axis=0)
        return camera.backproject_to_plane(px, z=in_cup_z if in_cup else props.cap_height / 2)

    rest_info = [(a, b, point(a, b, _inside(track.in_cup_px, np.nanmedian(cap[a:b], axis=0)))) for a, b in rests]
    # Releases: held, then away from the fingertips (and moving fast) for the next 0.1 s.
    away_k = max(2, int(0.1 * fps))
    releases = []
    for i in range(1, T - away_k):
        if held[i - 1] and not held[i:i + away_k].any() and speed[i:i + away_k].max() > 15 * r_px and \
                (not releases or i - releases[-1] > int(0.5 * fps)):
            # The hand follows the cap for a moment after letting go: the release is the last frame the cap was
            # between the fingertips.
            on = np.nonzero(dist[max(0, i - int(0.3 * fps)):i] < RELEASE_GAP * r_px)[0]
            releases.append(max(0, i - int(0.3 * fps)) + int(on[-1]) if len(on) else i - 1)
    throws = []
    for n, rel in enumerate(releases):
        before = [rb for rb in rest_info if rb[1] <= rel]
        if not before:
            continue
        a0, b0, _ = before[-1]
        # Lift-off: the cap leaves its resting spot (the rest itself can end earlier, as soon as the hand is near
        # it in the image); the pinch is the moment before.
        p_rest = np.nanmedian(cap[a0:b0], axis=0)
        off = np.linalg.norm(filled[b0:rel + 1] - p_rest, axis=1) > 1.5 * r_px
        moved = [i for i in range(len(off) - 2) if off[i:i + 3].all()]
        lift = b0 + (moved[0] if moved else 0)
        grasp = max(b0, lift - int(0.25 * fps))
        nxt = releases[n + 1] if n + 1 < len(releases) else T
        after = [ra for ra in rest_info if rel < ra[0] < nxt]
        if after:
            a1, b1, p1 = after[0]
            in_cup = _inside(track.in_cup_px, np.nanmedian(cap[a1:b1], axis=0))
            throws.append({"grasp": grasp, "lift": lift, "release": rel, "land_frame": a1, "land": p1,
                           "in_cup": in_cup})
        else:
            seg = np.arange(rel, nxt)
            seen_after = np.nonzero(seen[seg])[0]
            last = int(seg[seen_after[-1]]) if len(seen_after) else rel
            gone_in = n == len(releases) - 1 and _inside(track.in_cup_px, filled[last])
            # The opening's centre on the mouth's plane, then down to where the cap comes to rest.
            mouth_z = props.top_bottom_z + props.cup_height
            at_mouth = camera.backproject_to_plane(track.in_cup_px.mean(axis=0), z=mouth_z)
            land = at_mouth - [0, 0, mouth_z - in_cup_z] if gone_in else None
            throws.append({"grasp": grasp, "lift": lift, "release": rel, "land_frame": min(last + 1, T - 1),
                           "land": land, "in_cup": gone_in})
            if gone_in:  # the flight is seen until it passes the mouth
                throws[-1]["fit_end"] = (last, at_mouth)
    if throws and not any(th["in_cup"] for th in throws) and throws[-1]["land"] is not None:
        # The clip ends with the cap in the cup: the last landing is in it, even if the opening wasn't found.
        a1 = throws[-1]["land_frame"]
        b1 = next((rb for ra, rb, _ in rest_info if ra == a1), a1 + 1)
        throws[-1]["land"] = point(a1, b1, True)
        throws[-1]["in_cup"] = True
    # The hand is often missed while it throws, so a release can be found too early. Seen from above, the flight
    # is the stretch before the landing where the cap crosses the table in a straight line at a steady speed.
    if looking_down(camera):
        for th in throws:
            th["release"] = _flight_start(track, camera, th, fps, z=in_cup_z + props.cup_height)
    # A throw that went in, and everything after it, ends the clip (the thumbs-up follows).
    last_in = max((k for k, th in enumerate(throws) if th["in_cup"]), default=None)
    if last_in is not None:
        throws = throws[:last_in + 1]
    return throws, rest_info


def looking_down(camera: Camera, max_tilt_deg: float = 35.0) -> bool:
    """The camera looks at the table from above (its optical axis within max_tilt_deg of straight down)."""
    return bool(-(camera.R.T @ np.array([0.0, 0.0, 1.0]))[2] > np.cos(np.radians(max_tilt_deg)))


def _flight_start(track: SkillTrack, camera: Camera, th: dict, fps: float, z: float, tol: float = 0.015) -> int:
    """The earliest frame from which the cap moves at constant velocity across the table until it lands."""
    r, L = th["release"], th.get("fit_end", (th["land_frame"],))[0]
    if th["in_cup"] and track.in_cup_px is not None:  # up to where it drops into the opening (then it bounces)
        inside = [i for i in range(r, L + 1) if _inside(track.in_cup_px, track.obj_px[i])]
        L = inside[0] if inside else L
    seen = [i for i in range(r, L + 1) if np.isfinite(track.obj_px[i, 0])]
    if len(seen) < 4:
        return r
    xy = np.array([camera.backproject_to_plane(track.obj_px[i], z=z)[:2] for i in seen])
    t = track.t[seen]
    end = len(seen) - 1
    best = r
    for k in range(end - 3, -1, -1):  # grow the window backwards while the motion stays a straight line
        A = np.column_stack([t[k:end + 1], np.ones(end + 1 - k)])
        coef = np.linalg.lstsq(A, xy[k:end + 1], rcond=None)[0]
        if np.abs(A @ coef - xy[k:end + 1]).max() > tol:
            break
        best = seen[k]
    return best


def _fit_throw(track: SkillTrack, camera: Camera, pinch: np.ndarray, th: dict, fps: float,
               mouth_z: float, looking_down: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Cap position and velocity at a throw's release: the flight from the fingertips to where it landed.

    Seen from the side, the flight's curvature in the image gives its depth and the fit starts at the fingertips.
    From above (`looking_down`), the image shows where the cap is and how fast it moves across the table, but
    hardly its height: the release point is where the cap is seen, at the fingertips' height kept within 5 cm
    below to 15 cm above the top cup's mouth (the hand's depth while throwing is the least certain part of its
    path). If the fit then slides along the camera rays (no curvature to see), the ballistic flight to the landing
    point, with the time from the cap's speed across the table, is used."""
    t = track.t
    r, L = th["release"], th["land_frame"]
    frames = np.arange(r, L + 1)
    frames = frames[np.isfinite(track.obj_px[frames, 0])]
    p_rel = pinch[r]
    z_r = float(np.clip(p_rel[2], mouth_z - 0.05, mouth_z + 0.15))
    if looking_down:
        p_rel = camera.backproject_to_plane(track.obj_px[r], z=z_r) if np.isfinite(track.obj_px[r, 0]) else \
            np.array([p_rel[0], p_rel[1], z_r])
    t_land = t[L]
    if th["land"] is None:  # flew out of view: the flight seen until then, starting at the fingertips
        frames = frames[frames <= r + int(0.3 * fps)]
        if len(frames) < 3:
            return p_rel, np.zeros(3)
        uv = track.obj_px[frames]
        far = camera.backproject_to_plane(uv[-1], z=p_rel[2])
        v_init = (far - p_rel) / max(t[frames[-1]] - t[r], 1.0 / fps)
        return fit_flight(t[frames], uv, camera, t[r], p_rel, v_init)
    # It lands when it reaches its resting height: the top cup's bottom, or the table (or, out of sight in the
    # cup, it was last seen passing the mouth).
    end_point = th["land"]
    if "fit_end" in th:
        t_land, end_point = t[th["fit_end"][0]], th["fit_end"][1]
    tf = max(t_land - t[r], 1.0 / fps)
    early = frames[frames <= r + int(0.12 * fps)]
    if looking_down and len(early) >= 3:  # flight time from the cap's speed across the table after the release
        xy = np.array([camera.backproject_to_plane(track.obj_px[i], z=z_r)[:2] for i in early])
        v_h = np.polyfit(t[early] - t[r], xy, 1)[0]
        d = end_point[:2] - p_rel[:2]
        if np.linalg.norm(v_h) > 0.1:
            tf = float(np.clip(np.dot(d, v_h) / np.dot(v_h, v_h), 0.12, max(tf, 0.12)))
    v_init = (end_point - p_rel + np.array([0, 0, 0.5 * GRAVITY * tf * tf])) / tf
    if len(frames) < 3:
        return p_rel, v_init
    p0, v0 = fit_flight(t[frames], track.obj_px[frames], camera, t[r], p_rel, v_init, end=(t_land, end_point))
    if looking_down and (np.linalg.norm(p0 - p_rel) > 0.05 or np.linalg.norm(v0) > 4.0):
        return p_rel, v_init
    return p0, v0
