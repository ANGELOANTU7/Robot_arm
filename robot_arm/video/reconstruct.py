"""Turn per-frame detections into a clean, metric human demonstration in the table frame.

Steps: lift the pinch point to 3D -> fill gaps and smooth -> find grasp and release -> fix the depth scale with
the grasp anchor (pinch point == ball centre) -> remove the remaining offset with the release anchor (pinch point
directly above the cup).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from robot_arm.video.calibrate import Camera
from robot_arm.video.lift3d import aperture, pinch_point_cam


@dataclass
class RawTrack:
    """Per-frame detections from a video. Missing detections are NaN."""

    t: np.ndarray  # (T,) seconds
    hand_px: np.ndarray  # (T, 21, 2)
    hand_world: np.ndarray  # (T, 21, 3)
    ball_px: np.ndarray  # (T, 2)
    ball_r_px: np.ndarray  # (T,)
    cup_bottom_px: np.ndarray  # (T, 2)


@dataclass
class HumanDemo:
    """A human pick-and-drop demonstration in the table frame (metres, seconds)."""

    t: np.ndarray  # (T,)
    pinch: np.ndarray  # (T, 3) thumb/index midpoint
    aperture: np.ndarray  # (T,) thumb-index distance
    closed: np.ndarray  # (T,) bool, hand holding the ball
    ball: np.ndarray  # (3,) ball centre at the start
    cup: np.ndarray  # (3,) centre of the cup's base
    t_grasp: float
    t_release: float
    info: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        info = d.pop("info")
        np.savez(path, **d, info_keys=np.array(list(info.keys())), info_values=np.array(list(info.values()), float))

    @classmethod
    def load(cls, path: str | Path) -> "HumanDemo":
        z = np.load(path)
        info = dict(zip(z["info_keys"].tolist(), z["info_values"].tolist()))
        return cls(z["t"], z["pinch"], z["aperture"], z["closed"].astype(bool), z["ball"], z["cup"],
                   float(z["t_grasp"]), float(z["t_release"]), info)


def _fill_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation over invalid frames (edges are held constant)."""
    idx = np.arange(len(x))
    out = x.copy()
    flat = out.reshape(len(x), -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(idx, idx[valid], flat[valid, j])
    return out


def _smooth(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    window = max(5, int(round(seconds * fps)) | 1)
    if len(x) <= window:
        return x
    return savgol_filter(x, window, polyorder=2, axis=0)


def _crossing(signal: np.ndarray, start: int, threshold: float, above: bool) -> int | None:
    """First index >= start where the signal is above/below the threshold."""
    hits = np.nonzero(signal[start:] > threshold if above else signal[start:] < threshold)[0]
    return int(start + hits[0]) if len(hits) else None


def find_events(t: np.ndarray, aperture_m: np.ndarray, ball_px: np.ndarray, ball_px0: np.ndarray,
                ball_r_px: float) -> tuple[int, int, dict]:
    """Grasp and release frame indices.

    The clearest cue is the ball itself: it sits still until it is lifted. The grasp is the last time the
    thumb-index aperture closes before lift-off; the release is the first time it opens wide again afterwards.
    """
    fps = 1 / np.median(np.diff(t))
    with np.errstate(invalid="ignore"):
        moved = np.linalg.norm(ball_px - ball_px0, axis=1) > max(1.0 * ball_r_px, 5.0)
    k = max(2, int(0.15 * fps))
    sustained = np.array([moved[i:i + k].all() for i in range(len(moved) - k + 1)])
    if not sustained.any():
        raise RuntimeError("The ball never visibly moves away from its start position; check --ball-color, "
                           "or pass --grasp-time/--release-time.")
    lift = int(np.argmax(sustained))
    return events_from_lift(t, aperture_m, lift)


def events_from_lift(t: np.ndarray, aperture_m: np.ndarray, lift: int) -> tuple[int, int, dict]:
    fps = 1 / np.median(np.diff(t))
    hold = float(np.median(aperture_m[lift:lift + max(3, int(0.4 * fps))]))
    before = aperture_m[max(0, lift - int(2.0 * fps)):lift + 1]
    open_before = float(before.max())
    threshold = hold + 0.3 * max(open_before - hold, 0.005)
    above = np.nonzero(before > threshold)[0]
    grasp = max(0, lift - int(2.0 * fps)) + (int(above[-1]) + 1 if len(above) else 0)
    open_after = float(aperture_m[lift:].max())
    release = _crossing(aperture_m, lift, hold + 0.5 * max(open_after - hold, 0.005), above=True)
    if release is None:
        raise RuntimeError("Could not see the hand open to release the ball.")
    info = {"hold_aperture": hold, "open_aperture_before": open_before, "open_aperture_after": open_after,
            "t_lift": float(t[lift])}
    return min(grasp, lift), release, info


def reconstruct(raw: RawTrack, camera: Camera, ball_radius: float = 0.02, cup_radius: float = 0.04,
                grasp_time: float | None = None, release_time: float | None = None) -> HumanDemo:
    """grasp_time / release_time (seconds from the first frame with a hand) override the automatic events."""
    T = len(raw.t)
    fps = 1 / np.median(np.diff(raw.t))
    hand_ok = ~np.isnan(raw.hand_px).any(axis=(1, 2))
    if hand_ok.sum() < max(10, 0.5 * fps):
        raise RuntimeError(f"Hand detected in only {hand_ok.sum()}/{T} frames.")
    # Trim to the span where the hand is visible; inside it, most frames need a detection.
    first, last = np.nonzero(hand_ok)[0][[0, -1]]
    if hand_ok[first:last + 1].mean() < 0.6:
        raise RuntimeError(f"Hand lost in {100 * (1 - hand_ok[first:last + 1].mean()):.0f}% of the frames "
                           "while it is in view; improve lighting or keep the hand unoccluded.")
    sl = slice(first, last + 1)
    t = raw.t[sl] - raw.t[first]
    ok = hand_ok[sl]

    # Static objects: use the first half second, where nothing is occluded yet.
    head = slice(0, max(3, int(0.5 * fps)))
    ball_px0 = np.nanmedian(raw.ball_px[head], axis=0)
    ball_r_px = float(np.nanmedian(raw.ball_r_px[head]))
    cup_bottom_px = np.nanmedian(raw.cup_bottom_px[head], axis=0)
    if np.isnan(ball_px0).any() or np.isnan(cup_bottom_px).any():
        raise RuntimeError("Ball or cup not detected in the first frames (see --ball-color / --cup-color).")
    ball = camera.backproject_to_plane(ball_px0, z=ball_radius)
    cup_front = camera.backproject_to_plane(cup_bottom_px, z=0.0)
    view = cup_front - camera.center_table
    view[2] = 0
    cup = cup_front + cup_radius * view / np.linalg.norm(view)

    # Per-frame 3D pinch point (camera frame) and aperture.
    hand_px = _fill_gaps(raw.hand_px[sl], ok)
    hand_world = _fill_gaps(raw.hand_world[sl], ok)
    pinch_cam = np.array([pinch_point_cam(hand_px[i], hand_world[i], camera.K) for i in range(len(t))])
    ap = np.array([aperture(w) for w in hand_world])

    # Smooth in (pixel, depth) coordinates: MediaPipe's depth is much noisier than its pixel positions.
    depth = pinch_cam[:, 2]
    uv = (pinch_cam / depth[:, None]) @ camera.K.T
    uv = _smooth(uv[:, :2], fps, 0.15)
    depth = _smooth(depth, fps, 0.5)
    rays = np.linalg.solve(camera.K, np.column_stack([uv, np.ones(len(t))]).T).T
    pinch_cam = rays * depth[:, None]
    ap = _smooth(ap, fps, 0.2)

    if grasp_time is not None and release_time is not None:
        grasp, release = (int(np.argmin(np.abs(t - x))) for x in (grasp_time, release_time))
        event_info = {"manual_events": 1.0}
    else:
        grasp, release, event_info = find_events(t, ap, raw.ball_px[sl], ball_px0, ball_r_px)

    # Depth scale from the grasp anchor: the pinch point (scaled along its camera ray) must hit the ball centre.
    ball_cam = camera.to_cam(ball[None])[0]
    win = slice(max(0, grasp - 2), grasp + 3)
    p = pinch_cam[win]
    scale = float(np.sum(p @ ball_cam) / np.sum(p * p))
    pinch = camera.to_table(scale * pinch_cam)

    # Remaining offsets: at grasp the pinch is at the ball; at release it is right above the cup.
    off_grasp = ball - pinch[grasp]
    off_release = np.array([*(cup[:2] - pinch[release, :2]), 0.0])
    w = np.clip((np.arange(len(t)) - grasp) / max(release - grasp, 1), 0, 1)[:, None]
    pinch = pinch + (1 - w) * off_grasp + w * off_release

    closed = np.zeros(len(t), bool)
    closed[grasp:release] = True
    info = {"scale": scale, "grasp_offset_m": float(np.linalg.norm(off_grasp)),
            "release_offset_m": float(np.linalg.norm(off_release)), "fps": float(fps),
            "ball_r_px": ball_r_px, **event_info}
    return HumanDemo(t, pinch, ap, closed, ball, cup, float(t[grasp]), float(t[release]), info)
