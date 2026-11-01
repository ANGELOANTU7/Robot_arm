"""Hand-held clips: every frame warped onto one fixed virtual camera, using the marker sheet on the table.

The skill pipeline assumes a still camera: it finds the moved cup by comparing the first and last second, and lifts
the hand with one camera pose. A hand-held phone moves. Here each frame's camera pose is found from the sheet's
markers (the focal length is fixed for the clip), smoothed over time, and the frame is warped by the table plane's
homography onto the camera of a reference frame in the first second.

- Points on the table land exactly where the reference camera sees them.
- A hand-held phone mostly turns, and a pure turn is undone exactly for every point, on the table or not.
- What is left is the parallax of points above the table from the phone's translation: a 2 cm shift seen from
  40 cm moves a point 15 cm above the table by about 0.75 cm in the reference view.

The frames are warped onto a canvas `pad` larger on every side, with the principal point still at its centre, so
what drifts out of the reference view is kept.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

from robot_arm.video.calibrate import MARKER_MM, detect_markers, marker_corners_table


def plane_homography(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Table plane (z = 0, metres) -> image pixels."""
    R = Rotation.from_rotvec(rvec).as_matrix()
    return K @ np.column_stack([R[:, 0], R[:, 1], tvec])


def frame_poses(frames: list[np.ndarray], K: np.ndarray, marker_mm: float = MARKER_MM,
                max_err_px: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: rotation vector, translation (NaN where the sheet isn't seen well) and the number of markers."""
    corners = marker_corners_table(marker_mm)
    T = len(frames)
    rvecs, tvecs, counts = np.full((T, 3), np.nan), np.full((T, 3), np.nan), np.zeros(T, int)
    for i, frame in enumerate(frames):
        found = detect_markers(frame)
        counts[i] = len(found)
        if len(found) < 2:
            continue
        obj = np.concatenate([corners[m] for m in sorted(found)]).astype(np.float32)
        img = np.concatenate([found[m] for m in sorted(found)]).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        if np.linalg.norm(proj[:, 0] - img, axis=1).mean() < max_err_px:
            rvecs[i], tvecs[i] = rvec.ravel(), tvec.ravel()
    return rvecs, tvecs, counts


def smooth_poses(rvecs: np.ndarray, tvecs: np.ndarray, ref: int, fps: float, smooth_s: float = 0.0,
                 outlier: tuple[float, float] = (0.02, 0.01)) -> tuple[np.ndarray, np.ndarray]:
    """Fill the frames without a pose and replace the outliers: a pose more than `outlier` (rad, m) from the median of
    its neighbours (a frame with only two markers can give one that is far off). Not smoothed by default: the
    markers' corners are found to a fraction of a pixel, while a hand shakes the phone several times a second, and
    smoothing would leave that shake in the warped frames. Rotations are handled relative to the reference frame's,
    where they are small enough to interpolate and compare component-wise."""
    ok = np.isfinite(rvecs[:, 0])
    R_ref = Rotation.from_rotvec(rvecs[ref])
    rel = np.full_like(rvecs, np.nan)
    rel[ok] = (R_ref.inv() * Rotation.from_rotvec(rvecs[ok])).as_rotvec()
    idx = np.arange(len(rvecs))
    out = []
    for x, tol in zip((rel, tvecs), outlier):
        filled = np.column_stack([np.interp(idx, idx[ok], x[ok, k]) for k in range(3)])
        med = median_filter(filled, size=(5, 1), mode="nearest")
        bad = np.linalg.norm(filled - med, axis=1) > tol
        filled[bad] = med[bad]
        out.append(gaussian_filter1d(filled, sigma=smooth_s * fps, axis=0, mode="nearest") if smooth_s > 0
                   else filled)
    rel_s, t_s = out
    return (R_ref * Rotation.from_rotvec(rel_s)).as_rotvec(), t_s


def stabilize(frames: list[np.ndarray], K: np.ndarray, fps: float, marker_mm: float = MARKER_MM,
              pad: float = 0.15) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Frames warped onto the reference camera (the frame in the first second where the most markers are seen),
    on a canvas padded by `pad` of the frame size on each side. Returns (frames, K of the canvas, info)."""
    rvecs, tvecs, counts = frame_poses(frames, K, marker_mm)
    seen = np.isfinite(rvecs[:, 0])
    if seen.mean() < 0.5:
        raise RuntimeError(f"The marker sheet was seen in only {seen.mean():.0%} of the frames: a hand-held clip "
                           "needs it in view most of the time.")
    first = np.nonzero(seen[: max(1, int(fps))])[0]
    if not len(first):
        raise RuntimeError("The marker sheet was not seen in the first second.")
    ref = int(first[np.argmax(counts[first])])
    rvecs, tvecs = smooth_poses(rvecs, tvecs, ref, fps)

    h, w = frames[0].shape[:2]
    px, py = int(round(pad * w)), int(round(pad * h))
    size = (w + 2 * px, h + 2 * py)
    shift = np.array([[1.0, 0, px], [0, 1, py], [0, 0, 1]])
    K_canvas = shift @ K
    H_ref = plane_homography(K_canvas, rvecs[ref], tvecs[ref])
    out, moved = [], []
    for i, frame in enumerate(frames):
        H = H_ref @ np.linalg.inv(plane_homography(K, rvecs[i], tvecs[i]))
        out.append(cv2.warpPerspective(frame, H, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        c = H @ np.array([w / 2, h / 2, 1.0])
        moved.append(np.linalg.norm(c[:2] / c[2] - [w / 2 + px, h / 2 + py]))
    centres = np.array([-Rotation.from_rotvec(r).as_matrix().T @ t for r, t in zip(rvecs, tvecs)])
    info = {"stab_pose_frames": float(seen.mean()), "stab_ref_frame": ref,
            "stab_max_shift_px": float(np.max(moved)),
            "stab_camera_travel_cm": float(100 * np.linalg.norm(np.ptp(centres, axis=0)))}
    return out, K_canvas, info
