"""Per-frame detections: hand landmarks (MediaPipe) and colored ball / cup blobs (HSV thresholds)."""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/latest/hand_landmarker.task")
HAND_MODEL_PATH = Path.home() / ".cache" / "robot_arm" / "hand_landmarker.task"

# MediaPipe hand landmark indices.
WRIST, THUMB_TIP, INDEX_TIP, INDEX_MCP, MIDDLE_MCP = 0, 4, 8, 5, 9

# HSV ranges (OpenCV: H in 0..179). Pairs of (lower, upper); red wraps around 0 so it has two ranges.
COLOR_RANGES = {
    "orange": [((5, 120, 120), (22, 255, 255))],
    "yellow": [((22, 100, 120), (35, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "blue": [((95, 120, 50), (130, 255, 255))],
    "red": [((0, 120, 70), (6, 255, 255)), ((158, 120, 70), (179, 255, 255))],  # party cups read ~H 165-175
    "pink": [((140, 60, 120), (170, 255, 255))],
}


def ensure_hand_model() -> Path:
    if not HAND_MODEL_PATH.exists():
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH


@dataclass
class HandDetection:
    image_px: np.ndarray  # (21, 2) pixel coordinates
    world_m: np.ndarray  # (21, 3) metric landmarks around the hand centre, camera-aligned axes
    score: float
    handedness: str


class HandTracker:
    """MediaPipe HandLandmarker in video mode. Keeps the most confident hand in each frame."""

    def __init__(self, min_confidence: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence)
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def __call__(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandDetection | None:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                   timestamp_ms)
        if not result.hand_landmarks:
            return None
        scores = [hd[0].score for hd in result.handedness]
        i = int(np.argmax(scores))
        image_px = np.array([[lm.x * w, lm.y * h] for lm in result.hand_landmarks[i]])
        world_m = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]])
        return HandDetection(image_px, world_m, scores[i], result.handedness[i][0].category_name)

    def both(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
        """Every hand in the frame (up to two)."""
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                   timestamp_ms)
        return [HandDetection(np.array([[lm.x * w, lm.y * h] for lm in px]),
                              np.array([[lm.x, lm.y, lm.z] for lm in world]),
                              hd[0].score, hd[0].category_name)
                for px, world, hd in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)]

    def close(self):
        self._landmarker.close()


POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/"
                  "float16/latest/pose_landmarker_heavy.task")
POSE_MODEL_PATH = Path.home() / ".cache" / "robot_arm" / "pose_landmarker_heavy.task"

# MediaPipe pose landmark indices, by the model's own labels. Looking down from above, with the person sat at the
# table's edge, the model has no face or chest to tell the sides apart and comes out swapped: the user's right arm
# is the one labelled "left" (11, 13, 15). Pick the working arm by its wrist's distance to the working hand's
# wrist, never by the label.
SHOULDERS, ELBOWS, WRISTS = (11, 12), (13, 14), (15, 16)  # (labelled left, labelled right)


def ensure_pose_model() -> Path:
    if not POSE_MODEL_PATH.exists():
        POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe pose model to {POSE_MODEL_PATH} ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


@dataclass
class PoseDetection:
    image_px: np.ndarray  # (33, 2) pixel coordinates
    world_m: np.ndarray  # (33, 3) metric landmarks around the hips, camera-aligned axes
    visibility: np.ndarray  # (33,) the model's confidence that each landmark is in view and not occluded


class PoseTracker:
    """MediaPipe PoseLandmarker in video mode: the body's 33 landmarks (shoulders, elbows, wrists, hips, ...).
    Looking down at a table from the front, the head is mostly out of frame and the torso is hidden, so what
    matters is how often the shoulders and the elbow are found: check `visibility` before relying on them."""

    def __init__(self, min_confidence: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_pose_model())),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=min_confidence, min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence)
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def __call__(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseDetection | None:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                   timestamp_ms)
        if not result.pose_landmarks:
            return None
        lms, world = result.pose_landmarks[0], result.pose_world_landmarks[0]
        return PoseDetection(np.array([[lm.x * w, lm.y * h] for lm in lms]),
                             np.array([[lm.x, lm.y, lm.z] for lm in world]),
                             np.array([lm.visibility for lm in lms]))

    def close(self):
        self._landmarker.close()


@dataclass
class Blob:
    center: np.ndarray  # (2,) pixel centroid
    radius: float  # radius of the minimum enclosing circle, pixels
    bottom: np.ndarray  # (2,) lowest pixel of the blob (largest image y), at the centroid's column
    area: float


def color_mask(frame_bgr: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def detect_blob(frame_bgr: np.ndarray, color: str, min_area: float = 30.0,
                near: np.ndarray | None = None, max_jump: float | None = None, compact: bool = False,
                mask: np.ndarray | None = None, max_area: float = np.inf) -> Blob | None:
    """Largest blob of the given color (or the one closest to `near`, if given, within `max_jump` pixels).

    compact: only roundish blobs (a bottle cap seen from any side, not a pen). `mask` replaces the color mask."""
    mask = color_mask(frame_bgr, color) if mask is None else mask
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    if compact:
        w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
        candidates = [i for i in candidates if max(w[i], h[i]) <= 3 * min(w[i], h[i])
                      and stats[i, cv2.CC_STAT_AREA] >= 0.5 * w[i] * h[i]]
    if near is not None:
        candidates = [i for i in candidates if max_jump is None or np.linalg.norm(centroids[i] - near) < max_jump]
    if not candidates:
        return None
    if near is not None:
        best = min(candidates, key=lambda i: np.linalg.norm(centroids[i] - near))
    else:
        best = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(labels == best)
    (cx, cy), radius = cv2.minEnclosingCircle(np.stack([xs, ys], 1).astype(np.float32))
    col = np.abs(xs - centroids[best][0]) < max(2.0, 0.15 * radius)
    bottom = np.array([centroids[best][0], ys[col].max() if col.any() else ys.max()], float)
    return Blob(np.array(centroids[best], float), float(radius), bottom, float(stats[best, cv2.CC_STAT_AREA]))
