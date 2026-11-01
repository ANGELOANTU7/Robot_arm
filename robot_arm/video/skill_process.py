"""Skill clip (video file) -> SkillTrack -> SkillDemo, plus debug visualisations.

All cups have the same color (red party cups), so the cup that moves is told apart by what changes: the still
frames at the start and at the end (hand out of view) give the red pixels that are there in both (cups that stay
put), the ones only there at the start (where the moved cup was) and the ones only there at the end (where it
went). In between, the moved cup is the red that is not where a static cup is. The toss clips track the cap.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robot_arm.video.calibrate import SHEET_MM, Camera, calibrate
from robot_arm.video.cups import cup_centre, long_axis, two_cup_centres
from robot_arm.video.process import HAND_EDGES, read_video
from robot_arm.video.skills import Props, SkillDemo, SkillTrack, reconstruct_skill
from robot_arm.video.stabilize import stabilize
from robot_arm.video.tracking import HandTracker, color_mask, detect_blob

SKILLS = ("place_base", "place_top", "toss")


@dataclass
class SkillColors:
    cups: str = "red"
    cap: str = "blue"


def skill_from_name(path: str | Path) -> str:
    name = Path(path).stem.lower()
    for skill in SKILLS:
        if name.startswith(skill):
            return skill
    raise ValueError(f"Cannot tell the skill of {Path(path).name}: name clips place_base_*, place_top_* or toss_* "
                     "(or pass --skill).")


def largest_component(mask: np.ndarray, min_area: int = 200) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < min_area:
        return None
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def table_mask(frame: np.ndarray) -> np.ndarray:
    """The table's outline in the image, from its wood colour, to leave out red or blue things around it (a
    shirt's pattern, a bag). All of the image if there is no wooden table in view."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (9, 9), 0), cv2.COLOR_BGR2HSV)
    wood = cv2.inRange(hsv, np.array((6, 30, 170)), np.array((26, 200, 255)))  # brighter than skin
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    table = largest_component(wood, min_area=int(0.1 * wood.size))
    if table is None:
        return np.ones(frame.shape[:2], bool)
    ys, xs = np.nonzero(table)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [hull], 1)
    return mask > 0


def still_mask(frames: list[np.ndarray], color: str, where: np.ndarray | bool = True) -> np.ndarray:
    """Pixels of the color in most of the given (still) frames, inside `where`."""
    votes = sum((color_mask(f, color) > 0).astype(np.int32) for f in frames)
    return (votes * 2 > len(frames)) & where


def sheet_mask(camera: Camera, shape: tuple[int, int], margin: float = 0.015) -> np.ndarray:
    """Pixels of the (white) marker sheet."""
    hx, hy = SHEET_MM[0] / 2000 + margin, SHEET_MM[1] / 2000 + margin
    corners = camera.project(np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0.0]]))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 1)
    return mask > 0


def top_cup_opening(frame: np.ndarray, pyramid: np.ndarray, camera: Camera) -> np.ndarray:
    """Image polygon of the top cup's opening: seen from above, a mouth-up cup shows its white inside, next to
    the red of the pyramid (the red is only its outer wall and the base cups). Falls back to the red's hull."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 170) & (hsv[..., 1] < 60) & ~sheet_mask(camera, frame.shape)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    k = max(5, int(0.05 * frame.shape[1])) | 1
    near = cv2.dilate(pyramid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    best, best_area = None, 0
    for c in range(1, n):
        area = stats[c, cv2.CC_STAT_AREA]
        if area < 0.15 * frame.size / 3 and area > best_area and (near & (labels == c)).sum() > 0.02 * area:
            best, best_area = labels == c, area
    region = (best | pyramid) if best is not None else pyramid
    ys, xs = np.nonzero(region if best is None else best)
    return cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))[:, 0, :].astype(float)


def track_hands(frames: list[np.ndarray], fps: float) -> dict:
    """MediaPipe hand landmarks in every frame (NaN where no hand is found), as `hands` for the trackers."""
    T = len(frames)
    hand_px, hand_world = np.full((T, 21, 2), np.nan), np.full((T, 21, 3), np.nan)
    tracker = HandTracker()
    for i, frame in enumerate(frames):
        det = tracker(frame, int(round(i * 1000 / fps)))
        if det is not None:
            hand_px[i], hand_world[i] = det.image_px, det.world_m
    tracker.close()
    return {"hand_px": hand_px, "hand_world": hand_world}


def track_two_hands(frames: list[np.ndarray], fps: float) -> dict:
    """Both hands in every frame, as two continuous tracks (T, 2, 21, ...): a hand stays in its slot from frame to
    frame by the nearest wrist. NaN where a slot's hand isn't found."""
    T = len(frames)
    hand_px, hand_world = np.full((T, 2, 21, 2), np.nan), np.full((T, 2, 21, 3), np.nan)
    last = np.full((2, 2), np.nan)
    jump = 0.25 * frames[0].shape[1]
    tracker = HandTracker()
    for i, frame in enumerate(frames):
        dets = tracker.both(frame, int(round(i * 1000 / fps)))[:2]
        cost = np.array([[np.linalg.norm(d.image_px[0] - last[s]) for s in range(2)] for d in dets]).reshape(-1, 2)
        seen = ~np.isnan(last).any(axis=1)
        if not seen.any() and dets:  # slot 0 for the upper hand in the image
            order = sorted(range(len(dets)), key=lambda k: dets[k].image_px[0, 1])
            pairs = [(k, n) for n, k in enumerate(order)]
        elif len(dets) == 2:
            keep = np.nan_to_num(cost, nan=jump)
            pairs = [(0, 0), (1, 1)] if keep[0, 0] + keep[1, 1] <= keep[0, 1] + keep[1, 0] else [(0, 1), (1, 0)]
        elif dets:
            s = int(np.argmin(np.nan_to_num(cost[0], nan=jump)))
            pairs = [(0, s)]
        else:
            pairs = []
        for k, s in pairs:
            hand_px[i, s], hand_world[i, s] = dets[k].image_px, dets[k].world_m
            last[s] = dets[k].image_px[0]
    tracker.close()
    return {"hand_px": hand_px, "hand_world": hand_world}


def working_hand(hands: dict, obj_px: np.ndarray, fps: float) -> tuple[dict, dict]:
    """The hand track that is with the object while it moves (never by MediaPipe's left/right label). Falls back
    to the object's nearest track over the whole clip if it hardly moves."""
    seen = ~np.isnan(obj_px[:, 0])
    step = max(1, int(0.1 * fps))
    speed = np.full(len(obj_px), 0.0)
    speed[step:] = np.nan_to_num(np.linalg.norm(obj_px[step:] - obj_px[:-step], axis=1))
    moving = seen & (speed >= max(np.percentile(speed[seen], 75), 1e-6))
    dist = np.linalg.norm(hands["hand_px"][:, :, :9].mean(axis=2) - obj_px[:, None], axis=2)  # (T, 2)
    both_seen = ~np.isnan(dist).any(axis=1)
    frames = moving & both_seen if (moving & both_seen).sum() >= 5 else seen
    d = np.nanmedian(dist[frames], axis=0)
    d = np.nan_to_num(d, nan=np.inf)
    s = int(np.argmin(d))
    return {"hand_px": hands["hand_px"][:, s], "hand_world": hands["hand_world"][:, s]}, \
        {"hand_slot": s, "hand_margin_px": float(d[1 - s] - d[s]) if np.isfinite(d).all() else -1.0}


def track_skill(frames: list[np.ndarray], fps: float, skill: str, colors: SkillColors, camera: Camera,
                props: Props = Props(), hands: dict | None = None, second_cup: bool = False) -> SkillTrack:
    """Hand landmarks per frame (or `hands`, cached: hand_px, hand_world), the moved cup's blob centre (place) or
    the cap's centre (toss), and the marks from the still frames at the start and end (see SkillTrack).
    `second_cup`: a place_base clip of cup B, with cup A already standing where B ends up (marks["other"])."""
    T = len(frames)
    still = max(3, int(0.3 * fps))
    table = table_mask(frames[0])
    first, last = still_mask(frames[:still], colors.cups, table), still_mask(frames[-still:], colors.cups, table)
    marks, in_cup_px, static = {}, None, None
    kernel = np.ones((7, 7), np.uint8)
    on_table = lambda c: None if c is None else np.array([c[0], c[1], 0.0])  # noqa: E731
    if skill == "toss":
        pyramid = largest_component(first)
        if pyramid is None:
            raise RuntimeError("No cups found in the first frames; check the cup color (--check).")
        marks["axis"] = long_axis(pyramid, camera)
        in_cup_px = top_cup_opening(frames[0], pyramid, camera)
    else:
        static = cv2.dilate((first & last).astype(np.uint8), kernel) > 0
        start = largest_component(cv2.morphologyEx((first & ~last).astype(np.uint8), cv2.MORPH_OPEN, kernel))
        end = largest_component(cv2.morphologyEx((last & ~first).astype(np.uint8), cv2.MORPH_OPEN, kernel))
        # All cups start mouth-down: their outline is the mouth on the table.
        if start is not None:
            marks["start"] = on_table(cup_centre(start, camera, props.lip, props.mouth_radius))
        if skill == "place_base" and end is not None:
            marks["end"] = on_table(cup_centre(end, camera, props.lip, props.mouth_radius))
        if second_cup:
            place_beside_standing_cup(marks, first, last, start, end, camera, props)
        if skill == "place_top":
            moved = cv2.dilate(start.astype(np.uint8), kernel) > 0 if start is not None else np.zeros_like(first)
            bases = largest_component(first & ~moved)
            pair = two_cup_centres(bases, camera, props.lip, props.mouth_radius) if bases is not None else None
            if pair is not None:
                a, b = (on_table(c) for c in pair)
                marks["base_a"], marks["base_b"] = a, b
                mid = (a + b) / 2
                marks["end"] = mid + [0, 0, props.top_bottom_z]  # the top cup's base on the pyramid
                marks["mouth"] = mid + [0, 0, props.top_bottom_z + props.cup_height]
                marks["axis"] = (b - a) / np.linalg.norm(b - a)

    # The cap's expected size in the image, to tell it from other blue things (a mat, a pen).
    ends = camera.project(np.array([[0.0, 0, 0], [0.01, 0, 0]]))
    px_per_m = float(np.linalg.norm(ends[1] - ends[0]) * 100)
    cap_area = np.pi * (props.cap_radius * px_per_m) ** 2
    hands = track_hands(frames, fps) if hands is None else hands
    obj_px = np.full((T, 2), np.nan)
    obj_r = np.full(T, np.nan)
    last_cap, missing = None, 0
    for i, frame in enumerate(frames):
        if skill == "toss":
            near = last_cap if missing < int(0.1 * fps) else None  # search everywhere after losing it
            blue = cv2.bitwise_and(color_mask(frame, colors.cap), table.astype(np.uint8) * 255)
            cap = detect_blob(frame, colors.cap, min_area=0.2 * cap_area, max_area=4 * cap_area, near=near,
                              max_jump=0.12 * frame.shape[1], compact=True, mask=blue)
            if cap is not None:
                obj_px[i], obj_r[i], last_cap, missing = cap.center, cap.radius, cap.center, 0
            else:
                missing += 1
        else:
            moving = ((color_mask(frame, colors.cups) > 0) & ~static & table).astype(np.uint8) * 255
            cup = detect_blob(frame, colors.cups, min_area=300, mask=moving)
            if cup is not None:
                obj_px[i], obj_r[i] = cup.center, cup.radius
    picked = {}
    if hands["hand_px"].ndim == 4:  # both hands were tracked
        hands, picked = working_hand(hands, obj_px, fps)
    hand_px = np.full((T, 21, 2), np.nan)
    hand_world = np.full((T, 21, 3), np.nan)
    hand_px[:], hand_world[:] = hands["hand_px"][:T], hands["hand_world"][:T]
    return SkillTrack(np.arange(T) / fps, hand_px, hand_world, obj_px, obj_r, marks, in_cup_px, picked)


def components(mask: np.ndarray, min_area: int = 400, n: int = 2) -> list[np.ndarray]:
    """The n largest connected components of a mask (as masks), largest first."""
    k, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    order = [1 + int(i) for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) if stats[1 + i, cv2.CC_STAT_AREA] >= min_area]
    return [labels == i for i in order[:n]]


def place_beside_standing_cup(marks: dict, first: np.ndarray, last: np.ndarray, start: np.ndarray | None,
                              end: np.ndarray | None, camera: Camera, props: Props, reach: float = 0.03) -> None:
    """Set marks["other"] to the standing cup that touches the moved cup's end spot, and refine marks["end"] with
    it: where both cups form one red blob, their centres come from splitting it, as in track_base_pair."""
    if end is None:
        raise RuntimeError("Second cup: no end spot found.")
    ends = camera.project(np.array([[0.0, 0, 0], [0.01, 0, 0]]))
    px_per_m = float(np.linalg.norm(ends[1] - ends[0]) * 100)
    gap = cv2.distanceTransform((~end).astype(np.uint8), cv2.DIST_L2, 5)
    moved = cv2.dilate(start.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0 if start is not None else 0 * first
    standing = components(first & ~moved, n=6)
    near = [(float(gap[m].min()), i) for i, m in enumerate(standing)]
    if not near or min(near)[0] > reach * px_per_m:
        raise RuntimeError("Second cup: no standing cup touches the end spot.")
    cup_a = standing[min(near)[1]]
    other = cup_centre(cup_a, camera, props.lip, props.mouth_radius)
    n, labels = cv2.connectedComponents(last.astype(np.uint8))
    blob = labels == np.bincount(labels[end & last].ravel(), minlength=n)[1:].argmax() + 1
    pair = two_cup_centres(blob, camera, props.lip, props.mouth_radius) if (blob & cup_a).any() else None
    if pair is not None:
        i = int(np.linalg.norm(pair[1][:2] - other[:2]) < np.linalg.norm(pair[0][:2] - other[:2]))
        other, marks["end"] = pair[i], np.array([pair[1 - i][0], pair[1 - i][1], 0.0])
    marks["other"] = np.array([other[0], other[1], 0.0])


def track_base_pair(frames: list[np.ndarray], fps: float, colors: SkillColors, camera: Camera,
                    props: Props = Props(), hands: dict | None = None) -> list[SkillTrack]:
    """A base clip in which both base cups are put down, one after the other: one SkillTrack per cup, cut where
    the hand goes from the first to the second.

    The two start spots are the red only in the first still frames, the end spots the red only in the last ones
    (the two cups standing side by side, one blob). The cup whose start empties first is cup A; of the two end
    spots, the one that fills first is A's."""
    T = len(frames)
    still = max(3, int(0.3 * fps))
    first, last = still_mask(frames[:still], colors.cups), still_mask(frames[-still:], colors.cups)
    kernel = np.ones((7, 7), np.uint8)
    open_ = lambda m: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0  # noqa: E731
    starts = components(open_(first & ~last))
    ends = components(open_(last & ~first), n=1)
    if len(starts) < 2 or not ends:
        return []
    pair = two_cup_centres(ends[0], camera, props.lip, props.mouth_radius)
    if pair is None:
        return []
    # Split the end blob between the two cups: each pixel goes to the nearer cup centre (in the image).
    ys, xs = np.nonzero(ends[0])
    c_px = camera.project(np.array([[p[0], p[1], props.lip] for p in pair]))
    nearest = np.argmin(np.linalg.norm(np.column_stack([xs, ys])[:, None] - c_px[None], axis=2), axis=1)
    end_parts = []
    for j in range(2):
        m = np.zeros_like(ends[0])
        m[ys[nearest == j], xs[nearest == j]] = True
        end_parts.append(m)
    red = [color_mask(f, colors.cups) > 0 for f in frames]
    occ = lambda region: np.array([(r & region).sum() / max(region.sum(), 1) for r in red])  # noqa: E731
    k = max(3, int(0.2 * fps))

    hold = int(fps)  # a cup carried across another's spot covers it only briefly

    def emptied(o):  # first frame from which the spot stays (mostly) empty for a second
        runs = [i for i in range(len(o) - hold) if (o[i:i + hold] < 0.5).all()]
        return runs[0] if runs else len(o)

    def filled(o):  # first frame from which the spot stays (mostly) full for a second
        runs = [i for i in range(len(o) - hold) if (o[i:i + hold] > 0.6).all()]
        return runs[0] if runs else len(o)

    t_pick = [emptied(occ(m)) for m in starts]
    t_put = [filled(occ(m)) for m in end_parts]
    order_s = np.argsort(t_pick)
    order_e = np.argsort(t_put)
    split = int((t_put[order_e[0]] + t_pick[order_s[1]]) / 2)
    if not (k < split < T - k):
        return []
    static = cv2.dilate((first & last).astype(np.uint8), kernel) > 0
    dil = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0  # noqa: E731
    on_table = lambda c: np.array([c[0], c[1], 0.0])  # noqa: E731
    hands = track_hands(frames, fps) if hands is None else hands
    tracks = []
    for n, (f0, f1) in enumerate([(0, split), (split, T)]):
        s_mask, e_mask = starts[order_s[n]], end_parts[order_e[n]]
        other_start, other_end = starts[order_s[1 - n]], end_parts[order_e[1 - n]]
        exclude = static | dil(other_start) if n == 0 else static | dil(other_end)
        marks = {"start": on_table(cup_centre(s_mask, camera, props.lip, props.mouth_radius)),
                 "end": on_table(pair[order_e[n]])}
        if n == 1:
            marks["other"] = on_table(pair[order_e[0]])  # cup A, already in place
        obj_px, obj_r = np.full((f1 - f0, 2), np.nan), np.full(f1 - f0, np.nan)
        for i in range(f0, f1):
            moving = (red[i] & ~exclude).astype(np.uint8) * 255
            cup = detect_blob(frames[i], colors.cups, min_area=300, mask=moving)
            if cup is not None:
                obj_px[i - f0], obj_r[i - f0] = cup.center, cup.radius
        hand_px, hand_world = hands["hand_px"][f0:f1].copy(), hands["hand_world"][f0:f1].copy()
        tracks.append(SkillTrack(np.arange(f0, f1) / fps, hand_px, hand_world, obj_px, obj_r, marks))
    return tracks


def process_skill_video(path: str | Path, out_dir: str | Path, skill: str | None = None,
                        colors: SkillColors = SkillColors(), props: Props = Props(), marker_mm: float = 50.0,
                        focal_px: float | None = None, hfov_deg: float | None = None, debug: bool = True,
                        hands: dict | None = None, max_width: int = 1280, handheld: bool = False,
                        pad: float = 0.15, second_cup: bool = False,
                        single_cup: bool = False) -> tuple[SkillDemo, dict]:
    """`hands`: hand landmarks tracked earlier for this clip at its full resolution (hand_px, hand_world), to
    skip MediaPipe. `focal_px` is for the clip's full resolution too; frames wider than `max_width` are
    scaled down (a 1080x1920 clip of 900 frames is 6 GB in memory). `handheld`: the phone moved during the
    clip; the frames are first warped onto one fixed camera (see stabilize.py), `pad` larger on each side.
    `single_cup`: a base clip that puts down one cup only; `second_cup`: that cup is B, see track_skill."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skill = skill or skill_from_name(path)
    full_width = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_WIDTH)
    frames, fps = read_video(path, max_width)
    scale = frames[0].shape[1] / full_width if full_width else 1.0
    if hands is not None:
        hands = {**hands, "hand_px": np.asarray(hands["hand_px"]) * scale}
    focal_px = focal_px * scale if focal_px else None
    stab_info = {}
    if handheld:
        if hands is not None:
            raise ValueError("Hand landmarks tracked earlier are for the unwarped frames: don't pass them with "
                             "a hand-held clip.")
        first, _ = calibrate(frames[: int(fps)], marker_mm=marker_mm, focal_px=focal_px, hfov_deg=hfov_deg)
        frames, K, stab_info = stabilize(frames, first.K, fps, marker_mm, pad)
        focal_px, hfov_deg = float(K[0, 0]), None
    camera, calib_info = calibrate(frames[: int(fps)], marker_mm=marker_mm, focal_px=focal_px, hfov_deg=hfov_deg)
    calib_info.update(stab_info)
    np.savez(out_dir / f"{path.stem}_camera.npz", **camera.as_dict())
    two_cups = skill == "place_base" and not (single_cup or second_cup)
    if hands is None:
        hands = track_hands(frames, fps) if two_cups else track_two_hands(frames, fps)
    pieces = track_base_pair(frames, fps, colors, camera, props, hands) if two_cups else []
    if not pieces:  # one cup moved (or one cup thrown, with any re-throws)
        pieces = [track_skill(frames, fps, skill, colors, camera, props, hands, second_cup)]
    demos, errors = [], []
    for n, track in enumerate(pieces):
        try:
            demo = reconstruct_skill(track, camera, skill, props)
        except RuntimeError as e:  # e.g. the hand was hardly seen while placing one of the two cups
            errors.append(e)
            continue
        demo.info.update({f"calib_{k}": float(v) for k, v in calib_info.items() if isinstance(v, (int, float))})
        demo.info.update(track.info)
        if "other" in track.marks:
            demo.info["other_x"], demo.info["other_y"] = track.marks["other"][:2].tolist()
        name = path.stem if len(pieces) == 1 else f"{path.stem}_{'ab'[n]}"
        demo.save(out_dir / f"{name}.npz")
        if debug:
            f0 = int(round(track.t[0] * fps))
            save_skill_overlay(frames[f0:f0 + len(track.t)], track, demo, camera, out_dir / f"{name}_overlay.mp4", fps)
            save_skill_plot(demo, out_dir / f"{name}_trajectory.png")
        demos.append(demo)
    if not demos:
        raise errors[0]
    return demos, calib_info


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def save_skill_overlay(frames, track: SkillTrack, demo: SkillDemo, camera: Camera, path: Path, fps: float) -> None:
    """Video with landmarks, the tracked object, the marks, table axes, and the reconstructed palm path and
    palm frame."""
    palm_px = camera.project(demo.palm)
    axes = camera.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]))
    start_px, end_px = camera.project(demo.obj_start)[0], camera.project(demo.obj_end)[0]
    marks_px = {k: camera.project(np.asarray(v))[0] for k, v in track.marks.items() if k != "axis"}
    flight_px = None
    if demo.skill == "toss":
        dt = np.linspace(0, 0.5, 40)[:, None]
        flight = demo.release_pos + demo.release_vel * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2
        flight_px = camera.project(flight[flight[:, 2] > demo.obj_end[2] - 0.02]).astype(np.int32)
    out = []
    for i, frame in enumerate(frames):
        img = frame.copy()
        for k, color in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, _pt(axes[0]), _pt(axes[k]), color, 2)
        cv2.drawMarker(img, _pt(start_px), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(img, _pt(end_px), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        for name, p in marks_px.items():
            cv2.circle(img, _pt(p), 5, (255, 255, 0), -1)
            cv2.putText(img, name, _pt(p + [6, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if track.in_cup_px is not None:
            cv2.polylines(img, [track.in_cup_px.astype(np.int32)], True, (255, 255, 0), 1)
        if flight_px is not None and demo.t[min(i, len(demo.t) - 1)] >= demo.t_release:
            cv2.polylines(img, [flight_px], False, (0, 140, 255), 2)
        if not np.isnan(track.obj_px[i, 0]):
            cv2.circle(img, _pt(track.obj_px[i]), 5, (255, 0, 255), -1)
        if not np.isnan(track.hand_px[i]).any():
            pts = track.hand_px[i]
            for a, b in HAND_EDGES:
                cv2.line(img, _pt(pts[a]), _pt(pts[b]), (200, 200, 200), 1)
        if i < len(demo.t):
            cv2.polylines(img, [palm_px[: i + 1].astype(np.int32)], False, (0, 255, 255), 2)
            frame_px = camera.project(demo.palm[i] + 0.06 * demo.palm_rot[i].T)
            for k, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # fingers, thumb side, normal
                cv2.line(img, _pt(palm_px[i]), _pt(frame_px[k]), color, 2)
            held = demo.t_grasp <= demo.t[i] < demo.t_release
            cv2.putText(img, f"{demo.skill}  t={demo.t[i]:.2f}s  closure={demo.closure[i]:.2f}  "
                             f"{'HOLD' if held else ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, fps=fps, macro_block_size=1)


def save_skill_plot(demo: SkillDemo, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    held = (demo.t >= demo.t_grasp) & (demo.t < demo.t_release)
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*demo.palm.T, color="#1f77b4", label="palm centre")
    ax.plot(*demo.palm[held].T, color="#d62728", lw=3, label="holding")
    ax.scatter(*demo.obj_start, color="gray", s=60, label="object start")
    ax.scatter(*demo.obj_end, color="gold", s=60, label="top cup mouth" if demo.skill == "toss" else "cup end")
    if demo.skill == "toss":
        dt = np.linspace(0, 0.5, 30)[:, None]
        flight = demo.release_pos + demo.release_vel * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2
        flight = flight[flight[:, 2] > demo.obj_end[2] - 0.02]
        ax.plot(*flight.T, color="orange", ls="--", label="cap flight (fit)")
        if demo.base_axis is not None:
            a = demo.obj_end + 0.15 * np.array([[-1.0], [1.0]]) * demo.base_axis
            ax.plot(*a.T, color="red", label="base cups' line")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(demo.t, demo.closure, label="hand closure (0 open, 1 closed)")
    ax2.plot(demo.t, demo.palm[:, 2] * 5, label="palm height (x5, m)")
    for tt, name in ((demo.t_grasp, "grasp"), (demo.t_release, "release")):
        ax2.axvline(tt, color="gray", ls="--")
        ax2.text(tt, ax2.get_ylim()[1] * 0.95, name, rotation=90, va="top")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
