"""Side-by-side videos of dataset_2: each clip with everything tracked in it, next to the G1 retargeting that clip.

    G1_HAND=shadow G1_REAL=clean MUJOCO_GL=egl python scripts/g1_side_by_side.py --out try_videos/31_retarget_side_by_side

Left, the clip: both hands' MediaPipe landmarks, the arms (RTMW whole-body pose, cleaned: shoulder, elbow, wrist), the reconstructed
palm (path, frame and table-frame position), the cups and the cap with their estimated table positions, the table
markers and axes. Right, the G1 replaying that clip's demo (retarget.G1Retargeter). The sim is time-warped onto the
clip so the grasps and releases line up; its own clock is shown. Clips whose demo was rejected or never reconstructed
get the reason in place of the sim.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robot_arm.video.calibrate import Camera, detect_markers
from robot_arm.video.process import HAND_EDGES, read_video
from robot_arm.video.skills import Props, SkillDemo
from robot_arm.video.tracking import color_mask

FOLDER_SKILL = {"cup_a": "place_base", "cup_b": "place_base", "cup_c": "place_top", "throw": "toss"}
# Arm joints in MediaPipe's pose numbering: 11/12 shoulders, 13/14 elbows, 15/16 wrists (the person's left/right).
ARM_JOINTS = {"R": (12, 14, 16), "L": (11, 13, 15)}
H = 960  # output height; the clip is scaled to it, the sim is rendered at it
SIM_W = 720
TEXT = (255, 255, 255)


def put(img, text, org, scale=0.55, color=TEXT, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def pt(p):
    return int(round(p[0])), int(round(p[1]))


def on_plane(camera: Camera, uv, z: float) -> np.ndarray:
    """Table-frame point where the viewing ray through pixel uv crosses height z."""
    c, d = camera.center_table, camera.ray(uv)
    return c + d * (z - c[2]) / d[2]


def blobs(frame, color: str, area: tuple, max_n: int, max_aspect: float, min_fill: float):
    """Compact blobs of a colour: area in range, not elongated, filling most of their box."""
    n, _, stats, cent = cv2.connectedComponentsWithStats(color_mask(frame, color))
    ok = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if area[0] <= a <= area[1] and max(w, h) / max(min(w, h), 1) <= max_aspect and a / (w * h) >= min_fill:
            ok.append(i)
    ok = sorted(ok, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max_n]
    return [(stats[i, :4], cent[i]) for i in ok]


ON_TABLE = (-0.30, 0.60, -0.90, 0.25)  # x and y limits (table frame, m) where cups and the cap can be


def track_pose(frames, cache: Path) -> np.ndarray:
    """(T, 33, 3) arm joints in MediaPipe's numbering (11-16: shoulders, elbows, wrists), pixels and confidence, NaN
    elsewhere. From RTMW whole-body (rtmlib): on these overhead clips it found the working wrist within 8-14 px of the
    hand tracker's, against 20-97 px for MediaPipe pose, and its elbow didn't jump (0-16 jumps of 40 px per clip,
    against 22-90)."""
    if cache.exists():
        return np.load(cache)["pose"]
    from rtmlib import Wholebody

    model = Wholebody(mode="performance", backend="onnxruntime", device="cpu")
    out = np.full((len(frames), 33, 3), np.nan)
    coco = {5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16}  # COCO body index -> MediaPipe index
    for i, f in enumerate(frames):
        kps, scores = model(f)
        if len(kps):
            b = int(np.argmax(scores[:, :23].mean(1)))  # the person (body and feet points)
            for c, m in coco.items():
                out[i, m] = [*kps[b, c], scores[b, c]]
    np.savez_compressed(cache, pose=out)
    return out


def clean_track(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3, jump: float = 25.0, gap: int = 6,
                window: int = 7) -> np.ndarray:
    """One joint's (T, 2) pixel track: drop unsure points and points that leap away from their neighbours (median of
    the frames around), fill short gaps, then smooth without lag (the whole clip is known)."""
    from scipy.signal import savgol_filter

    xy = np.where((conf > min_conf)[:, None], xy, np.nan)
    med = np.stack([np.array([np.nanmedian(xy[max(0, i - 3):i + 4, k]) if np.isfinite(xy[max(0, i - 3):i + 4, k]).any()
                              else np.nan for i in range(len(xy))]) for k in range(2)], 1)
    xy[np.linalg.norm(xy - med, axis=1) > jump] = np.nan
    t, ok = np.arange(len(xy)), np.isfinite(xy[:, 0])
    if ok.sum() < 2:
        return xy
    filled = np.stack([np.interp(t, t[ok], xy[ok, k]) for k in range(2)], 1)
    # Leave gaps longer than `gap` frames empty.
    run, last = np.zeros(len(xy), int), -10 ** 9
    for i in range(len(xy)):
        last = i if ok[i] else last
        run[i] = i - last
    nxt, after = np.zeros(len(xy), int), 10 ** 9
    for i in range(len(xy) - 1, -1, -1):
        after = i if ok[i] else after
        nxt[i] = after - i
    filled[(run + nxt > gap) & ~ok] = np.nan
    good = np.isfinite(filled[:, 0])
    if good.sum() > window:
        filled[good] = savgol_filter(filled[good], window, 2, axis=0)
    return filled


def clean_pose(pose: np.ndarray, hands: np.ndarray | None) -> np.ndarray:
    """Cleaned arm joints; each wrist snaps onto the hand tracker's wrist when that hand is at it, so arm and hand
    stay joined."""
    out = pose.copy()
    for j in range(11, 17):
        out[:, j, :2] = clean_track(pose[:, j, :2], np.nan_to_num(pose[:, j, 2]))
        out[:, j, 2] = np.where(np.isfinite(out[:, j, 0]), 1.0, np.nan)
    if hands is not None:
        for i in range(min(len(out), len(hands))):
            for j in (15, 16):
                if not np.isfinite(out[i, j, 0]):
                    continue
                d = [np.linalg.norm(hands[i, s, 0] - out[i, j, :2]) for s in range(2)]
                s = int(np.nanargmin(d)) if np.isfinite(d).any() else None
                if s is not None and d[s] < 60:
                    out[i, j, :2] = hands[i, s, 0]
    # With the shoulder out of frame the model can still put one on the table between elbow and hand. An arm can't
    # fold its shoulder back towards its wrist (elbow angle under 60 degrees, or the shoulder nearer the wrist than
    # the elbow is), so such a shoulder isn't drawn.
    for s, e, w in ARM_JOINTS.values():
        up, fore = out[:, s, :2] - out[:, e, :2], out[:, w, :2] - out[:, e, :2]
        cos = np.sum(up * fore, 1) / (np.linalg.norm(up, axis=1) * np.linalg.norm(fore, axis=1) + 1e-9)
        bad = (cos > 0.5) | (np.linalg.norm(out[:, s, :2] - out[:, w, :2], axis=1) < 1.2 * np.linalg.norm(fore, axis=1))
        out[bad, s] = np.nan
    return out


# ---------------------------------------------------------------- the sim side

def sim_rollout(skill: str, demos: list[SkillDemo], styles: list[dict], camera: str, seed: int):
    """Run the G1 on the clip's demo(s). Returns frames, per-step (stage, grip), the outcome and the step dt."""
    from robot_arm.humanoid.envs import TOSS, G1PyramidEnv
    from robot_arm.humanoid.retarget import G1Retargeter, expert_params, throw_style, toss_params

    kw = dict(render_mode="rgb_array", camera=camera, width=SIM_W * 3 // 4, height=H * 3 // 4)  # the offscreen buffer is 720 high
    toss, place, stop = None, {}, None
    if skill == "toss":
        style = toss_params(throw_style(demos[0]))
        human = {k: round(v, 3) for k, v in throw_style(demos[0]).items()}
        for s in styles:  # the aim-calibrated version of this style, if the dataset run made one
            if all(abs(s.get("human", {}).get(k, 1e9) - v) < 2e-3 for k, v in human.items()):
                style = s
        toss = expert_params(style)
        env = G1PyramidEnv(task="toss", **kw)
    elif skill == "place_top":
        place, env = {2: demos[0]}, G1PyramidEnv(task="place", stage=2, **kw)
    elif len(demos) == 2:  # both base cups in one clip
        place, stop = {0: demos[0], 1: demos[1]}, 2
        env = G1PyramidEnv(task="full", **kw)
    else:
        place, env = {0: demos[0]}, G1PyramidEnv(task="place", stage=0, **kw)
    policy = G1Retargeter(place, toss, env.cfg)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    frames, log = [env.render()], [(env.stage, env.grip)]
    info = {"is_success": False, "fail_reason": None}
    for _ in range(env.max_steps):
        obs, _, term, trunc, info = env.step(policy(obs))
        frames.append(env.render())
        log.append((env.stage, env.grip))
        if stop is not None and env.stage >= stop:
            info = {**info, "is_success": True}
            break
        if term or trunc:
            break
    if skill == "toss":
        outcome = "cap in the top cup" if info["is_success"] else f"missed: {info['fail_reason'] or 'timeout'}"
    else:
        outcome = "placed" if info["is_success"] else f"failed: {info['fail_reason'] or 'timeout'}"
    return frames, np.array(log), outcome, env.cfg.control_dt, TOSS


def sim_events(log: np.ndarray, skill: str, toss_stage: int) -> list[int]:
    """Steps of each grasp (grip closes) and release (grip opens) in the stages the clip shows."""
    grip = log[:, 1] > 0
    if skill == "toss":
        grip &= log[:, 0] >= toss_stage
    ev = []
    for i in range(1, len(grip)):
        if grip[i] != grip[i - 1]:
            ev.append(i)
    return ev


def time_warp(human_keys, sim_keys, t_video: np.ndarray) -> np.ndarray:
    """Sim step for each video time: piecewise linear through matching key times (grasp, release, ...)."""
    if len(human_keys) == len(sim_keys) and np.all(np.diff(human_keys) > 0) and np.all(np.diff(sim_keys) > 0):
        return np.interp(t_video, human_keys, sim_keys)
    return np.interp(t_video, [human_keys[0], human_keys[-1]], [sim_keys[0], sim_keys[-1]])


# ---------------------------------------------------------------- the clip side

def working_arm(pose, demos, camera, t_video) -> str:
    """'R' or 'L' (MediaPipe's labels): the arm whose wrist stays closest to the reconstructed palm."""
    dist = {"R": [], "L": []}
    for d in demos:
        for t, p in zip(d.t, camera.project(d.palm)):
            i = int(np.argmin(np.abs(t_video - t)))
            for side, (_, _, w) in ARM_JOINTS.items():
                if not np.isnan(pose[i, w]).any():
                    dist[side].append(np.linalg.norm(pose[i, w, :2] - p))
    return min(dist, key=lambda k: np.median(dist[k]) if dist[k] else np.inf)


def draw_clip(frame, i, t, hands, pose, demos, camera, skill, props: Props, arm: str = "R"):
    img = frame.copy()
    # Table frame and markers: what the camera calibration is based on.
    for c in detect_markers(frame).values():
        cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
    ax = camera.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]))
    for k, col in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(img, pt(ax[0]), pt(ax[k]), col, 2)
    put(img, "table x/y/z", pt(ax[0] + [6, 18]), 0.45)
    # Cups and cap, with their table positions (cup: top of a mouth-down cup, cap: its centre).
    for (x, y, w, h), c in blobs(frame, "red", (1500, 60000), 4, 2.2, 0.35):
        p = on_plane(camera, c, props.cup_height * 0.5)
        if not (ON_TABLE[0] < p[0] < ON_TABLE[1] and ON_TABLE[2] < p[1] < ON_TABLE[3]):
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 255), 2)
        put(img, f"cup ({100 * p[0]:.0f}, {100 * p[1]:.0f}) cm", (x, max(14, y - 6)), 0.45, (120, 160, 255))
    for (x, y, w, h), c in blobs(frame, "blue", (80, 4000), 3, 1.8, 0.5):
        p = on_plane(camera, c, props.cap_height * 0.5)
        if not (ON_TABLE[0] < p[0] < ON_TABLE[1] and ON_TABLE[2] < p[1] < ON_TABLE[3]):
            continue
        cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 200, 0), 2)
        put(img, f"cap ({100 * p[0]:.0f}, {100 * p[1]:.0f}) cm", (x, y + h + 18), 0.45, (255, 220, 120))
    # Arms (pose) and hands (21 landmarks each).
    seen, xy = np.nan_to_num(pose[i, :, 2]) > 0.5, pose[i, :, :2]
    for side, (s, e, w) in ARM_JOINTS.items():
        col = (0, 165, 255) if side == arm else (150, 150, 150)
        for a, b in ((s, e), (e, w)):
            if seen[a] and seen[b]:
                cv2.line(img, pt(xy[a]), pt(xy[b]), col, 4)
        for j, name in ((s, "shoulder"), (e, "elbow"), (w, "wrist")):
            if seen[j]:
                cv2.circle(img, pt(xy[j]), 7, col, -1)
                if side == arm:
                    put(img, name, pt(xy[j] + [9, -9]), 0.5, (0, 200, 255))
    if hands is not None and i < len(hands):
        for slot in range(2):
            h = hands[i, slot]
            if np.isnan(h).any():
                continue
            for a, b in HAND_EDGES:
                cv2.line(img, pt(h[a]), pt(h[b]), (0, 255, 0), 2)
            for j, p in enumerate(h):
                cv2.circle(img, pt(p), 4 if j in (4, 8, 12, 16, 20) else 3, (0, 0, 255) if j in (4, 8) else (255, 255, 255), -1)
    # The reconstructed palm: path so far, its frame, and where it is on the table.
    state = ""
    for d in demos:
        if not d.t[0] <= t <= d.t[-1] + 1e-6:
            continue
        k = int(np.clip(np.searchsorted(d.t, t), 0, len(d.t) - 1))
        path = camera.project(d.palm[: k + 1]).astype(np.int32)
        cv2.polylines(img, [path], False, (0, 255, 255), 2)
        c = camera.project(d.palm[k])[0]
        f = camera.project(d.palm[k] + 0.06 * d.palm_rot[k].T)
        for n, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, pt(c), pt(f[n]), col, 3)
        cv2.circle(img, pt(c), 7, (0, 255, 255), -1)
        p = d.palm[k]
        put(img, f"palm ({100 * p[0]:.0f}, {100 * p[1]:.0f}, {100 * p[2]:.0f}) cm", pt(c + [12, 20]), 0.5, (0, 255, 255))
        state = "HOLDING" if d.t_grasp <= t < d.t_release else ("RELEASED" if t >= d.t_release else "REACHING")
        if skill == "toss" and t >= d.t_release:
            dt = np.linspace(0, 0.5, 40)[:, None]
            fl = d.release_pos + d.release_vel * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2
            fl = fl[fl[:, 2] > d.obj_end[2] - 0.02]
            if len(fl) > 1:
                cv2.polylines(img, [camera.project(fl).astype(np.int32)], False, (0, 140, 255), 2)
        closure = float(d.closure[k])
        x0, y0 = 10, img.shape[0] - 30
        cv2.rectangle(img, (x0, y0), (x0 + 200, y0 + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (x0, y0), (x0 + int(200 * closure), y0 + 14), (0, 255, 255), -1)
        put(img, f"hand closure {closure:.2f}", (x0 + 210, y0 + 12), 0.5)
        for name, pos in (("start", d.obj_start), ("end", d.obj_end)):
            q = camera.project(pos)[0]
            cv2.drawMarker(img, pt(q), (255, 255, 255) if name == "start" else (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            put(img, f"object {name}", pt(q + [10, -8]), 0.45)
    return img, state


# ---------------------------------------------------------------- one clip

def make_clip(job) -> str:
    video, out_dir, landmarks, demo_dir, styles_path, camera_name, seed = job
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    video, out_dir = Path(video), Path(out_dir)
    folder, name = video.parent.name, video.stem
    target = out_dir / f"{folder}_{name}.mp4"
    if target.exists():
        return f"{name}: exists"
    skill = FOLDER_SKILL[folder]
    frames, fps = read_video(video, 720)
    cam_file = Path(demo_dir) / f"{name}_camera.npz"
    camera = None
    if cam_file.exists():
        c = np.load(cam_file)
        camera = Camera(c["K"], c["R"], c["t"], int(c["size"][0]), int(c["size"][1]))
    full_w = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_WIDTH) or frames[0].shape[1]
    lm = Path(landmarks) / f"{folder}_{name}.npz"
    hands = np.load(lm)["hand_px"] * (frames[0].shape[1] / full_w) if lm.exists() else None
    pose = clean_pose(track_pose(frames, Path(landmarks) / f"{folder}_{name}_rtmw.npz"), hands)

    # This clip's demo(s), and whether they were used.
    from robot_arm.humanoid.dataset import demo_problem
    # Demos the old fixed-pelvis G1 could not reproduce were moved to unreproducible/; they are replayed all the same.
    paths, note = [], ""
    for folder_ in (Path(demo_dir), Path(demo_dir) / "unreproducible"):
        paths = sorted(p for p in folder_.glob(f"{name}*.npz") if p.stem in (name, f"{name}_a", f"{name}_b"))
        if paths:
            note = "  (left out of training: unreproducible)" if folder_.name == "unreproducible" else ""
            break
    demos, problem = [], None
    for p in paths:
        d = SkillDemo.load(p)
        demos.append(d)
        problem = problem or demo_problem(d)
    if not paths:
        summary = Path(demo_dir) / "summary.json"
        err = json.loads(summary.read_text()).get(video.name, {}).get("error") if summary.exists() else None
        problem = f"not reconstructed: {err}" if err else "not reconstructed"

    t_video = np.arange(len(frames)) / fps
    sim, idx, outcome, sim_dt = None, None, "", 0.05
    if demos and not problem:
        styles = json.loads(Path(styles_path).read_text()) if styles_path and Path(styles_path).exists() else []
        sim, log, outcome, sim_dt, toss_stage = sim_rollout(skill, demos, styles, camera_name, seed)
        ev = sim_events(log, skill, toss_stage)
        hk = [0.0] + [x for d in demos for x in (d.t_grasp, d.t_release)] + [t_video[-1]]
        sk = [0] + ev[: len(hk) - 2] + [len(sim) - 1]
        idx = np.clip(np.round(time_warp(hk, sk, t_video)).astype(int), 0, len(sim) - 1)

    props = Props()
    arm = working_arm(pose, demos, camera, t_video) if demos and camera is not None else "R"
    out = []
    for i, f in enumerate(frames):
        if camera is not None:
            left, state = draw_clip(f, i, t_video[i], hands, pose, demos, camera, skill, props, arm)
        else:
            left, state = f.copy(), ""
        put(left, f"{folder}/{name}  {skill}  t={t_video[i]:.2f}s  {state}", (10, 28), 0.7, (0, 255, 255), 2)
        left = cv2.resize(left, (int(round(left.shape[1] * H / left.shape[0])), H), interpolation=cv2.INTER_AREA)
        if sim is not None:
            right = cv2.resize(cv2.cvtColor(sim[idx[i]], cv2.COLOR_RGB2BGR), (SIM_W, H), interpolation=cv2.INTER_LINEAR)
            put(right, "G1 retargeting this clip (realistic robot)" + note, (10, 28), 0.65 if not note else 0.5, (0, 255, 255), 2)
            put(right, f"sim t={idx[i] * sim_dt:.2f}s (synced to the clip at grasp & release)", (10, 54), 0.5)
            if i >= len(frames) - int(fps * 1.5):
                put(right, f"result: {outcome}", (10, 82), 0.65, (0, 255, 0) if "placed" in outcome or "in the" in outcome else (0, 0, 255), 2)
        else:
            right = np.full((H, SIM_W, 3), 30, np.uint8)
            put(right, "not retargeted", (20, H // 2 - 20), 0.9, (0, 200, 255), 2)
            import textwrap
            for n, line in enumerate(textwrap.wrap(problem or "", 50)):
                put(right, line, (20, H // 2 + 20 + 26 * n), 0.6)
        out.append(cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_BGR2RGB))
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(target, out, fps=fps, macro_block_size=1, quality=7)
    return f"{name}: {skill}, {'sim ' + outcome if sim is not None else 'not retargeted: ' + str(problem)}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", type=Path)
    p.add_argument("--dataset", type=Path, default=Path("../dataset_2/videos"))
    p.add_argument("--demos", type=Path, default=Path("data/processed/dataset2"))
    p.add_argument("--landmarks", type=Path, default=Path("data/landmarks"))
    p.add_argument("--styles", type=Path, default=Path("data/processed/ds2_toss4k/g1_styles.json"))
    p.add_argument("--out", type=Path, default=Path("try_videos/31_retarget_side_by_side"))
    p.add_argument("--camera", default="front", choices=["front", "side", "over"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    videos = args.videos or sorted(args.dataset.glob("*/*.MOV"))
    jobs = [(str(v), str(args.out), str(args.landmarks), str(args.demos), str(args.styles), args.camera, args.seed)
            for v in videos]
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(make_clip, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
