"""Vision-language-action data for the G1 pyramid: what a VLA sees (the robot's head and wrist cameras, its own
joint readings, a spoken instruction per stage) next to the retargeted experts' actions.

With the "robot" state the VLA gets no object positions: the cups and the cap are only in the images. The
"perceived" state adds the robot's estimate of where the object to grab and its target are. The full observation is stored too,
for an RL critic that may use what the robot can't measure.
"""

import json
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from robot_arm.humanoid.envs import OBS_GRIP, OBS_HAND_ROT, OBS_OBJ_TO_TARGET, OBS_SITE, OBS_TO_OBJECT

CAMERAS = ("head", "wrist")
IMG = 224
INSTRUCTIONS = (
    "Place the first red cup mouth-down on the table.",
    "Place the second red cup mouth-down next to the first one.",
    "Turn the third cup over and stack it on the two cups.",
    "Throw the blue cap into the top cup.",
    "Give a thumbs-up.",
)
# "robot": joint angles of the waist and arm (10), hand joints (7), grasp point (3), hand orientation (6), grip
# command (1). "perceived" adds what the robot's perception estimates (with the camera's errors on the realistic
# robot): hand to the object to grab (3) and that object to its target (3).
STATE_SETS = {"robot": np.r_[0:10, 20:27, OBS_SITE, OBS_HAND_ROT, OBS_GRIP]}
STATE_SETS["perceived"] = np.r_[STATE_SETS["robot"], OBS_TO_OBJECT, OBS_OBJ_TO_TARGET]


def vla_state(obs: np.ndarray, kind: str = "robot") -> np.ndarray:
    return np.asarray(obs)[..., STATE_SETS[kind]].astype(np.float32)


def encode(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def decode(buf: bytes | np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


_ENVS = {}


def _vla_worker(job):
    """One retargeted-expert episode with camera images; returns the stages it completed, each as (stage, obs,
    actions, jpeg frames per camera)."""
    from robot_arm.humanoid.envs import G1PyramidEnv
    from robot_arm.humanoid.retarget import G1Retargeter, expert_params, rollout

    task, place, style, seed, noise, stop_stage = job
    if task not in _ENVS:
        _ENVS[task] = G1PyramidEnv(task=task, render_mode="rgb_array", width=IMG, height=IMG)
    env = _ENVS[task]
    rng = np.random.default_rng(seed)
    policy = G1Retargeter(place, expert_params(style))
    options = {"stage": int(rng.integers(3))} if task == "place" else None
    obs, actions, stages, done, info, frames = rollout(env, policy, rng, noise=noise, seed=seed, options=options,
                                                       stop_stage=stop_stage, render=CAMERAS)
    out = []
    for stage in done:
        idx = np.flatnonzero(stages == stage)
        jpgs = [[encode(frames[i][c]) for i in idx] for c in range(len(CAMERAS))]
        out.append((int(stage), obs[idx].astype(np.float32), actions[idx].astype(np.float32), jpgs))
    return out, sorted(done), info.get("fail_reason")


def write_shard(path: Path, segments: list) -> int:
    """Segments (episode id, stage, obs, actions, jpegs) as one npz: images are concatenated jpeg bytes plus offsets."""
    obs, act, stage, ep, step, blobs, offsets = [], [], [], [], [], [[] for _ in CAMERAS], [[] for _ in CAMERAS]
    pos = [0] * len(CAMERAS)
    for e, s, o, a, jpgs in segments:
        obs.append(o), act.append(a), stage.append(np.full(len(o), s, np.int8)), ep.append(np.full(len(o), e))
        step.append(np.arange(len(o)))
        for c in range(len(CAMERAS)):
            for j in jpgs[c]:
                offsets[c].append((pos[c], len(j)))
                blobs[c].append(j)
                pos[c] += len(j)
    np.savez(path, obs=np.concatenate(obs), actions=np.concatenate(act), stage=np.concatenate(stage),
             episode=np.concatenate(ep), step=np.concatenate(step),
             **{f"jpg_{c}": np.frombuffer(b"".join(blobs[k]), np.uint8) for k, c in enumerate(CAMERAS)},
             **{f"off_{c}": np.array(offsets[k], np.int64) for k, c in enumerate(CAMERAS)})
    return int(sum(len(o) for o in obs))


def generate_vla(jobs: list[tuple], out_dir: str | Path, workers: int = 8, shard_episodes: int = 200,
                 first_episode: int = 0) -> dict:
    """Run the jobs (see dataset.g1_jobs) and write the completed stages in shards of about `shard_episodes`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments, n_shard, steps, by_stage, fails = [], len(list(out_dir.glob("shard_*.npz"))), 0, {}, {}
    with get_context("spawn").Pool(workers) as pool:
        for e, (segs, done, reason) in enumerate(pool.imap(_vla_worker, jobs, chunksize=1)):
            for s, o, a, jpgs in segs:
                segments.append((first_episode + e, s, o, a, jpgs))
                by_stage[s] = by_stage.get(s, 0) + 1
            if reason:
                fails[reason] = fails.get(reason, 0) + 1
            if (e + 1) % shard_episodes == 0 or e + 1 == len(jobs):
                if segments:
                    steps += write_shard(out_dir / f"shard_{n_shard:04d}.npz", segments)
                    n_shard += 1
                segments = []
                print(f"{e + 1}/{len(jobs)} episodes, {steps} steps kept, stages done {dict(sorted(by_stage.items()))},"
                      f" fails {fails}", flush=True)
    return {"episodes": len(jobs), "steps": steps, "stages_done": by_stage, "fails": fails}


def load_shards(folder: str | Path) -> dict:
    """All shards of a folder, concatenated; the jpeg offsets point into one buffer per camera."""
    parts = [np.load(p) for p in sorted(Path(folder).glob("shard_*.npz"))]
    if not parts:
        raise FileNotFoundError(f"No VLA shards in {folder}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("obs", "actions", "stage", "episode", "step")}
    for c in CAMERAS:
        base, offs = 0, []
        for p in parts:
            o = p[f"off_{c}"].copy()
            o[:, 0] += base
            offs.append(o)
            base += len(p[f"jpg_{c}"])
        out[f"jpg_{c}"] = np.concatenate([p[f"jpg_{c}"] for p in parts])
        out[f"off_{c}"] = np.concatenate(offs)
    return out


def save_summary(out_dir: str | Path, summary: dict) -> None:
    Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=1, default=str))
