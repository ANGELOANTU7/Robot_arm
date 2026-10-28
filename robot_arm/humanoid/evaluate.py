"""Evaluate G1 pyramid policies: success rates per task and optional rollout videos."""

from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from robot_arm.humanoid.envs import OBS_STAGE, TOSS, G1PyramidEnv


class StagePolicy:
    """Routes each observation to the place policy (stages 0-2) or the toss policy (stages 3-4: throw, thumbs-up)."""

    def __init__(self, place, toss):
        self.place, self.toss = place, toss

    def reset(self):
        for p in (self.place, self.toss):
            if hasattr(p, "reset"):
                p.reset()

    def __call__(self, obs):
        return self.toss(obs) if int(np.argmax(obs[OBS_STAGE])) >= TOSS else self.place(obs)


def make_policy(spec: str, device: str = "cpu"):
    """'scripted', 'replay:<folder of human skill demos>', or 'place=<ckpt>,toss=<ckpt>' (either may be
    'scripted')."""
    from robot_arm.humanoid.scripted import PlaceExpert, TossExpert

    if spec.startswith("replay:"):
        from robot_arm.humanoid.dataset import load_replay_policy
        return load_replay_policy(spec.split(":", 1)[1])
    parts = {"place": "scripted", "toss": "scripted"}
    if spec != "scripted":
        for item in spec.split(","):
            key, value = item.split("=", 1)
            parts[key.strip()] = value.strip()

    def load(name, scripted_cls):
        if name == "scripted":
            return scripted_cls()
        from robot_arm.policy import load_checkpoint
        return load_checkpoint(name, device=device)

    return StagePolicy(load(parts["place"], PlaceExpert), load(parts["toss"], TossExpert))


def _episode(args):
    task, stage, spec, seed, video, camera = args
    env = G1PyramidEnv(task=task, stage=stage, render_mode="rgb_array" if video else None, camera=camera)
    policy = make_policy(spec)
    if hasattr(policy, "start_at"):  # replay: cycle through the human demos across episodes
        policy.start_at(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    frames = [env.render()] if video else []
    for t in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if video:
            frames.append(env.render())
        if terminated or truncated:
            break
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, frames, fps=env.metadata["render_fps"], macro_block_size=1)
    reason = info["fail_reason"] or ("ok" if info["is_success"] else "timeout")
    return bool(info["is_success"]), int(info["stage"]), reason, int(info["misses"])


def evaluate_g1(task: str = "full", spec: str = "scripted", episodes: int = 50, stage: int | None = None,
                seed: int = 20_000, workers: int = 4, video_dir: str | None = None, videos: int = 2,
                camera: str = "front") -> dict:
    jobs = []
    for i in range(episodes):
        video = str(Path(video_dir) / f"{task}_{i:03d}.mp4") if video_dir and i < videos else None
        jobs.append((task, stage, spec, seed + i, video, camera))
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_episode, jobs)
    success = [r[0] for r in results]
    return {
        "task": task, "policy": spec, "episodes": episodes,
        "success_rate": float(np.mean(success)),
        "reached_stage": dict(sorted(Counter(r[1] for r in results).items())),
        "outcomes": dict(Counter(r[2] for r in results)),
        "misses_per_episode": float(np.mean([r[3] for r in results])),
    }
