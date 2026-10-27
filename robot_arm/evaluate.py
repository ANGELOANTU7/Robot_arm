"""Roll out a policy on fixed-seed layouts, report the success rate and optionally save videos."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from robot_arm.env import PandaBallCupEnv


def load_policy(name: str, device: str = "cpu"):
    """'scripted' or a path to a checkpoint saved by bc.py / ppo.py."""
    if name == "scripted":
        from robot_arm.scripted import ScriptedPolicy
        return ScriptedPolicy()
    from robot_arm.policy import load_checkpoint
    return load_checkpoint(name, device=device)


def evaluate(policy, episodes: int = 100, seed: int = 10_000, video_dir: str | Path | None = None,
             videos: int = 3, camera: str = "side") -> dict:
    """Success rate over `episodes` layouts. Seeds start at 10 000 so they never overlap training layouts."""
    env = PandaBallCupEnv(render_mode="rgb_array" if video_dir else None, camera=camera)
    successes, returns, lengths = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        record = video_dir is not None and ep < videos
        frames = [env.render()] if record else []
        total, success_at = 0.0, None
        for t in range(env.cfg.max_episode_steps):
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            if record:
                frames.append(env.render())
            if info["is_success"] and success_at is None:
                success_at = t + 1
            if terminated or truncated:
                break
        successes.append(info["is_success"])
        returns.append(total)
        lengths.append(success_at or env.cfg.max_episode_steps)
        if record:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            tag = "success" if info["is_success"] else "fail"
            imageio.mimsave(Path(video_dir) / f"episode{ep:03d}_{tag}.mp4", frames,
                            fps=env.metadata["render_fps"], macro_block_size=1)
    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_steps_to_success": float(np.mean([l for l, s in zip(lengths, successes) if s])) if any(successes) else None,
        "episodes": episodes,
    }
