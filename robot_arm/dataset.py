"""Generate the robot demonstration dataset from human demos (parallel), and load it for training."""

from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from robot_arm.video.reconstruct import HumanDemo

_ENV = None


def load_human_demos(folder: str | Path) -> list[HumanDemo]:
    paths = sorted(p for p in Path(folder).glob("*.npz") if not p.stem.endswith("_camera"))
    if not paths:
        raise FileNotFoundError(f"No human demos (*.npz) in {folder}")
    return [HumanDemo.load(p) for p in paths]


def _worker(job):
    global _ENV
    from robot_arm.env import PandaBallCupEnv
    from robot_arm.retarget import rollout

    if _ENV is None:
        _ENV = PandaBallCupEnv()
    demo, ball, cup, seed, noise = job
    rng = np.random.default_rng(seed)
    obs, actions, success, _ = rollout(_ENV, demo, ball, cup, rng, noise=noise, seed=seed)
    return obs.astype(np.float32), actions.astype(np.float32), success


def generate(demos: list[HumanDemo], layouts_per_demo: int = 50, noise: float = 0.3, clean_fraction: float = 0.2,
             seed: int = 0, workers: int = 4) -> dict:
    """Roll out every human demo on many random layouts; keep the successful episodes."""
    from robot_arm.retarget import canonical_layout, sample_robot_layout

    rng = np.random.default_rng(seed)
    jobs, meta = [], []
    for d_idx, demo in enumerate(demos):
        for k in range(layouts_per_demo):
            ball, cup = canonical_layout(demo) if k == 0 else sample_robot_layout(demo, rng)
            ep_noise = 0.0 if rng.random() < clean_fraction else noise
            jobs.append((demo, ball, cup, int(rng.integers(1 << 30)), ep_noise))
            meta.append((d_idx, ball, cup, ep_noise))

    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_worker, jobs, chunksize=4)

    obs, actions, episode = [], [], []
    ep_demo, ep_ball, ep_cup, ep_success, ep_noise = [], [], [], [], []
    for (o, a, ok), (d_idx, ball, cup, nz) in zip(results, meta):
        ep_demo.append(d_idx), ep_ball.append(ball), ep_cup.append(cup), ep_success.append(ok), ep_noise.append(nz)
        if ok:
            obs.append(o), actions.append(a), episode.append(np.full(len(o), len(ep_success) - 1))
    if not obs:
        raise RuntimeError("No successful robot rollouts; check the human demos (plots/overlays).")
    return {
        "obs": np.concatenate(obs), "actions": np.concatenate(actions), "episode": np.concatenate(episode),
        "ep_demo": np.array(ep_demo), "ep_ball": np.array(ep_ball), "ep_cup": np.array(ep_cup),
        "ep_success": np.array(ep_success), "ep_noise": np.array(ep_noise),
    }


@dataclass
class DemoDataset:
    obs: np.ndarray
    actions: np.ndarray
    episode: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "DemoDataset":
        z = np.load(path)
        return cls(z["obs"], z["actions"], z["episode"])

    def split(self, val_fraction: float = 0.1, seed: int = 0) -> tuple["DemoDataset", "DemoDataset"]:
        """Split by episode so validation layouts are unseen."""
        eps = np.unique(self.episode)
        rng = np.random.default_rng(seed)
        val_eps = rng.choice(eps, max(1, int(len(eps) * val_fraction)), replace=False)
        val = np.isin(self.episode, val_eps)
        return (DemoDataset(self.obs[~val], self.actions[~val], self.episode[~val]),
                DemoDataset(self.obs[val], self.actions[val], self.episode[val]))
