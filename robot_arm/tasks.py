"""The learning tasks: which environment a policy acts in, its dimensions, and how it is evaluated.

- "panda": the Panda ball-in-cup task.
- "g1-place": the G1 places one cup of the pyramid (stage 0, 1 or 2; the observation says which).
- "g1-toss": the G1 picks the cap off the table and throws it into the top cup of a built pyramid (again after a
  miss), then shows a thumbs-up.

Evaluation takes a checkpoint path so it can run in worker processes.
"""

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym


@dataclass(frozen=True)
class TaskSpec:
    name: str
    obs_dim: int
    act_dim: int
    grip_obs: int  # index of the previous grip command in the observation (for the BC switch weighting)
    make_env: Callable[[], gym.Env]
    evaluate: Callable[..., dict]  # (checkpoint, episodes, seed, workers) -> {"success_rate": ...}


def _panda_env():
    from robot_arm.env import PandaBallCupEnv
    return PandaBallCupEnv()


def _panda_eval(checkpoint, episodes: int = 50, seed: int = 50_000, workers: int = 1) -> dict:
    from robot_arm.evaluate import evaluate
    from robot_arm.policy import load_checkpoint
    return evaluate(load_checkpoint(checkpoint), episodes=episodes, seed=seed)


def _g1_env(task: str):
    def make():
        from robot_arm.humanoid.envs import G1PyramidEnv
        return G1PyramidEnv(task=task)
    return make


def _g1_eval(task: str):
    def evaluate(checkpoint, episodes: int = 50, seed: int = 50_000, workers: int = 4) -> dict:
        from robot_arm.humanoid.evaluate import evaluate_g1
        return evaluate_g1(task, f"{task}={checkpoint}", episodes, seed=seed, workers=workers)
    return evaluate


def get_task(name: str) -> TaskSpec:
    if name == "panda":
        from robot_arm.env import ACT_DIM, OBS_DIM, OBS_GRIP
        return TaskSpec(name, OBS_DIM, ACT_DIM, OBS_GRIP, _panda_env, _panda_eval)
    if name in ("g1-place", "g1-toss"):
        from robot_arm.humanoid.envs import ACT_DIM, OBS_DIM, OBS_GRIP
        task = name.split("-")[1]
        return TaskSpec(name, OBS_DIM, ACT_DIM, OBS_GRIP, _g1_env(task), _g1_eval(task))
    raise ValueError(f"Unknown task {name!r}: use panda, g1-place or g1-toss")


TASKS = ("panda", "g1-place", "g1-toss")
