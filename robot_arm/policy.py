"""Gaussian MLP actor-critic shared by behavior cloning and PPO."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robot_arm.env import ACT_DIM, OBS_DIM, OBS_GRIP


def mlp(sizes: list[int], out_gain: float = 1.0) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, out_gain if last else np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor: obs -> Gaussian over actions (state-independent std). Critic: obs -> value.

    Observations are normalised with fixed statistics taken from the demonstrations.
    """

    def __init__(self, hidden: int = 256, layers: int = 3, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 grip_obs: int = OBS_GRIP, task: str = "panda"):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.obs_dim, self.act_dim, self.grip_obs, self.task = obs_dim, act_dim, grip_obs, task
        self.fast_weight = 10.0 if task == "g1-toss" else 1.0  # see bc_loss
        self.actor = mlp([obs_dim] + [hidden] * layers + [act_dim], out_gain=0.01)
        self.critic = mlp([obs_dim] + [hidden] * layers + [1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_obs_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(np.maximum(std, 1e-3), dtype=torch.float32))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.obs_mean) / self.obs_std).clamp(-10, 10)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(self.normalize(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.normalize(obs)).squeeze(-1)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalize(obs))

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"state_dict": self.state_dict(), "hidden": self.hidden, "layers": self.layers,
                    "obs_dim": self.obs_dim, "act_dim": self.act_dim, "grip_obs": self.grip_obs, "task": self.task,
                    **extra}, path)

    @classmethod
    def for_task(cls, task: str, hidden: int = 256, layers: int = 3) -> "ActorCritic":
        from robot_arm.tasks import get_task

        spec = get_task(task)
        return cls(hidden, layers, spec.obs_dim, spec.act_dim, spec.grip_obs, task)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ActorCritic":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["hidden"], ckpt["layers"], ckpt.get("obs_dim", OBS_DIM), ckpt.get("act_dim", ACT_DIM),
                    ckpt.get("grip_obs", OBS_GRIP), ckpt.get("task", "panda"))
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)


def bc_loss(model: "ActorCritic", obs: torch.Tensor, act: torch.Tensor, switch_weight: float = 20.0) -> torch.Tensor:
    """MSE to the demo actions, with rare but decisive steps upweighted.

    - Gripper switches: each demo has only two (close on the object, open at the target) among ~150 steps.
      Without the extra weight the regressed gripper output drifts towards "keep doing what you did" right
      where it must switch.
    - Fast steps (tasks with `fast_weight` > 1, i.e. the toss): the throw itself is a handful of steps after a
      long, slow approach, and its precision decides everything.
    """
    err = (model.mean_action(obs) - act) ** 2
    switch = (obs[:, model.grip_obs] > 0) != (act[:, -1] > 0)
    w = torch.ones_like(err)
    w[:, -1] = 1.0 + (switch_weight - 1.0) * switch.float()
    if model.fast_weight > 1:
        fast = act[:, :3].norm(dim=-1) > 0.3
        w = w * (1.0 + (model.fast_weight - 1.0) * fast.float())[:, None]
    return (err * w).sum() / w.sum()


class TorchPolicy:
    """numpy obs -> numpy action, deterministic (the Gaussian's mean), for evaluation and viewing."""

    def __init__(self, model: ActorCritic, device: str = "cpu"):
        self.model, self.device = model.eval(), device

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.model.mean_action(x).clamp(-1, 1).cpu().numpy()


def load_checkpoint(path: str | Path, device: str = "cpu") -> TorchPolicy:
    return TorchPolicy(ActorCritic.load(path, device), device)
