"""Behavior cloning: fit the actor's mean to the retargeted demonstration actions."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from robot_arm.dataset import DemoDataset
from robot_arm.policy import ActorCritic, bc_loss
from robot_arm.tasks import get_task


def train_bc(data_path: str | Path, out_dir: str | Path, epochs: int = 60, batch_size: int = 1024,
             lr: float = 1e-3, hidden: int = 256, layers: int = 3, eval_every: int = 10, eval_episodes: int = 30,
             device: str | None = None, seed: int = 0, task: str = "panda", eval_workers: int = 4) -> dict:
    """Train on a demo dataset of `task` (see robot_arm.tasks); keep the checkpoint that succeeds most often."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = DemoDataset.load(data_path).split(0.1, seed)
    spec = get_task(task)
    model = ActorCritic.for_task(task, hidden, layers).to(device)
    model.set_obs_stats(train.obs.mean(0), train.obs.std(0))
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731
    obs_t, act_t = to(train.obs), to(train.actions)
    obs_v, act_v = to(val.obs), to(val.actions)

    opt = torch.optim.AdamW(model.actor.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(obs_t) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)
    history, best = [], {"success_rate": -1.0}
    print(f"BC on {len(obs_t)} transitions ({len(np.unique(train.episode))} episodes), device={device}")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(obs_t), device=device)
        total = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            loss = bc_loss(model, obs_t[idx], act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        with torch.no_grad():
            val_loss = bc_loss(model, obs_v, act_v).item()
        row = {"epoch": epoch, "train_loss": total / steps_per_epoch, "val_loss": val_loss}
        if epoch % eval_every == 0 or epoch == epochs:
            _set_std_from_residuals(model, obs_t, act_t)
            model.save(out_dir / "candidate.pt", source="bc", epoch=epoch)
            stats = spec.evaluate(out_dir / "candidate.pt", episodes=eval_episodes, seed=50_000,
                                  workers=eval_workers)
            row["success_rate"] = stats["success_rate"]
            if stats["success_rate"] >= best["success_rate"]:
                best = {**row}
                shutil.copy(out_dir / "candidate.pt", out_dir / "policy.pt")
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))

    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {"best": best, "history": history}


@torch.no_grad()
def _set_std_from_residuals(model: ActorCritic, obs: torch.Tensor, act: torch.Tensor) -> None:
    """Initial exploration std for PPO: how far the demos scatter around the BC mean (clipped)."""
    resid = (model.mean_action(obs) - act).std(0)
    model.log_std.copy_(resid.clamp(0.05, 0.5).log())
