"""Chunked policies: predict the next `horizon` actions from the last `obs_horizon` observations.

Two heads share the data windows, the wrapper and the checkpoint format:
- "diffusion": a conditional 1D U-Net that denoises an action chunk (Diffusion Policy, state-based). Trained as
  DDPM noise prediction, sampled with DDIM, evaluated with an exponential moving average of the weights;
- "chunk_mlp": the same chunk regressed directly with an L1 loss: the baseline the diffusion head has to beat.

Why chunks. The retargeted expert waits on timers (close for 8 steps, open for `OPEN_STEPS`, align for 20), so
nearly the same state is labelled "wait" at one step and "go" at the next, and a single-step regressor averages
them and stalls. A chunk commits to a whole stretch of motion, and the diffusion head draws one coherent way out of
the several ways the demos reach the same situation, where a mean-squared-error fit blends them into a path nobody
took.

Windows never cross a stage change: the data is split into (episode, stage) segments, the observation history is
padded with the segment's first observation and the action chunk is masked past the segment's last step. The
wrapper does the same at run time (it forgets its history and queued actions when the stage changes).
"""

import copy
import json
import math
import shutil
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robot_arm.policy import mlp
from robot_arm.tasks import get_task

STAGE_OBS = (60, 65)  # the G1 observation's one-hot stage (humanoid.envs.OBS_STAGE)


# ------------------------------------------------------------------------------------------------ networks

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        x = t.float()[:, None] * freqs[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int, groups: int = 8):
        super().__init__(nn.Conv1d(cin, cout, kernel, padding=kernel // 2), nn.GroupNorm(groups, cout), nn.Mish())


class ResBlock(nn.Module):
    """Two conv blocks with the conditioning (step embedding + observations) applied as a scale and a bias."""

    def __init__(self, cin: int, cout: int, cond_dim: int, kernel: int):
        super().__init__()
        self.b1, self.b2 = ConvBlock(cin, cout, kernel), ConvBlock(cout, cout, kernel)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        return self.b2(scale * self.b1(x) + bias) + self.skip(x)


class UNet1D(nn.Module):
    """Noise prediction for an action chunk (B, horizon, act_dim) given the step and the observations."""

    def __init__(self, act_dim: int, obs_cond_dim: int, down_dims=(64, 128, 256), kernel: int = 5, step_dim: int = 64):
        super().__init__()
        dims = [act_dim, *down_dims]
        pairs = list(zip(dims[:-1], dims[1:]))
        cond = step_dim + obs_cond_dim
        self.step_encoder = nn.Sequential(SinusoidalEmbedding(step_dim), nn.Linear(step_dim, 4 * step_dim), nn.Mish(),
                                          nn.Linear(4 * step_dim, step_dim))
        self.down = nn.ModuleList([nn.ModuleList([
            ResBlock(cin, cout, cond, kernel), ResBlock(cout, cout, cond, kernel),
            nn.Conv1d(cout, cout, 3, stride=2, padding=1) if i < len(pairs) - 1 else nn.Identity()])
            for i, (cin, cout) in enumerate(pairs)])
        self.mid = nn.ModuleList([ResBlock(dims[-1], dims[-1], cond, kernel) for _ in range(2)])
        self.up = nn.ModuleList([nn.ModuleList([
            ResBlock(cout * 2, cin, cond, kernel), ResBlock(cin, cin, cond, kernel),
            nn.ConvTranspose1d(cin, cin, 4, stride=2, padding=1)])
            for cin, cout in reversed(pairs[1:])])
        self.out = nn.Sequential(ConvBlock(dims[1], dims[1], kernel), nn.Conv1d(dims[1], act_dim, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.step_encoder(t), obs_cond], dim=-1)
        x = x.transpose(1, 2)
        skips = []
        for res1, res2, down in self.down:
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for res in self.mid:
            x = res(x, cond)
        for res1, res2, up in self.up:
            x = res2(res1(torch.cat([x, skips.pop()], dim=1), cond), cond)
            x = up(x)
        return self.out(x).transpose(1, 2)


class ChunkModel(nn.Module):
    """What both heads share: the observation normalisation and the chunk shape."""

    def __init__(self, obs_dim: int, act_dim: int, obs_horizon: int, horizon: int):
        super().__init__()
        self.obs_dim, self.act_dim, self.obs_horizon, self.horizon = obs_dim, act_dim, obs_horizon, horizon
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_obs_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(np.maximum(std, 1e-3), dtype=torch.float32))

    def condition(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, obs_horizon, obs_dim) -> normalised and flattened (B, obs_horizon * obs_dim)."""
        return ((obs - self.obs_mean) / self.obs_std).clamp(-10, 10).flatten(1)


def _randn(shape, generator: torch.Generator | None, device) -> torch.Tensor:
    return torch.randn(shape, generator=generator).to(device)  # on the CPU: the same draw on every device


class DiffusionHead(ChunkModel):
    def __init__(self, obs_dim: int, act_dim: int, obs_horizon: int = 2, horizon: int = 16, down_dims=(64, 128, 256),
                 kernel: int = 5, train_steps: int = 100):
        super().__init__(obs_dim, act_dim, obs_horizon, horizon)
        self.net = UNet1D(act_dim, obs_horizon * obs_dim, down_dims, kernel)
        self.train_steps = train_steps
        s = 0.008  # the squared-cosine noise schedule
        f = torch.cos((torch.arange(train_steps + 1) / train_steps + s) / (1 + s) * math.pi / 2) ** 2
        self.register_buffer("alpha_bar", (f[1:] / f[0]).clamp(1e-5, 0.9999))

    def loss(self, obs: torch.Tensor, act: torch.Tensor, mask: torch.Tensor,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """Noise-prediction error over the valid steps of the chunk."""
        t = torch.randint(0, self.train_steps, (len(act),), generator=generator).to(act.device)
        eps = _randn(act.shape, generator, act.device)
        ab = self.alpha_bar[t][:, None, None]
        noisy = ab.sqrt() * act + (1 - ab).sqrt() * eps
        err = ((self.net(noisy, t, self.condition(obs)) - eps) ** 2).mean(-1)
        return (err * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, steps: int = 10, generator: torch.Generator | None = None) -> torch.Tensor:
        """A chunk (B, horizon, act_dim) drawn with deterministic DDIM from `steps` of the training steps."""
        cond = self.condition(obs)
        x = _randn((len(obs), self.horizon, self.act_dim), generator, obs.device)
        ts = np.linspace(self.train_steps - 1, 0, steps).round().astype(int)
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_next = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.ones_like(ab)
            eps = self.net(x, torch.full((len(x),), int(t), device=x.device), cond)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)  # actions live in [-1, 1]
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x


class ChunkMLPHead(ChunkModel):
    def __init__(self, obs_dim: int, act_dim: int, obs_horizon: int = 2, horizon: int = 16, hidden: int = 512,
                 layers: int = 3):
        super().__init__(obs_dim, act_dim, obs_horizon, horizon)
        self.net = mlp([obs_horizon * obs_dim] + [hidden] * layers + [horizon * act_dim], out_gain=0.01)

    def loss(self, obs: torch.Tensor, act: torch.Tensor, mask: torch.Tensor,
             generator: torch.Generator | None = None) -> torch.Tensor:
        err = (self._chunk(obs) - act).abs().mean(-1)  # unclamped: a prediction past 1 still gets pulled back
        return (err * mask).sum() / mask.sum().clamp(min=1)

    def _chunk(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.condition(obs)).reshape(len(obs), self.horizon, self.act_dim)

    def sample(self, obs: torch.Tensor, steps: int = 0, generator: torch.Generator | None = None) -> torch.Tensor:
        return self._chunk(obs).clamp(-1, 1)


def build_model(cfg: dict) -> ChunkModel:
    common = dict(obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], obs_horizon=cfg["obs_horizon"],
                  horizon=cfg["horizon"])
    if cfg["kind"] == "diffusion":
        return DiffusionHead(**common, down_dims=tuple(cfg["down_dims"]), kernel=cfg["kernel"],
                             train_steps=cfg["train_steps"])
    return ChunkMLPHead(**common, hidden=cfg["hidden"], layers=cfg["layers"])


# ------------------------------------------------------------------------------------------------ run time

class ChunkPolicy:
    """numpy obs -> numpy action, like policy.TorchPolicy. It plans a chunk, executes the first `n_exec` actions
    open-loop and plans again. The noise of each plan is seeded from the observation it starts from, so a
    given episode evaluates the same way every time."""

    def __init__(self, model: ChunkModel, n_exec: int = 8, ddim_steps: int = 10, device: str = "cpu",
                 stage_obs: tuple[int, int] | None = STAGE_OBS):
        if device == "cpu":
            torch.set_num_threads(1)  # evaluation runs one of these per worker process
        self.model, self.device = model.to(device).eval(), device
        self.n_exec, self.ddim_steps, self.stage_obs = n_exec, ddim_steps, stage_obs
        self.reset()

    def reset(self):
        self.history, self.queue, self.stage = [], deque(), None

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.stage_obs:
            stage = int(np.argmax(obs[self.stage_obs[0]:self.stage_obs[1]]))
            if stage != self.stage:  # the training windows start afresh with a stage
                self.history, self.queue, self.stage = [], deque(), stage
        obs = np.asarray(obs, np.float32)
        self.history = (self.history + [obs])[-self.model.obs_horizon:]
        if not self.queue:
            window = [self.history[0]] * (self.model.obs_horizon - len(self.history)) + self.history
            generator = torch.Generator().manual_seed(zlib.crc32(obs.tobytes()))
            x = torch.as_tensor(np.stack(window), device=self.device)[None]
            chunk = self.model.sample(x, self.ddim_steps, generator)[0, :self.n_exec]
            self.queue.extend(chunk.clamp(-1, 1).cpu().numpy())
        return self.queue.popleft()


def save_checkpoint(path: str | Path, model: ChunkModel, cfg: dict, **extra) -> None:
    torch.save({"state_dict": model.state_dict(), **cfg, **extra}, path)


def load_chunk_policy(ckpt: dict, device: str = "cpu") -> ChunkPolicy:
    model = build_model(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    return ChunkPolicy(model, ckpt["n_exec"], ckpt["ddim_steps"], device, ckpt.get("stage_obs"))


# ------------------------------------------------------------------------------------------------ training

class ChunkData:
    """Demo steps as windows: (history of observations, chunk of actions, which chunk steps are real)."""

    def __init__(self, paths: str | list[str], obs_horizon: int, horizon: int, device: str):
        obs, act, episode, key, offset = [], [], [], [], 0
        for path in paths.split(",") if isinstance(paths, str) else paths:
            d = np.load(path)
            obs.append(d["obs"])
            act.append(d["actions"])
            episode.append(d["episode"] + offset)
            key.append(d["episode"] + offset if "stage" not in d else (d["episode"] + offset) * 8 + d["stage"])
            offset += int(d["episode"].max()) + 1
        self.obs, self.act = np.concatenate(obs).astype(np.float32), np.concatenate(act).astype(np.float32)
        self.episode, key = np.concatenate(episode), np.concatenate(key)
        n = len(key)
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], n] - 1
        seg = np.repeat(np.arange(len(starts)), ends - starts + 1)
        self.n, self.obs_horizon, self.horizon, self.device = n, obs_horizon, horizon, device
        to = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
        self.obs_t, self.act_t = to(self.obs), to(self.act)
        self.seg_start, self.seg_end = to(starts[seg]).long(), to(ends[seg]).long()
        self._obs_back = torch.arange(obs_horizon - 1, -1, -1, device=device)
        self._act_ahead = torch.arange(horizon, device=device)

    def split(self, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample indices of the training and validation episodes."""
        episodes = np.unique(self.episode)
        val = np.random.default_rng(seed).choice(episodes, max(1, int(fraction * len(episodes))), replace=False)
        is_val = np.isin(self.episode, val)
        return (torch.as_tensor(np.flatnonzero(~is_val), device=self.device),
                torch.as_tensor(np.flatnonzero(is_val), device=self.device))

    def window(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o_idx = torch.maximum(idx[:, None] - self._obs_back, self.seg_start[idx][:, None])
        a_idx = idx[:, None] + self._act_ahead
        valid = a_idx <= self.seg_end[idx][:, None]
        a_idx = torch.minimum(a_idx, self.seg_end[idx][:, None])
        return self.obs_t[o_idx], self.act_t[a_idx], valid.float()


class EMA:
    """Weights averaged over the training steps (decay rising from 0 towards 0.9999); the averaged copy is what
    gets evaluated and saved."""

    def __init__(self, model: nn.Module, power: float = 0.75):
        self.model, self.power, self.step = copy.deepcopy(model).eval(), power, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        decay = min(0.9999, 1 - (1 + self.step) ** -self.power)
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p, 1 - decay)


def train_chunked(data_path: str | list[str], out_dir: str | Path, kind: str = "diffusion", task: str = "g1-place",
                  epochs: int = 40, batch_size: int = 256, lr: float = 3e-4, obs_horizon: int = 2, horizon: int = 16,
                  n_exec: int = 8, down_dims=(64, 128, 256), hidden: int = 512, ddim_steps: int = 10,
                  eval_every: int = 10, eval_episodes: int = 30, eval_workers: int = 6, device: str | None = None,
                  seed: int = 0) -> dict:
    """Train a chunked policy on a demo dataset of `task`; keep the checkpoint that succeeds most often."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available()
                        else "cpu")
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = get_task(task)
    data = ChunkData(data_path, obs_horizon, horizon, device)
    train_idx, val_idx = data.split(0.1, seed)
    cfg = {"kind": kind, "task": task, "obs_dim": spec.obs_dim, "act_dim": spec.act_dim, "obs_horizon": obs_horizon,
           "horizon": horizon, "n_exec": n_exec, "ddim_steps": ddim_steps, "down_dims": list(down_dims), "kernel": 5,
           "train_steps": 100, "hidden": hidden, "layers": 3, "stage_obs": STAGE_OBS if task.startswith("g1") else None}
    model = build_model(cfg).to(device)
    model.set_obs_stats(data.obs[train_idx.cpu().numpy()].mean(0), data.obs[train_idx.cpu().numpy()].std(0))
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)
    steps_per_epoch = max(1, len(train_idx) // batch_size)
    total, warmup = epochs * steps_per_epoch, 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{kind} on {len(train_idx)} windows ({len(np.unique(data.episode))} episodes), {n_params / 1e6:.1f}M "
          f"parameters, device={device}", flush=True)
    history, best = [], {"success_rate": -1.0}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        running = 0.0
        for i in range(steps_per_epoch):
            loss = model.loss(*data.window(perm[i * batch_size:(i + 1) * batch_size]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            running += loss.item()
        with torch.no_grad():
            g = torch.Generator().manual_seed(1234)  # the same noise for every epoch's validation
            sel = val_idx[:8192]
            val_loss = ema.model.loss(*data.window(sel), generator=g).item()
        row = {"epoch": epoch, "train_loss": running / steps_per_epoch, "val_loss": val_loss}
        if epoch % eval_every == 0 or epoch == epochs:
            for e_buf, buf in zip(ema.model.buffers(), model.buffers()):
                e_buf.copy_(buf)
            save_checkpoint(out_dir / "candidate.pt", ema.model.cpu(), cfg, epoch=epoch)
            ema.model.to(device)
            stats = spec.evaluate(out_dir / "candidate.pt", episodes=eval_episodes, seed=50_000,
                                  workers=eval_workers)
            row["success_rate"] = stats["success_rate"]
            if stats["success_rate"] >= best["success_rate"]:
                best = {**row}
                shutil.copy(out_dir / "candidate.pt", out_dir / "policy.pt")
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {"best": best, "history": history}
