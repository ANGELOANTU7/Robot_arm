"""SmolVLA on the G1 pyramid: the model, its training data and batched closed-loop control of many envs.

Needs LeRobot (the `.venv-vla` environment: `pip install "lerobot[smolvla]==0.6.1"` with torch < 2.12).
The pretrained SmolVLA (`lerobot/smolvla_base`) is fine-tuned on the image demos of `vla.py`: two cameras, the
robot's state (vla.STATE_SETS), one instruction per stage, chunks of actions. State is normalised here, not by
LeRobot's processors; actions are already in [-1, 1].
"""

from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from robot_arm.humanoid.vla import CAMERAS, IMG, INSTRUCTIONS, STATE_SETS, decode, load_shards, vla_state

PRETRAINED = "lerobot/smolvla_base"
IMAGE_KEYS = tuple(f"observation.images.camera{k + 1}" for k in range(len(CAMERAS)))


def make_config(chunk: int = 16, n_exec: int = 8, res: int = 384, state: str = "robot", **overrides):
    """SmolVLA settings for the G1. `res`: the size images are padded and resized to for the vision encoder
    (pretrained at 512; 384 gives 36 tokens a camera and trains 1.7x faster, and the renders are only 224 px)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    inputs = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG)) for k in IMAGE_KEYS}
    dim = len(STATE_SETS[state])
    inputs["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(dim,))
    if dim > 32:  # pretrained width is 32; the state projection is then trained from scratch
        overrides.setdefault("max_state_dim", 48)
    return SmolVLAConfig(input_features=inputs, output_features={"action": PolicyFeature(type=FeatureType.ACTION,
                                                                                         shape=(7,))},
                         chunk_size=chunk, n_action_steps=n_exec, load_vlm_weights=False,
                         resize_imgs_with_padding=(res, res), **overrides), state


def build_policy(cfg, init: str | None = PRETRAINED):
    """A SmolVLA policy with the pretrained weights or a fine-tuned checkpoint folder. Tensors whose shape changed
    (the state projection, for a wider state) keep their fresh initialisation."""
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_file

    policy = SmolVLAPolicy(cfg)
    if init:
        path = Path(init) if Path(init).exists() else Path(snapshot_download(init))
        state = load_file(path / "model.safetensors")
        own = policy.state_dict()
        skip = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
        if skip:
            print(f"not loaded (shape changed): {skip}")
        state = {k: v for k, v in state.items() if k not in skip}
        missing, unexpected = policy.load_state_dict(state, strict=False)
        missing = [k for k in missing if k not in skip]
        if missing or unexpected:
            raise RuntimeError(f"weights of {init} don't fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return policy


class Tokens:
    """The instructions, tokenised once: stage -> (tokens, mask)."""

    def __init__(self, cfg, device="cpu"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
        enc = tok([s + "\n" for s in INSTRUCTIONS], padding="longest", padding_side="right", return_tensors="pt")
        self.tokens, self.mask = enc["input_ids"].to(device), enc["attention_mask"].bool().to(device)

    def __call__(self, stage: torch.Tensor):
        return self.tokens[stage], self.mask[stage]


class VLAData(torch.utils.data.Dataset):
    """Samples (images, state, stage, action chunk, padding) from VLA shards. A chunk never crosses into another
    stage or episode; the steps past the segment's end are padding."""

    def __init__(self, folders: str | list[str], chunk: int = 16, stats: dict | None = None, state: str = "robot"):
        if isinstance(folders, str):
            folders = folders.split(",")
        parts = [load_shards(f) for f in folders]
        self.d = parts[0] if len(parts) == 1 else _concat(parts)
        self.chunk = chunk
        seg = np.r_[True, (np.diff(self.d["episode"]) != 0) | (np.diff(self.d["stage"]) != 0)]
        start = np.flatnonzero(seg)
        ends = np.r_[start[1:], len(seg)]
        self.end = np.repeat(ends, np.diff(np.r_[start, len(seg)]))
        self.state = vla_state(self.d["obs"], state)
        self.stats = stats or {"mean": self.state.mean(0), "std": self.state.std(0) + 1e-3}
        self.episode = self.d["episode"]

    def __len__(self):
        return len(self.state)

    def image(self, cam: str, i: int) -> np.ndarray:
        o, n = self.d[f"off_{cam}"][i]
        return decode(self.d[f"jpg_{cam}"][o:o + n])

    def __getitem__(self, i):
        idx = np.arange(i, i + self.chunk)
        pad = idx >= self.end[i]
        act = self.d["actions"][np.minimum(idx, self.end[i] - 1)]
        out = {k: torch.from_numpy(self.image(c, i)).permute(2, 0, 1) for k, c in zip(IMAGE_KEYS, CAMERAS)}
        out["observation.state"] = torch.from_numpy((self.state[i] - self.stats["mean"]) / self.stats["std"])
        out["stage"] = int(self.d["stage"][i])
        out["action"] = torch.from_numpy(act)
        out["action_is_pad"] = torch.from_numpy(pad)
        return out


def _concat(parts: list[dict]) -> dict:
    out = {}
    for k in ("obs", "actions", "stage", "step"):
        out[k] = np.concatenate([p[k] for p in parts])
    base, eps = 0, []
    for p in parts:
        eps.append(p["episode"] + base)
        base += int(p["episode"].max()) + 1
    out["episode"] = np.concatenate(eps)
    for c in CAMERAS:
        b, offs = 0, []
        for p in parts:
            o = p[f"off_{c}"].copy()
            o[:, 0] += b
            offs.append(o)
            b += len(p[f"jpg_{c}"])
        out[f"jpg_{c}"] = np.concatenate([p[f"jpg_{c}"] for p in parts])
        out[f"off_{c}"] = np.concatenate(offs)
    return out


def to_batch(batch: dict, tokens: Tokens, device) -> dict:
    """A collated batch as SmolVLA wants it: images in [0, 1] on the device, the stage's instruction tokens."""
    out = {}
    for k in IMAGE_KEYS:
        out[k] = batch[k].to(device, non_blocking=True).float() / 255.0
    out["observation.state"] = batch["observation.state"].to(device, non_blocking=True).float()
    stage = torch.as_tensor(batch["stage"], device=device)
    out["observation.language.tokens"], out["observation.language.attention_mask"] = tokens(stage)
    for k in ("action", "action_is_pad"):
        if k in batch:
            out[k] = batch[k].to(device, non_blocking=True)
    return out


def save(policy, stats: dict, out_dir: str | Path, state: str = "robot", **extra) -> None:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in policy.state_dict().items()}, out_dir / "model.safetensors")
    np.savez(out_dir / "stats.npz", mean=stats["mean"], std=stats["std"])
    torch.save({"chunk": policy.config.chunk_size, "n_exec": policy.config.n_action_steps,
                "res": policy.config.resize_imgs_with_padding[0], "state": state, **extra},
               out_dir / "meta.pt")


def load(out_dir: str | Path, device="cuda", n_exec: int | None = None):
    meta = torch.load(Path(out_dir) / "meta.pt", weights_only=False)
    cfg, state = make_config(meta["chunk"], n_exec or meta["n_exec"], meta.get("res", 512), meta.get("state", "robot"))
    policy = build_policy(cfg, str(out_dir)).to(device).eval()
    s = np.load(Path(out_dir) / "stats.npz")
    return policy, {"mean": s["mean"], "std": s["std"], "state": state}


# ---- many envs stepped in parallel, the VLA batched on the GPU ----

def _env_proc(conn, task, stage, seed_base):
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from robot_arm.humanoid.envs import G1PyramidEnv

    env = G1PyramidEnv(task=task, stage=stage, render_mode="rgb_array", width=IMG, height=IMG)

    def views():
        return np.stack([env.render(c) for c in CAMERAS])

    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            obs, info = env.reset(seed=arg)
            conn.send((obs, views(), {"stage": env.stage}))
        elif cmd == "step":
            obs, r, term, trunc, info = env.step(arg)
            conn.send((obs, views(), r, term, trunc, info))
        elif cmd == "close":
            env.close()
            conn.close()
            return


class EnvPool:
    """N G1PyramidEnvs in their own processes, each returning its observation and camera views."""

    def __init__(self, n: int, task: str, stage: int | None = None):
        ctx = get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_env_proc, args=(b, task, stage, 0), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def reset(self, i: int, seed: int):
        self.conns[i].send(("reset", seed))

    def step(self, i: int, action):
        self.conns[i].send(("step", np.asarray(action, np.float32)))

    def recv(self, i: int):
        return self.conns[i].recv()

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=10)


class VLAController:
    """Chunked control of a batch of envs: whenever an env's queue of actions is empty (or its stage changed), it gets
    a new chunk; all envs that need one are planned together."""

    def __init__(self, policy, stats: dict, n: int, device="cuda", n_exec: int | None = None):
        self.policy, self.stats, self.device = policy, stats, device
        self.tokens = Tokens(policy.config, device)
        self.n_exec = n_exec or policy.config.n_action_steps
        self.queues = [[] for _ in range(n)]
        self.stage = [-1] * n

    def reset(self, i: int):
        self.queues[i], self.stage[i] = [], -1

    @torch.no_grad()
    def plan(self, obs: list[np.ndarray], views: list[np.ndarray], stages: list[int]) -> np.ndarray:
        """Action chunks (B, chunk, 7) for these envs."""
        state = (vla_state(np.stack(obs), self.stats.get("state", "robot")) - self.stats["mean"]) / self.stats["std"]
        im = torch.from_numpy(np.stack(views)).to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0
        batch = {k: im[:, j] for j, k in enumerate(IMAGE_KEYS)}
        batch["observation.state"] = torch.from_numpy(state).to(self.device).float()
        st = torch.as_tensor(stages, device=self.device)
        batch["observation.language.tokens"], batch["observation.language.attention_mask"] = self.tokens(st)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = self.policy.predict_action_chunk(batch)
        return chunk.float().clamp(-1, 1).cpu().numpy()

    def act(self, ids: list[int], obs: list[np.ndarray], views: list[np.ndarray], stages: list[int]) -> dict:
        need = [k for k, i in enumerate(ids) if not self.queues[i] or self.stage[i] != stages[k]]
        if need:
            chunks = self.plan([obs[k] for k in need], [views[k] for k in need], [stages[k] for k in need])
            for k, c in zip(need, chunks):
                self.queues[ids[k]] = list(c[: self.n_exec])
                self.stage[ids[k]] = stages[k]
        return {i: self.queues[i].pop(0) for i in ids}


def evaluate(policy, stats: dict, task: str, stage: int | None = None, episodes: int = 48, n_envs: int = 24,
             seed: int = 20_000, device="cuda", pool: EnvPool | None = None, residual=None) -> dict:
    """Closed-loop success of the VLA, with the same seeds as g1_evaluate.py (seed + episode number).
    `residual` is (model, scale) from scripts/train_vla_rl.py: its mean correction is added to the VLA action."""
    from collections import Counter

    from robot_arm.humanoid.envs import OBS_STAGE

    own = pool is None
    pool = pool or EnvPool(n_envs, task, stage)
    n = len(pool.conns)
    ctl = VLAController(policy, stats, n, device)
    todo = list(range(episodes))
    state, results = {}, []
    for i in range(min(n, episodes)):
        pool.reset(i, seed + todo[0])
        state[i] = {"ep": todo.pop(0)}
    for i in list(state):
        obs, views, _ = pool.recv(i)
        state[i].update(obs=obs, views=views)
        ctl.reset(i)
    while state:
        ids = list(state)
        acts = ctl.act(ids, [state[i]["obs"] for i in ids], [state[i]["views"] for i in ids],
                       [int(np.argmax(state[i]["obs"][OBS_STAGE])) for i in ids])
        if residual is not None:
            model, scale = residual
            x = torch.as_tensor(np.stack([np.r_[state[i]["obs"], acts[i]] for i in ids]), dtype=torch.float32,
                                device=device)
            with torch.no_grad():
                res = model.dist(x).mean.cpu().numpy()
            acts = {i: np.clip(acts[i] + scale * r, -1, 1) for i, r in zip(ids, res)}
        for i in ids:
            pool.step(i, acts[i])
        for i in ids:
            obs, views, r, term, trunc, info = pool.recv(i)
            if term or trunc:
                results.append((bool(info["is_success"]), int(info["stage"]),
                                info["fail_reason"] or ("ok" if info["is_success"] else "timeout")))
                if todo:
                    state[i] = {"ep": todo.pop(0)}
                    pool.reset(i, seed + state[i]["ep"])
                    obs, views, _ = pool.recv(i)
                    ctl.reset(i)
                else:
                    del state[i]
                    continue
            state[i].update(obs=obs, views=views)
    if own:
        pool.close()
    return {"task": task, "stage": stage, "episodes": episodes,
            "success_rate": float(np.mean([r[0] for r in results])),
            "reached_stage": dict(sorted(Counter(r[1] for r in results).items())),
            "outcomes": dict(Counter(r[2] for r in results))}
