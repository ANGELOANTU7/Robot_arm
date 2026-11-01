"""Chunked policies: data windows, both heads fit a simple expert, the wrapper's queue, checkpoints."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_arm.diffusion import (ChunkData, ChunkPolicy, DiffusionHead, build_model, load_chunk_policy,  # noqa: E402
                                 save_checkpoint)
from robot_arm.policy import load_checkpoint  # noqa: E402

OBS_DIM, ACT_DIM, STAGE = 8, 2, (6, 8)


def make_data(path, episodes=12, steps=30, seed=0):
    """A point driven to a target by the expert (a clipped proportional step): two stages per episode."""
    rng = np.random.default_rng(seed)
    obs, act, ep, stage = [], [], [], []
    for e in range(episodes):
        for s in range(2):
            pos, target = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
            for _ in range(steps):
                a = np.clip(5 * (target - pos), -1, 1)
                obs.append([*pos, *target, 0, 0, *np.eye(2)[s]])
                act.append(a)
                ep.append(e)
                stage.append(s)
                pos = pos + 0.1 * a
    np.savez(path, obs=np.array(obs, np.float32), actions=np.array(act, np.float32), episode=np.array(ep),
             stage=np.array(stage))


def test_windows_stop_at_stage_boundaries(tmp_path):
    make_data(tmp_path / "d.npz", episodes=2, steps=10)
    data = ChunkData(str(tmp_path / "d.npz"), obs_horizon=2, horizon=16, device="cpu")
    obs, act, valid = data.window(torch.tensor([0, 9, 10, 19]))
    assert obs.shape == (4, 2, OBS_DIM) and act.shape == (4, 16, ACT_DIM)
    assert torch.equal(obs[0, 0], obs[0, 1])  # the first step of a segment repeats its observation
    assert torch.equal(obs[2, 0], obs[2, 1])  # ... and so does the first step of the next stage
    assert valid.sum(1).tolist() == [10, 1, 10, 1]  # the chunk is cut at the segment's last step


@pytest.mark.parametrize("kind,steps", [("diffusion", 1500), ("chunk_mlp", 400)])  # diffusion needs the longer fit
def test_head_learns_the_expert(kind, steps, tmp_path):
    torch.manual_seed(0)
    make_data(tmp_path / "d.npz")
    data = ChunkData(str(tmp_path / "d.npz"), 2, 16, "cpu")
    cfg = dict(kind=kind, task="g1-place", obs_dim=OBS_DIM, act_dim=ACT_DIM, obs_horizon=2, horizon=16, n_exec=8,
               ddim_steps=10, down_dims=[16, 32, 64], kernel=5, train_steps=100, hidden=128, layers=2,
               stage_obs=STAGE)
    model = build_model(cfg)
    model.set_obs_stats(data.obs.mean(0), data.obs.std(0))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    all_idx = torch.arange(data.n)
    first = None
    for step in range(steps):
        batch = all_idx[torch.randint(0, data.n, (128,))]
        loss = model.loss(*data.window(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.6 * first
    obs, act, valid = data.window(all_idx[::7][:64])
    pred = model.eval().sample(obs, 10, torch.Generator().manual_seed(0))
    err = (((pred - act) ** 2).mean(-1) * valid).sum() / valid.sum()
    assert err < 0.3 * ((act ** 2).mean(-1) * valid).sum() / valid.sum()  # far better than predicting zeros
    assert pred.abs().max() <= 1.0 + 1e-6


def test_wrapper_queue_stage_reset_and_checkpoint(tmp_path):
    cfg = dict(kind="diffusion", task="g1-place", obs_dim=OBS_DIM, act_dim=ACT_DIM, obs_horizon=2, horizon=16,
               n_exec=8, ddim_steps=4, down_dims=[16, 32, 64], kernel=5, train_steps=100, hidden=0, layers=0,
               stage_obs=STAGE)
    model = build_model(cfg)
    save_checkpoint(tmp_path / "p.pt", model, cfg)
    policy = load_checkpoint(tmp_path / "p.pt")  # through the policy module's dispatch
    assert isinstance(policy, ChunkPolicy)
    obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
    obs[STAGE[0]:] = [1, 0]
    first = [policy(obs) for _ in range(8)]
    assert len(policy.queue) == 0 and all(a.shape == (ACT_DIM,) for a in first)
    policy.reset()
    again = [policy(obs) for _ in range(8)]  # the noise is seeded from the observation: the same chunk
    assert np.allclose(first, again)
    policy(obs)  # starts a new chunk (7 actions left over from it)
    assert len(policy.queue) == 7
    obs[STAGE[0]:] = [0, 1]
    policy(obs)  # a new stage drops what was queued and plans again
    assert len(policy.queue) == 7 and len(policy.history) == 1
    same = load_chunk_policy(torch.load(tmp_path / "p.pt", weights_only=False))
    assert np.allclose(same(obs), ChunkPolicy(model, 8, 4, stage_obs=STAGE)(obs))
