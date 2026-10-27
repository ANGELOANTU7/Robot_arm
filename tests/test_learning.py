"""Smoke tests: BC and PPO run end to end on a tiny dataset and produce loadable checkpoints."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_arm.bc import train_bc  # noqa: E402
from robot_arm.env import PandaBallCupEnv  # noqa: E402
from robot_arm.evaluate import evaluate, load_policy  # noqa: E402
from robot_arm.ppo import PPOConfig, train_ppo  # noqa: E402
from robot_arm.retarget import canonical_layout, rollout  # noqa: E402
from robot_arm.synthetic import synthetic_demo  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    env = PandaBallCupEnv()
    rng = np.random.default_rng(0)
    obs, act, ep = [], [], []
    for i in range(3):
        demo = synthetic_demo(np.random.default_rng(i))
        o, a, ok, _ = rollout(env, demo, *canonical_layout(demo), rng, noise=0.3, seed=i)
        obs.append(o), act.append(a), ep.append(np.full(len(o), i))
    path = tmp_path_factory.mktemp("data") / "demos.npz"
    np.savez(path, obs=np.concatenate(obs), actions=np.concatenate(act), episode=np.concatenate(ep))
    return path


def test_bc_then_ppo(tiny_dataset, tmp_path):
    result = train_bc(tiny_dataset, tmp_path / "bc", epochs=2, batch_size=128, eval_every=2, device="cpu")
    assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"] * 1.5
    policy = load_policy(str(tmp_path / "bc" / "policy.pt"))
    action = policy(PandaBallCupEnv().reset(seed=0)[0])
    assert action.shape == (4,) and np.all(np.abs(action) <= 1)

    cfg = PPOConfig(num_envs=2, rollout_steps=32, iterations=2, minibatch=32, critic_warmup=1,
                    eval_every=2, eval_episodes=1, bc_batch=32)
    out = train_ppo(tmp_path / "bc" / "policy.pt", tiny_dataset, tmp_path / "ppo", cfg, device="cpu")
    assert len(out["history"]) == 2
    assert evaluate(load_policy(str(tmp_path / "ppo" / "policy.pt")), episodes=1)["episodes"] == 1
