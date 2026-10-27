"""PPO fine-tuning of the BC policy, with a decaying behavior-cloning term on the demos (DAPG-style).

Rajeswaran et al. 2018, "Learning Complex Dexterous Manipulation with Deep RL and Demonstrations": start from a
BC policy and add lambda_k * BC loss to the policy-gradient loss, with lambda_k decaying so RL can eventually
improve on the demonstrations. The critic is warmed up first with the actor frozen, so early noisy advantages do
not wreck the BC policy.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from robot_arm.dataset import DemoDataset
from robot_arm.env import PandaBallCupEnv
from robot_arm.evaluate import evaluate
from robot_arm.policy import ActorCritic, TorchPolicy, bc_loss


@dataclass
class PPOConfig:
    num_envs: int = 16
    rollout_steps: int = 512  # per env per iteration
    iterations: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 5
    minibatch: int = 2048
    clip: float = 0.2
    target_kl: float = 0.02  # stop the epoch loop early if the policy moves further than this
    actor_lr: float = 5e-5
    critic_lr: float = 3e-4
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.0
    reward_scale: float = 0.1
    critic_warmup: int = 10  # iterations with the actor frozen
    bc_coef: float = 0.5  # lambda_0
    bc_decay: float = 0.99  # lambda_k = max(lambda_0 * decay^k, bc_min)
    bc_min: float = 0.02
    bc_batch: int = 1024
    min_log_std: float = -3.0
    eval_every: int = 20
    eval_episodes: int = 50
    seed: int = 0


def make_envs(n: int, seed: int) -> gym.vector.VectorEnv:
    fns = [lambda: PandaBallCupEnv() for _ in range(n)]
    envs = gym.vector.AsyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, context="spawn")
    envs.reset(seed=seed)
    return envs


def train_ppo(init_checkpoint: str | Path, demos_path: str | Path | None, out_dir: str | Path,
              cfg: PPOConfig = PPOConfig(), device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))

    model = ActorCritic.load(init_checkpoint, device)
    actor_params = list(model.actor.parameters()) + [model.log_std]
    actor_opt = torch.optim.Adam(actor_params, lr=cfg.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=cfg.critic_lr, eps=1e-5)
    demo_obs = demo_act = None
    if demos_path and cfg.bc_coef > 0:
        demos = DemoDataset.load(demos_path)
        demo_obs = torch.as_tensor(demos.obs, dtype=torch.float32, device=device)
        demo_act = torch.as_tensor(demos.actions, dtype=torch.float32, device=device)

    envs = make_envs(cfg.num_envs, cfg.seed)
    N, T = cfg.num_envs, cfg.rollout_steps
    obs_buf = torch.zeros((T, N, model.obs_mean.shape[0]), device=device)
    act_buf = torch.zeros((T, N, model.log_std.shape[0]), device=device)
    logp_buf, rew_buf, done_buf, val_buf = (torch.zeros((T, N), device=device) for _ in range(4))

    obs, _ = envs.reset(seed=cfg.seed)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_return, ep_success = np.zeros(N), np.zeros(N, bool)
    recent_success, recent_return = [], []
    history, best_eval = [], -1.0
    start = time.time()

    init_eval = evaluate(TorchPolicy(model, device), episodes=cfg.eval_episodes, seed=60_000)
    print(f"initial (BC) deterministic success: {init_eval['success_rate']:.2f}")
    best_eval = init_eval["success_rate"]
    model.save(out_dir / "policy.pt", source="bc-init")

    for it in range(1, cfg.iterations + 1):
        # ---------------------------------------------------------------- rollout
        for t in range(T):
            with torch.no_grad():
                dist = model.dist(obs)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                value = model.value(obs)
            next_obs, raw_reward, terminated, truncated, info = envs.step(action.clamp(-1, 1).cpu().numpy())
            reward = raw_reward * cfg.reward_scale
            done = terminated | truncated
            if truncated.any():
                # Bootstrap through time limits: the episode did not really end.
                final = np.stack([o if tr else np.zeros_like(next_obs[0])
                                  for o, tr in zip(info["final_obs"], truncated)])
                with torch.no_grad():
                    v_final = model.value(torch.as_tensor(final, dtype=torch.float32, device=device)).cpu().numpy()
                reward = reward + cfg.gamma * v_final * truncated
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action, logp, value
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            done_buf[t] = torch.as_tensor(done, dtype=torch.float32, device=device)

            ep_return += raw_reward
            ep_success |= np.asarray(info.get("is_success", np.zeros(N, bool)), bool)
            if done.any():
                final_success = np.asarray(info["final_info"]["is_success"], bool) if "final_info" in info else ep_success
                for i in np.nonzero(done)[0]:
                    recent_success.append(bool(final_success[i]))
                    recent_return.append(ep_return[i])
                    ep_return[i], ep_success[i] = 0.0, False
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        # ---------------------------------------------------------------- GAE
        with torch.no_grad():
            next_value = model.value(obs)
            adv = torch.zeros_like(rew_buf)
            last = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nv = next_value if t == T - 1 else val_buf[t + 1]
                nonterminal = 1.0 - done_buf[t]
                delta = rew_buf[t] + cfg.gamma * nv * nonterminal - val_buf[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
                adv[t] = last
            returns = adv + val_buf

        b_obs, b_act = obs_buf.reshape(T * N, -1), act_buf.reshape(T * N, -1)
        b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), returns.reshape(-1)

        # ---------------------------------------------------------------- update
        train_actor = it > cfg.critic_warmup
        bc_coef = max(cfg.bc_coef * cfg.bc_decay ** max(0, it - cfg.critic_warmup), cfg.bc_min) if cfg.bc_coef else 0.0
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "bc_loss": 0.0, "clipfrac": 0.0, "kl": 0.0}
        n_updates, stop = 0, False
        for _ in range(cfg.epochs):
            if stop:
                break
            perm = torch.randperm(T * N, device=device)
            for s in range(0, T * N, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                v_loss = 0.5 * ((model.value(b_obs[idx]) - b_ret[idx]) ** 2).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
                critic_opt.step()
                stats["v_loss"] += v_loss.item()

                if train_actor:
                    dist = model.dist(b_obs[idx])
                    logp = dist.log_prob(b_act[idx]).sum(-1)
                    ratio = (logp - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                    loss = pg_loss - cfg.entropy_coef * dist.entropy().sum(-1).mean()
                    if demo_obs is not None and bc_coef > 1e-4:
                        j = torch.randint(len(demo_obs), (cfg.bc_batch,), device=device)
                        demo_loss = bc_loss(model, demo_obs[j], demo_act[j])
                        loss = loss + bc_coef * demo_loss
                        stats["bc_loss"] += demo_loss.item()
                    actor_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                    actor_opt.step()
                    with torch.no_grad():
                        model.log_std.clamp_(min=cfg.min_log_std)
                    with torch.no_grad():
                        kl = ((ratio - 1) - (logp - b_logp[idx])).mean().item()  # low-variance KL estimate
                    stats["pg_loss"] += pg_loss.item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    stats["kl"] += kl
                    stop = kl > cfg.target_kl
                n_updates += 1
                if stop:
                    break

        row = {"iter": it, "steps": it * T * N, "time_s": round(time.time() - start, 1),
               "success": float(np.mean(recent_success[-100:])) if recent_success else None,
               "return": float(np.mean(recent_return[-100:])) if recent_return else None,
               "std": model.log_std.exp().mean().item(), "bc_coef": bc_coef,
               **{k: v / n_updates for k, v in stats.items()}}
        if it % cfg.eval_every == 0 or it == cfg.iterations:
            ev = evaluate(TorchPolicy(model, device), episodes=cfg.eval_episodes, seed=60_000)
            row["eval_success"] = ev["success_rate"]
            if ev["success_rate"] >= best_eval:
                best_eval = ev["success_rate"]
                model.save(out_dir / "policy.pt", source="ppo", iteration=it)
            model.save(out_dir / "last.pt", source="ppo", iteration=it)
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    envs.close()
    return {"initial_eval": init_eval["success_rate"], "best_eval": best_eval, "history": history}
