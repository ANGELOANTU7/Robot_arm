"""Watch a policy in the interactive MuJoCo viewer (needs a display).

    python scripts/view.py --policy scripted
    python scripts/view.py --policy runs/ppo/policy.pt
"""

import argparse
import time

import mujoco.viewer

from robot_arm.env import PandaBallCupEnv
from robot_arm.evaluate import load_policy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="scripted")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = PandaBallCupEnv()
    policy = load_policy(args.policy)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            if hasattr(policy, "reset"):
                policy.reset()
            for _ in range(env.cfg.max_episode_steps):
                start = time.time()
                obs, _, terminated, truncated, info = env.step(policy(obs))
                viewer.sync()
                if not viewer.is_running():
                    return
                time.sleep(max(0.0, env.cfg.control_dt - (time.time() - start)))
                if terminated or truncated:
                    break
            print(f"episode {ep}: {'success' if info['is_success'] else 'fail'}")


if __name__ == "__main__":
    main()
