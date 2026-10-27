"""Watch the G1 build the cup pyramid and throw the cap in, in the interactive MuJoCo viewer (needs a display).

    python scripts/g1_view.py
    python scripts/g1_view.py --policy "place=runs/g1_place_bc/policy.pt,toss=runs/g1_toss_ppo/policy.pt"
"""

import argparse
import time

import mujoco.viewer

from robot_arm.humanoid.envs import G1PyramidEnv
from robot_arm.humanoid.evaluate import make_policy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="full", choices=["place", "toss", "full"])
    p.add_argument("--policy", default="scripted")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = G1PyramidEnv(task=args.task)
    policy = make_policy(args.policy)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            policy.reset()
            for _ in range(env.max_steps):
                start = time.time()
                obs, _, terminated, truncated, info = env.step(policy(obs))
                viewer.sync()
                if not viewer.is_running():
                    return
                time.sleep(max(0.0, env.cfg.control_dt - (time.time() - start)))
                if terminated or truncated:
                    break
            print(f"episode {ep}: {'success' if info['is_success'] else info['fail_reason'] or 'timeout'}")


if __name__ == "__main__":
    main()
