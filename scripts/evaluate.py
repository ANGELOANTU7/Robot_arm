"""Evaluate a policy: success rate on 100 fixed random layouts, plus a few rollout videos.

    python scripts/evaluate.py --policy scripted
    python scripts/evaluate.py --policy runs/bc/policy.pt --videos runs/bc/videos
"""

import argparse
import json

from robot_arm.evaluate import evaluate, load_policy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="scripted", help="'scripted' or a checkpoint path")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--videos", default=None, help="directory to save the first few rollouts as mp4")
    p.add_argument("--num-videos", type=int, default=3)
    p.add_argument("--camera", default="side", choices=["front", "top", "side"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    policy = load_policy(args.policy, device=args.device)
    stats = evaluate(policy, episodes=args.episodes, video_dir=args.videos,
                     videos=args.num_videos, camera=args.camera)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
