"""Evaluate the G1 cup-pyramid policies.

    python scripts/g1_evaluate.py --task full                                  # scripted experts, whole task
    python scripts/g1_evaluate.py --task place --stage 1                       # one placement stage
    python scripts/g1_evaluate.py --task toss --policy "toss=runs/g1_toss_ppo/policy.pt"
    python scripts/g1_evaluate.py --task full --policy "place=runs/g1_place_bc/policy.pt,toss=runs/g1_toss_ppo/policy.pt" --videos runs/videos

--policy is 'scripted' or 'place=<checkpoint|scripted>,toss=<checkpoint|scripted>'.
"""

import argparse
import json

from robot_arm.humanoid.evaluate import evaluate_g1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="full", choices=["place", "toss", "full"])
    p.add_argument("--stage", type=int, default=None, help="place task only: 0, 1 or 2 (default: random)")
    p.add_argument("--policy", default="scripted")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--videos", default=None, help="directory for the first few rollouts as mp4")
    p.add_argument("--num-videos", type=int, default=2)
    p.add_argument("--camera", default="front", choices=["front", "side", "over"])
    args = p.parse_args()
    stats = evaluate_g1(args.task, args.policy, args.episodes, args.stage, workers=args.workers,
                        video_dir=args.videos, videos=args.num_videos, camera=args.camera)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
