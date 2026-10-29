# Videos of the G1's tries

Recorded with `scripts/g1_record_attempts.py`: every try is kept, failures included. Left: side view, right: front view. The stage the robot is in and the outcome are written on the video.

## 03_learned_place_policy_top_cup

Policy: `place=runs/g1_place_ppo2/policy.pt,toss=scripted`

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [place_95000_knocked-a-placed-cup.mp4](03_learned_place_policy_top_cup/place_95000_knocked-a-placed-cup.mp4) | place | knocked-a-placed-cup | turn over + place top cup | 0 |
| [place_95001_knocked-a-placed-cup.mp4](03_learned_place_policy_top_cup/place_95001_knocked-a-placed-cup.mp4) | place | knocked-a-placed-cup | turn over + place top cup | 0 |

## 02_retargeted_from_your_clips_full_task

Policy: `replay:data/processed/human_pyramid`

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [full_95000_timeout.mp4](02_retargeted_from_your_clips_full_task/full_95000_timeout.mp4) | full | timeout | turn over + place top cup | 0 |
| [full_95001_cap-fell-off-the-table.mp4](02_retargeted_from_your_clips_full_task/full_95001_cap-fell-off-the-table.mp4) | full | cap-fell-off-the-table | throw the cap | 0 |

## 01_scripted_experts_full_task

Policy: `scripted`

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [full_95000_timeout.mp4](01_scripted_experts_full_task/full_95000_timeout.mp4) | full | timeout | turn over + place top cup | 0 |
| [full_95001_timeout.mp4](01_scripted_experts_full_task/full_95001_timeout.mp4) | full | timeout | turn over + place top cup | 0 |
