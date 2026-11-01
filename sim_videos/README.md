# Videos of the G1's tries

Recorded with `scripts/g1_record_attempts.py`: every try is kept, failures included. Left: side view, right: front view. The stage the robot is in and the outcome are written on the video.

## 07_learned_full_sequence

Policy: `place=runs/chunk_2k/policy.pt,toss=runs/toss4k_dp_e80/policy.pt`

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [full_20000_success.mp4](07_learned_full_sequence/full_20000_success.mp4) | full | success | thumbs-up | 0 |
| [full_20001_knocked-a-placed-cup.mp4](07_learned_full_sequence/full_20001_knocked-a-placed-cup.mp4) | full | knocked-a-placed-cup | throw the cap | 0 |
| [full_20002_knocked-a-placed-cup.mp4](07_learned_full_sequence/full_20002_knocked-a-placed-cup.mp4) | full | knocked-a-placed-cup | throw the cap | 0 |
| [full_20003_success.mp4](07_learned_full_sequence/full_20003_success.mp4) | full | success | thumbs-up | 0 |
| [full_20004_knocked-a-placed-cup.mp4](07_learned_full_sequence/full_20004_knocked-a-placed-cup.mp4) | full | knocked-a-placed-cup | throw the cap | 0 |
| [full_20005_success.mp4](07_learned_full_sequence/full_20005_success.mp4) | full | success | thumbs-up | 0 |
| [full_20006_success.mp4](07_learned_full_sequence/full_20006_success.mp4) | full | success | thumbs-up | 0 |
| [full_20007_success.mp4](07_learned_full_sequence/full_20007_success.mp4) | full | success | thumbs-up | 0 |
| [full_20008_timeout.mp4](07_learned_full_sequence/full_20008_timeout.mp4) | full | timeout | thumbs-up | 0 |
| [full_20009_knocked-a-placed-cup.mp4](07_learned_full_sequence/full_20009_knocked-a-placed-cup.mp4) | full | knocked-a-placed-cup | throw the cap | 0 |
| [full_20010_timeout.mp4](07_learned_full_sequence/full_20010_timeout.mp4) | full | timeout | throw the cap | 1 |
| [full_20011_timeout.mp4](07_learned_full_sequence/full_20011_timeout.mp4) | full | timeout | thumbs-up | 0 |
| [full_20012_success.mp4](07_learned_full_sequence/full_20012_success.mp4) | full | success | thumbs-up | 0 |
| [full_20013_success.mp4](07_learned_full_sequence/full_20013_success.mp4) | full | success | thumbs-up | 0 |
| [full_20014_success.mp4](07_learned_full_sequence/full_20014_success.mp4) | full | success | thumbs-up | 0 |
| [full_20015_success.mp4](07_learned_full_sequence/full_20015_success.mp4) | full | success | thumbs-up | 0 |

## 06_learned_BC_policies_full_task

Policy: `place=runs/v3/g1_place_bc/policy.pt,toss=runs/v3/g1_toss_bc/policy.pt`

The place and throw policies learned (BC) from your clips: placing works part of the time; the learned throw doesn't get the cap pinched.

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [full_95010_timeout.mp4](06_learned_BC_policies_full_task/full_95010_timeout.mp4) | full | timeout | place cup A | 0 |
| [full_95011_timeout.mp4](06_learned_BC_policies_full_task/full_95011_timeout.mp4) | full | timeout | throw the cap | 0 |

## 05_scripted_throw_and_rethrows

Policy: `scripted`

Recorded before the cap's fallback spot was moved 20 cm away from the pyramid: in these tries the cap lies right next to the pyramid and the fingers hit it while picking the cap up.

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [toss_95000_knocked-a-placed-cup.mp4](05_scripted_throw_and_rethrows/toss_95000_knocked-a-placed-cup.mp4) | toss | knocked-a-placed-cup | throw the cap | 0 |
| [toss_95001_knocked-a-placed-cup.mp4](05_scripted_throw_and_rethrows/toss_95001_knocked-a-placed-cup.mp4) | toss | knocked-a-placed-cup | throw the cap | 0 |
| [toss_95002_knocked-a-placed-cup.mp4](05_scripted_throw_and_rethrows/toss_95002_knocked-a-placed-cup.mp4) | toss | knocked-a-placed-cup | throw the cap | 0 |

## 04_scripted_touching_base_cups_full_task

Policy: `scripted`

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [full_95000_timeout.mp4](04_scripted_touching_base_cups_full_task/full_95000_timeout.mp4) | full | timeout | throw the cap | 0 |
| [full_95001_timeout.mp4](04_scripted_touching_base_cups_full_task/full_95001_timeout.mp4) | full | timeout | throw the cap | 0 |
| [full_95002_success.mp4](04_scripted_touching_base_cups_full_task/full_95002_success.mp4) | full | success | thumbs-up | 0 |

## 03_learned_place_policy_top_cup

Policy: `place=runs/g1_place_ppo2/policy.pt,toss=scripted`

Recorded with the base cups 2 cm apart, before they were changed to touch (the top cup then often tipped into the gap between them).

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [place_95000_knocked-a-placed-cup.mp4](03_learned_place_policy_top_cup/place_95000_knocked-a-placed-cup.mp4) | place | knocked-a-placed-cup | turn over + place top cup | 0 |
| [place_95001_knocked-a-placed-cup.mp4](03_learned_place_policy_top_cup/place_95001_knocked-a-placed-cup.mp4) | place | knocked-a-placed-cup | turn over + place top cup | 0 |

## 02_retargeted_from_your_clips_full_task

Policy: `replay:data/processed/human_pyramid`

Recorded with the base cups 2 cm apart, before they were changed to touch (the top cup then often tipped into the gap between them).

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [full_95000_timeout.mp4](02_retargeted_from_your_clips_full_task/full_95000_timeout.mp4) | full | timeout | turn over + place top cup | 0 |
| [full_95001_cap-fell-off-the-table.mp4](02_retargeted_from_your_clips_full_task/full_95001_cap-fell-off-the-table.mp4) | full | cap-fell-off-the-table | throw the cap | 0 |

## 01_scripted_experts_full_task

Policy: `scripted`

Recorded with the base cups 2 cm apart, before they were changed to touch (the top cup then often tipped into the gap between them).

| Video | Task | Outcome | Reached | Missed throws |
|---|---|---|---|---|
| [full_95000_timeout.mp4](01_scripted_experts_full_task/full_95000_timeout.mp4) | full | timeout | turn over + place top cup | 0 |
| [full_95001_timeout.mp4](01_scripted_experts_full_task/full_95001_timeout.mp4) | full | timeout | turn over + place top cup | 0 |
