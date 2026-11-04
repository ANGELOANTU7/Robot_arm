# Code guide

A running guide to the code, built part by part. Read order: 1 task, 2 control, 3 learning, 4 training data, 5 video.

Stages are 0–2 cups, 3 throw, 4 thumbs-up, so reaching stage 3 means the pyramid is built. `chunk_2k` built it 32/32
times in whole sequences; on 96 sequences it built 85 (89%).

## Part 1: the task (`humanoid/config.py`, `scene.py`, `objects.py`, `envs.py`)

`G1PyramidEnv` is one class with three tasks: `place` (one cup), `toss` (throw and thumbs-up from a built pyramid), and
`full` (everything). It runs at 20 steps a second, with 25 physics steps per control step. Time limits: 10 s for `place`
(200 steps), 20 s for `toss` (400), 50 s shared across all stages for `full` (1,000).

**Observation (81 numbers):**

| Index | Content |
|---|---|
| 0–9 | Waist and arm joint angles |
| 10–19 | Their speeds, scaled by 0.1 |
| 20–26 | Hand joints |
| 27–29 | Grasp-point position |
| 30–35 | Hand orientation |
| 36–53 | Each cup's position and mouth direction |
| 54–59 | Cap position and velocity |
| 60–64 | Current stage, one-hot |
| 65–67 | Target position |
| 68–70 | Vector from grasp point to the object to grab |
| 71–73 | Vector from the object to its target |
| 74 | Grip command |
| 75–80 | Gap between the commanded and actual hand pose |

**Action (7 numbers, each -1 to 1):** hand translation (3), hand rotation (3), grip (1). Actions move a stored command
pose, not the joints; `controller.py` turns that pose into joint targets. A leash keeps the command close to the real
hand. Top speed is 30 cm/s for the cups and 1.6 m/s in the throw.

**Reward:** 10 for finishing a stage, plus the change in `_potential()`, a progress score. Because only the change is
paid, progress sums to almost nothing over an episode, and a policy can't earn more by stalling.

**Success:**

| Stage | Passes when |
|---|---|
| 0 Cup A | Within 4 cm of its target, upright, still, released, hand 10 cm away |
| 1 Cup B | Same release checks; its spacing from cup A is in range |
| 2 Top cup | Same release checks; resting on the base cups, and all of it holds for 0.5 s |
| 3 Throw | Cap in the cup, pyramid standing, released at least 5 cm from the cup |
| 4 Thumbs-up | Fist with thumb up, hand clear, held for 0.5 s, cap still in the cup |

An episode ends early if something falls off the table, a placed cup is knocked, the simulation goes unstable, or the
cap is dropped in from too close. A missed throw isn't a failure: once the cap stops, the robot can throw again.

## Part 2: control (`humanoid/controller.py`, `hands.py`, `scripted.py`)

Path of one action: `step()` moves the command pose, `G1HandController.apply()` solves inverse kinematics for the 10
waist and arm joints and sends them to the position servos, and `hand_targets()` turns the single grip number into 18
hand actuator targets. Five interpolated IK solves run per control step, each with 8 iterations.

- **IK (`solve`):** damped least squares. The waist counts as 3 times more expensive than the arm, and spare freedom
  pulls the arm toward a comfortable posture (ready, or the pinch posture in the throw). `lag` cancels servo delay.
- **Grip:** blends between a pose pair chosen by stage: `wrap` for cups, `pinch` for the cap, `thumbs_up` at the end.
- **`hands.py`:** data for each hand, Dex3 or Shadow. Experiments here use Shadow (`G1_HAND=shadow`).
- **`PlaceExpert`:** a phase sequence: rise, over_pre, down_pre, approach, close, lift, (to_flip, flip for the top
  cup), carry, lower, open, (back), up. From carry on it servos the cup, not the hand, so slip in the grip is corrected.
- **`TossExpert`:** rise, neutral, traverse, turn, over, down, pinch, lift, windup, settle, swing, release. The release
  speed comes from projectile physics; after a miss, the next aim shifts by 80% of the miss.
- **Their part in the training data (details in Part 4):** the data comes from retargeted experts in `retarget.py`.
  Cup motion tracks paths from the human clips, `PlaceExpert` takes over to turn the top cup over and put it down, and
  the throw is `TossExpert` with settings from the human throw style. `toss` episodes also use `PlaceExpert` at reset,
  so each starts from a scripted top-cup placement.

## Part 3: learning (`policy.py`, `bc.py`, `dataset.py`, `diffusion.py`, `ppo.py`, `tasks.py`, `scripts/train_*.py`)

Throw data: `ds2_toss4k/g1_toss.npz` holds 351,352 steps from 1,745 successful episodes. 10% of episodes are held out
for validation, split by episode. Observations are normalised with statistics from the data.

| Model | File | Input to output | Parameters | Loss |
|---|---|---|---|---|
| BC | `policy.py`, `bc.py` | 1 observation to 1 action (network 81-256-256-256-7) | 154k | Weighted squared error |
| Chunked MLP | `diffusion.py`, `ChunkMLPHead` | 2 observations to 16 actions; 8 run | 666k | Absolute error |
| Diffusion | `diffusion.py`, `DiffusionHead` | 2 observations to 16 actions; 8 run | 4.67M | Noise-prediction error |
| PPO | `ppo.py` | Same network as BC, plus a critic | 154k + 153k | PPO loss + BC term |

- **`bc_loss`:** grip-switch steps (1.4% of the data) count 20 times; fast steps (9.6%) count 10 times in the throw.
- **Diffusion training:** add random noise to a demo action chunk, train the U-Net to predict that noise.
- **Diffusion at run time (`sample`):** start from noise, take 10 denoising steps (DDIM), run the first 8 actions, then
  plan again. A plan takes about 26 ms on the CPU. Evaluation uses an averaged copy of the weights (EMA).
- **`ChunkPolicy`:** plans are not interrupted by feedback, so 8 actions (0.4 s) run without looking. History and queued
  actions are cleared when the stage changes.
- **PPO loop:** collect 512 steps per environment with noisy actions; estimate how much better each action was than
  expected (GAE); update the critic and, after 10 warm-up rounds, the actor. `target_kl` stops an update that moves too
  far. The BC weight falls from 0.5 to only about 0.3 by round 60, so demonstrations still count at the end.
- **Checkpoint choice:** training keeps the best of the periodic evaluations (seeds 50000 or 60000). The final 96-try
  tests use seed 20000, so they are separate from that choice.
