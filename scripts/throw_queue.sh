#!/bin/bash
# Finishes the throw comparison without any Claude session attached. Safe to run again: every step is skipped once
# its result exists. Status in logs/throw_queue.log; final table in logs/throw_summary.txt; docs/THROW_RL_HANDOVER.md
# explains the rest. Start it detached:
#     setsid nohup bash scripts/throw_queue.sh > logs/throw_queue.log 2>&1 < /dev/null &
export G1_HAND=shadow OMP_NUM_THREADS=1
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
D4=data/processed/ds2_toss4k/g1_toss.npz
D6=data/processed/ds2_toss6k_s1/g1_toss.npz
say() { echo "$(date '+%H:%M:%S') $*"; }

# Let whatever is already running finish (two quiet checks: its scripts start the next step a moment later).
# The pattern is anchored: unanchored, it also matches any shell whose command text mentions these scripts.
quiet=0
while [ $quiet -lt 2 ]; do
    if pgrep -f '^python scripts/(train_|g1_evaluate|g1_retarget)' > /dev/null; then
        quiet=0
    else
        quiet=$((quiet + 1))
    fi
    sleep 30
done
say "earlier jobs have stopped"

# A diffusion run writes policy.pt and its "best:" line only at the end, so a killed one starts again.
train_dp() {  # name, then the extra options
    local n=$1; shift
    grep -q '^best:' "logs/q_$n.log" 2>/dev/null && return
    [ -f "runs/$n/policy.pt" ] && [ -f "runs/$n/.complete" ] && return
    [ -f "runs/$n/policy.pt" ] && { touch "runs/$n/.complete"; return; }
    say "training $n"
    python scripts/train_diffusion.py --task g1-toss --eval-workers 4 --out "runs/$n" "$@" > "logs/q_$n.log" 2>&1 \
        && touch "runs/$n/.complete" || say "FAILED training $n"
}
train_dp toss4k_dp_s1 --epochs 40 --eval-every 40 --seed 1 --data $D4 &
train_dp toss4k_dp_e80 --epochs 80 --eval-every 80 --data $D4 &
train_dp toss4k_dp_wide --epochs 40 --eval-every 40 --down-dims 96,192,384 --data $D4 &
train_dp toss4k_dp_wide_s1 --epochs 40 --eval-every 40 --down-dims 96,192,384 --seed 1 --data $D4 &
train_dp toss10k_dp --epochs 40 --eval-every 40 --data "$D4,$D6" &
wait

# PPO keeps its best checkpoint in policy.pt as it goes, so an interrupted run is tested as it stands, not rerun.
evaluate() {  # log name, checkpoint
    grep -q 'success_rate' "logs/$1.log" 2>/dev/null && return
    [ -f "$2" ] || { say "MISSING $2"; return; }
    say "testing $2"
    python scripts/g1_evaluate.py --task toss --policy "toss=$2" --episodes 96 --workers 8 > "logs/$1.log" 2>&1 \
        || say "FAILED testing $2"
}
evaluate ev_toss4k_dp_s196 runs/toss4k_dp_s1/policy.pt &
evaluate ev_toss4k_dp_e8096 runs/toss4k_dp_e80/policy.pt &
evaluate ev_toss4k_dp_wide96 runs/toss4k_dp_wide/policy.pt &
wait
evaluate ev_toss4k_dp_wide_s196 runs/toss4k_dp_wide_s1/policy.pt &
evaluate ev_toss10k_dp96 runs/toss10k_dp/policy.pt &
evaluate ev_ppo_small_final96 runs/toss4k_ppo_small/policy.pt &
wait
evaluate ev_ppo_mid_final96 runs/toss4k_ppo_mid/policy.pt

python - <<'EOF' > logs/throw_summary.txt
import json, pathlib, re
names = {"ev_toss4k_bc96": "BC (the control)", "ev_toss4k_dp96": "Diffusion", "ev_toss4k_dp_s196": "Diffusion, seed 1",
         "ev_toss4k_dp_e8096": "Diffusion, 80 epochs", "ev_toss4k_dp_wide96": "Diffusion, wide",
         "ev_toss4k_dp_wide_s196": "Diffusion, wide, seed 1", "ev_toss10k_dp96": "Diffusion, 10k episodes",
         "ev_ppo_small_i40_96": "PPO small noise, round 40", "ev_ppo_small_final96": "PPO small noise, best",
         "ev_ppo_mid_final96": "PPO mid noise, best"}
print("Throw, 96 tries on the same layouts (each score is about +-7 points)")
for log, name in names.items():
    p = pathlib.Path(f"logs/{log}.log")
    m = re.findall(r'"ok":\s*(\d+)', p.read_text()) if p.exists() else []
    print(f"{name:30s} {m[-1] + '/96' if m else 'no result'}")
EOF
say "FINISHED"
