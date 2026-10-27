# Robot_arm: teach a simulated Panda to drop a ball in a cup from your own videos

Work in progress. Milestone 1 (simulation core) is in place:

- `robot_arm/scene.py`: MuJoCo Menagerie Franka Panda, a 4 cm ball and a cup
- `robot_arm/env.py`: Gymnasium env `PandaBallCup-v0` (Cartesian delta actions + gripper)
- `robot_arm/scripted.py`: waypoint baseline (100/100 successes)

```bash
pip install -e ".[dev]"
python scripts/evaluate.py --policy scripted     # downloads the Panda model on first run
python scripts/view.py --policy scripted         # interactive viewer (needs a display)
pytest
```
