"""Task constants shared by the simulation, the video pipeline and the learners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    # Ball: a 4 cm ping-pong-sized ball.
    ball_radius: float = 0.02
    ball_mass: float = 0.01

    # Cup: open cylinder made of a bottom disc and a ring of wall boxes.
    cup_inner_radius: float = 0.045
    cup_wall: float = 0.006
    cup_height: float = 0.08
    cup_segments: int = 16

    # Where the ball and cup may be placed (robot base frame, metres).
    workspace_x: tuple[float, float] = (0.35, 0.65)
    workspace_y: tuple[float, float] = (-0.30, 0.30)
    min_ball_cup_dist: float = 0.18
    max_ball_cup_dist: float = 0.45

    # Control: 20 Hz policy on a 500 Hz simulation.
    sim_dt: float = 0.002
    control_dt: float = 0.05
    episode_seconds: float = 12.0
    max_step: float = 0.015  # max TCP displacement per control step (m)
    # The joint servos cover ~40% of a commanded step within one control period, so the IK target is placed
    # this many steps ahead of the measured TCP to make the TCP actually move ~max_step per step.
    target_lead: float = 2.5

    # TCP height limits (keep the fingers off the table, stay in reach).
    tcp_z_min: float = 0.016
    tcp_z_max: float = 0.55

    @property
    def frame_skip(self) -> int:
        return int(round(self.control_dt / self.sim_dt))

    @property
    def max_episode_steps(self) -> int:
        return int(round(self.episode_seconds / self.control_dt))


CFG = TaskConfig()

# Panda arm joint angles of the Menagerie "home" keyframe (gripper pointing down).
HOME_QPOS = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
