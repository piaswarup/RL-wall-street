# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wall-climbing task for a Vertigo-style robot: two 360-degree tilting propellers, free-rolling wheels.

Every 20 ms the policy outputs, for each propeller, the force it wants that propeller to put on the robot, as a
2D vector in the robot's side view (body x = forward, body z = up out of the chassis):

    [front_fx, front_fz, rear_fx, rear_fz]   (each in [-1, 1])

The environment turns each vector into a tilt angle (servo target, shortest way round) and a thrust magnitude
(|vector| * thrust_max). Thrust only pushes one way: the props blow air away from the robot.

Success = drive to the wall, flip onto it, climb until the "hold" command arrives, then hold still on the wall.
The policy (actor) only sees what the real robot can sense: the IMU (gyro, accelerometer, gravity direction), the
two forward IR distance sensors, the climb/hold command, and its own previous commands. The critic additionally sees the
full simulator state; it is only used during training.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from vertigo_rl.assets import VERTIGO_CFG

WALL_X = 1.0
"""x position [m] of the wall face, relative to each environment origin."""

WALL_SIZE = (0.2, 2.0, 3.0)
"""Wall box size [m] (thickness, width, height)."""

SURFACE_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)

ROBOT_MASS = 2.0
"""[kg] total robot mass (must match scripts/make_vertigo_urdf.py)."""


@configclass
class WallClimbPhysicsCfg(PresetCfg):
    """Physics presets. ``default`` (Newton/MuJoCo-Warp) needs no Isaac Sim and runs on any CUDA GPU."""

    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # per-environment contact budget: 4 wheels + chassis against floor and wall
            njmax=128,
            nconmax=32,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class WallClimbSceneCfg(InteractiveSceneCfg):
    """Floor + one wall per environment + the robot."""

    robot: ArticulationCfg = VERTIGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # a big static box instead of GroundPlaneCfg: the default ground plane USD is downloaded from NVIDIA's asset
    # server, which may not be reachable from cluster compute nodes
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(400.0, 400.0, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=SURFACE_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.32, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    wall: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=WALL_SIZE,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=SURFACE_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.65)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(WALL_X + WALL_SIZE[0] / 2, 0.0, WALL_SIZE[2] / 2)),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class WallClimbEnvCfg(DirectRLEnvCfg):
    """Configuration for the Vertigo wall-climbing environment."""

    # env: physics at 200 Hz, policy at 50 Hz
    decimation = 4
    episode_length_s = 12.0
    history_length = 4  # actor sees the last N sensor frames (it has no velocity or position sensor)
    action_space = 4
    observation_space = 80  # actor size; recomputed by the env from the sensor switches below
    state_space = 36  # critic (privileged) observation size

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physics=WallClimbPhysicsCfg())
    scene: WallClimbSceneCfg = WallClimbSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # ---- robot interface (names must match the URDF) ----
    tilt_joint_names = ["front_tilt", "rear_tilt"]
    wheel_joint_names = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
    wheel_body_names = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
    prop_body_names = ["front_prop", "rear_prop"]
    wheel_radius = 0.06  # [m] must match the URDF

    # ---- propulsion: REPLACE WITH MEASURED VALUES ----
    thrust_max = 0.8 * ROBOT_MASS * 9.81  # [N] per propeller. PLACEHOLDER (15.7 N) until thrust data arrives
    thrust_time_constant = 0.05  # [s] first-order motor/prop spin-up lag. PLACEHOLDER
    prop_torque_coeff = 0.016  # [m] reaction (yaw) torque per Newton of thrust; props counter-rotate
    min_thrust_for_tilt = 0.05  # below this |action vector| the tilt target is left unchanged

    # ---- sensors: REPLACE WITH DATASHEET VALUES ----
    # two forward-facing IR sensors at the front of the chassis (body frame, metres)
    ir_positions = ((0.18, 0.08, 0.0), (0.18, -0.08, 0.0))
    ir_range = (0.03, 0.8)  # [m] min/max range. PLACEHOLDER (Sharp-style IR); a ToF sensor would be ~(0.03, 4.0)
    ir_noise_std = 0.01  # [m]
    gyro_noise_std = 0.02  # [rad/s]  ICM-45686 + vibration
    gravity_noise_std = 0.02  # [-]    error of the attitude (gravity direction) estimate
    accel_noise_std = 0.05  # [g]    ICM-45686 accelerometer + prop vibration

    # ---- what the policy (actor) may see ----
    use_accelerometer = True  # the ICM-45686 has one
    use_wheel_encoders = False  # NOT on the robot yet: encoders on the free wheels (hardware upgrade option)
    encoder_noise_std = 0.05  # [rad/s]
    privileged_actor = False  # True = actor sees the full simulator state (diagnostic upper bound, not deployable)

    # ---- task ----
    wall_x = WALL_X
    wall_top = 2.6  # [m] climb target height (just below the top of the 3 m wall)
    start_mode_probs = (0.5, 0.25, 0.25)  # floor / on wall climbing / on wall holding (curriculum)
    floor_start_distance = (0.3, 1.2)  # [m] from the wall face
    wall_start_height = (0.3, 1.5)  # [m]
    hold_height = (0.6, 2.0)  # [m] height at which the "hold" command is issued during an episode
    contact_tolerance = 0.015  # [m] wheel counts as touching a surface within this gap
    airborne_time_limit = 0.3  # [s] terminate if no wheel touches floor or wall for this long (no flying)
    terminate_on_prop_strike = True  # end the episode if the propeller side of the robot faces the wall
    wall_start_push_angle = 55.0  # [deg] wall starts begin mid-climb: props pushing up+in, thrust ~ supporting weight

    # ---- rewards ----
    rew_progress = 10.0  # climb mode: per metre closed toward the wall / up the wall
    rew_wall_contact = 1.0  # all four wheels on the wall, per second
    rew_hold_still = 3.0  # hold mode: exp(-|v|^2 / 0.02), per second
    rew_hold_position = 2.0  # hold mode: exp(-|z - z_hold| / 0.05), per second
    rew_thrust = -0.2  # mean squared normalised thrust, per second
    rew_action_rate = -0.05  # squared action change, per second
    rew_ang_vel = -0.02  # squared body angular velocity, per second
    rew_terminated = -3.0  # once, when the robot falls / flips / flies / leaves the arena

    def __post_init__(self) -> None:
        """Set the viewer camera."""
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-2.5, -3.0, 2.0), lookat=(WALL_X, 0.0, 0.8))
