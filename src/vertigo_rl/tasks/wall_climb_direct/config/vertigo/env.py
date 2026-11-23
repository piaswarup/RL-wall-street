# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow wall-climbing environment for a Vertigo-style robot."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, sample_uniform, wrap_to_pi

if TYPE_CHECKING:
    from .env_cfg import WallClimbEnvCfg

# robot on the wall: rotated -90 deg about world Y, so body +x points up and the belly faces the wall
_ON_WALL_QUAT_XYZW = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))

FLOOR, WALL_CLIMB, WALL_HOLD = 0, 1, 2
GRAVITY = 9.81


def actor_frame_dim(cfg: WallClimbEnvCfg) -> int:
    """gyro 3 + gravity 3 + IR 2 + command 1 + tilt command sin/cos 4 + last action 4 (+ accel 3) (+ encoders 4)."""
    return 17 + 3 * cfg.use_accelerometer + 4 * cfg.use_wheel_encoders


class WallClimbEnv(DirectRLEnv):
    """Drive to a wall, flip onto it, climb, and hold when commanded, using two 360-degree tilting propellers.

    Action (4, each in [-1, 1]): the force each propeller should put on the robot, in the body x-z plane
        [front_fx, front_fz, rear_fx, rear_fz]
    -> tilt target = direction (servo turns the short way round), thrust = |vector| * thrust_max (one-way only).

    Thrust is applied along each propeller link's local -Z axis (the props blow air along +Z, away from the robot).
    """

    cfg: WallClimbEnvCfg

    def __init__(self, cfg: WallClimbEnvCfg, render_mode: str | None = None, **kwargs):
        # actor observation size follows the sensor switches (they may be changed from the command line)
        cfg.observation_space = cfg.state_space if cfg.privileged_actor else actor_frame_dim(cfg) * cfg.history_length
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        tilt_ids, _ = self.robot.find_joints(self.cfg.tilt_joint_names, preserve_order=True)
        wheel_body_ids, _ = self.robot.find_bodies(self.cfg.wheel_body_names, preserve_order=True)
        wheel_joint_ids, _ = self.robot.find_joints(self.cfg.wheel_joint_names, preserve_order=True)
        self._wheel_joint_ids = torch.tensor(wheel_joint_ids, device=self.device)
        prop_ids, _ = self.robot.find_bodies(self.cfg.prop_body_names, preserve_order=True)
        self._tilt_ids = torch.tensor(tilt_ids, device=self.device)
        self._wheel_body_ids = torch.tensor(wheel_body_ids, device=self.device)
        self._prop_ids = torch.tensor(prop_ids, device=self.device)
        self._weight = float(self.robot.data.body_mass.torch[0].sum()) * GRAVITY  # [N]
        self._unit_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        self._prop_strike = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        n, dev = self.num_envs, self.device
        self.actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.tilt_target = torch.zeros(n, 2, device=dev)
        self._thrust_cmd = torch.zeros(n, 2, device=dev)
        self.thrust = torch.zeros(n, 2, device=dev)  # actual (lagged) thrust [N]
        self._forces = torch.zeros(n, 2, 3, device=dev)
        self._torques = torch.zeros(n, 2, 3, device=dev)
        self._spin_dir = torch.tensor([1.0, -1.0], device=dev)  # counter-rotating props
        self._thrust_alpha = min(1.0, self.physics_dt / self.cfg.thrust_time_constant)

        self._ir_pos_b = torch.tensor(self.cfg.ir_positions, device=dev)  # (2, 3)
        self.ir_dist = torch.full((n, 2), self.cfg.ir_range[1], device=dev)
        self._obs_history = torch.zeros(n, self.cfg.history_length, actor_frame_dim(self.cfg), device=dev)
        self._prev_lin_vel_w = torch.zeros(n, 3, device=dev)
        self._history_needs_fill = torch.ones(n, dtype=torch.bool, device=dev)

        # task state
        self.command = torch.zeros(n, device=dev)  # 0 = climb, 1 = hold
        self._command_prev = torch.zeros(n, device=dev)
        self.hold_height = torch.zeros(n, device=dev)
        self.z_hold = torch.zeros(n, device=dev)  # height where holding started
        self.start_mode = torch.zeros(n, dtype=torch.long, device=dev)
        self.climb_target_w = torch.zeros(n, 3, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self._dist = torch.zeros(n, device=dev)
        self.max_height = torch.zeros(n, device=dev)
        self.airborne_time = torch.zeros(n, device=dev)
        self.ever_on_wall = torch.zeros(n, dtype=torch.bool, device=dev)
        self.wheels_on_floor = torch.zeros(n, 4, dtype=torch.bool, device=dev)
        self.wheels_on_wall = torch.zeros(n, 4, dtype=torch.bool, device=dev)
        self.last_episode_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_episode_on_wall = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_episode_hold_issued = torch.zeros(n, dtype=torch.bool, device=dev)

        self._episode_sums = {
            key: torch.zeros(n, device=dev)
            for key in ("progress", "wall_contact", "hold_still", "hold_position", "thrust", "action_rate", "ang_vel",
                        "terminated")
        }

    # ------------------------------------------------------------------ actions

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1.0, 1.0))
        vec = self.actions.view(-1, 2, 2)  # (N, prop, [fx, fz])
        mag = torch.linalg.norm(vec, dim=-1).clamp(max=1.0)
        # the force on the robot points along -prop_z = -(sin t, cos t) in body (x, z)  =>  t = atan2(-fx, -fz)
        desired = torch.atan2(-vec[..., 0], -vec[..., 1])
        current = self.robot.data.joint_pos.torch[:, self._tilt_ids]
        target = current + wrap_to_pi(desired - current)  # shortest way round
        steer = mag > self.cfg.min_thrust_for_tilt
        self.tilt_target.copy_(torch.where(steer, target, self.tilt_target))
        self._thrust_cmd = mag * self.cfg.thrust_max

    def _apply_action(self) -> None:
        self.robot.actuators.target_command.set_position_index(value=self.tilt_target, joint_ids=self._tilt_ids)
        self.thrust += self._thrust_alpha * (self._thrust_cmd - self.thrust)
        self._forces[..., 2] = -self.thrust
        self._torques[..., 2] = self.cfg.prop_torque_coeff * self.thrust * self._spin_dir
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._forces, torques=self._torques, body_ids=self._prop_ids, is_global=False
        )

    # ---------------------------------------------------------------- sensing

    def _update_ir(self) -> None:
        """Ray-cast the two forward IR sensors against the floor (z = 0) and the wall face (x = wall_x)."""
        data = self.robot.data
        quat = data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1)
        origin = data.root_pos_w.torch.unsqueeze(1) - self.scene.env_origins.unsqueeze(1)
        origin = origin + quat_apply(quat, self._ir_pos_b.expand(self.num_envs, -1, -1))
        fwd = torch.zeros_like(origin)
        fwd[..., 0] = 1.0
        ray = quat_apply(quat, fwd)
        lo, hi = self.cfg.ir_range
        far = torch.full_like(origin[..., 0], 1e3)
        # wall
        t_wall = (self.cfg.wall_x - origin[..., 0]) / torch.where(ray[..., 0].abs() > 1e-6, ray[..., 0], far)
        hit = origin + t_wall.unsqueeze(-1) * ray
        wall_ok = (t_wall > 0) & (hit[..., 2] > 0) & (hit[..., 2] < 3.0) & (hit[..., 1].abs() < 1.0)
        # floor
        t_floor = -origin[..., 2] / torch.where(ray[..., 2] < -1e-6, ray[..., 2], -far)
        floor_ok = t_floor > 0
        dist = torch.minimum(torch.where(wall_ok, t_wall, far), torch.where(floor_ok, t_floor, far))
        self.ir_dist.copy_(dist.clamp(lo, hi))

    def _update_contacts(self) -> None:
        wheel_pos = self.robot.data.body_pos_w.torch[:, self._wheel_body_ids] - self.scene.env_origins.unsqueeze(1)
        r, tol = self.cfg.wheel_radius, self.cfg.contact_tolerance
        self.wheels_on_floor.copy_(wheel_pos[..., 2] < r + tol)
        self.wheels_on_wall.copy_(wheel_pos[..., 0] > self.cfg.wall_x - r - tol)

    # ------------------------------------------------------------- observations

    def _get_observations(self) -> dict:
        data = self.robot.data
        self._update_ir()
        gyro = data.root_ang_vel_b.torch + self.cfg.gyro_noise_std * torch.randn(self.num_envs, 3, device=self.device)
        gravity = data.projected_gravity_b.torch + self.cfg.gravity_noise_std * torch.randn(
            self.num_envs, 3, device=self.device
        )
        ir = self.ir_dist + self.cfg.ir_noise_std * torch.randn_like(self.ir_dist)
        parts = [
            gyro,  # 3  IMU gyro
            gravity,  # 3  IMU attitude estimate (which way is down, in the robot frame)
            ir,  # 2  front IR distances
            self.command.unsqueeze(-1),  # 1  0 = climb, 1 = hold
            torch.sin(self.tilt_target),  # 2  where the robot told the props to point
            torch.cos(self.tilt_target),  # 2
            self.actions,  # 4
        ]
        lin_vel_w = data.root_lin_vel_w.torch
        if self.cfg.use_accelerometer:
            # accelerometer = specific force (acceleration minus gravity) in the body frame, in g
            accel_w = (lin_vel_w - self._prev_lin_vel_w) / self.step_dt
            accel_w[:, 2] += GRAVITY
            accel_b = quat_apply_inverse(data.root_quat_w.torch, accel_w) / GRAVITY
            parts.append(accel_b + self.cfg.accel_noise_std * torch.randn_like(accel_b))  # 3
        self._prev_lin_vel_w.copy_(lin_vel_w)
        if self.cfg.use_wheel_encoders:
            enc = data.joint_vel.torch[:, self._wheel_joint_ids]
            parts.append(0.05 * (enc + self.cfg.encoder_noise_std * torch.randn_like(enc)))  # 4
        frame = torch.cat(parts, dim=-1)
        # rolling history (oldest first); freshly reset envs get their history filled with the current frame
        fill = self._history_needs_fill
        self._obs_history.copy_(torch.where(
            fill.view(-1, 1, 1),
            frame.unsqueeze(1).expand_as(self._obs_history),
            torch.cat((self._obs_history[:, 1:], frame.unsqueeze(1)), dim=1),
        ))
        self._history_needs_fill[:] = False

        # critic: full simulator state (training only)
        root_pos = data.root_pos_w.torch - self.scene.env_origins
        tilt = data.joint_pos.torch[:, self._tilt_ids]
        critic = torch.cat(
            (
                torch.stack((root_pos[:, 0] - self.cfg.wall_x, root_pos[:, 1], root_pos[:, 2]), dim=-1),  # 3
                data.root_lin_vel_b.torch,  # 3
                data.root_ang_vel_b.torch,  # 3
                data.projected_gravity_b.torch,  # 3
                torch.sin(tilt),  # 2
                torch.cos(tilt),  # 2
                0.1 * data.joint_vel.torch[:, self._tilt_ids],  # 2
                self.thrust / self.cfg.thrust_max,  # 2
                self.command.unsqueeze(-1),  # 1
                (root_pos[:, 2] - torch.where(self.command > 0, self.z_hold, self.hold_height)).unsqueeze(-1),  # 1
                self.wheels_on_floor.float(),  # 4
                self.wheels_on_wall.float(),  # 4
                self.actions,  # 4
                self.ir_dist,  # 2
            ),
            dim=-1,
        )
        actor = critic if self.cfg.privileged_actor else self._obs_history.flatten(1)
        return {"policy": actor, "critic": critic}

    # ------------------------------------------------------------ dones/rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        root_pos = data.root_pos_w.torch - self.scene.env_origins
        self._update_contacts()
        on_wall = self.wheels_on_wall.all(dim=-1)
        self.ever_on_wall |= on_wall
        touching = self.wheels_on_floor.any(dim=-1) | self.wheels_on_wall.any(dim=-1)
        self.airborne_time.copy_(torch.where(touching, torch.zeros_like(self.airborne_time), self.airborne_time + self.step_dt))

        # issue the "hold" command once the robot is fully on the wall at the hold height
        self._command_prev.copy_(self.command)
        start_hold = (self.command == 0) & on_wall & (root_pos[:, 2] >= self.hold_height)
        self.z_hold.copy_(torch.where(start_hold, root_pos[:, 2], self.z_hold))
        self.command.copy_(torch.where(start_hold, torch.ones_like(self.command), self.command))

        self._dist.copy_(torch.linalg.norm(self.climb_target_w - data.root_pos_w.torch, dim=-1))
        torch.maximum(self.max_height, root_pos[:, 2], out=self.max_height)

        flying = self.airborne_time > self.cfg.airborne_time_limit
        fell = (self.max_height > 0.6) & (root_pos[:, 2] < 0.2)
        upside_down = (data.projected_gravity_b.torch[:, 2] > 0.7) & (root_pos[:, 2] < 0.3)
        # top (propeller side) facing the wall while close to it = the spinning props would strike the wall
        body_z_w = quat_apply(data.root_quat_w.torch, self._unit_z)
        prop_strike = (body_z_w[:, 0] > 0.5) & (root_pos[:, 0] > self.cfg.wall_x - 0.3) & self.cfg.terminate_on_prop_strike
        self._prop_strike = prop_strike
        out_of_bounds = (
            (root_pos[:, 2] > 2.9)
            | (torch.abs(root_pos[:, 1]) > 1.0)
            | (root_pos[:, 0] < self.cfg.wall_x - 2.5)
            | (root_pos[:, 0] > self.cfg.wall_x + 0.1)
        )
        terminated = flying | fell | upside_down | prop_strike | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _success(self) -> torch.Tensor:
        """Holding: command is 'hold', all wheels on the wall, nearly still, close to where it started holding."""
        data = self.robot.data
        z = data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        speed = torch.linalg.norm(data.root_lin_vel_w.torch, dim=-1)
        return (
            (self.command > 0)
            & self.wheels_on_wall.all(dim=-1)
            & (speed < 0.1)
            & ((z - self.z_hold).abs() < 0.1)
        )

    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt
        data = self.robot.data
        climbing = self._command_prev == 0
        holding = self.command > 0
        z = data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        speed_sq = torch.sum(torch.square(data.root_lin_vel_w.torch), dim=-1)
        terms = {
            "progress": self.cfg.rew_progress * (self.prev_dist - self._dist) * climbing.float(),
            "wall_contact": self.cfg.rew_wall_contact * self.wheels_on_wall.all(dim=-1).float() * dt,
            "hold_still": self.cfg.rew_hold_still * torch.exp(-speed_sq / 0.02) * holding.float() * dt,
            "hold_position": self.cfg.rew_hold_position * torch.exp(-(z - self.z_hold).abs() / 0.05) * holding.float()
            * dt,
            "thrust": self.cfg.rew_thrust * torch.mean(torch.square(self.thrust / self.cfg.thrust_max), dim=-1) * dt,
            "action_rate": self.cfg.rew_action_rate * torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
            * dt,
            "ang_vel": self.cfg.rew_ang_vel * torch.sum(torch.square(data.root_ang_vel_b.torch), dim=-1) * dt,
            "terminated": self.cfg.rew_terminated * self.reset_terminated.float(),
        }
        self.prev_dist.copy_(self._dist)
        reward = torch.zeros(self.num_envs, device=self.device)
        for key, value in terms.items():
            reward += value
            self._episode_sums[key] += value
        return reward

    # -------------------------------------------------------------------- reset

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # episode statistics (computed before the state is overwritten)
        success = self._success()[env_ids] & self.reset_time_outs[env_ids]
        self.last_episode_success[env_ids] = success
        self.last_episode_on_wall[env_ids] = self.ever_on_wall[env_ids]
        self.last_episode_hold_issued[env_ids] = self.command[env_ids] > 0
        log = self.extras.setdefault("log", {})
        for key, value in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = value[env_ids].mean() / self.max_episode_length_s
            value[env_ids] = 0.0
        floor_starts = self.start_mode[env_ids] == FLOOR
        log["Metrics/success_rate"] = success.float().mean()
        log["Metrics/floor_start_reached_wall"] = (
            (self.ever_on_wall[env_ids] & floor_starts).float().sum() / floor_starts.float().sum().clamp(min=1)
        )
        log["Metrics/max_height"] = self.max_height[env_ids].mean()
        log["Episode_Termination/terminated"] = self.reset_terminated[env_ids].float().mean()
        log["Episode_Termination/flying"] = (self.airborne_time[env_ids] > self.cfg.airborne_time_limit).float().mean()
        log["Episode_Termination/prop_strike"] = self._prop_strike[env_ids].float().mean()

        super()._reset_idx(env_ids)
        n = len(env_ids)
        dev = self.device
        origins = self.scene.env_origins[env_ids]

        # --- start mode: floor facing the wall / on the wall climbing / on the wall already holding
        probs = torch.tensor(self.cfg.start_mode_probs, device=dev)
        mode = torch.multinomial(probs / probs.sum(), n, replacement=True)
        on_wall = mode != FLOOR
        self.start_mode[env_ids] = mode

        pos = torch.zeros(n, 3, device=dev)
        pos[:, 1] = sample_uniform(-0.3, 0.3, (n,), dev)
        contact_gap = self.cfg.wheel_radius + 0.002
        floor_dist = sample_uniform(*self.cfg.floor_start_distance, (n,), dev)
        wall_height = sample_uniform(*self.cfg.wall_start_height, (n,), dev)
        pos[:, 0] = torch.where(on_wall, self.cfg.wall_x - contact_gap, self.cfg.wall_x - floor_dist)
        pos[:, 2] = torch.where(on_wall, wall_height, torch.full_like(wall_height, contact_gap))

        zeros = torch.zeros(n, device=dev)
        floor_quat = quat_from_euler_xyz(zeros, zeros, sample_uniform(-0.3, 0.3, (n,), dev))
        wall_quat = torch.tensor(_ON_WALL_QUAT_XYZW, device=dev).expand(n, 4)
        quat = torch.where(on_wall.unsqueeze(-1), wall_quat, floor_quat)

        root_pose = torch.cat((pos + origins, quat), dim=-1)
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()

        # wall starts begin "mid-climb": props pushing up and into the wall with thrust ~ supporting the weight
        # (push direction phi from the wall normal <=> tilt = -phi; force = (sin phi up, -cos phi into the wall))
        phi = math.radians(self.cfg.wall_start_push_angle)
        hover_thrust = min(self._weight / (2 * math.sin(phi)), self.cfg.thrust_max)
        hover_mag = hover_thrust / self.cfg.thrust_max
        tilt0 = torch.where(on_wall, torch.full_like(zeros, -phi), zeros).unsqueeze(-1).expand(-1, 2)
        joint_pos[:, self._tilt_ids] = tilt0
        thrust0 = torch.where(on_wall, torch.full_like(zeros, hover_thrust), zeros).unsqueeze(-1).expand(-1, 2)
        act0 = torch.tensor([math.sin(phi), -math.cos(phi)] * 2, device=dev) * hover_mag
        act0 = torch.where(on_wall.unsqueeze(-1), act0.expand(n, -1), torch.zeros(n, 4, device=dev))
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)

        # --- command: climb until the hold height, or already holding
        hold_height = sample_uniform(*self.cfg.hold_height, (n,), dev)
        climbing_on_wall = mode == WALL_CLIMB
        hold_height = torch.where(climbing_on_wall, torch.maximum(hold_height, wall_height + 0.2), hold_height)
        self.hold_height[env_ids] = hold_height
        self.command[env_ids] = (mode == WALL_HOLD).float()
        self.z_hold[env_ids] = wall_height

        target = torch.stack(
            (torch.full_like(zeros, self.cfg.wall_x - contact_gap), pos[:, 1], torch.full_like(zeros, self.cfg.wall_top)),
            dim=-1,
        )
        self.climb_target_w[env_ids] = target + origins
        self.prev_dist[env_ids] = torch.linalg.norm(target - pos, dim=-1)
        self._dist[env_ids] = self.prev_dist[env_ids]
        self.max_height[env_ids] = pos[:, 2]
        self.airborne_time[env_ids] = 0.0
        self.ever_on_wall[env_ids] = on_wall
        self.thrust[env_ids] = thrust0
        self.tilt_target[env_ids] = tilt0
        self.actions[env_ids] = act0
        self.prev_actions[env_ids] = act0
        self._prev_lin_vel_w[env_ids] = 0.0
        self._history_needs_fill[env_ids] = True
