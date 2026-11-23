"""Physics sanity checks for the Vertigo wall-climb task (run before burning training/cluster hours).

Runs scripted (non-learned) commands and checks the robot does what physics says it should. Actions are the
force each propeller should put on the robot, in the body x-z plane: [front_fx, front_fz, rear_fx, rear_fz].
On the wall, body +x points UP the wall and body -z points INTO the wall.

  1. zero thrust on the wall                   -> falls
  2. thrust up + into the wall, above weight   -> climbs and stays on the wall
  3. thrust up + into the wall, below weight   -> slides down, slower than free fall (free wheels give no grip)
  4. thrust forward + down on the floor        -> rolls toward the wall on its free wheels
  5. tilt command across +/-180 deg            -> servo takes the short way round

Usage:
    uv run python scripts/smoke_test.py
"""

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=32)
add_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402

sys.argv = sys.argv[:1]  # resolve_task_config re-parses sys.argv with Hydra; our flags are already consumed

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import vertigo_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "VertigoRl-Wall-Climb-Vertigo-Direct"


def push(fx: float, fz: float, magnitude: float) -> list[float]:
    """Action for one propeller: push direction (fx, fz) in the body frame, scaled to `magnitude` of thrust_max."""
    norm = math.hypot(fx, fz)
    return [magnitude * fx / norm, magnitude * fz / norm]


@torch.inference_mode()
def run_phase(env, action: list[float], seconds: float) -> dict[str, torch.Tensor]:
    """Step with a constant action and return start/end root positions (env-local)."""
    u = env.unwrapped
    act = torch.tensor(action, device=u.device).expand(u.num_envs, -1)
    start = (u.robot.data.root_pos_w.torch - u.scene.env_origins).clone()
    for _ in range(int(seconds / u.step_dt)):
        obs, rew, *_ = env.step(act)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all(), "non-finite observation"
        assert torch.isfinite(rew).all(), "non-finite reward"
    end = (u.robot.data.root_pos_w.torch - u.scene.env_origins).clone()
    return {"start": start, "end": end}


def main():
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 60.0  # no timeouts during the checks
    env_cfg.airborne_time_limit = 1e3  # no "flying" terminations: we want to watch the robot fall
    args_cli.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        weight = 2.0 * 9.81
        print(f"[smoke] obs {env.observation_space}, critic {u.state_space}, actions {env.action_space}")
        print(f"[smoke] thrust_max per prop = {u.cfg.thrust_max:.1f} N, robot weight = {weight:.1f} N")
        results = {}

        @torch.inference_mode()
        def reset(start_mode: int, height: float = 1.0):
            probs = [0.0, 0.0, 0.0]
            probs[start_mode] = 1.0
            u.cfg.start_mode_probs = tuple(probs)
            u.cfg.wall_start_height = (height, height)
            env.reset()

        def wall_gap(r):
            return (u.cfg.wall_x - r["end"][:, 0]).mean().item()

        # 1) zero thrust on the wall
        reset(start_mode=2)
        r = run_phase(env, [0.0] * 4, 0.4)
        dz = (r["end"][:, 2] - r["start"][:, 2]).mean().item()
        results["zero thrust on wall falls"] = (dz < -0.3, f"dz={dz:+.2f} m in 0.4 s (free fall = -0.78 m)")

        # 2) push up the wall (+x) and into it (-z), upward part > weight
        up, into = 0.85, -0.53
        mag_hi = 1.0
        up_force = 2 * mag_hi * up * u.cfg.thrust_max
        reset(start_mode=1, height=0.8)
        r = run_phase(env, push(up, into, mag_hi) * 2, 1.0)
        dz = (r["end"][:, 2] - r["start"][:, 2]).mean().item()
        results["upward thrust > weight climbs"] = (
            dz > 0.2 and wall_gap(r) < 0.1,
            f"up force {up_force:.1f} N: dz={dz:+.2f} m in 1 s, wall gap={wall_gap(r):.3f} m",
        )

        # 3) same direction, upward part < weight
        mag_lo = 0.6 * weight / (2 * up * u.cfg.thrust_max)
        reset(start_mode=1, height=1.5)
        r = run_phase(env, push(up, into, mag_lo) * 2, 0.4)
        dz = (r["end"][:, 2] - r["start"][:, 2]).mean().item()
        results["upward thrust < weight slides slower"] = (
            -0.6 < dz < -0.05 and wall_gap(r) < 0.1,
            f"up force {0.6 * weight:.1f} N: dz={dz:+.2f} m in 0.4 s, wall gap={wall_gap(r):.3f} m",
        )

        # 4) floor: push forward (+x) and a little down (-z) -> rolls on the free wheels
        reset(start_mode=0)
        r = run_phase(env, push(1.0, -0.3, 0.3) * 2, 0.5)
        dx = (r["end"][:, 0] - r["start"][:, 0]).mean().item()
        z_end = r["end"][:, 2].mean().item()
        results["forward thrust rolls on floor"] = (dx > 0.1 and z_end < 0.1, f"dx={dx:+.2f} m in 0.5 s, z={z_end:.3f} m")

        # 5) tilt wrap: target 170 deg, then -170 deg -> joint should go to +190 deg, not swing back to -170
        reset(start_mode=0)
        tiny = 0.1  # small thrust so the robot stays put; enough to steer the servo

        def tilt_cmd(deg):
            t = math.radians(deg)  # force direction = -(sin t, cos t)
            return [-tiny * math.sin(t), -tiny * math.cos(t)] * 2

        run_phase(env, tilt_cmd(170), 1.0)
        run_phase(env, tilt_cmd(-170), 0.5)
        tilt = math.degrees(u.robot.data.joint_pos.torch[:, u._tilt_ids].mean().item())
        results["tilt wraps the short way"] = (abs(tilt - 190) < 10, f"joint = {tilt:.0f} deg (want 190)")

        print("\n================ smoke test ================")
        ok = True
        for name, (passed, detail) in results.items():
            ok &= passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:38s} {detail}")
        print("=============================================")
        env.close()
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
