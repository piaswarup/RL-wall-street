"""Evaluate a trained policy and report what it does with the propellers.

Runs one full episode per robot for each start mode (floor / on wall climbing / on wall holding) and prints:
  * reached wall, hold command issued, success (holding still on the wall at the end), terminated (fell/flew)
  * per phase: average push direction and thrust of each propeller

Push direction is the direction of the force a propeller puts on the robot, in the robot's side view:
  0 deg = straight into the surface the robot is on (belly side), +90 deg = forward (= UP when on the wall),
  -90 deg = backward, 180 deg = away from the surface.

Usage (after `isaaclab play` has exported the policy once):
    uv run python scripts/evaluate.py --policy logs/rsl_rl/vertigo_wall_climb/<run>/exported/policy.pt
Pass the same env.* overrides the policy was trained with, e.g. env.use_wheel_encoders=True.
"""

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="Path to exported/policy.pt (TorchScript).")
parser.add_argument("--num_envs", type=int, default=512)
add_launcher_args(parser)
args_cli, overrides = parser.parse_known_args()

import sys  # noqa: E402

# resolve_task_config re-parses sys.argv with Hydra: keep only config overrides such as env.use_wheel_encoders=True
sys.argv = sys.argv[:1] + overrides

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import vertigo_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "VertigoRl-Wall-Climb-Vertigo-Direct"
PHASES = ("floor", "flipping", "climbing", "holding")


@torch.inference_mode()
def run_episodes(env, policy, start_mode: int) -> dict:
    u = env.unwrapped
    probs = [0.0, 0.0, 0.0]
    probs[start_mode] = 1.0
    u.cfg.start_mode_probs = tuple(probs)
    obs, _ = env.reset()
    n = u.num_envs
    done_once = torch.zeros(n, dtype=torch.bool, device=u.device)
    terminated = torch.zeros_like(done_once)
    # per phase: sum of (sin, cos) of push direction for each prop, thrust for each prop, step count
    sincos = torch.zeros(4, 2, 2, device=u.device)
    thrust = torch.zeros(4, 2, device=u.device)
    counts = torch.zeros(4, device=u.device)

    for _ in range(int(u.max_episode_length)):
        obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
        alive = ~done_once
        on_wall = u.wheels_on_wall.all(-1)
        phase = torch.ones(n, dtype=torch.long, device=u.device)  # flipping / in between
        phase[u.wheels_on_floor.any(-1) & ~u.wheels_on_wall.any(-1)] = 0
        phase[on_wall & (u.command == 0)] = 2
        phase[on_wall & (u.command > 0)] = 3
        push = -u.robot.data.joint_pos.torch[:, u._tilt_ids]  # push direction = -tilt (see module docstring)
        for p in range(4):
            m = alive & (phase == p)
            sincos[p, :, 0] += torch.sin(push[m]).sum(0)
            sincos[p, :, 1] += torch.cos(push[m]).sum(0)
            thrust[p] += u.thrust[m].sum(0)
            counts[p] += m.sum()
        ended = (term | trunc) & alive
        terminated |= term & alive
        done_once |= ended
        if done_once.all():
            break

    return {
        "reached_wall": u.last_episode_on_wall.float().mean().item(),
        "hold_issued": u.last_episode_hold_issued.float().mean().item(),
        "success": u.last_episode_success.float().mean().item(),
        "terminated": terminated.float().mean().item(),
        "push_deg": torch.rad2deg(torch.atan2(sincos[..., 0], sincos[..., 1])).cpu(),
        "thrust": (thrust / counts.clamp(min=1).unsqueeze(-1)).cpu(),
        "frac": (counts / counts.sum()).cpu(),
    }


def main():
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    args_cli.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        policy = torch.jit.load(args_cli.policy, map_location=u.device).eval()
        print(f"\n================ evaluation ({u.num_envs} robots per start) ================")
        print(f"thrust_max = {u.cfg.thrust_max:.1f} N per prop, robot weight = {2.0 * 9.81:.1f} N")
        for label, mode in (("start on floor", 0), ("start on wall, climbing", 1), ("start on wall, holding", 2)):
            r = run_episodes(env, policy, mode)
            print(
                f"\n{label}: reached wall {r['reached_wall']:.1%} | hold issued {r['hold_issued']:.1%} | "
                f"SUCCESS {r['success']:.1%} | fell/flew {r['terminated']:.1%}"
            )
            print(f"  {'phase':10s} {'time':>5s} {'front push':>11s} {'rear push':>10s} {'front T':>8s} {'rear T':>7s}")
            for p, name in enumerate(PHASES):
                if r["frac"][p] < 1e-3:
                    continue
                fp, rp = r["push_deg"][p].tolist()
                tf, tr = r["thrust"][p].tolist()
                print(f"  {name:10s} {r['frac'][p]:5.0%} {fp:10.0f}° {rp:9.0f}° {tf:7.1f}N {tr:6.1f}N")
        print("\n(push: 0° = into the surface, +90° = forward / up the wall, 180° = away from the surface)")
        env.close()


if __name__ == "__main__":
    main()
