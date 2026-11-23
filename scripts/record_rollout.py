"""Run a trained policy and save what the robot does, step by step, for plotting/animation.

Saves one .npz per start mode with, at every policy step (50 Hz), for one robot:
  body position (x, z), side-view angles of the body front and top axes, propeller tilt angles, thrust,
  climb/hold command, wheel contacts, and whether the episode ended.

Usage:
    uv run python scripts/record_rollout.py --policy logs/.../exported/policy.pt --out videos/data/B [env.* overrides]
    uv run python scripts/animate_rollouts.py videos/data/A videos/data/B ...
"""

import argparse
import math
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="Path to exported/policy.pt (TorchScript).")
parser.add_argument("--out", required=True, help="Output folder; writes floor.npz and wall.npz.")
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args_cli, overrides = parser.parse_known_args()

import sys  # noqa: E402

sys.argv = sys.argv[:1] + overrides  # Hydra overrides such as env.use_wheel_encoders=True

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import vertigo_rl.tasks  # noqa: E402,F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "VertigoRl-Wall-Climb-Vertigo-Direct"


@torch.inference_mode()
def record(env, policy, start_mode: int) -> dict:
    u = env.unwrapped
    probs = [0.0, 0.0, 0.0]
    probs[start_mode] = 1.0
    u.cfg.start_mode_probs = tuple(probs)
    obs, _ = env.reset()
    ex = torch.tensor([1.0, 0.0, 0.0], device=u.device).expand(u.num_envs, 3)
    ez = torch.tensor([0.0, 0.0, 1.0], device=u.device).expand(u.num_envs, 3)
    log = {k: [] for k in ("x", "z", "pitch", "up", "tilt", "thrust", "command", "on_wall", "on_floor", "done")}
    for _ in range(int(u.max_episode_length) - 1):
        obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
        d = u.robot.data
        pos = (d.root_pos_w.torch - u.scene.env_origins)[0]
        xb = quat_apply(d.root_quat_w.torch, ex)[0]  # body forward axis in the world
        log["x"].append(pos[0].item())
        log["z"].append(pos[2].item())
        zb = quat_apply(d.root_quat_w.torch, ez)[0]  # body "top" axis (propeller side) in the world
        log["pitch"].append(math.atan2(xb[2].item(), xb[0].item()))  # side-view angle of the body x axis
        log["up"].append(math.atan2(zb[2].item(), zb[0].item()))  # side-view angle of the body z axis
        log["tilt"].append(d.joint_pos.torch[0, u._tilt_ids].cpu().numpy())
        log["thrust"].append(u.thrust[0].cpu().numpy() / u.cfg.thrust_max)
        log["command"].append(u.command[0].item())
        log["on_wall"].append(u.wheels_on_wall[0].sum().item())
        log["on_floor"].append(u.wheels_on_floor[0].sum().item())
        log["done"].append(bool(term[0] or trunc[0]))
        if term[0] or trunc[0]:
            # the env has already auto-reset this robot: drop the post-reset sample, keep the last real one
            for v in log.values():
                v.pop()
            break
    out = {k: np.array(v) for k, v in log.items()}
    out["dt"] = np.array(u.step_dt)
    out["wall_x"] = np.array(u.cfg.wall_x)
    out["terminated"] = np.array(bool(term[0]))
    return out


def main():
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    args_cli.device = env_cfg.sim.device
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        policy = torch.jit.load(args_cli.policy, map_location=env.unwrapped.device).eval()
        for name, mode in (("floor", 0), ("wall", 1)):
            data = record(env, policy, mode)
            np.savez(out_dir / f"{name}.npz", **data)
            print(f"[record] {out_dir / name}.npz: {len(data['x'])} steps, terminated={bool(data['terminated'])}")
        env.close()


if __name__ == "__main__":
    main()
