"""Run an exported hovering policy directly in a Genesis scene."""

import argparse
import json
from pathlib import Path

import torch

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_quat_by_quat
from genesis.utils.leapp.deployment import LeappDeployment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", type=Path, default=Path("out/hover_leapp/policy/policy.yaml"), help="Pipeline YAML file"
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Simulate on the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the scene viewer")
    parser.add_argument("-s", "--steps", type=int, default=4500, help="Number of policy steps")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/hover_deploy"), help="Rollout report directory"
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=1, logging_level="warning")
    env_cfg = json.loads((args.bundle.parent / "scene.json").read_text())
    bindings = json.loads((args.bundle.parent / "bindings.json").read_text())
    scene = gs.Scene.load(args.bundle.parent / "scene.gscene", show_viewer=args.vis)
    scene.build(n_envs=1)
    drone = scene.get_entity(bindings["drone"])

    def reset_scene():
        drone.set_pos(env_cfg["base_init_pos"], zero_velocity=True)
        drone.set_quat(env_cfg["base_init_quat"], zero_velocity=True)
        drone.zero_all_dofs_velocity()

    deployment = LeappDeployment(scene, args.bundle, {"drone": drone}, reset_callback=reset_scene)
    deployment.reset()
    command = torch.tensor([[0.0, 0.0, 1.0]], dtype=gs.tc_float, device=gs.device)
    position_error = 0.0
    height_min = env_cfg["base_init_pos"][2]
    episode_steps = round(env_cfg["episode_length_s"] / scene.dt)
    inv_initial_quat = inv_quat(torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device))
    failed_episodes = 0
    completed_episodes = 0
    episode_lengths = []
    i_episode_step = 0
    try:
        for i_step in range(args.steps):
            command[:, 0] = 0.5 if i_episode_step >= episode_steps // 2 else 0.0
            command[:, 1] = -0.5 if i_episode_step >= episode_steps // 2 else 0.0
            outputs = deployment.step({"command": command})
            if not all(torch.isfinite(value).all() for value in outputs.values()):
                raise RuntimeError("The exported policy produced non-finite actuator targets.")
            position = drone.get_pos()
            position_error += (position - command).square().sum().item()
            height_min = min(height_min, position[:, 2].min().item())
            base_euler = quat_to_xyz(transform_quat_by_quat(inv_initial_quat, drone.get_quat()), rpy=True, degrees=True)
            rel_pos = command - position
            is_failed = (
                (
                    (base_euler[:, 0].abs() > env_cfg["termination_if_roll_greater_than"])
                    | (base_euler[:, 1].abs() > env_cfg["termination_if_pitch_greater_than"])
                    | (rel_pos[:, 0].abs() > env_cfg["termination_if_x_greater_than"])
                    | (rel_pos[:, 1].abs() > env_cfg["termination_if_y_greater_than"])
                    | (rel_pos[:, 2].abs() > env_cfg["termination_if_z_greater_than"])
                    | (position[:, 2] < env_cfg["termination_if_close_to_ground"])
                    | scene.get_error_envs_mask()
                )
                .any()
                .item()
            )
            i_episode_step += 1
            if is_failed or i_episode_step == episode_steps:
                failed_episodes += is_failed
                completed_episodes += not is_failed
                episode_lengths.append(i_episode_step)
                deployment.reset()
                i_episode_step = 0
    finally:
        deployment.close()
    report = {
        "steps": args.steps,
        "position_rmse_m": (position_error / args.steps) ** 0.5,
        "minimum_base_height_m": height_min,
        "completed_episodes": completed_episodes,
        "failed_episodes": failed_episodes,
        "episode_lengths": episode_lengths,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rollout.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
