"""Export a trained walking policy with its observation and actuator computations."""

import argparse
import json
import pickle
from pathlib import Path

import torch

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat
from genesis.utils.leapp.deployment import LeappDeployment
from genesis.utils.leapp.export import export_policy
from genesis.utils.leapp.types import TensorMetadata

from go2_env import Go2Env
from go2_policy import compute_observation, compute_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-walking", help="Training directory under logs/")
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to export")
    parser.add_argument("-g", "--gpu", action="store_true", help="Use the GPU for scene state and eager inference")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Held-out replay steps after export")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out/go2_leapp"), help="Export directory")
    parser.add_argument("--rollout-steps", type=int, default=3000, help="Native closed-loop reference steps")
    args = parser.parse_args()
    if args.steps <= 0 or args.rollout_steps <= 0:
        parser.error("--steps and --rollout-steps must be positive")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=1, logging_level="warning")
    log_dir = Path("logs") / args.exp_name
    with (log_dir / "cfgs.pkl").open("rb") as stream:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(stream)
    reward_cfg["reward_scales"] = {}
    env = Go2Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    runner.load(str(log_dir / f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    def policy_step(entities, inputs, state):
        robot = entities["robot"]
        inv_base_quat = inv_quat(robot.get_quat())
        observation = compute_observation(
            transform_by_quat(robot.get_ang(), inv_base_quat),
            transform_by_quat(env.global_gravity, inv_base_quat),
            inputs["command"],
            robot.get_dofs_position(env.motors_dof_idx),
            robot.get_dofs_velocity(env.motors_dof_idx),
            state["previous_actions"],
            env.default_dof_pos,
            env.obs_scales,
            env.commands_scale,
        )
        actions, targets = compute_targets(
            policy({"policy": observation}),
            state["previous_actions"],
            env.default_dof_pos,
            clip_actions=env_cfg["clip_actions"],
            action_scale=env_cfg["action_scale"],
            simulate_action_latency=env.simulate_action_latency,
        )
        robot.control_dofs_position(targets[:, env.actions_dof_idx], slice(6, 18))
        return {"previous_actions": actions}

    def reset_scene():
        env.reset()
        # Synchronize public getters with the initial buffers populated by the training environment
        env.robot.set_qpos(env.init_qpos, zero_velocity=True)

    def samples():
        for speed in (0.5, 0.3, 0.6, 0.4, 0.5):
            env.commands[:, 0] = speed
            env._update_observation()
            yield {"command": env.commands.clone()}
            env.step(policy(env.get_observations()))

    with torch.no_grad():
        bundle = export_policy(
            policy_step,
            {"robot": env.robot},
            samples(),
            {"previous_actions": torch.zeros_like(env.actions)},
            args.output_dir,
            step_dt=env.dt,
            input_semantics={
                "command": TensorMetadata(
                    kind="frame/twist",
                    system_axis=0,
                    component_names=("x", "y", "angular_z"),
                    expressed_in="robot/base",
                    is_setpoint=True,
                )
            },
        )
        (bundle.parent / "scene.json").write_text(json.dumps(env_cfg, indent=2) + "\n")
        deployment = LeappDeployment(env.scene, bundle, {"robot": env.robot}, reset_callback=reset_scene)
        deployment.reset()
        error_max = 0.0
        state_error_max = 0.0
        for i_step in range(args.steps):
            if i_step == args.steps // 2:
                deployment.reset()
            env.commands[:, 0] = 0.25 if i_step % 20 < 10 else 0.5
            env._update_observation()
            actions = policy(env.get_observations())
            _, targets = compute_targets(
                actions,
                env.last_actions,
                env.default_dof_pos,
                clip_actions=env_cfg["clip_actions"],
                action_scale=env_cfg["action_scale"],
                simulate_action_latency=env.simulate_action_latency,
            )
            outputs = deployment.inference.run_policy(deployment.read_inputs({"command": env.commands}))
            actual = outputs[deployment.inference.outputs[0]]
            expected = targets[:, env.actions_dof_idx].to(actual.device)
            error_max = max(error_max, (actual - expected).abs().max().item())
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
            feedback = deployment.inference.value_dict["policy"]["state_previous_actions_in"]
            expected_actions = torch.clip(actions, -env_cfg["clip_actions"], env_cfg["clip_actions"]).to(
                feedback.device
            )
            state_error_max = max(state_error_max, (feedback - expected_actions).abs().max().item())
            torch.testing.assert_close(feedback, expected_actions, rtol=0, atol=1e-5)
            env.step(actions)
            if env.reset_buf.any():
                deployment.reset()
        deployment.close()
        env.reset()
        episode_steps = round(env_cfg["episode_length_s"] / env.dt)
        i_episode_step = 0
        failed_episodes = 0
        completed_episodes = 0
        episode_lengths = []
        error_sum = 0.0
        height_min = env_cfg["base_init_pos"][2]
        for _ in range(args.rollout_steps):
            env.commands[:, 0] = 0.25 if i_episode_step >= episode_steps // 2 else 0.5
            env.commands[:, 1:] = 0
            env._update_observation()
            command = env.commands.clone()
            actions = policy(env.get_observations())
            env.step(actions)
            base_velocity = transform_by_quat(env.robot.get_vel(), inv_quat(env.robot.get_quat()))
            error_sum += (base_velocity[:, :2] - command[:, :2]).square().sum().item()
            height_min = min(height_min, env.robot.get_pos()[:, 2].min().item())
            is_failed = (env.reset_buf & ~env.extras["time_outs"].bool()).any().item()
            i_episode_step += 1
            if is_failed or i_episode_step == episode_steps:
                failed_episodes += is_failed
                completed_episodes += not is_failed
                episode_lengths.append(i_episode_step)
                env.reset()
                i_episode_step = 0
        native_rollout = {
            "steps": args.rollout_steps,
            "velocity_rmse_m_s": (error_sum / args.rollout_steps) ** 0.5,
            "minimum_base_height_m": height_min,
            "failed_episodes": failed_episodes,
            "completed_episodes": completed_episodes,
            "episode_lengths": episode_lengths,
        }
    report = {
        "steps": args.steps,
        "actuator_max_abs_error": error_max,
        "feedback_max_abs_error": state_error_max,
        "checkpoint": str(log_dir / f"model_{args.ckpt}.pt"),
    }
    report["native_rollout"] = native_rollout
    (bundle.parent / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"bundle": str(bundle), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
