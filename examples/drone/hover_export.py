"""Export a trained hovering policy with its observation and actuator computations."""

import argparse
import copy
import json
import pickle
from pathlib import Path

import torch

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.leapp.deployment import LeappDeployment
from genesis.utils.leapp.environment import export_environment

from hover_env import HoverEnv


class _ActuatorRecorder:
    def set_propellers_rpm(self, targets):
        self.targets = targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="drone-hovering", help="Training directory under logs/")
    parser.add_argument("--ckpt", type=int, default=300, help="Checkpoint iteration to export")
    parser.add_argument("-g", "--gpu", action="store_true", help="Use the GPU for scene state and eager inference")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Held-out replay steps after export")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out/hover_leapp"), help="Export directory")
    parser.add_argument("--rollout-steps", type=int, default=4500, help="Native closed-loop reference steps")
    args = parser.parse_args()
    if args.steps <= 0 or args.rollout_steps <= 0:
        parser.error("--steps and --rollout-steps must be positive")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=1, logging_level="warning")
    log_dir = Path("logs") / args.exp_name
    with (log_dir / "cfgs.pkl").open("rb") as stream:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(stream)
    reward_cfg["reward_scales"] = {}
    env = HoverEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    runner.load(str(log_dir / f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    with torch.no_grad():
        bundle = export_environment(env, policy, args.output_dir)
        (bundle.parent / "scene.json").write_text(json.dumps(env_cfg, indent=2) + "\n")
        deployment = LeappDeployment(env.scene, bundle, {"drone": env.drone}, reset_callback=env.reset)
        deployment.reset()
        command_snapshot = env.commands.clone()
        error_max = 0.0
        state_error_max = 0.0
        previous_actions = torch.zeros_like(env.actions)
        for i_step in range(args.steps):
            if i_step == args.steps // 2:
                deployment.reset()
                command_snapshot = env.commands.clone()
                previous_actions.zero_()
            actions = policy(env.get_observations())
            reference = copy.copy(env)
            reference.drone = _ActuatorRecorder()
            reference._apply_actions(actions)
            expected = reference.drone.targets
            outputs = deployment.inference.run_policy(deployment.read_inputs({"command": command_snapshot}))
            actual = outputs[deployment.inference.outputs[0]]
            expected = expected.to(actual.device)
            error_max = max(error_max, (actual - expected).abs().max().item())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0.02)
            feedback = deployment.inference.value_dict["policy"]
            expected_actions = reference.actions
            for key, expected_state in (
                ("state_actions_in", expected_actions),
                ("state_last_actions_in", previous_actions),
            ):
                actual_state = feedback[key]
                expected_state = expected_state.to(actual_state.device)
                state_error_max = max(state_error_max, (actual_state - expected_state).abs().max().item())
                torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=2e-6)
            previous_actions = expected_actions.clone()
            env.commands[:, 0] = 0.25 if i_step % 20 < 10 else -0.25
            # Target resampling follows observation-relative position updates in the environment
            command_snapshot = env.commands.clone()
            env.step(actions)
            if env.reset_buf.any():
                deployment.reset()
                command_snapshot = env.commands.clone()
                previous_actions.zero_()
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
            env.commands[:, 0] = 0.5 if i_episode_step >= episode_steps // 2 else 0.0
            env.commands[:, 1] = -0.5 if i_episode_step >= episode_steps // 2 else 0.0
            env.commands[:, 2] = 1.0
            command = env.commands.clone()
            reference = copy.copy(env)
            reference.rel_pos = command - env.base_pos
            reference.last_actions = env.obs_buf[:, -env.num_actions :]
            reference._update_observation()
            env.step(policy(reference.get_observations()))
            position = env.drone.get_pos()
            error_sum += (position - command).square().sum().item()
            height_min = min(height_min, position[:, 2].min().item())
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
            "position_rmse_m": (error_sum / args.rollout_steps) ** 0.5,
            "minimum_base_height_m": height_min,
            "failed_episodes": failed_episodes,
            "completed_episodes": completed_episodes,
            "episode_lengths": episode_lengths,
        }
    report = {
        "steps": args.steps,
        "actuator_max_abs_error_rpm": error_max,
        "feedback_max_abs_error": state_error_max,
        "checkpoint": str(log_dir / f"model_{args.ckpt}.pt"),
    }
    report["native_rollout"] = native_rollout
    (bundle.parent / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"bundle": str(bundle), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
