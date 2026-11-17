"""Observation and actuator computations shared by training and policy export."""

import torch


def compute_observation(
    base_ang_vel,
    projected_gravity,
    commands,
    dof_pos,
    dof_vel,
    previous_actions,
    default_dof_pos,
    obs_scales,
    commands_scale,
):
    """Build the walking policy observation in the configured joint order."""
    return torch.concatenate(
        (
            base_ang_vel * obs_scales["ang_vel"],
            projected_gravity,
            commands * commands_scale,
            (dof_pos - default_dof_pos) * obs_scales["dof_pos"],
            dof_vel * obs_scales["dof_vel"],
            previous_actions,
        ),
        dim=-1,
    )


def compute_targets(actions, previous_actions, default_dof_pos, *, clip_actions, action_scale, simulate_action_latency):
    """Clip actions and convert the selected action step into joint position targets."""
    actions = torch.clip(actions, -clip_actions, clip_actions)
    exec_actions = previous_actions if simulate_action_latency else actions
    target_dof_pos = exec_actions * action_scale + default_dof_pos
    return actions, target_dof_pos
