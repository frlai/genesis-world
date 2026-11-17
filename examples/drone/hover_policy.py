"""Observation and actuator computations shared by training and policy export."""

import torch


def compute_observation(rel_pos, base_quat, base_lin_vel, base_ang_vel, previous_actions, obs_scales):
    """Build the hovering policy observation from the relative target and base state."""
    return torch.cat(
        [
            torch.clip(rel_pos * obs_scales["rel_pos"], -1, 1),
            base_quat,
            torch.clip(base_lin_vel * obs_scales["lin_vel"], -1, 1),
            torch.clip(base_ang_vel * obs_scales["ang_vel"], -1, 1),
            previous_actions,
        ],
        axis=-1,
    )


def compute_targets(actions, *, clip_actions):
    """Clip actions and convert them to propeller speeds in revolutions per minute (RPM)."""
    actions = torch.clip(actions, -clip_actions, clip_actions)
    propellers_rpm = (1 + actions * 0.8) * 14468.429183500699
    return actions, propellers_rpm
