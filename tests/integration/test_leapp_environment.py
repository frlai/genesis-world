import pytest
import torch

import genesis as gs
from genesis.utils.leapp import EnvironmentInput, TensorMetadata, policy_environment
from genesis.utils.leapp.types import dof_names

from ..utils.assertions import assert_allclose, assert_equal

pytest.importorskip("leapp")

from genesis.utils.leapp.deployment import LeappDeployment
from genesis.utils.leapp.environment import export_environment


class _ArmEnvironment:
    def __init__(self):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=1,
                gravity=(0.0, 0.0, 0.0),
            ),
            show_viewer=False,
        )
        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file="urdf/simple/two_link_arm.urdf",
                fixed=True,
            ),
        )
        self.scene.build(n_envs=1)
        self.dt = self.scene.dt
        self.commands = torch.tensor([[0.1, -0.2]], dtype=gs.tc_float, device=gs.device)
        self.elapsed_s = torch.tensor([[0.3]], dtype=gs.tc_float, device=gs.device)
        self.history = torch.full((1, 2, 2), 0.2, dtype=gs.tc_float, device=gs.device)
        self.previous_actions = torch.full((1, 2), -0.1, dtype=gs.tc_float, device=gs.device)
        self.observation = torch.zeros((1, 11), dtype=gs.tc_float, device=gs.device)
        self.buffers = {"history": self.history, "previous_actions": self.previous_actions}
        self.events = []
        self.fail = False

    def read_state(self):
        self.position = self.robot.get_dofs_position([1, 0])
        self.velocity = self.robot.get_dofs_velocity([1, 0])
        self.events.append("read")

    def update_observation(self):
        self.history.copy_(torch.cat((self.history[:, 1:], self.position[:, None]), dim=1))
        self.observation.copy_(
            torch.cat(
                (
                    self.history.flatten(1),
                    self.commands,
                    self.velocity * 0.1,
                    self.buffers["previous_actions"],
                    self.elapsed_s,
                ),
                dim=-1,
            )
        )
        self.events.append("observe")

    @policy_environment(
        entities=("robot",),
        inputs=(
            EnvironmentInput(
                "command",
                "commands",
                TensorMetadata(
                    kind="joint/position", system_axis=0, element_names=("interArm", "baseHinge"), is_setpoint=True
                ),
            ),
            EnvironmentInput("time", "elapsed_s", TensorMetadata(kind="time/duration", system_axis=0, unit="s")),
        ),
        state=("history", "previous_actions"),
        prepare=("read_state", "update_observation"),
        action="apply_action",
    )
    def get_observations(self):
        return self.observation

    def apply_action(self, actions):
        if self.fail:
            self.history.add_(10)
            raise ValueError("Deliberate environment capture failure")
        clipped = torch.clip(actions, -0.4, 0.4)
        self.targets = clipped * 0.25 + self.previous_actions * 0.1
        self.robot.control_dofs_force(self.targets, [1, 0])
        self.previous_actions.copy_(clipped)
        self.events.append("act")

    def step(self, actions):
        self.apply_action(actions)
        self.scene.step()
        self.read_state()
        self.update_observation()


@pytest.fixture
def policy():
    model = torch.nn.Sequential(torch.nn.Linear(11, 2), torch.nn.Tanh()).to(device=gs.device, dtype=gs.tc_float)
    with torch.no_grad():
        model[0].weight.copy_(torch.linspace(-0.5, 0.5, 22, device=gs.device).reshape(2, 11))
        model[0].bias.copy_(torch.tensor([0.15, -0.25], device=gs.device))
    return model.eval()


@pytest.mark.required
@pytest.mark.precision("32")
def test_export_original_environment(tmp_path, record_property, policy):
    env = _ArmEnvironment()
    tensor_names = ("commands", "elapsed_s", "history", "previous_actions", "observation")
    original_tensors = {name: getattr(env, name) for name in tensor_names}
    original_values = {name: value.clone() for name, value in original_tensors.items()}
    initial_position = env.robot.get_dofs_position().clone()
    samples = [
        {"command": torch.full_like(env.commands, value), "time": torch.full_like(env.elapsed_s, value + 0.5)}
        for value in (0.1, -0.2, 0.3)
    ]
    bundle = export_environment(env, policy, tmp_path, sample_inputs=samples)
    for name in tensor_names:
        assert getattr(env, name) is original_tensors[name]
        assert_equal(getattr(env, name), original_values[name])
    assert env.events == []
    assert env.buffers["history"] is original_tensors["history"]
    assert env.buffers["previous_actions"] is original_tensors["previous_actions"]
    assert_equal(env.robot.get_dofs_position(), initial_position)
    assert_equal(env.robot.get_dofs_control_force(), [[0.0, 0.0]])
    assert_equal(env.scene.get_time(), 0)

    def reset():
        env.scene.reset()
        for name in tensor_names:
            getattr(env, name).copy_(original_values[name])
        env.events.clear()

    deployment = LeappDeployment(env.scene, bundle, {"robot": env.robot}, reset_callback=reset)
    node = deployment.inference.nodes["policy"]
    descriptions = {item["name"]: item for item in node.input_descriptions}
    command = descriptions["external_command"]
    assert command["axes"][-1]["names"] == ["interArm", "baseHinge"]
    assert command["is_setpoint"]
    assert descriptions["external_time"]["unit"] == "s"
    assert descriptions["robot_get_dofs_position_0"]["genesis_connection"]["dof_names"] == list(
        reversed(dof_names(env.robot))
    )
    error_max = 0.0
    feedback_error_max = 0.0
    with torch.no_grad():
        for _ in range(2):
            deployment.reset()
            for index, value in enumerate((-0.7, 0.8, 0.25, -0.4)):
                env.robot.set_dofs_position([value, -value])
                env.robot.set_dofs_velocity([value * 0.3, value * 0.7])
                env.commands.copy_(torch.tensor([[value, -value]], device=gs.device))
                env.elapsed_s.fill_(0.017 + index * 0.13)
                env.read_state()
                env.update_observation()
                env.apply_action(policy(env.get_observations()))
                expected = env.targets.clone()
                outputs = deployment.inference.run_policy(
                    deployment.read_inputs({"command": env.commands, "time": env.elapsed_s})
                )
                actual = outputs[deployment.inference.outputs[0]]
                error_max = max(error_max, (actual - expected.to(actual.device)).abs().max().item())
                assert_allclose(actual, expected, tol=5e-8)
                feedback = deployment.inference.value_dict["policy"]
                for key, expected_state in (("history", env.history), ("previous_actions", env.previous_actions)):
                    actual_state = feedback[f"state_{key}_in"]
                    feedback_error_max = max(
                        feedback_error_max, (actual_state - expected_state.to(actual_state.device)).abs().max().item()
                    )
                    assert_allclose(actual_state, expected_state, tol=5e-8)
            assert env.events == ["read", "observe", "act"] * 4

        deployment.reset()
        env.read_state()
        env.update_observation()
        env.apply_action(policy(env.get_observations()))
        expected = env.targets.clone()
        deployment.step({"command": env.commands, "time": env.elapsed_s})
        assert_allclose(env.robot.get_dofs_control_force([1, 0]), expected, tol=5e-8)
        assert_allclose(env.scene.get_time(), env.dt, atol=1e-8)
        assert torch.any(env.robot.get_dofs_velocity() != 0)
    deployment.close()
    record_property("actuator_max_abs_error", error_max)
    record_property("feedback_max_abs_error", feedback_error_max)


@pytest.mark.required
@pytest.mark.precision("32")
def test_environment_capture_failure_isolated(tmp_path, policy):
    env = _ArmEnvironment()
    history = env.history.clone()
    previous = env.previous_actions.clone()
    env.fail = True
    with pytest.raises(ValueError, match="Deliberate environment capture failure"):
        export_environment(env, policy, tmp_path)
    assert_equal(env.history, history)
    assert_equal(env.previous_actions, previous)
    assert env.events == []
    assert_equal(env.scene.get_time(), 0)
    env.fail = False
    bundle = export_environment(env, policy, tmp_path, capture_steps=2)
    assert_allclose(env.scene.get_time(), env.dt, atol=1e-8)
    deployment = LeappDeployment(env.scene, bundle, {"robot": env.robot})
    deployment.step({"command": env.commands, "time": env.elapsed_s})
    assert_allclose(env.scene.get_time(), 2 * env.dt, atol=1e-8)
    deployment.close()


@pytest.mark.required
@pytest.mark.precision("32")
def test_environment_rejects_cached_state_views(tmp_path, policy):
    env = _ArmEnvironment()
    previous = env.previous_actions.clone()
    history = env.history.clone()
    env.buffers["previous_actions"] = env.previous_actions.view_as(env.previous_actions)
    with pytest.raises(ValueError, match="share storage"):
        export_environment(env, policy, tmp_path)
    assert_equal(env.previous_actions, previous)
    assert_equal(env.history, history)
    assert env.events == []
    assert_equal(env.scene.get_time(), 0)

    env.buffers["previous_actions"] = env.previous_actions

    def samples():
        yield {"command": env.commands, "time": env.elapsed_s}
        env.buffers["previous_actions"] = env.previous_actions.view_as(env.previous_actions)
        yield {"command": env.commands, "time": env.elapsed_s}

    with pytest.raises(ValueError, match="share storage"):
        export_environment(env, policy, tmp_path, sample_inputs=samples())
    assert_equal(env.previous_actions, previous)
    assert_equal(env.history, history)
    assert env.events == []
    assert_equal(env.scene.get_time(), 0)
