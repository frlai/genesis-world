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
    def __init__(self, with_extra=False):
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
        self.extra = None
        if with_extra:
            self.extra = self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(2.0, 0.0, 0.0),
                    size=(0.1, 0.1, 0.1),
                    fixed=True,
                ),
            )
        self.scene.build(n_envs=1)
        self.dt = self.scene.dt
        self.commands = torch.tensor([[0.1, -0.2]], dtype=gs.tc_float, device=gs.device)
        self.elapsed_s = torch.tensor([[0.3]], dtype=gs.tc_float, device=gs.device)
        self.history = torch.zeros((1, 2, 2), dtype=gs.tc_float, device=gs.device)
        self.actions = torch.zeros((1, 2), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.observation = torch.zeros((1, 11), dtype=gs.tc_float, device=gs.device)
        self.buffers = {"last_actions": self.last_actions}
        self.events = []
        self.fail = False
        self.physics_steps = 1

    def step(self, actions):
        if self.extra is not None:
            self.extra.set_pos((3.0, 0.0, 0.0), zero_velocity=True)
        self.actions = torch.clip(actions, -0.4, 0.4)
        self.targets = self.actions * 0.25 + self.last_actions * 0.1
        self.robot.control_dofs_force(self.targets, [1, 0])
        for _ in range(self.physics_steps):
            self.scene.step()
        position = self.robot.get_dofs_position([1, 0])
        velocity = self.robot.get_dofs_velocity([1, 0])
        self.history.copy_(torch.cat((self.history[:, 1:], position[:, None]), dim=1))
        if self.fail:
            raise ValueError("Deliberate environment capture failure")
        self._resample_commands()
        self._reset_idx(torch.zeros(1, dtype=torch.bool, device=gs.device))
        self.observation.copy_(
            torch.cat(
                (self.history.flatten(1), self.commands, velocity * 0.1, self.buffers["last_actions"], self.elapsed_s),
                dim=-1,
            )
        )
        self.last_actions.copy_(self.actions)
        self.events.append("observe")
        return self.get_observations(), None, None, {}

    def _resample_commands(self):
        self.commands.add_(0.05)
        self.events.append("resample")

    def _reset_idx(self, mask):
        self.last_actions[mask] = 0
        self.events.append("reset")

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
        state=("actions", "last_actions", "history"),
        skip=("_reset_idx", "_resample_commands"),
    )
    def get_observations(self):
        return self.observation


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
    tensor_names = ("commands", "elapsed_s", "history", "actions", "last_actions", "observation")
    original_tensors = {name: getattr(env, name) for name in tensor_names}
    original_values = {name: value.clone() for name, value in original_tensors.items()}
    samples = [
        {"command": torch.full_like(env.commands, value), "time": torch.full_like(env.elapsed_s, value + 0.5)}
        for value in (0.1, -0.2, 0.3)
    ]
    bundle = export_environment(env, policy, tmp_path, sample_inputs=samples)
    for name in tensor_names:
        assert getattr(env, name) is original_tensors[name]
        assert_equal(getattr(env, name), original_values[name])
    assert env.events == []
    assert env.buffers["last_actions"] is original_tensors["last_actions"]
    assert_equal(env.robot.get_dofs_position(), [[0.0, 0.0]])
    assert_equal(env.robot.get_dofs_control_force(), [[0.0, 0.0]])
    assert_equal(env.scene.get_time(), 0)

    def reset():
        env.scene.reset()
        for name in tensor_names:
            getattr(env, name).copy_(original_values[name])
        env.events.clear()
        # The original step produces the reference observation from the current raw state.
        env.step(torch.zeros_like(env.actions))

    deployment = LeappDeployment(env.scene, bundle, {"robot": env.robot}, reset_callback=reset)
    descriptions = {item["name"]: item for item in deployment.inference.nodes["policy"].input_descriptions}
    assert descriptions["external_command"]["axes"][-1]["names"] == ["interArm", "baseHinge"]
    assert descriptions["external_command"]["is_setpoint"]
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
                actions = policy(env.get_observations())
                expected_state = {
                    "actions": torch.clip(actions, -0.4, 0.4),
                    "last_actions": env.actions.clone(),
                    "history": env.history.clone(),
                }
                outputs = deployment.inference.run_policy(
                    deployment.read_inputs({"command": env.commands, "time": env.elapsed_s})
                )
                actual = outputs[deployment.inference.outputs[0]]
                feedback = deployment.inference.value_dict["policy"]
                for key, expected in expected_state.items():
                    actual_state = feedback[f"state_{key}_in"]
                    feedback_error_max = max(
                        feedback_error_max, (actual_state - expected.to(actual_state.device)).abs().max().item()
                    )
                    assert_allclose(actual_state, expected, tol=5e-8)
                env.commands.copy_(torch.tensor([[value, -value]], device=gs.device))
                env.elapsed_s.fill_(0.017 + index * 0.13)
                env.step(actions)
                error_max = max(error_max, (actual - env.targets.to(actual.device)).abs().max().item())
                assert_allclose(actual, env.targets, tol=5e-8)
                assert_allclose(env.robot.get_dofs_control_force([1, 0]), actual, tol=5e-8)
            assert env.events == ["resample", "reset", "observe"] * 5

        deployment.reset()
        expected_actions = policy(env.get_observations())
        env.step(expected_actions)
        expected = env.targets.clone()
        deployment.reset()
        deployment.step({"command": env.commands, "time": env.elapsed_s})
        assert_allclose(env.robot.get_dofs_control_force([1, 0]), expected, tol=5e-8)
        assert_allclose(env.scene.get_time(), 2 * env.dt, atol=1e-8)
        assert torch.any(env.robot.get_dofs_velocity() != 0)
    deployment.close()
    record_property("actuator_max_abs_error", error_max)
    record_property("feedback_max_abs_error", feedback_error_max)


@pytest.mark.required
@pytest.mark.precision("32")
def test_environment_capture_failure_isolated(tmp_path, policy):
    env = _ArmEnvironment()
    env.fail = True
    with pytest.raises(ValueError, match="Deliberate environment capture failure"):
        export_environment(env, policy, tmp_path)
    assert_equal(env.history, torch.zeros_like(env.history))
    assert_equal(env.actions, torch.zeros_like(env.actions))
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
@pytest.mark.parametrize("physics_steps", [0, 2])
def test_environment_requires_one_physics_boundary(tmp_path, policy, physics_steps):
    env = _ArmEnvironment()
    env.physics_steps = physics_steps
    with pytest.raises(ValueError, match="scene.step|physics"):
        export_environment(env, policy, tmp_path)
    assert env.events == []
    assert_equal(env.scene.get_time(), 0)


@pytest.mark.required
@pytest.mark.precision("32")
def test_environment_rejects_cached_state_views(tmp_path, policy):
    env = _ArmEnvironment()
    env.buffers["last_actions"] = env.last_actions.view_as(env.last_actions)
    with pytest.raises(ValueError, match="share storage"):
        export_environment(env, policy, tmp_path)
    env.buffers["last_actions"] = env.last_actions

    def samples():
        yield {"command": env.commands, "time": env.elapsed_s}
        env.buffers["last_actions"] = env.last_actions.view_as(env.last_actions)
        yield {"command": env.commands, "time": env.elapsed_s}

    with pytest.raises(ValueError, match="share storage"):
        export_environment(env, policy, tmp_path, sample_inputs=samples())
    assert_equal(env.actions, torch.zeros_like(env.actions))
    assert_equal(env.history, torch.zeros_like(env.history))
    assert env.events == []
    assert_equal(env.scene.get_time(), 0)


@pytest.mark.required
@pytest.mark.precision("32")
def test_environment_blocks_undeclared_entity_writes(tmp_path, policy):
    env = _ArmEnvironment(with_extra=True)
    position = env.extra.get_pos().clone()
    with pytest.raises(ValueError, match="not a declared environment export boundary"):
        export_environment(env, policy, tmp_path)
    assert_equal(env.extra.get_pos(), position)
    assert env.events == []
    assert_equal(env.scene.get_time(), 0)
