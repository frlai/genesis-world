# Export complete policy pipelines

The optional LEAPP integration exports tensor computations from raw Genesis state and external commands through a policy to actuator targets. The exported ONNX model includes preprocessing, policy normalization, and postprocessing. A deployment runner reads raw state and applies the exported targets to a supplied scene.

Install Genesis with its optional export dependency and the trainer used by the examples:

```bash
python -m pip install -e '.[leapp]' 'rsl-rl-lib>=5.0.0,<6' tensordict tensorboard
```

The integration uses LEAPP 0.8.1. Install the PyTorch build appropriate for your device before installing Genesis. Ordinary simulation and training can run with the export extra omitted.

## Export an annotated environment

For environments with an annotated observation method, export the original inference path with one call:

```python
from genesis.utils.leapp.environment import export_environment

# env is freshly initialized; policy includes its existing inference normalization.
bundle = export_environment(env, policy, "out/policy")
```

The exporter discovers the environment's named entities, external inputs, state attributes, and existing inference methods. It collects five samples by advancing the native environment between captures. Each capture runs on an owned copy of its tensor attributes, intercepting public getters and actuator setters. The original observation method and policy callable execute unchanged. No policy callback or parallel observation implementation is required.

Annotate the environment's existing `get_observations` method with its inference boundaries:

```python
from genesis.utils.leapp import EnvironmentInput, TensorMetadata, policy_environment


@policy_environment(
    entities=("robot",),
    inputs=(EnvironmentInput("command", "commands", semantics=TensorMetadata(system_axis=0)),),
    state=("actions", "last_actions"),
    action="actions",
    skip=("_reset_idx", "_resample_commands"),
)
def get_observations(self):
    return self.obs_buf  # Keep the environment's actual return type and body.
```

The exporter runs the original `step` method around its `scene.step()` boundary on owned environment copies. It reads raw state, follows the original observation and history updates, calls the policy on the returned observations, and captures the next original action-processing prefix. Physics is not advanced during capture. `action` names the tensor attribute defining action shape; `skip` names reset and command-sampling methods that must not run on the capture copy. The Go2 and drone environment bodies, scene construction, comments, and execution order remain unchanged. Observations after `scene.step()` must derive from decorated raw getters or declared environment input/state attributes. Tensor locals calculated before the physics boundary and reused afterward are rejected, preventing discarded values from becoming export constants. Termination decisions remain outside the exported policy; scene-error masks may only support unused native termination bookkeeping.

The feedback boundary is at the physics boundary, before the original post-step buffer and observation updates. Both examples carry `actions` and `last_actions`: observations read their original history slot, then the existing history assignment copies the previous action before the new policy output is processed. This preserves Go2's actuator delay and the drone's older observation action. Reset restores the exported initial values. Export a freshly initialized environment for its canonical initial state, or pass an explicit `initial_state={...}` when exporting an environment that has already stepped. A live snapshot is not necessarily the task's reset state.

Declare every external input and persistent tensor state. Other configuration tensors are constants. Ordinary custom objects are shared read-only by contract and must not be mutated during capture; access to undeclared Genesis objects is rejected. Repeated references to one tensor are supported. Create tensor views inside the inference methods; distinct cached tensors sharing storage produce a binding error.

Optional `sample_inputs` gives callers control of sample collection; otherwise `capture_steps` defaults to five. The callback API below remains useful for standalone tensor pipelines that have no environment methods.

## Export a standalone tensor pipeline

For a standalone pipeline, implement one inference callback. Entity bindings supplied to it provide annotated public getters and control setters. Command tensors and feedback state are explicit inputs. Pass tensor-valued actuator targets and return every declared next-state tensor. Keep scene stepping, command generation, rewards, and episode decisions in the caller.

```python
import torch

from genesis.utils.leapp import TensorMetadata
from genesis.utils.leapp.export import export_policy

# scene and arm are already built. model is your trained, evaluation-mode policy.
# Select the controllable joints in the model's order before capture.
dofs = [arm.get_joint(name).dof_start for name in joint_names]


def policy_step(entities, inputs, state):
    robot = entities['arm']
    position = robot.get_dofs_position(dofs)
    velocity = robot.get_dofs_velocity(dofs)
    observation = torch.cat((position - offset, velocity * velocity_scale, inputs['command']), dim=-1)
    action = model(observation)
    target = torch.clamp(action, -1.0, 1.0) * action_scale + offset
    target = 0.8 * target + 0.2 * state['previous_target']
    robot.control_dofs_position(target, dofs)
    return {'previous_target': target}


bundle = export_policy(
    policy_step,
    {'arm': arm},
    command_samples,
    {'previous_target': initial_target},
    'out/arm_policy',
    step_dt=scene.dt,
)
```

`command_samples` is an iterable of dictionaries, each containing the callback's external tensors. Supply several representative, changing samples. A generator can update the real scene between yields. Capture intercepts control writes, so the callback itself advances no physics. Select a fixed deployment batch size before capture.

The result is a pipeline YAML path. Keep its directory together: the ONNX model, external model data when present, feedback initialization, and the sibling `.genesis.json` controller/timing description are required for deployment.

Built-in annotations cover base position, quaternion, linear and angular velocity, joint position/velocity, joint position/velocity/force control, and drone propeller RPM. The metadata preserves ordered joint selections, Genesis quaternion order, coordinate frames, and physical units. Use `input_semantics={'command': TensorMetadata(...)}` to describe external commands. For a command shaped `(num_envs, num_components)`, set `system_axis=0` and provide component names in their numerical order. Leave the system axis unspecified for shared or scalar inputs. Quantity labels describe physical boundaries, rather than normalized mixed observation vectors. LEAPP 0.8.1 accepts `is_setpoint=True` on command inputs only. Actuator ports preserve their target meaning through the control setter and value-parameter binding.

To extend an entity API, register its raw getter with `tensor_input` or its actuator argument with `tensor_output` from `genesis.utils.leapp.annotations`. These decorators return the original function. Export adapters consume their metadata only during capture. Additional preprocessing can use supported PyTorch or NumPy operations inside the same callback.

Feedback uses named tensors with fixed shapes and dtypes. Explicitly include action history, delays, or other persistent tensor state in the callback contract. Reset restores the exported initial values. Tensor-dependent Python branches and variable shapes require a separately supported tracing strategy.

## Direct deployment

```python
from genesis.utils.leapp.deployment import LeappDeployment

runner = LeappDeployment(scene, bundle, {'arm': arm}, reset_callback=reset_scene)
runner.reset()
try:
    for command in commands:
        runner.step({'command': command})
finally:
    runner.close()
```

The scene is built by the application. Bind entities with the names used at export. The runner resolves joint names, restores controller parameters, checks timestep compatibility, runs the exported preprocessing/model/postprocessing, and applies targets at the exported control interval. `reset_scene` restores the simulation state through public APIs. `close()` releases the runner while scene ownership stays with the application.

`runner.read_inputs(commands)` returns the exact raw inputs expected by `runner.inference`. This supports held-out numerical replay without applying actuator targets or advancing physics. The training framework and original policy class are unnecessary when loading the exported bundle.

## Train, export, and redeploy Go2

Run these commands from the repository root. Training uses the existing 4,096-environment, 101-iteration defaults. Export and deployment use the CPU unless `--gpu` is passed.

```bash
python examples/locomotion/go2_train.py -e go2-leapp --seed 1
python examples/locomotion/go2_export.py -e go2-leapp --ckpt 100 -o out/go2_leapp
python examples/locomotion/go2_deploy.py --bundle out/go2_leapp/policy/policy.yaml
```

The exporter records five capture samples and validates 100 held-out transitions, including command changes and reset. Go2's previous-action feedback preserves its one-step actuation delay. The native reference and independent deployment each run 3,000 steps, spanning three 20-second episodes when the policy succeeds. Reports include actuator/feedback error, velocity tracking, and episode failures.

## Train, export, and redeploy a drone

Training uses the existing 8,192-environment, 301-iteration defaults.

```bash
python examples/drone/hover_train.py -e hover-leapp --seed 1
python examples/drone/hover_export.py -e hover-leapp --ckpt 300 -o out/hover_leapp
python examples/drone/hover_deploy.py --bundle out/hover_leapp/policy/policy.yaml
```

The complete model includes relative-target computation, frame transforms, observation clipping, action history, and conversion to propeller RPM. Two feedback tensors preserve the training example's observation-history update order. Native numerical replay uses the command snapshot that formed each observation because the training environment resamples reached targets after updating relative position.

The native reference and independent deployment each run 4,500 steps, spanning three 15-second episodes when successful, with target changes and reset. Reports include target tracking, minimum altitude, episode failures, actuator error in RPM, and action-feedback error.

Each exporter writes `validation.json`, the example's reset/metric configuration in `scene.json`, and logical-to-scene entity names in `bindings.json` beside the pipeline. It also calls Genesis's existing `Scene.export()` to write the authored scene and assets to `scene.gscene`. Standalone launchers use `Scene.load()` and build that scene, resolve entity names, and run `LeappDeployment`; they import neither the training environment nor the trainer or original policy class. Each launcher writes `rollout.json` under its output directory. Unsupported scene content follows the native serialization API's errors and warnings. General exporters and runners have no task-name dispatch or custom scene reconstruction.

Measured full-training and deployment results are recorded in [VALIDATION.md](VALIDATION.md).
