# Training and deployment validation

Measured on 2026-11-17 with an NVIDIA RTX 4090, Python 3.12.11, PyTorch 2.15.0+cu132, LEAPP 0.8.1, RSL-RL 5.5.2, ONNX 1.23.2, ONNX Runtime 1.32.0, NumPy 2.5.5, and TensorBoard 2.21.0. ONNX inference used the CPU provider, including comparisons against CUDA eager inference. This report establishes numerical correctness and task behavior, not inference speed.

The upstream source was `08bad0a58a228014a1cad9a340da65892d406f1c`. Full training was repeated at example revision `00fd4d7e`, with the native environment files unchanged except for one metadata import and a decorator. Stripping those additions reproduces both upstream files byte-for-byte, including comments; all 14 functions in each file also have identical signatures and body ASTs. The one-call exporter captures the original `step` around its physics boundary, preserving observation, history, policy, and actuator computations. No scene-construction or inference helper was extracted. Source snapshots and hashes are retained with the training artifacts.

## Full training

Both final policies trained from scratch with seed 1 and the original defaults:

| Example | Environments | Iterations | Transitions | Final mean reward | Final mean episode length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Go2 | 4,096 | 101 | 9,928,704 | 15.94 | 994.61 |
| Drone | 8,192 | 301 | 246,579,200 | 2.96 | 1,472.47 |

All 1,818 Go2 TensorBoard scalar values across 18 series and all 5,117 drone values across 17 series were finite. Each series covers all 101 or 301 iterations. Configuration pickles, intermediate/final checkpoints, TensorBoard curves, and full stdout logs are retained outside git.

Final checkpoint SHA-256:

- Go2 `model_100.pt`: `609b667178d7d2510801011d7fc4fdc7dfb8903d74b198c2e2f274e098ad7c14`.
- Drone `model_300.pt`: `7df3d1ef62de07ec723d9456d6a32dc069a0688b5471b8d1fc033da78810c81b`.

Genesis's existing `Scene.export()` / `Scene.load()` also passed a direct roundtrip check for both authored scenes. Entity types, ordered link/joint names, and timestep were preserved. After restoring initial pose and controller settings, 20 matched physics steps had maximum qpos error **0.0** for both tasks. The portable `.gscene` files include the assets; standalone launchers need neither the training environment nor a duplicate scene constructor.

## Export and held-out replay

Each trained policy was captured with five samples using its complete saved native configuration, including reward computations; the exporter does not clear or replace the reward configuration. Each CPU/GPU export then passed 100 held-out transitions with changing commands and an explicit reset halfway through. Raw state was shared between the native and exported computations; actuator targets and every feedback tensor were compared before advancing physics.

| Example / eager device | Actuator maximum absolute error | Feedback maximum absolute error |
| --- | ---: | ---: |
| Go2 / CPU | 2.3842e-7 rad | 1.4305e-6 |
| Go2 / CUDA | 4.7684e-7 rad | 1.4305e-6 |
| Drone / CPU | 0.010254 RPM | 8.3447e-7 |
| Drone / CUDA | 0.011719 RPM | 1.0431e-6 |

Assertions use zero relative tolerance. Go2 absolute bounds are 2e-6 rad for targets and 1e-5 for action feedback. These retain calibration across both trained checkpoint sets evaluated in this work, whose largest measured errors were 9.54e-7 rad and 3.64e-6 respectively; the current checkpoint has smaller errors. Drone bounds are 0.02 RPM and 2e-6 for action feedback, respectively 1.7 and 1.9 times the largest measured errors. These bounds cover measured CPU/CUDA differences while remaining small relative to actuator magnitudes.

Feedback validation exposed the Go2 training reset's deferred forward update: immediately after `env.reset()`, cached observations describe the initial pose while public getters can still expose the preceding pose. The export validation callback synchronizes the physical state using public `set_qpos(..., zero_velocity=True)` before comparing raw getter values. No training reset behavior changed. Both declared feedback attributes are captured at the physics boundary before the original post-step observation computations; the normal environment methods perform their original in-place history writes. Drone replay retains the command snapshot used to form the cached relative-position observation and preserves both action-history slots.

The original Go2 step also reads base position and linear velocity for reward bookkeeping. LEAPP reports those raw ports as unused by the policy computation; no reward computations or simulator stepping enter the exported ONNX model.

## Independent closed-loop deployment

Fresh processes loaded the bundles and used Genesis's native `Scene.load()` / `build()` without the training environments or original policy classes. An import blocker rejected any import of `rsl_rl`, `go2_env`, `hover_env`, or either exporter. Both processes completed successfully. The launchers now also reject non-finite actuator targets and non-finite report metrics.

Each rollout exercised command changes halfway through each episode and reset at the original episode duration. Go2 used 0.5 then 0.25 m/s forward velocity; the drone used targets `(0, 0, 1)` then `(0.5, -0.5, 1)` m. Termination checks match the native task's angle/position/height criteria and scene error mask.

| Task | Steps / duration | Native tracking RMSE | Exported tracking RMSE | Native / exported minimum height | Completed / failed episodes |
| --- | --- | ---: | ---: | --- | --- |
| Go2 | 3,000 / 60 s | 0.17566905 m/s | 0.17566913 m/s | 0.29403979 / 0.29403973 m | 3 / 0 for both |
| Drone | 4,500 / 45 s | 0.10855213 m | 0.10855360 m | 0.94120061 / 0.94120067 m | 3 / 0 for both |

These task metrics use CPU native and deployment rollouts. CUDA native rollouts also completed three episodes with zero failures: Go2 tracking RMSE 0.17566865 m/s and drone tracking RMSE 0.10855366 m. Long trajectories are evaluated by task metrics; floating-point trajectory equality is not required.

## General exporter and deployment checks

The final integrated native-export suite passed **17 tests in 124.14 seconds** with `--backend cpu --dev -n 2`. Six independent native-step deployment tests also passed on CUDA in 51.47 seconds. These exercise original inline step bodies, TensorDict observations, in-place history, repeated aliases, scalar commands, reset, physical actuator effects, native sampling, capture failure cleanup, and rejection of undeclared Genesis objects. Other focused exporter/deployment checks cover quaternion transforms, normalization, joint ordering, controller saturation, and control decimation. The earlier required control/API/tensor baseline passed 67 cases.

Source checks confirm metadata-only changes: all 14 functions in each example environment preserve their signatures, bodies, comments, and scene construction. The rigid/drone entity boundary annotations likewise preserve the upstream function bodies. Repository Ruff checks and formatting passed on the changed examples and documentation.

The native capture contract requires observation dependencies to come from decorated raw getters or declared environment input/state attributes after the physics boundary. Pre-physics derived locals reused afterward are unsupported. Termination decisions and scene-error masks are not policy inputs; native termination bookkeeping is pruned from the exported computation. Distinct cached tensor views sharing storage are rejected; repeated references to the same tensor are preserved. Ordinary custom objects remain shared read-only by contract, and undeclared Genesis objects cannot be accessed through the capture copy. These limits are explicit rather than reasons to freeze sample-dependent values silently.

Run the focused suite with the optional extra installed:

```bash
PYTHONPATH="$PWD" python -m pytest --backend cpu --dev tests/integration/test_leapp_export.py tests/integration/test_leapp_deployment.py
PYTHONPATH="$PWD" python -m pytest --backend gpu --dev tests/integration/test_leapp_export.py tests/integration/test_leapp_deployment.py
```

## Reproduce and locate evidence

From the repository root, after the installation in [README.md](README.md):

```bash
python examples/locomotion/go2_train.py -e leapp_go2_native_20261117 --seed 1
python examples/drone/hover_train.py -e leapp_hover_native_20261117 --seed 1
python examples/locomotion/go2_export.py -e leapp_go2_native_20261117 -o out/leapp_go2_native
python examples/drone/hover_export.py -e leapp_hover_native_20261117 -o out/leapp_hover_native
python examples/locomotion/go2_export.py --gpu -e leapp_go2_native_20261117 -o out/leapp_go2_native_gpu
python examples/drone/hover_export.py --gpu -e leapp_hover_native_20261117 -o out/leapp_hover_native_gpu
python examples/locomotion/go2_deploy.py --bundle out/leapp_go2_native/policy/policy.yaml
python examples/drone/hover_deploy.py --bundle out/leapp_hover_native/policy/policy.yaml
```

A consolidated copy is retained under `/home/frlai/workspaces/projs/genesis-world/out/leapp_validation/environment_export/`: `training/` contains checkpoints, configurations and curves; `bundles/` contains CPU/CUDA exports and rollout reports; `logs/` contains agent validation logs; `tested_sources/` preserves the exact example sources. Integrated test logs and dependency versions are at the archive root.

The original evidence lives in `/home/frlai/workspaces/projs/genesis-world-examples`:

- `logs/leapp_{go2,hover}_native_20261117/`: configurations, checkpoints, curves, and source diffs.
- `out/leapp_go2_native/` and `out/leapp_hover_native/`: CPU export bundles, validation reports, and deployment reports.
- `out/leapp_{go2,hover}_native_gpu/`: CUDA capture bundles and validation reports.
- `/tmp/genesis-leapp/examples/`: full training/export/deployment stdout, `training_native_evidence.json`, `native_environment_source_parity.json`, `probe_native_scene_roundtrip.py`, `native_scene_roundtrip.json`, the standalone import-blocking harness, and `native_tested_sources/` snapshots.

Generated models and training artifacts are intentionally excluded from source control. This validation covers these two learned policies at batch size one and the documented fixed-shape contract; it uses the existing native scene format for its supported authored scenes; it does not add arbitrary-scene reconstruction or establish hardware deployment.
