# Training and deployment validation

Measured on 2026-11-17 with an NVIDIA RTX 4090, Python 3.12.11, PyTorch 2.15.0+cu132, LEAPP 0.8.1, RSL-RL 5.5.2, ONNX 1.23.2, ONNX Runtime 1.32.0, NumPy 2.5.5, and TensorBoard 2.21.0. ONNX inference used the CPU provider, including comparisons against CUDA eager inference. This report establishes numerical correctness and task behavior, not inference speed.

The upstream source was `08bad0a58a228014a1cad9a340da65892d406f1c`. Training was repeated after replacing handwritten export callbacks with annotated original environment methods at example revision `f8462b1a`. The one-call `export_environment(env, policy, output_dir)` path invokes the original buffer updates, observation construction, history updates, policy callable, and actuator processing. The retained training logs include RSL-RL's source diff; the acceptance archive includes the tested example sources and their SHA-256 hashes. Physics, rewards, termination rules, action ordering, and training configurations were unchanged.

## Full training

Both final policies trained from scratch with seed 1 and the original defaults:

| Example | Environments | Iterations | Transitions | Final mean reward | Final mean episode length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Go2 | 4,096 | 101 | 9,928,704 | 15.94 | 994.61 |
| Drone | 8,192 | 301 | 246,579,200 | 2.96 | 1,472.47 |

All 1,818 Go2 TensorBoard scalar values across 18 series and all 5,117 drone values across 17 series were finite. Each series covers all 101 or 301 iterations. Configuration pickles, intermediate/final checkpoints, TensorBoard curves, and full stdout logs are retained outside git.

Final checkpoint SHA-256:

- Go2 `model_100.pt`: `dda662fcc3c2808a78d1cbad79ff24e13af74ef3b46e91977f721c4b467e088f`.
- Drone `model_300.pt`: `7df3d1ef62de07ec723d9456d6a32dc069a0688b5471b8d1fc033da78810c81b`.

The original upstream environments and the edited environments were also run side by side with identical seeded scenes and 20 changing action inputs in two environments per task, including an explicit reset halfway through. Observations had maximum absolute difference **0.0** for both tasks; rewards, termination flags, actions, and last-action buffers matched.

## Export and held-out replay

Each trained policy was captured with five samples. Each CPU/GPU export then passed 100 held-out transitions with changing commands and an explicit reset halfway through. Raw state was shared between the native and exported computations; actuator targets and every feedback tensor were compared before advancing physics.

| Example / eager device | Actuator maximum absolute error | Feedback maximum absolute error |
| --- | ---: | ---: |
| Go2 / CPU | 2.3842e-7 rad | 9.5367e-7 |
| Go2 / CUDA | 3.5763e-7 rad | 1.4305e-6 |
| Drone / CPU | 0.010254 RPM | 8.3447e-7 |
| Drone / CUDA | 0.011719 RPM | 1.0431e-6 |

Assertions use zero relative tolerance. Go2 absolute bounds are 2e-6 rad for targets and 1e-5 for action feedback. These retain calibration across both trained checkpoint sets evaluated in this work, whose largest measured errors were 9.54e-7 rad and 3.64e-6 respectively; the current checkpoint has smaller errors. Drone bounds are 0.02 RPM and 2e-6 for action feedback, respectively 1.7 and 1.9 times the largest measured errors. These bounds cover measured CPU/CUDA differences while remaining small relative to actuator magnitudes.

Feedback validation exposed the Go2 training reset's deferred forward update: immediately after `env.reset()`, cached observations describe the initial pose while public getters can still expose the preceding pose. The export validation callback synchronizes the physical state using public `set_qpos(..., zero_velocity=True)` before comparing raw getter values. No training reset behavior changed. Both declared feedback attributes are captured at the pre-observation phase; the normal environment methods perform their original in-place history writes. Drone replay retains the command snapshot used to form the cached relative-position observation and preserves both action-history slots.

The original Go2 buffer update also reads base position and linear velocity for reward bookkeeping. LEAPP reports those raw ports as unused by the policy computation; no reward computations or simulator stepping enter the exported ONNX model.

## Independent closed-loop deployment

Fresh processes loaded the bundles and built scenes without the training environments or original policy classes. An import blocker rejected any import of `rsl_rl`, `go2_env`, `hover_env`, or either exporter. Both processes completed successfully. The launchers now also reject non-finite actuator targets and non-finite report metrics.

Each rollout exercised command changes halfway through each episode and reset at the original episode duration. Go2 used 0.5 then 0.25 m/s forward velocity; the drone used targets `(0, 0, 1)` then `(0.5, -0.5, 1)` m. Termination checks match the native task's angle/position/height criteria and scene error mask.

| Task | Steps / duration | Native tracking RMSE | Exported tracking RMSE | Native / exported minimum height | Completed / failed episodes |
| --- | --- | ---: | ---: | --- | --- |
| Go2 | 3,000 / 60 s | 0.16484628 m/s | 0.16484610 m/s | 0.29390013 / 0.29389995 m | 3 / 0 for both |
| Drone | 4,500 / 45 s | 0.10855213 m | 0.10855360 m | 0.94120061 / 0.94120067 m | 3 / 0 for both |

These task metrics use CPU native and deployment rollouts. CUDA native rollouts also completed three episodes with zero failures: Go2 tracking RMSE 0.16484629 m/s and drone tracking RMSE 0.10855366 m. Long trajectories are evaluated by task metrics; floating-point trajectory equality is not required.

## General exporter and deployment checks

The six exporter cases passed on CPU and CUDA simulation, including unbatched and batched tensors, scalar commands, command-free pipelines, joint permutations, quaternion transforms, normalization, explicit feedback, reset, failed-capture cleanup, and ordinary Genesis imports with LEAPP blocked. Maximum observed target error was 2.98e-8 and feedback error was 1.19e-7; the assertions use 1e-7 and 2e-7 respectively. A deliberately halved exported target failed the held-out comparison.

The four deployment cases passed on CPU and CUDA simulation with Genesis debug checks enabled. They verify physical actuator effects, controller gains and saturation limits, command changes, feedback reset, and repeated propeller writes across control decimation. Maximum normalized physics error was 1.40e-8 against a 5e-8 bound. Intentionally applying propeller targets only once per control period caused the analytical velocity check to fail.

The final environment-level suite passed **14 tests in 51.39 seconds** with `--backend cpu --dev -n 2`. The three independent environment cases also passed on CUDA with the final snapshot implementation. They cover original method execution, in-place history updates, repeated tensor references, external commands/time, semantics, reset, source isolation, and failed capture. Their largest measured actuator and feedback errors were 1.49e-8 and 2.98e-8, both checked at 5e-8.

A cached distinct tensor view exposed a 0.01415 replay error despite passing export validation. The exporter now rejects shared-storage views with an actionable error before capture and if they appear between samples. The negative tests verify both paths. Both trained examples passed another 100-step replay with the guard enabled, with unchanged errors. Create views inside the original inference methods so their relationship is traced.

An AST comparison against upstream confirmed that all 159 rigid-entity functions and 11 drone-entity functions preserve their signatures, bodies, and existing decorators. Only metadata annotations and imports were added. Before adding the environment-level entry point, the required control/API/tensor-utility baseline passed 67 cases and the combined exporter/deployment/control/API suite passed 39 cases. The core engine remained unchanged in this update. Complete-diff Ruff checks, formatting, documentation snippet syntax, and local links passed. Both newly copied learned-policy bundles loaded and ran from a new directory with training imports blocked.

Run the focused suite with the optional extra installed:

```bash
PYTHONPATH="$PWD" python -m pytest --backend cpu --dev tests/integration/test_leapp_export.py tests/integration/test_leapp_deployment.py tests/integration/test_leapp_environment.py
PYTHONPATH="$PWD" python -m pytest --backend gpu --dev tests/integration/test_leapp_export.py tests/integration/test_leapp_deployment.py tests/integration/test_leapp_environment.py
```

## Reproduce and locate evidence

From the repository root, after the installation in [README.md](README.md):

```bash
python examples/locomotion/go2_train.py -e leapp_go2_environment_20261117 --seed 1
python examples/drone/hover_train.py -e leapp_hover_environment_20261117 --seed 1
python examples/locomotion/go2_export.py -e leapp_go2_environment_20261117 -o out/leapp_go2_environment
python examples/drone/hover_export.py -e leapp_hover_environment_20261117 -o out/leapp_hover_environment
python examples/locomotion/go2_export.py --gpu -e leapp_go2_environment_20261117 -o out/leapp_go2_environment_gpu
python examples/drone/hover_export.py --gpu -e leapp_hover_environment_20261117 -o out/leapp_hover_environment_gpu
python examples/locomotion/go2_deploy.py --bundle out/leapp_go2_environment/policy/policy.yaml
python examples/drone/hover_deploy.py --bundle out/leapp_hover_environment/policy/policy.yaml
```

A consolidated copy is retained under `/home/frlai/workspaces/projs/genesis-world/out/leapp_validation/environment_export/`: `training/` contains checkpoints, configurations and curves; `bundles/` contains CPU/CUDA exports and rollout reports; `logs/` contains agent validation logs; `tested_sources/` preserves the exact example sources. Integrated test logs and dependency versions are at the archive root.

The original evidence lives in `/home/frlai/workspaces/projs/genesis-world-examples`:

- `logs/leapp_{go2,hover}_environment_20261117/`: configurations, checkpoints, curves, and source diffs.
- `out/leapp_go2_environment/` and `out/leapp_hover_environment/`: CPU export bundles, validation reports, and deployment reports.
- `out/leapp_{go2,hover}_environment_gpu/`: CUDA capture bundles and validation reports.
- `/tmp/genesis-leapp/examples/`: full training/export/deployment stdout, `training_environment_evidence.json`, `compare_environment_methods_reset.py`, `compare_environment_methods_reset.json`, the standalone import-blocking harness, and `environment_tested_sources/` snapshots.

Generated models and training artifacts are intentionally excluded from source control. This validation covers these two learned policies at batch size one and the documented fixed-shape contract; it does not establish arbitrary-scene reconstruction or hardware deployment.
