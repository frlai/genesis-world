# Training and deployment validation

Measured on 2026-11-17 with an NVIDIA RTX 4090, Python 3.12.11, PyTorch 2.15.0+cu132, LEAPP 0.8.1, RSL-RL 5.5.2, ONNX 1.23.2, ONNX Runtime 1.32.0, NumPy 2.5.5, and TensorBoard 2.21.0. ONNX inference used the CPU provider, including comparisons against CUDA eager inference. This report establishes numerical correctness and task behavior, not inference speed.

The upstream source was `08bad0a58a228014a1cad9a340da65892d406f1c`. Training was repeated after extracting the shared scene and inference helpers. The retained training logs include RSL-RL's source diff; the acceptance archive includes the tested example sources and their SHA-256 hashes. Physics, rewards, termination rules, action ordering, and training configurations were unchanged.

## Full training

Both final policies trained from scratch with seed 1 and the original defaults:

| Example | Environments | Iterations | Transitions | Final mean reward | Final mean episode length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Go2 | 4,096 | 101 | 9,928,704 | 15.20 | 1,001.00 |
| Drone | 8,192 | 301 | 246,579,200 | 2.96 | 1,472.47 |

All 1,818 Go2 TensorBoard scalar values across 18 series and all 5,117 drone values across 17 series were finite. Each series covers all 101 or 301 iterations. Configuration pickles, intermediate/final checkpoints, TensorBoard curves, and full stdout logs are retained outside git.

Final checkpoint SHA-256:

- Go2 `model_100.pt`: `b77bb9d2a6682b4778ec8c0e0dc227e7eeeccf3b375790a3e2e846506fc2ef9c`.
- Drone `model_300.pt`: `7df3d1ef62de07ec723d9456d6a32dc069a0688b5471b8d1fc033da78810c81b`.

The original upstream environments and the edited environments were also run side by side with identical seeded scenes and 20 changing action inputs in two environments per task. Observations had maximum absolute difference **0.0** for both tasks; rewards and termination flags matched.

## Export and held-out replay

Each trained policy was captured with five samples. Each CPU/GPU export then passed 100 held-out transitions with changing commands and an explicit reset halfway through. Raw state was shared between the native and exported computations; actuator targets and every feedback tensor were compared before advancing physics.

| Example / eager device | Actuator maximum absolute error | Feedback maximum absolute error |
| --- | ---: | ---: |
| Go2 / CPU | 7.1526e-7 rad | 2.6524e-6 |
| Go2 / CUDA | 9.5367e-7 rad | 3.6359e-6 |
| Drone / CPU | 0.010254 RPM | 8.3447e-7 |
| Drone / CUDA | 0.011719 RPM | 1.0431e-6 |

Assertions use zero relative tolerance. Go2 absolute bounds are 2e-6 rad for targets and 1e-5 for action feedback, respectively 2.1 and 2.8 times the largest measured errors. Drone bounds are 0.02 RPM and 2e-6 for action feedback, respectively 1.7 and 1.9 times the largest measured errors. These bounds cover measured CPU/CUDA differences while remaining small relative to actuator magnitudes.

Feedback validation exposed the Go2 training reset's deferred forward update: immediately after `env.reset()`, cached observations describe the initial pose while public getters can still expose the preceding pose. The export validation callback synchronizes the physical state using public `set_qpos(..., zero_velocity=True)` before comparing raw getter values. No training reset behavior changed. Drone replay retains the command snapshot used to form the cached relative-position observation and preserves both action-history slots.

## Independent closed-loop deployment

Fresh processes loaded the bundles and built scenes without the training environments or original policy classes. An import blocker rejected any import of `rsl_rl`, `go2_env`, `hover_env`, or either exporter. Both processes completed successfully. The launchers now also reject non-finite actuator targets and non-finite report metrics.

Each rollout exercised command changes halfway through each episode and reset at the original episode duration. Go2 used 0.5 then 0.25 m/s forward velocity; the drone used targets `(0, 0, 1)` then `(0.5, -0.5, 1)` m. Termination checks match the native task's angle/position/height criteria and scene error mask.

| Task | Steps / duration | Native tracking RMSE | Exported tracking RMSE | Native / exported minimum height | Completed / failed episodes |
| --- | --- | ---: | ---: | --- | --- |
| Go2 | 3,000 / 60 s | 0.17797388 m/s | 0.17797876 m/s | 0.29445943 / 0.29445896 m | 3 / 0 for both |
| Drone | 4,500 / 45 s | 0.10855213 m | 0.10855360 m | 0.94120061 / 0.94120067 m | 3 / 0 for both |

These task metrics use CPU native and deployment rollouts. CUDA native rollouts also completed three episodes with zero failures: Go2 tracking RMSE 0.17797353 m/s and drone tracking RMSE 0.10855366 m. Long trajectories are evaluated by task metrics; floating-point trajectory equality is not required.

## Reproduce and locate evidence

From the repository root, after the installation in [README.md](README.md):

```bash
python examples/locomotion/go2_train.py -e leapp_go2_final_20261117 --seed 1
python examples/drone/hover_train.py -e leapp_hover_final_20261117 --seed 1
python examples/locomotion/go2_export.py -e leapp_go2_final_20261117 -o out/leapp_go2_final_sync
python examples/drone/hover_export.py -e leapp_hover_final_20261117 -o out/leapp_hover_final
python examples/locomotion/go2_export.py --gpu -e leapp_go2_final_20261117 -o out/leapp_go2_gpu
python examples/drone/hover_export.py --gpu -e leapp_hover_final_20261117 -o out/leapp_hover_gpu
python examples/locomotion/go2_deploy.py --bundle out/leapp_go2_final_sync/policy/policy.yaml
python examples/drone/hover_deploy.py --bundle out/leapp_hover_final/policy/policy.yaml
```

The original evidence lives in `/home/frlai/workspaces/projs/genesis-world-examples`:

- `logs/leapp_{go2,hover}_final_20261117/`: configurations, checkpoints, curves, and source diffs.
- `out/leapp_go2_final_sync/` and `out/leapp_hover_final/`: CPU export bundles, validation reports, and deployment reports.
- `out/leapp_{go2,hover}_gpu/`: CUDA capture bundles and validation reports.
- `/tmp/genesis-leapp/examples/`: full training/export/deployment stdout, `training_evidence.json`, `compare_envs.py`, `compare_envs.json`, the standalone import-blocking harness, and tested-source snapshots.

Generated models and training artifacts are intentionally excluded from source control. This validation covers these two learned policies at batch size one and the documented fixed-shape contract; it does not establish arbitrary-scene reconstruction or hardware deployment.
