# Autonomous Drone Navigation with Deep Reinforcement Learning

This project studies visual autonomous drone navigation in Microsoft AirSim for the COMP9444 Neural Networks and Deep Learning project. It compares DQN, PPO trained from scratch, and PPO with curriculum learning under controlled interaction budgets.

The final experimental scope uses two environments:

- Blocks as the simple controlled baseline scene.
- AirSimNH as the complex scene.

## RL Task

At every environment step, the agent receives:

- A normalised `84 x 84` AirSim depth image.
- Relative target position `(target - drone position)`.
- Current linear velocity `(vx, vy, vz)`.

The six discrete actions are:

| Action | Command |
|---:|---|
| 0 | Forward |
| 1 | Left |
| 2 | Right |
| 3 | Up |
| 4 | Down |
| 5 | Hover |

The current agent does not build a map or store explicit obstacle coordinates. It learns a mapping from the current depth image and navigation state to an action. Collision information is used for reward and termination, not included directly in the observation.

Current reward configuration:

| Component | Value |
|---|---:|
| Step penalty | `-0.05` |
| Progress reward scale | `2.0` |
| Goal reward | `+100` |
| Collision penalty | `-100` |
| Altitude violation penalty | `-100` |
| Altitude hold penalty | `-0.25 * abs(z - target_z)` per step |
| Altitude boundary margin | `1.5 m` |
| Boundary proximity penalty | up to `-1.0` per metre inside the margin |
| Timeout penalty | `-25` |
| Goal radius | `2 m` |
| Valid NED altitude | `-10 <= z <= -1` |

AirSim uses NED coordinates, so a more negative `z` value means a greater altitude.

PPO internally multiplies all environment rewards by `0.1` before critic and GAE calculations. This does not change which policy is optimal, but it prevents the `+/-100` terminal rewards from producing very large value gradients.

## Stable PPO Revision

The first AirSimNH comparison found that PPO Scratch and PPO Curriculum both failed at 45,000 interactions. Offline checkpoint inspection showed that the shared `Tanh` representation was almost completely saturated:

| Checkpoint | Hidden activation saturation |
|---|---:|
| PPO Scratch 5k | `100%` |
| PPO Scratch 45k | `100%` |
| Curriculum Stage 2 final | `82.4%` |
| Curriculum Stage 3 final | `99.8%` |

The saturated networks were almost insensitive to the depth image and target state. The stable revision therefore adds:

- Per-sample feature normalisation immediately before the shared `Tanh`.
- Orthogonal CNN/hidden initialisation and a small actor-output initialisation.
- PPO reward scaling of `0.1`.
- Huber critic loss instead of unbounded value MSE.
- Activation saturation, explained variance, and maximum action-probability diagnostics.
- `ppo_best.pt`, selected by rolling success, unsafe rate, and final distance.
- Optimizer reset by default when a curriculum stage changes target.
- Deterministic Stage 1 and Stage 2 curriculum gates.
- Continuous altitude-hold/boundary penalties and an explicit timeout penalty.

Old checkpoints remain valid for reproducing their existing evaluation behavior, but `train_ppo.py` deliberately refuses to resume them into a stable run. Start the stable curriculum from Stage 1 so that every transferred checkpoint uses the corrected representation and reward scale.

Run the PPO regression tests without starting AirSim:

```powershell
python -B -m unittest discover -s tests -v
```

## Project Structure

```text
rl_drone_navigation/
  README.md
  requirements.txt
  airsim_settings_sample.json
  run_ppo_training.ps1
  run_comparison_experiment.ps1
  src/
    airsim_drone_env.py
    check_setup.py
    dqn_agent.py
    evaluate.py
    experiment_paths.py
    list_scenes.py
    manual_control.py
    plot_results.py
    ppo_agent.py
    smoke_test_env.py
    summarize_experiments.py
    train_dqn.py
    train_ppo.py
  notebooks/
    COMP9444_AirSim_Drone_Navigation.ipynb
  tests/
    test_ppo_agent.py
  experiments/
    <scenario>/
      dqn/<run-name>/
      ppo/<run-name>/
```

## Setup

Recommended environment:

```powershell
conda create -n airsim-rl python=3.10 -y
conda activate airsim-rl
cd D:\AirSim\rl_drone_navigation
pip install -r requirements.txt
python -m ipykernel install --user --name airsim-rl --display-name "Python (airsim-rl)"
```

AirSim uses the old `msgpack-rpc-python` package. If importing AirSim fails with `No module named 'tornado.platform.auto'`, install the compatible versions:

```powershell
pip install "tornado==4.5.3" "ipykernel==5.5.6" "jupyter-client==7.1.2"
```

Start a scene, wait for it to load, and test the connection:

```powershell
D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe
python src\check_setup.py --connect
```

List installed scene executables:

```powershell
python src\list_scenes.py
```

`--scenario` is an output label. It does not switch the Unreal scene. The correct scene executable must be running before a direct training command is used.

## Manual Route Selection

Start the required scene, then run:

```powershell
conda activate airsim-rl
cd D:\AirSim\rl_drone_navigation
python src\manual_control.py
```

Controls:

| Key | Action |
|---|---|
| `W/S` | Forward/backward |
| `A/D` | Left/right |
| `R/F` | Up/down |
| `Q/E` | Rotate left/right |
| `P` | Print and save the current coordinates |
| `H` | Hover and print coordinates |
| `L` | Land and exit |
| `Esc` | Hover and exit |

Saved coordinates are appended to `results/selected_targets.txt`.

The RL action space has no backward or yaw action. When checking whether an RL route is feasible, avoid relying on `S`, `Q`, or `E`.

Visible foliage is not automatically a physical obstacle. Unreal assets only generate AirSim collisions when their collision geometry and collision responses are enabled. Use trunks, walls, buildings, rocks, or other objects that have been verified with `simGetCollisionInfo()` for obstacle-avoidance experiments.

## Final Comparison Design

The experiment has three methods:

| Method | Training task | Total budget |
|---|---|---:|
| DQN Scratch | Train directly on the final target | `45,000 steps` |
| PPO Scratch | Train directly on the final target | `45,000 steps` |
| PPO Curriculum | Three progressively harder targets | `45,000 steps` |

This design supports two controlled comparisons:

1. DQN Scratch versus PPO Scratch measures the algorithm difference.
2. PPO Scratch versus PPO Curriculum measures the effect of curriculum learning.

All three methods use the same final task, start position, observation, action space, reward, maximum episode length, interaction budget, seeds, and deterministic evaluation procedure.

### Why Total Steps Are Used

`max_steps * episodes` is only a theoretical upper limit. Episodes can finish early because of success, collision, or altitude violation. Therefore, equal episode counts do not guarantee equal training data.

The `--total-steps` option counts new interactions in the current run:

```text
one step = observe -> choose action -> move drone -> receive reward
```

For resumed PPO curriculum stages, the budget is reset for each stage. For example, Stage 3 receives 30,000 new interactions even though the checkpoint already contains the 15,000 interactions from Stages 1 and 2.

## Confirmed AirSimNH Route

The approved AirSimNH coordinates use a common safe altitude of `z=-3.0`:

```text
Start   = (85.413,  -15.334, -3.0)
Stage 1 = (95.190,  -14.491, -3.0)
Stage 2 = (107.635, -10.842, -3.0)
Stage 3 = (117.756, -19.034, -3.0)
```

Distances measured from the common start:

| Stage | Approximate distance | New interactions | Max steps per episode |
|---|---:|---:|---:|
| Stage 1 | `9.81 m` | `5,000` | `70` |
| Stage 2 | `22.67 m` | `10,000` | `110` |
| Stage 3 | `32.55 m` | `30,000` | `150` |
| Curriculum total |  | `45,000` |  |

DQN Scratch and PPO Scratch train directly from the common start to Stage 3 for 45,000 interactions with `max_steps=150`.

### Coordinate Smoke Test

Before formal training:

```powershell
python src\smoke_test_env.py `
  --steps 3 --action 5 --require-clean `
  --start-x 85.413 --start-y -15.334 --start-z -3.0 `
  --target-x 117.756 --target-y -19.034 --target-z -3.0
```

The initial distance should be approximately `32.55 m`, spawn error should remain below `0.75 m`, and no new collision or altitude violation should occur.

## Running the Comparison

The recommended entry point is:

```powershell
.\run_comparison_experiment.ps1
```

The configuration section at the top of the script contains:

- Scene label and executable.
- Seed.
- A run tag that keeps stable experiments separate from legacy outputs.
- Method enable/disable switches.
- Confirmed start and target coordinates.
- Stage and scratch interaction budgets.
- PPO hyperparameters.
- Stable PPO reward scale, value loss, best-model window, and curriculum gates.
- The last curriculum stage to run (`1`, `2`, or `3`).
- Evaluation episode counts.
- Automatic scene startup and shutdown settings.

Default method switches:

```powershell
$RunTag = "stable_v2"
$RunDqnScratch = $true
$RunPpoScratch = $true
$RunPpoCurriculum = $true
$CurriculumLastStage = 3
```

The script performs the following workflow:

1. Rejects any run whose output directory already exists.
2. Starts the configured AirSim scene if the RPC port is not open.
3. Runs a clean-spawn hover smoke test.
4. Trains each enabled method sequentially.
5. Saves checkpoints every 5,000 new interactions.
6. Saves `ppo_best.pt` from rolling training performance.
7. Evaluates the Stage 1/2 best model and stops if a curriculum gate fails.
8. Transfers best PPO weights while resetting the optimizer between stages.
9. Evaluates the comparable 30,000-step checkpoints.
10. Evaluates final and best PPO checkpoints at 45,000 steps.
11. Records the duration of every training run and curriculum stage.
12. Closes the configured AirSim scene in a `finally` block, including after failures.

To run only one method on a machine, disable the other two switches. Example for DQN only:

```powershell
$RunDqnScratch = $true
$RunPpoScratch = $false
$RunPpoCurriculum = $false
```

Never reuse a run name. Change `$Seed` or `$RunTag` before starting a new experiment. The stable defaults use `$RunTag = "stable_v2"`, so the original failed runs remain untouched.

### First Stable Validation Run

Do not immediately repeat all three 45k experiments. First run only the corrected Curriculum Stage 1:

```powershell
$RunTag = "stable_v2_stage1_pilot"
$RunDqnScratch = $false
$RunPpoScratch = $false
$RunPpoCurriculum = $true
$CurriculumLastStage = 1
$RequireCurriculumGates = $true
```

Then run:

```powershell
.\run_comparison_experiment.ps1
```

The runner trains 5,000 interactions, evaluates `ppo_best.pt` for 20 deterministic episodes, records the gate result, and closes AirSim. Continue to a full experiment only when all of the following hold:

| Stage 1 pilot check | Required value |
|---|---:|
| Deterministic success rate | `>= 80%` |
| Collision plus altitude unsafe rate | `<= 20%` |
| `activation_saturation` | `< 5%` throughout training |
| Value loss | finite, without sustained growth |
| Final altitude | remains inside `-10 <= z <= -1` |

If the gate passes, use a fresh tag such as `stable_v2_full`, restore `$CurriculumLastStage = 3`, and run the full comparison. A failed gate is a useful diagnostic result; do not bypass it by setting `$RequireCurriculumGates = $false` until the failure has been inspected.

## Checkpoints and Fair Comparisons

Scratch checkpoints are saved at:

```text
experiments/<scenario>/dqn/scratch_33m_45k_seed7_stable_v2/models/dqn_step_0030000.pt
experiments/<scenario>/dqn/scratch_33m_45k_seed7_stable_v2/models/dqn_final.pt

experiments/<scenario>/ppo/scratch_33m_45k_seed7_stable_v2/models/ppo_step_0030000.pt
experiments/<scenario>/ppo/scratch_33m_45k_seed7_stable_v2/models/ppo_best.pt
experiments/<scenario>/ppo/scratch_33m_45k_seed7_stable_v2/models/ppo_final.pt
```

For PPO Curriculum, the first 15,000 interactions are collected in Stages 1 and 2. Therefore, the fair 30,000-total-step checkpoint is 15,000 interactions into Stage 3:

```text
experiments/<scenario>/ppo/curriculum_stage03_33m_30k_seed7_stable_v2/models/ppo_step_0015000.pt
```

The final curriculum model has received:

```text
5,000 + 10,000 + 30,000 = 45,000 total interactions
```

The primary final comparison is at 45,000 total interactions. The 30,000-step comparison is used to discuss sample efficiency.

## Direct Training Commands

The comparison runner is preferred, but the training scripts can also be called directly after AirSim is started.

DQN Scratch:

```powershell
python src\train_dqn.py `
  --scenario airsimnh --run-name scratch_33m_45k_seed7_stable_v2 `
  --total-steps 45000 --max-steps 150 `
  --start-x 85.413 --start-y -15.334 --start-z -3.0 `
  --target-x 117.756 --target-y -19.034 --target-z -3.0 `
  --checkpoint-every-steps 5000 --seed 7
```

PPO Scratch:

```powershell
python src\train_ppo.py `
  --scenario airsimnh --run-name scratch_33m_45k_seed7_stable_v2 `
  --total-steps 45000 --max-steps 150 --rollout-steps 500 `
  --start-x 85.413 --start-y -15.334 --start-z -3.0 `
  --target-x 117.756 --target-y -19.034 --target-z -3.0 `
  --learning-rate 1e-4 --batch-size 64 --update-epochs 4 `
  --reward-scale 0.1 --value-loss huber `
  --best-window 20 --best-min-episodes 20 `
  --checkpoint-every-steps 5000 --seed 7
```

The curriculum commands are generated by `run_comparison_experiment.ps1`. Stage 2 and Stage 3 load the previous `ppo_best.pt` weights and reset Adam state. Pass `--resume-optimizer` only for a true continuation of the same target and reward configuration, not for a curriculum stage change.

The older `run_ppo_training.ps1` remains available for a single episode-budget PPO run. It is not the preferred runner for the controlled DQN/PPO comparison.

## Seeds

The formal seed set is:

```text
7, 17, 27
```

Seeds control:

- Neural-network initialisation.
- PPO action sampling and minibatch order.
- DQN epsilon exploration.
- DQN replay-buffer sampling.
- Python, NumPy, and PyTorch random-number generation.

Seeds do not change the scene, start point, target point, obstacles, reward, or interaction budget. AirSim physics and GPU execution may still prevent bit-for-bit reproducibility.

Run every method with every seed in both scenes:

| Scene | DQN Scratch | PPO Scratch | PPO Curriculum |
|---|---|---|---|
| Blocks | Seeds 7/17/27 | Seeds 7/17/27 | Seeds 7/17/27 |
| AirSimNH | Seeds 7/17/27 | Seeds 7/17/27 | Seeds 7/17/27 |

Blocks should use separately calibrated coordinates with approximately the same 10 m, 23 m, and 33 m distances. All methods within Blocks must use exactly the same Blocks coordinates.

## Three-Machine Allocation

A balanced allocation is:

| Machine | Jobs |
|---|---|
| Machine A | Blocks seeds 7 and 17, all three methods |
| Machine B | AirSimNH seeds 7 and 17, all three methods |
| Machine C | Blocks seed 27 and AirSimNH seed 27, all three methods |

Each machine receives six complete method/seed jobs. Install both scenes on Machine C or reassign its jobs while keeping the same code revision and configuration.

Run only one AirSim process and one training process at a time on each machine.

## Evaluation

Deterministic evaluation uses:

```powershell
python src\evaluate.py `
  --algorithm ppo --scenario airsimnh `
  --run-name scratch_33m_45k_seed7_stable_v2_best `
  --model experiments\airsimnh\ppo\scratch_33m_45k_seed7_stable_v2\models\ppo_best.pt `
  --episodes 50 --max-steps 150 `
  --start-x 85.413 --start-y -15.334 --start-z -3.0 `
  --target-x 117.756 --target-y -19.034 --target-z -3.0
```

During evaluation:

- DQN selects the action with the largest Q-value and does not use epsilon exploration.
- PPO selects the action with the largest policy logit instead of sampling.

### Inference Video Recording

Use `run_inference_recording.ps1` to launch a scene, load one checkpoint, record inference, and close the scene automatically. Its configuration section controls the model, route, policy mode, number of attempts, and video settings.

```powershell
.\run_inference_recording.ps1
```

The current default records the Stage 2 stable PPO checkpoint on the 23 m route. It uses `stochastic` mode and stops after the first successful attempt because this checkpoint achieved high sampled-policy training success but failed its deterministic gate. This recording demonstrates the sampled policy; it must not be reported as deterministic evaluation performance.

Set `$PolicyMode = "deterministic"` to record deployment-style PPO inference, or use DQN with deterministic mode. Each invocation creates a timestamped directory beside the selected run:

```text
experiments/<scenario>/<algorithm>/<run-name>/recordings/<mode>_<timestamp>/
  episode_001_success.mp4
  episode_001_success_preview.jpg
  inference_steps.csv
  inference_episodes.csv
  inference_summary.json
```

The MP4 contains the drone's front RGB camera, current action, cumulative reward, distance, position, outcome, and a colour inset of the depth observation supplied to the policy. This is an AirSim camera recording rather than a desktop capture of the Unreal Engine window.

Required metrics:

| Metric | Desired direction |
|---|---|
| Success rate | Higher |
| Collision rate | Lower |
| Altitude violation rate | Lower |
| Average reward | Higher |
| Average final distance | Lower |
| Average successful steps | Lower when success remains high |
| Steps to reach 50%/80% success | Lower |
| Path length | Lower when success remains high |
| Minimum observed depth | Higher safety clearance |
| Dominant action fraction | Avoid one-action collapse |

Report `mean +/- standard deviation` across seeds. Learning-curve x-axes should use environment interactions rather than episodes.

Stable PPO update logs also contain:

| Diagnostic | Interpretation |
|---|---|
| `activation_saturation` | Fraction of shared hidden values with `abs(value) > 0.99`; target `< 0.05` |
| `activation_abs_mean` | Magnitude of the shared representation after normalisation |
| `max_action_probability` | Mean confidence in the most probable action |
| `explained_variance` | Critic fit quality; higher is better, but task metrics remain primary |

Recommended performance targets are:

| Evaluation setting | Success | Collision | Altitude violation |
|---|---:|---:|---:|
| Fixed start/target sanity test | `>= 90%` | `<= 5%` | `<= 2%` |
| AirSimNH final route | `>= 70%` initially; aim for `>= 85%` | `<= 15%` | `<= 5%` |
| Perturbed start or unseen route | `>= 60%` | `<= 20%` | `<= 5%` |

The current final target still supplies only one end position. If stable PPO passes Stage 1 and Stage 2 but repeatedly fails Stage 3, the next experiment should represent the manually verified route as sequential waypoints or use simulator-only route-progress shaping. This tests whether Euclidean distance reward is penalising the necessary lateral detour; it should be treated as a separate reward/task ablation rather than mixed into the first stable PPO validation.

## Experiment Outputs

Each named run contains:

```text
experiments/<scenario>/<algorithm>/<run-name>/
  metadata.json
  models/
    <algorithm>_final.pt
    ppo_best.pt               # PPO only
    step checkpoints
  results/
    training_log.csv
    training_curves.png
    training_summary.json
    evaluation_log.csv
    ppo_update_log.csv        # PPO only
```

Each successful DQN/PPO training run writes `training_summary.json` containing:

```text
started_at
completed_at
elapsed_seconds
elapsed_hours
completed_episodes
new_environment_interactions
requested_total_steps
agent_cumulative_steps        # resumed PPO runs
best_model                    # PPO only
best_window_metrics           # PPO only
```

The comparison runner also writes one timing table per scenario and seed:

```text
experiments/<scenario>/comparison_seed7_stable_v2_training_times.csv
```

Its rows separately identify:

```text
DQN Scratch final task
PPO Scratch final task
PPO Curriculum Stage 1
PPO Curriculum Stage 2
PPO Curriculum Stage 3
```

The CSV includes algorithm, method, stage, run name, completion status, timestamps, elapsed seconds, and elapsed hours. Evaluation time is excluded. Failed training commands are recorded with `status=failed` before the runner closes AirSim.

Collect scenario directories from all machines into one central `experiments` directory. Scenario, algorithm, run name, and seed prevent path conflicts.

Generate the summary table:

```powershell
python src\summarize_experiments.py
```

This creates `experiments\summary.csv`. It is generated output and can be recreated from the individual `evaluation_log.csv` files.

## Notebook and Rubric Evidence

Open:

```text
D:\AirSim\rl_drone_navigation\notebooks\COMP9444_AirSim_Drone_Navigation.ipynb
```

The final notebook should include:

1. Problem statement and motivation for visual autonomous navigation.
2. AirSim task, observation, action, reward, and termination definitions.
3. Depth-image examples and analysis of scene/task difficulty.
4. DQN, PPO, and curriculum-learning methods with sources and implementation details.
5. DQN-versus-PPO and PPO-Scratch-versus-Curriculum controlled comparisons.
6. Blocks-versus-AirSimNH comparison.
7. Training curves, final metric tables, and mean/standard deviation over seeds.
8. Discussion of collision geometry, fixed routes, discrete actions, lack of memory/SLAM, compute limits, and simulator-to-real transfer.

These items map directly to the COMP9444 rubric sections for RL task exploration, models and methods, results, discussion, and reproducibility.
