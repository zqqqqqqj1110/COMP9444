# Autonomous Drone Navigation with Deep Reinforcement Learning

This project scaffold is designed for the COMP9444 AirSim drone navigation task. It defines an AirSim-based reinforcement learning environment, DQN and PPO baselines, training/evaluation scripts, plotting utilities, and a rubric-aligned notebook skeleton.

## Project Structure

```text
rl_drone_navigation/
  README.md
  requirements.txt
  airsim_settings_sample.json
  src/
    airsim_drone_env.py
    check_setup.py
    dqn_agent.py
    experiment_paths.py
    evaluate.py
    plot_results.py
    ppo_agent.py
    train_dqn.py
    train_ppo.py
  notebooks/
    COMP9444_AirSim_Drone_Navigation.ipynb
  experiments/
    blocks/
      dqn/
        models/
        results/
      ppo/
        models/
        results/
```

## Setup

The current machine has Python available, but the RL dependencies are not installed yet. AirSim's Python package is usually easier to run in a Python 3.10 or 3.11 environment than in a very new Python version.

Recommended setup:

```powershell
conda create -n airsim-rl python=3.10 -y
conda activate airsim-rl
cd D:\AirSim\rl_drone_navigation
pip install -r requirements.txt
python -m ipykernel install --user --name airsim-rl --display-name "Python (airsim-rl)"
```

AirSim's Python client uses `msgpack-rpc-python`, which is an old RPC library. The requirements file pins Tornado and the notebook kernel packages to versions that still work with AirSim. If you already installed the dependencies before this pin was added, run:

```powershell
pip install "tornado==4.5.3" "ipykernel==5.5.6" "jupyter-client==7.1.2"
```

Then start the AirSim Blocks simulator:

```powershell
D:\AirSim\Blocks\WindowsNoEditor\Blocks.exe
```

In a second terminal, check the connection:

```powershell
cd D:\AirSim\rl_drone_navigation
conda activate airsim-rl
python src\check_setup.py --connect
```

If the connection check shows `Retry connection over the limit` or `WSAECONNREFUSED`, the Python dependencies are working but the simulator is not listening on AirSim's RPC port. Keep the Blocks window open and wait until the 3D scene has fully loaded before running the connection check again.

Run a short environment smoke test:

```powershell
python src\smoke_test_env.py --steps 5
```

## Experiment Layout

Each AirSim scene and algorithm has its own output directory:

```text
experiments/<scenario>/<algorithm>/
  metadata.json
  models/
    <algorithm>_final.pt
    checkpoint files
  results/
    training_log.csv
    training_curves.png
    evaluation_log.csv
```

Examples:

```text
experiments/blocks/dqn/
experiments/blocks/ppo/
experiments/forest/dqn/
experiments/forest/ppo/
```

Important: `--scenario` is the experiment label used for output folders. It does not automatically switch the Unreal/AirSim map. Start the correct AirSim environment first, wait for the 3D scene to load, then run training/evaluation with the matching scenario name.

AirSim environments do not have to be stored below the Blocks directory. For example:

```text
D:\AirSim\Forest\WindowsNoEditor\Forest.exe
```

you can list available scene executables with:

```powershell
python src\list_scenes.py
```

start that `.exe`, then train with:

```powershell
python src\train_ppo.py --scenario forest --episodes 200 --target-x 20 --target-y 0 --target-z -3
```

## Three-Machine Experiment Plan

This section defines the distributed experiment protocol for the final project. The goal is to train and evaluate the same visual PPO navigation system in six non-baseline AirSim environments while a third machine runs controlled baselines in Blocks.

### 1. Research Questions

The experiments should answer the following questions:

1. Can PPO learn short, long, and diagonal visual navigation tasks in AirSim?
2. How much does navigation performance change across urban, open, coastal, and mountainous environments?
3. Does curriculum learning improve training stability compared with starting directly on the difficult task?
4. How does PPO compare with DQN under the same observation, action, reward, start, and target settings?
5. Does a policy trained for one target generalise to mirrored and longer targets in the same scene?

The PPO observation contains an `84 x 84` normalised depth image and a six-value state vector containing relative target position and drone velocity. The action space contains six discrete actions: forward, left, right, up, down, and hover. Therefore, this is visual navigation with additional low-dimensional navigation state, not a camera-only policy.

### 2. Available Scenes

The current scene executables are:

| Scenario label | Executable |
|---|---|
| `blocks` | `D:\AirSim\Blocks\WindowsNoEditor\Blocks.exe` |
| `airsimnh` | `D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe` |
| `africa` | `D:\AirSim\Africa\WindowsNoEditor\Africa_001.exe` |
| `landscapemountains` | `D:\AirSim\LandscapeMountains\WindowsNoEditor\LandscapeMountains.exe` |
| `abandonedpark` | `D:\AirSim\AbandonedPark\WindowsNoEditor\AbandonedPark.exe` |
| `coastline` | `D:\AirSim\Coastline\WindowsNoEditor\Coastline.exe` |
| `msbuild2018` | `D:\AirSim\MSBuild2018\WindowsNoEditor\MSBuild2018.exe` |

Use the lowercase scenario label in output paths even if the executable uses capital letters. Run `python src\list_scenes.py` to discover scene executables after adding or moving an environment.

### 3. Machine Allocation

Each machine runs one AirSim process and one training process at a time. Running multiple AirSim scenes concurrently on one GPU is not part of this protocol.

| Machine | Assigned scenes | Experimental role |
|---|---|---|
| Machine A | AirSimNH, Africa, LandscapeMountains | Urban, open terrain, and mountainous environments |
| Machine B | AbandonedPark, Coastline, MSBuild2018 | Cluttered park, coastal, and complex urban environments |
| Machine C | Blocks | Controlled PPO and DQN baseline experiments |

The assignment gives Machines A and B a mixture of scene types rather than placing all difficult environments on one machine. Scenes assigned to one machine are run sequentially.

### 4. Reproducibility Rules

Before starting the distributed experiment:

1. Put the same project revision on all three machines.
2. Use Python 3.10 and install the same `requirements.txt` on every machine.
3. Use the same AirSim `settings.json`, camera configuration, image resolution, action duration, speed, reward function, and altitude limits.
4. Use the same seed for the first pass: `7`.
5. Record the GPU, CPU, Python version, PyTorch version, scene build, start coordinates, target coordinates, and training duration for each machine.
6. Do not change the reward function after one machine has started. If a reward change is necessary, all affected runs must be repeated under a new experiment version.
7. Never reuse a run name. The runner intentionally stops if the output directory already exists.

Current reward values in `DroneEnvConfig` are:

| Reward component | Value |
|---|---:|
| Per-step penalty | `-0.05` |
| Progress reward scale | `2.0` |
| Goal reward | `+100` |
| Collision penalty | `-100` |
| Altitude violation penalty | `-100` |
| Goal radius | `2 m` |
| Valid AirSim NED altitude | `-10 <= z <= -1` |

AirSim uses NED coordinates. A more negative `z` value means a greater altitude. Do not interpret `z=-10` as being below `z=-3`.

### 5. Scene Calibration

The same absolute coordinates cannot be assumed to be safe in every Unreal scene. Before training, manually inspect each environment and record one collision-free start point. Keep the relative task geometry the same across scenes.

| Scenario | Start `(sx, sy, sz)` | Stage 1 target | Stage 2 target | Stage 3 target | Verified |
|---|---|---|---|---|---|
| `airsimnh` | `(0, 0, -3)` | `(5, 0, -3)` | `(10, 0, -3)` | `(10, 5, -3)` | Yes for current curriculum |
| `africa` | To measure | `(sx+5, sy, sz)` | `(sx+10, sy, sz)` | `(sx+10, sy+5, sz)` | No |
| `landscapemountains` | To measure | `(sx+5, sy, sz)` | `(sx+10, sy, sz)` | `(sx+10, sy+5, sz)` | No |
| `abandonedpark` | To measure | `(sx+5, sy, sz)` | `(sx+10, sy, sz)` | `(sx+10, sy+5, sz)` | No |
| `coastline` | To measure | `(sx+5, sy, sz)` | `(sx+10, sy, sz)` | `(sx+10, sy+5, sz)` | No |
| `msbuild2018` | To measure | `(sx+5, sy, sz)` | `(sx+10, sy, sz)` | `(sx+10, sy+5, sz)` | No |
| `blocks` | `(0, 0, -3)` unless recalibrated | `(5, 0, -3)` | `(10, 0, -3)` | `(10, 5, -3)` | Recheck before baseline |

For each scene:

1. Start the executable and wait until the environment is fully loaded.
2. Run `python src\check_setup.py --connect`.
3. Manually inspect the proposed route and obtain safe start and target coordinates.
4. Put those coordinates in the configuration section of `run_ppo_training.ps1`.
5. Keep `$RunSmokeTest = $true`. The runner resets the drone, checks spawn error, captures the observation, and performs a short hover test before training.
6. Confirm that the depth image is not blank and that the initial distance matches the intended task distance.

If the smoke test immediately terminates, do not train. Check spawn height, nearby collision geometry, stale collision state, and whether the target is reachable.

### 6. PPO Curriculum

Every PPO scene uses the same three-stage curriculum. A stage resumes both the model and optimiser from the previous stage, but writes to a new run directory.

| Stage | Relative target from start | Episodes | Max steps per episode | Rollout steps | Learning rate | Initial model |
|---|---:|---:|---:|---:|---:|---|
| Stage 1 | `(5, 0, 0)` | `150` | `100` | `512` | `1e-4` | None; train from scratch |
| Stage 2 | `(10, 0, 0)` | `200` | `150` | `512` | `1e-4` | Stage 1 `ppo_final.pt` |
| Stage 3 | `(10, 5, 0)` | `250` | `200` | `512` | `7.5e-5` | Stage 2 `ppo_final.pt` |

Keep the following PPO parameters fixed in all scenes:

```powershell
$BatchSize = 64
$UpdateEpochs = 4
$CheckpointEvery = 10
$Seed = 7
$EvaluateAfterTraining = $true
$EvaluationEpisodes = 50
$RunSmokeTest = $true
$AutoStartScene = $true
$CloseSceneAfterRun = $true
```

The current scripts stop after a specified number of episodes. Because failed episodes can finish early, equal episode counts do not guarantee equal interaction counts. Use the `steps` column in every `training_log.csv` to report the actual number of environment interactions. For a strict final PPO-versus-DQN comparison, match total interaction counts or clearly report this limitation.

#### Stage 1 configuration

Edit only the configuration block at the top of `run_ppo_training.ps1`:

```powershell
$Scenario = "Africa"
$SceneExe = "D:\AirSim\Africa\WindowsNoEditor\Africa_001.exe"
$RunName = "stage01_5m_seed7"
$ResumeModel = ""

$Episodes = 150
$MaxSteps = 100
$RolloutSteps = 512
$StartX = 0.0  # Replace with the calibrated scene start.
$StartY = 0.0
$StartZ = -3.0
$TargetX = $StartX + 5.0
$TargetY = $StartY
$TargetZ = $StartZ
$LearningRate = 1e-4
```

PowerShell variable assignments are evaluated from top to bottom. Set `$StartX`, `$StartY`, and `$StartZ` before expressions that use them, or enter the calculated target coordinates directly.

#### Stage 2 configuration

```powershell
$RunName = "stage02_10m_seed7"
$ResumeModel = "D:\AirSim\rl_drone_navigation\experiments\africa\ppo\stage01_5m_seed7\models\ppo_final.pt"

$Episodes = 200
$MaxSteps = 150
$RolloutSteps = 512
$StartX = 0.0  # Use the same calibrated start as Stage 1.
$StartY = 0.0
$StartZ = -3.0
$TargetX = $StartX + 10.0
$TargetY = $StartY
$TargetZ = $StartZ
$LearningRate = 1e-4
```

#### Stage 3 configuration

```powershell
$RunName = "stage03_diagonal_10x5m_seed7"
$ResumeModel = "D:\AirSim\rl_drone_navigation\experiments\africa\ppo\stage02_10m_seed7\models\ppo_final.pt"

$Episodes = 250
$MaxSteps = 200
$RolloutSteps = 512
$StartX = 0.0  # Use the same calibrated start as Stages 1 and 2.
$StartY = 0.0
$StartZ = -3.0
$TargetX = $StartX + 10.0
$TargetY = $StartY + 5.0
$TargetZ = $StartZ
$LearningRate = 7.5e-5
```

Run each configured stage from the project directory:

```powershell
conda activate airsim-rl
cd D:\AirSim\rl_drone_navigation
.\run_ppo_training.ps1
```

With `$CloseSceneAfterRun = $true`, the scene started by the runner is closed after training and evaluation. Cleanup also runs when training or evaluation fails. If the scene was already open before the script started, verify that it has closed before launching the next environment.

### 7. Stage Acceptance Criteria

The automatic evaluation after each stage should use 50 deterministic episodes. A stage is considered ready for progression when:

```text
Success rate >= 80%
Collision rate <= 5%
Altitude violation rate <= 5%
```

Also inspect the final 20% of training episodes rather than relying only on the average over the complete run. Early exploration failures can make the full-run success rate look poor even when the final policy is stable.

If a stage fails the acceptance criteria:

1. Inspect `training_curves.png` and the final 40 episodes in `training_log.csv`.
2. Check whether failure is caused mainly by collisions, altitude termination, timeouts, or stopping near the goal radius.
3. Verify the start and target positions again in the actual scene.
4. Continue from the best valid checkpoint with a lower learning rate only when the policy is improving.
5. Do not silently change rewards for one scene. Record any changed experiment as a new version and repeat the corresponding baseline.

### 8. Blocks Baseline on Machine C

Blocks is the controlled baseline environment. It should not be included in the six-scene PPO generalisation average.

The required algorithm comparison is:

| Baseline | Training | Purpose |
|---|---|---|
| PPO | Same fixed Blocks task | Main policy-gradient result |
| DQN | Same fixed Blocks task | Value-based RL comparison from the relevant literature |

For a direct comparison, run both algorithms from `(0, 0, -3)` to `(20, 0, -3)` with `300` episodes, `300` maximum steps, and seed `7`. Use separate output directories so existing results are not overwritten.

PPO baseline:

```powershell
python src\train_ppo.py `
  --scenario blocks `
  --run-name baseline_20m_seed7 `
  --episodes 300 `
  --max-steps 300 `
  --rollout-steps 512 `
  --learning-rate 1e-4 `
  --target-x 20 --target-y 0 --target-z -3 `
  --start-x 0 --start-y 0 --start-z -3 `
  --seed 7
```

DQN baseline:

```powershell
python src\train_dqn.py `
  --scenario blocks `
  --episodes 300 `
  --max-steps 300 `
  --target-x 20 --target-y 0 --target-z -3 `
  --seed 7 `
  --results-dir experiments\blocks\dqn\baseline_20m_seed7\results `
  --models-dir experiments\blocks\dqn\baseline_20m_seed7\models
```

The current DQN trainer does not support `--run-name`, custom start coordinates, or checkpoint resume. The explicit result and model directories above isolate the run. Keep the default start `(0, 0, -3)` for this comparison.

The existing top-level Blocks outputs are not a valid direct comparison: the stored PPO metadata uses target `(66.692, -12.265, -1.2)`, while the stored DQN metadata uses `(20, 0, -3)`. Keep those files as pilot results, but use the new `baseline_20m_seed7` runs for the comparison table.

Evaluate both new baseline models on the identical 20 m task:

```powershell
python src\evaluate.py --algorithm ppo --scenario blocks `
  --model experiments\blocks\ppo\baseline_20m_seed7\models\ppo_final.pt `
  --episodes 50 --max-steps 300 `
  --target-x 20 --target-y 0 --target-z -3 `
  --start-x 0 --start-y 0 --start-z -3 `
  --results-dir experiments\blocks\ppo\baseline_20m_seed7_eval\results

python src\evaluate.py --algorithm dqn --scenario blocks `
  --model experiments\blocks\dqn\baseline_20m_seed7\models\dqn_final.pt `
  --episodes 50 --max-steps 300 `
  --target-x 20 --target-y 0 --target-z -3 `
  --start-x 0 --start-y 0 --start-z -3 `
  --results-dir experiments\blocks\dqn\baseline_20m_seed7_eval\results
```

The direct baseline commands do not manage the Unreal process. After all Blocks training and evaluation jobs finish, close it with:

```powershell
Get-Process Blocks -ErrorAction SilentlyContinue | Stop-Process
```

A random-action policy and an always-forward policy are useful optional references because they expose whether a fixed straight target is too easy. They require a small baseline evaluator that is not currently included in this repository; do not describe them as completed experiments until that evaluator has been implemented and run.

### 9. Generalisation Evaluation

After Stage 3, evaluate the same final model on three target types. Use separate `--results-dir` values because `evaluate.py` writes a file named `evaluation_log.csv` and otherwise overwrites the previous evaluation.

| Evaluation | Relative target | Meaning |
|---|---:|---|
| In-distribution | `(10, 5, 0)` | Same geometry as Stage 3 training |
| Mirrored | `(10, -5, 0)` | Tests left/right directional generalisation |
| Longer | `(15, 0, 0)` | Tests distance generalisation |

Example for an AirSimNH model with start `(0, 0, -3)`:

```powershell
$Model = "experiments\airsimnh\ppo\stage03_diagonal_10x5m_seed7\models\ppo_final.pt"

python src\evaluate.py --algorithm ppo --scenario airsimnh --model $Model `
  --episodes 50 --max-steps 200 `
  --start-x 0 --start-y 0 --start-z -3 `
  --target-x 10 --target-y 5 --target-z -3 `
  --results-dir experiments\airsimnh\ppo\stage03_diagonal_10x5m_seed7_eval_in_distribution\results

python src\evaluate.py --algorithm ppo --scenario airsimnh --model $Model `
  --episodes 50 --max-steps 200 `
  --start-x 0 --start-y 0 --start-z -3 `
  --target-x 10 --target-y -5 --target-z -3 `
  --results-dir experiments\airsimnh\ppo\stage03_diagonal_10x5m_seed7_eval_mirrored\results

python src\evaluate.py --algorithm ppo --scenario airsimnh --model $Model `
  --episodes 50 --max-steps 250 `
  --start-x 0 --start-y 0 --start-z -3 `
  --target-x 15 --target-y 0 --target-z -3 `
  --results-dir experiments\airsimnh\ppo\stage03_diagonal_10x5m_seed7_eval_longer\results
```

For a non-zero scene start, convert each relative target into absolute coordinates before running evaluation.

### 10. Metrics and Analysis

The required metrics are:

| Metric | Definition | Desired direction |
|---|---|---|
| Success rate | Successful episodes divided by evaluation episodes | Higher |
| Collision rate | Episodes ending in a new collision divided by evaluation episodes | Lower |
| Altitude violation rate | Episodes leaving the valid altitude band divided by evaluation episodes | Lower |
| Average reward | Mean cumulative episode reward | Higher |
| Average final distance | Mean target distance when an episode ends | Lower |
| Average steps | Mean actions taken per episode | Lower only when success remains high |
| Training interactions | Sum of training episode steps | Report for fairness |
| Wall-clock time | Training start-to-finish duration | Report as computational cost |

Path length and path efficiency are recommended metrics, but the current environment does not log the full trajectory. If trajectory logging is added, define:

```text
path efficiency = straight-line start-to-target distance / actual flown path length
```

Only calculate path efficiency for successful episodes, and report its mean and standard deviation.

### 11. Seed Strategy

Use two rounds to control computation cost:

**Round 1: scene coverage**

- Run every scene once with seed `7`.
- Use the result to identify setup errors, impossible routes, and unstable scenes.
- Do not repeatedly tune one scene using its final evaluation set.

**Round 2: reproducibility**

- Repeat `blocks`, `airsimnh`, and `landscapemountains` with seeds `17` and `27`.
- These three environments represent the controlled baseline, an urban scene, and a difficult terrain scene.
- Report `mean +/- standard deviation` over seeds `7`, `17`, and `27` for these representative environments.

If time permits, extend the three-seed evaluation to every scene. New run names must include the seed, for example `stage03_diagonal_10x5m_seed17`.

### 12. Execution Order

Recommended order for the three machines:

| Order | Machine A | Machine B | Machine C |
|---:|---|---|---|
| 1 | Calibrate all assigned scenes | Calibrate all assigned scenes | Recheck Blocks start and route |
| 2 | AirSimNH curriculum/evaluation | AbandonedPark curriculum/evaluation | PPO Blocks baseline seed 7 |
| 3 | Africa curriculum/evaluation | Coastline curriculum/evaluation | DQN Blocks baseline seed 7 |
| 4 | LandscapeMountains curriculum/evaluation | MSBuild2018 curriculum/evaluation | Blocks seeds 17 and 27 |
| 5 | AirSimNH seeds 17 and 27 | LandscapeMountains seeds 17 and 27 | Verify and package baseline results |

Do not wait until all stages finish before checking results. Review the smoke test and Stage 1 evaluation before committing hours to later stages.

### 13. Result Collection

Each machine returns only its assigned experiment directories plus a short machine-information text file:

```text
Machine A:
  experiments/airsimnh/
  experiments/africa/
  experiments/landscapemountains/

Machine B:
  experiments/abandonedpark/
  experiments/coastline/
  experiments/msbuild2018/

Machine C:
  experiments/blocks/
```

Because scenario names are unique across machines, these directories can be copied into the same central `experiments` directory without path conflicts. Do not copy one machine's complete `experiments` folder over another machine's folder.

After collection, generate the project summary:

```powershell
python src\summarize_experiments.py
```

This creates `experiments\summary.csv`. The file is generated output and can be recreated from evaluation logs. It may be deleted during cleanup, but keep it when preparing the notebook because it is convenient for tables and plots.

The current summary script recognises evaluation logs at `experiments/<scenario>/<algorithm>/results/` and `experiments/<scenario>/<algorithm>/<run-name>/results/`. The generalisation examples deliberately use the second layout so that all three evaluations appear in `summary.csv`.

### 14. Expected Figures and Tables

The final notebook should contain at least:

1. A table describing observations, actions, rewards, termination conditions, and curriculum stages.
2. PPO training reward and success curves for each scene.
3. A Blocks PPO-versus-DQN table using identical evaluation settings.
4. A grouped bar chart of success, collision, and altitude violation rates across scenes.
5. A table comparing in-distribution, mirrored, and longer-target evaluation.
6. Mean and standard deviation across three seeds for the representative scenes.
7. Example depth images from at least one simple and one complex scene.
8. A discussion of compute limits, episode-versus-step budget differences, discrete actions, fixed starts, simulator realism, and transfer to real drones.

This evidence maps directly to the COMP9444 rubric sections for RL task exploration, models and methods, results, and discussion.

## DQN Training

Start Blocks first, then run:

```powershell
cd D:\AirSim\rl_drone_navigation
conda activate airsim-rl
python src\train_dqn.py --scenario blocks --episodes 200 --target-x 20 --target-y 0 --target-z -3
```

Training writes:

- `experiments/blocks/dqn/results/training_log.csv`
- `experiments/blocks/dqn/results/training_curves.png`
- model checkpoints in `experiments/blocks/dqn/models/`

## PPO Training

The easiest option is the configurable PowerShell runner. Edit the configuration section at the top of `run_ppo_training.ps1`, then run:

```powershell
.\run_ppo_training.ps1
```

It can start the selected scene, wait for AirSim, train PPO, run evaluation, and save each run separately:

The runner also performs a short hover smoke test before training. Configure `StartX`, `StartY`, and `StartZ` for each scene; training stops before writing a run if the spawn is unsafe.

For curriculum learning, set `ResumeModel` in `run_ppo_training.ps1` to the previous stage's `ppo_final.pt`. The new run loads both the network and optimizer state while keeping its models and results in a separate run directory.

Set `CloseSceneAfterRun = $true` to close the configured AirSim scene after training and evaluation. The cleanup also runs when training or evaluation fails.

```text
experiments/<scenario>/ppo/<run-name>/models/
experiments/<scenario>/ppo/<run-name>/results/
```

For direct command-line training, start the AirSim scene first, then run:

```powershell
python src\train_ppo.py --scenario blocks --run-name baseline_seed7 --episodes 200 --max-steps 300 --rollout-steps 256 --target-x 20 --target-y 0 --target-z -3
```

Continue from an earlier PPO stage:

```powershell
python src\train_ppo.py --scenario airsimnh --run-name stage02_10m_seed7 --resume-model experiments\airsimnh\ppo\stage01_5m_seed7\models\ppo_final.pt --episodes 200 --max-steps 150 --rollout-steps 512 --target-x 10 --target-y 0 --target-z -3
```

The next curriculum stage can introduce a diagonal target before increasing the forward distance:

```powershell
python src\train_ppo.py --scenario airsimnh --run-name stage03_diagonal_10x5m_seed7 --resume-model experiments\airsimnh\ppo\stage02_10m_seed7\models\ppo_final.pt --episodes 250 --max-steps 200 --rollout-steps 512 --learning-rate 7.5e-5 --target-x 10 --target-y 5 --target-z -3
```

PPO writes:

- `experiments/blocks/ppo/<run-name>/results/training_log.csv`
- `experiments/blocks/ppo/<run-name>/results/ppo_update_log.csv`
- `experiments/blocks/ppo/<run-name>/results/training_curves.png`
- model checkpoints in `experiments/blocks/ppo/<run-name>/models/`

## Evaluation

DQN:

```powershell
python src\evaluate.py --algorithm dqn --scenario blocks --episodes 20
```

PPO:

```powershell
python src\evaluate.py --algorithm ppo --scenario blocks --run-name baseline_seed7 --episodes 20
```

Evaluation writes:

- `experiments/<scenario>/<algorithm>/results/evaluation_log.csv`
- with `--run-name`: `experiments/<scenario>/<algorithm>/<run-name>/results/evaluation_log.csv`

You can still evaluate a specific checkpoint manually:

```powershell
python src\evaluate.py --algorithm ppo --scenario blocks --model experiments\blocks\ppo\models\ppo_update_0010.pt --episodes 20
```

After evaluating multiple scene/algorithm pairs, summarize them:

```powershell
python src\summarize_experiments.py
```

This writes:

- `experiments/summary.csv`

The summary includes a `run_name` column for named PPO runs.

## Notebook

Open:

```text
D:\AirSim\rl_drone_navigation\notebooks\COMP9444_AirSim_Drone_Navigation.ipynb
```

The notebook is organized according to the marking rubric:

1. Introduction, motivation, and problem statement
2. Data source / RL task definition
3. Exploratory analysis of the RL task
4. Models and methods
5. Results
6. Discussion
7. Writing and reproducibility notes

## Notes

- The environment uses AirSim's NED coordinate system, where negative `z` means higher altitude.
- DQN and PPO both use depth images plus relative target/velocity features.
- Keep scenario names short and consistent, such as `blocks`, `forest`, `neighborhood`, or `city`.
