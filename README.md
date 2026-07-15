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

If you download a new environment under `D:\AirSim\Blocks`, for example:

```text
D:\AirSim\Blocks\Forest\WindowsNoEditor\Forest.exe
```

you can list available scene executables with:

```powershell
python src\list_scenes.py
```

start that `.exe`, then train with:

```powershell
python src\train_ppo.py --scenario forest --episodes 200 --target-x 20 --target-y 0 --target-z -3
```

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
