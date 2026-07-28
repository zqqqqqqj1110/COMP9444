# Autonomous Drone Navigation with Deep Reinforcement Learning

This COMP9444 project studies visual autonomous drone navigation in Microsoft
AirSim. It compares a vanilla DQN baseline, PPO trained directly on the final
route, and PPO trained with a three-stage curriculum.

The current formal experiment uses one fixed route in the AirSimNH scene and
Seed 7. There is no offline dataset: observations and rewards are collected
online from AirSim while the agent interacts with the simulator.

## Final Result

All methods used the same final route, observation, action space, reward,
45,000-interaction budget, and maximum episode length. Checkpoints were selected
with validation episodes and then tested for 50 fresh-seed episodes.

Test configuration:

```text
Scene:       AirSimNH
Train seed:  7
Test seed:   20007
Start:       (85.413, -15.334, -3.0)
Target:      (117.756, -19.034, -3.0)
Distance:    approximately 32.55 m
Max steps:   150
```

Primary deterministic results:

| Method | Selected checkpoint | Success | Collision | Timeout | Average steps | Final distance |
|---|---|---:|---:|---:|---:|---:|
| DQN Scratch | `dqn_step_0040000.pt` | `16%` | `72%` | `14%` | `86.80` | `4.47 m` |
| **PPO Scratch** | **`ppo_step_0042500.pt`** | **`98%`** | **`2%`** | **`0%`** | **`52.28`** | **`2.01 m`** |
| PPO Curriculum | `ppo_step_0020000.pt` in Stage 3 | `68%` | `2%` | `30%` | `83.46` | `14.66 m` |

PPO stochastic-policy diagnostics:

| Method | Success | Collision | Timeout | Average steps | Final distance |
|---|---:|---:|---:|---:|---:|
| PPO Scratch | `66%` | `34%` | `0%` | `64.08` | `3.26 m` |
| PPO Curriculum | `46%` | `38%` | `16%` | `97.12` | `4.21 m` |

The formal comparison table is:

```text
experiments/airsimnh/validated_comparison_seed7_test_seed20007.csv
```

For this route and seed, PPO Scratch is the strongest method. It learned a
repeatable lateral detour around the blocking house. DQN usually followed the
direct goal direction and collided, while Curriculum PPO was safe when it
succeeded but timed out frequently.

These results demonstrate fixed-route performance, not generalisation to unseen
routes or scenes. Multiple seeds and perturbed start/target tests are future
work.

## Pretrained Models

The three selected inference weights are published separately from the large
experiment directories:

| Model | Repository path | Deterministic success |
|---|---|---:|
| **PPO Scratch (recommended)** | `pretrained/airsimnh/ppo_scratch_seed7.pt` | **`98%`** |
| PPO Curriculum | `pretrained/airsimnh/ppo_curriculum_seed7.pt` | `68%` |
| DQN Scratch baseline | `pretrained/airsimnh/dqn_scratch_seed7.pt` | `16%` |

SHA256 checksums:

```text
ppo_scratch_seed7.pt
4F23F3EF8E5E7706224591CEECDCE9D3B413B35A89CF79341FFE3989FF6A9989

ppo_curriculum_seed7.pt
E5D07915E31853A72F24EA81E211A56228D992ED1379F5E57B314FCE1D6CDF67

dqn_scratch_seed7.pt
BC5E3C44D83A850CA3009E32E2A7909BDCFAEA73A943471CF5E6B8D897A9C92D
```

The `.pt` files are Git LFS objects. After cloning the repository:

```powershell
git lfs install
git lfs pull
```

If `git lfs pull` is skipped, the apparent model file may only be a small text
pointer and PyTorch will not be able to load it.

## RL Task

Each observation contains:

- One normalised `84 x 84` front depth image.
- Relative target position `(target - position)`, scaled by fixed constants.
- Linear velocity `(vx, vy, vz)`, also scaled.

The image and six-dimensional navigation state are processed by a CNN and a
fully connected network. The agent does not receive a map or explicit obstacle
coordinates. Collision information is used only for reward and termination.

Discrete actions:

| ID | Action |
|---:|---|
| 0 | Forward |
| 1 | Left |
| 2 | Right |
| 3 | Up |
| 4 | Down |
| 5 | Hover |

Reward and termination configuration:

| Component | Value |
|---|---:|
| Step penalty | `-0.05` |
| Progress reward | `2.0 * distance reduction` |
| Goal reward | `+100` |
| Collision penalty | `-100` |
| Altitude violation penalty | `-100` |
| Timeout penalty | `-25` |
| Altitude hold penalty | `-0.25 * abs(z - target_z)` |
| Goal radius | `2 m` |
| Valid NED altitude | `-10 <= z <= -1` |

AirSim uses NED coordinates, so a more negative `z` means a greater altitude.

## Methods

### DQN Baseline

The baseline is a vanilla DQN with:

- A CNN depth encoder and six-dimensional navigation state.
- A six-value Q-function output.
- Uniform replay buffer.
- Target network updated every 1,000 interactions.
- Epsilon-greedy exploration from `1.0` towards `0.05`.
- Huber TD loss and gradient clipping.

It does not use Double DQN, Dueling DQN, prioritized replay, or n-step returns.
The report should therefore describe it as a vanilla DQN visual-navigation
baseline, not an optimized state-of-the-art DQN.

### Stable PPO

PPO uses the same observation and actions. The shared representation feeds:

- An actor that outputs six action logits.
- A critic that estimates the state value.

The stable implementation includes:

- Layer normalization before the shared `Tanh`.
- Orthogonal initialization.
- Reward scaling of `0.1` inside PPO.
- Huber critic loss.
- Four update epochs per rollout.
- Linear entropy decay from `0.01` to `0.001`.
- Activation saturation, explained variance, entropy, KL, and action-confidence
  diagnostics.

These changes fixed the hidden-layer saturation observed in the original PPO
implementation.

### PPO Curriculum

Curriculum PPO transfers policy weights through progressively longer targets:

| Stage | Target | New interactions | Max steps |
|---|---|---:|---:|
| 1 | `(95.190, -14.491, -3.0)` | `5,000` | `70` |
| 2 | `(107.635, -10.842, -3.0)` | `10,000` | `110` |
| 3 | `(117.756, -19.034, -3.0)` | `30,000` | `150` |

The optimizer is reset when the target changes. The full curriculum consumes
the same 45,000 environment interactions as Scratch PPO.

Stage 1 and Stage 2 have separate `_gate` directories containing evaluation
logs only. Those gates determined whether training could continue. Stage 3 is
the final task, so its training and final evaluation are stored together and
there is no Stage 3 `_gate` directory.

The words `stage2_pilot` and `stage3_pilot` are historical run tags. They do not
represent different algorithms or PPO versions.

## Experimental Protocol

The primary comparison is performed at 45,000 consumed interactions:

| Comparison | Question |
|---|---|
| DQN Scratch vs PPO Scratch | Which algorithm learns the final route more effectively? |
| PPO Scratch vs PPO Curriculum | Does progressive target difficulty improve PPO? |

Raw rewards from Curriculum Stages 1 and 2 must not be compared directly with
Scratch rewards because the target distances and episode limits are different.
All final models are tested on the same Stage 3 route.

Checkpoint selection is separated from final testing:

```text
Selection Stage 1: all checkpoints x 5 episodes, seed 7
Selection Stage 2: top 3 checkpoints x 30 episodes, seed 10007
Final test:         selected model x 50 episodes, seed 20007
```

Selection episodes are validation data. Only the independent final-test rows
should be reported as final performance.

Training times:

| Method | Training time |
|---|---:|
| DQN Scratch | `7.41 h` |
| PPO Scratch | `6.50 h` |
| PPO Curriculum Stage 1 + 2 + 3 | approximately `6.51 h` |

Evaluation and checkpoint-selection time is excluded.

## Setup

Create the Python environment:

```powershell
conda create -n airsim-rl python=3.10 -y
conda activate airsim-rl
cd D:\AirSim\rl_drone_navigation
pip install -r requirements.txt
python -m ipykernel install --user --name airsim-rl --display-name "Python (airsim-rl)"
```

AirSim depends on an old RPC stack. If importing AirSim fails with
`No module named 'tornado.platform.auto'`, install:

```powershell
pip install "tornado==4.5.3" "ipykernel==5.5.6" "jupyter-client==7.1.2"
```

Start AirSimNH and verify the connection:

```powershell
D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe
python src\check_setup.py --connect
```

`--scenario` controls experiment output paths; it does not launch or switch the
Unreal scene.

The AirSimNH Unreal environment is not included in this repository. Download
and start a compatible AirSimNH build before inference.

Run the regression tests without AirSim:

```powershell
python -B -m unittest discover -s tests -v
```

## Running Experiments

### New Controlled Comparison

Edit the configuration block at the top of:

```text
run_comparison_experiment.ps1
```

Use a new seed or run tag. Existing run directories are intentionally rejected
to prevent accidental result overwrite. Then run:

```powershell
.\run_comparison_experiment.ps1
```

The runner starts AirSim, checks the spawn, trains enabled methods, saves
checkpoints, performs two-stage selection and evaluation, records training
times, and closes AirSim in a `finally` block.

### Select Existing Checkpoints

`run_checkpoint_selection.ps1` applies the common two-stage protocol to the
three completed Seed 7 runs and writes a unified comparison CSV. This has
already been completed for the current results.

For a new set of runs, update the run names and test-seed offsets before using:

```powershell
.\run_checkpoint_selection.ps1
```

### Evaluate One Model

Example deterministic evaluation of the selected PPO Scratch model:

```powershell
python src\evaluate.py `
  --algorithm ppo --scenario airsimnh `
  --run-name ppo_scratch_manual_test `
  --model pretrained\airsimnh\ppo_scratch_seed7.pt `
  --policy-mode deterministic --episodes 20 --seed 30007 `
  --max-steps 150 `
  --start-x 85.413 --start-y -15.334 --start-z -3.0 `
  --target-x 117.756 --target-y -19.034 --target-z -3.0
```

PPO supports `deterministic`, `stochastic`, or `both`. DQN evaluation is
deterministic because epsilon exploration is disabled during deployment.

### Record Inference

Edit the configuration block in `run_inference_recording.ps1`, then run:

```powershell
.\run_inference_recording.ps1
```

For the best current model, use:

```text
Algorithm:   ppo
PolicyMode:  deterministic
Model:       pretrained/airsimnh/ppo_scratch_seed7.pt
Start:       (85.413, -15.334, -3.0)
Target:      (117.756, -19.034, -3.0)
MaxSteps:    150
```

Recordings contain RGB video, an inset of the depth input, action, reward,
distance, position, and outcome.

### Inspect a Route Manually

Start the scene and run:

```powershell
python src\manual_control.py
```

Useful controls:

| Key | Action |
|---|---|
| `W/S` | Forward/backward |
| `A/D` | Left/right |
| `R/F` | Up/down |
| `Q/E` | Rotate |
| `P` | Print and save coordinates |
| `H` | Hover and print coordinates |
| `L` | Land and exit |
| `Esc` | Hover and exit |

The RL policy has no backward or yaw action, so a manually selected RL route
must not depend on those controls.

## Important Files

```text
run_comparison_experiment.ps1   train and evaluate controlled comparisons
run_checkpoint_selection.ps1    select checkpoints and run final tests
run_inference_recording.ps1     record policy inference

pretrained/airsimnh/             selected Git LFS inference weights

src/airsim_drone_env.py         observation, action, reward, termination
src/dqn_agent.py                DQN network and replay learning
src/ppo_agent.py                PPO actor-critic and optimization
src/train_dqn.py                DQN training entry point
src/train_ppo.py                PPO training entry point
src/evaluate.py                 deterministic/stochastic evaluation
src/sweep_checkpoints.py        two-stage checkpoint selection
src/trajectory_logging.py       action and reward-component logs

notebooks/COMP9444_AirSim_Drone_Navigation.ipynb
```

## Output Structure

Each training run contains:

```text
experiments/<scenario>/<algorithm>/<run-name>/
  metadata.json
  models/
    <algorithm>_final.pt
    <algorithm>_step_*.pt
    dqn_best_deterministic.pt       # DQN only
    ppo_best.pt                     # PPO rolling-training candidate
    ppo_best_deterministic.pt       # PPO selected validation model
  results/
    training_log.csv
    training_action_log.csv         # PPO only, one row per interaction
    training_curves.png
    training_summary.json
    ppo_update_log.csv              # PPO only
    checkpoint_sweep_stage1.csv
    checkpoint_sweep_stage2.csv
    checkpoint_sweep_two_stage_summary.json
```

Evaluation directories contain:

```text
evaluation_deterministic_log.csv
evaluation_deterministic_trajectory.csv
evaluation_stochastic_log.csv
evaluation_stochastic_trajectory.csv
evaluation_mode_comparison.csv
evaluation_summary.json
```

Trajectory logs record every action, before/after position, target distance,
reward sign and magnitude, individual reward components, terminal outcome, and
collision object.

## GitHub Upload

Do not upload every checkpoint from `experiments/`. The three completed training
runs contain more than 1 GB of intermediate `.pt` files. `.gitignore` excludes
those checkpoints, recordings, and large step-level trajectory logs.

The ignore rules prevent new generated checkpoints from being added. They do
not remove legacy checkpoints that were already committed in older Git history.
History cleanup is a separate maintainer operation and should not be performed
casually on a shared branch.

Only the selected files under `pretrained/airsimnh/` are intended for GitHub.
They are configured for Git LFS in `.gitattributes`. Before committing:

```powershell
git lfs install
git lfs status
git status --short
```

Check that the three `pretrained/*.pt` files appear as LFS objects and that
unwanted experiment checkpoints are not staged.

## Limitations

- The formal table contains only Seed 7.
- Training and testing use one fixed AirSimNH route.
- Test seeds separate selection from testing but do not change the scene layout.
- The policy has one front depth frame and no recurrent memory or SLAM.
- The discrete action space has no backward or yaw action.
- AirSim collision geometry may differ from visible meshes, especially foliage.
- Simulator performance does not establish real-drone transfer.
- PPO received stabilization changes that are not all mirrored in vanilla DQN;
  conclusions should refer to the implemented systems, not all PPO and DQN
  variants.

Generalisation should be evaluated with safe start/target perturbations,
additional routes, different scenes, and multiple training seeds. Improving
generalisation would require training-time randomisation or multi-route
training; repeated inference alone does not update the policy.

## Notebook

The project notebook is:

```text
notebooks/COMP9444_AirSim_Drone_Navigation.ipynb
```

It should use the final comparison CSV and include:

1. Problem statement and motivation.
2. AirSim task, observation, actions, reward, and termination analysis.
3. Depth-image examples and route difficulty.
4. DQN, PPO, stable PPO changes, and curriculum method.
5. Training curves and the validated final comparison.
6. Discussion of the DQN failure mode, Curriculum timeouts, limitations, and
   future generalisation experiments.

## References

1. Shah, S., Dey, D., Lovett, C., and Kapoor, A. "AirSim: High-Fidelity Visual
   and Physical Simulation for Autonomous Vehicles." Field and Service
   Robotics, 2018.
2. Mnih, V. et al. "Human-level control through deep reinforcement learning."
   Nature, 2015.
3. Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:
   1707.06347, 2017.
