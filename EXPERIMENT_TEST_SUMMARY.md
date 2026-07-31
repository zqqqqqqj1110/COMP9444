# Experimental Evaluation of Deep Reinforcement Learning for Autonomous Drone Navigation

## Abstract

This report presents the experimental evaluation of three deep reinforcement learning approaches for visual autonomous drone navigation in Microsoft AirSim: a vanilla Deep Q-Network trained from scratch (DQN Scratch), Proximal Policy Optimisation trained directly on the final route (PPO Scratch), and PPO trained through a three-stage distance curriculum (PPO Curriculum). The agents share the same multimodal observation, discrete action space, reward function, simulator, final task, and total interaction budget.

On the original AirSimNH route, deterministic PPO Scratch achieved a 98% success rate over 50 held-out evaluation episodes, compared with 68% for PPO Curriculum and 16% for DQN Scratch. PPO Scratch was consequently selected for extended route-generalisation testing. It achieved 100% success on two small route perturbations, a large-offset shortened route, and an unseen route in another area of the same map. However, it achieved 0% success on a longer combined-offset route and on a geometrically more difficult unseen route. The results show that PPO Scratch learns a robust policy within part of the training distribution and can transfer to some unseen routes, but it does not provide route-independent navigation or general-purpose global planning.

## 1. Navigation Task

### 1.1 Environment and objective

The task is performed in the AirSimNH Unreal environment. At the beginning of each episode, the drone is placed at a fixed three-dimensional start coordinate with zero yaw. It must enter a 2 m radius around the target while avoiding collisions and remaining within the permitted altitude interval. An episode terminates upon success, collision, altitude violation, or after 150 decision steps.

The original route is:

| Parameter | Value |
|---|---|
| Start position | `(85.413, -15.334, -3.000)` |
| Target position | `(117.756, -19.034, -3.000)` |
| Straight-line distance | approximately `32.55 m` |
| Maximum episode length | `150 steps` |
| Goal radius | `2 m` |
| Valid NED altitude | `-10 <= z <= -1` |

AirSim uses North-East-Down coordinates; a more negative `z` value therefore represents greater altitude.

### 1.2 Observation space

Each observation contains:

- one normalised `84 x 84` forward-facing depth image;
- the relative target vector `(target position - drone position)`; and
- the drone linear velocity `(vx, vy, vz)`.

The relative target vector and velocity form a six-dimensional navigation state. The policy therefore receives explicit local goal direction and distance information in addition to visual obstacle information. It does not receive a global map, obstacle coordinates, memory of the full trajectory, or a conventional path produced by a planner.

### 1.3 Action space and motion execution

The action space is discrete:

| ID | Action | Executed command |
|---:|---|---|
| 0 | Forward | body-frame `vx = 2 m/s` |
| 1 | Left | body-frame `vy = -2 m/s` |
| 2 | Right | body-frame `vy = 2 m/s` |
| 3 | Up | `vz = -1 m/s` |
| 4 | Down | `vz = 1 m/s` |
| 5 | Hover | zero commanded velocity |

Each action is applied for 0.35 s. The policy selects a discrete motion primitive, while the simulator executes that primitive with a fixed velocity. The agent has no explicit backward or yaw action, which limits recovery manoeuvres and fine-grained replanning in difficult geometry.

### 1.4 Reward function

The non-terminal reward at each step is formed from:

`reward = step penalty + progress reward + altitude-hold penalty + altitude-margin penalty`

The configured components are:

| Component | Value |
|---|---:|
| Step penalty | `-0.05` |
| Progress reward | `2.0 x reduction in target distance` |
| Goal reward | `+100` |
| Collision penalty | `-100` |
| Altitude-violation penalty | `-100` |
| Timeout penalty | `-25` |
| Altitude-hold penalty | `-0.25 x abs(z - target_z)` |

This shaping favours immediate reduction in Euclidean target distance while penalising unsafe and inefficient behaviour. It supports reactive goal following and obstacle avoidance, but does not directly reward construction of a globally valid path around previously unseen obstacle layouts.

## 2. Learning Methods

### 2.1 DQN Scratch

The DQN baseline combines a convolutional depth encoder with the six-dimensional navigation state and outputs one Q-value for each of the six actions. It uses epsilon-greedy exploration, a uniform replay buffer, a target network, Huber temporal-difference loss, and gradient clipping. Experience is stored and sampled continuously during training; the network is not trained only after all episodes have finished.

The implementation is intentionally a vanilla DQN baseline. It does not include Double DQN, a duelling architecture, prioritised replay, or multi-step returns.

### 2.2 PPO Scratch

PPO uses a shared visual and state representation followed by an actor and critic. The actor produces logits for the six discrete actions, while the critic estimates state value. Training data are collected in 500-step on-policy rollouts. Each rollout is divided into minibatches and used for four optimisation epochs before the updated policy collects the next rollout.

The implementation uses layer normalisation, orthogonal initialisation, reward scaling by 0.1 inside PPO, Huber critic loss, and linear entropy-coefficient decay from 0.01 to 0.001. PPO Scratch allocates all 45,000 environment interactions to the final route.

### 2.3 PPO Curriculum

PPO Curriculum transfers policy weights through three progressively longer targets:

| Stage | Target | New interactions | Maximum steps |
|---|---|---:|---:|
| 1 | `(95.190, -14.491, -3.000)` | 5,000 | 70 |
| 2 | `(107.635, -10.842, -3.000)` | 10,000 | 110 |
| 3 | `(117.756, -19.034, -3.000)` | 30,000 | 150 |

Policy parameters are transferred between stages, while the optimiser is reset when the target changes. The total budget remains 45,000 interactions. The important distinction from PPO Scratch is the distribution of collected experience: only 30,000 curriculum interactions occur on the final route, whereas all 45,000 scratch interactions occur there. Both PPO variants still update the model repeatedly from finite rollouts during flight.

### 2.4 Training scale

| Method | Environment interactions | Completed episodes | Training time |
|---|---:|---:|---:|
| DQN Scratch | 45,000 | 658 | 7.41 h |
| PPO Scratch | 45,000 | 459 | 6.50 h |
| PPO Curriculum | 5,000 + 10,000 + 30,000 | recorded by stage | approximately 6.51 h |

All reported models were trained with seed 7. This controls a major source of software randomness and makes the runs reproducible, but a single training seed does not measure variation across independently learned policies.

## 3. Evaluation Protocol

Checkpoint selection was separated from final testing:

| Phase | Protocol |
|---|---|
| Selection stage 1 | all checkpoints, 5 validation episodes each |
| Selection stage 2 | top three checkpoints, 30 validation episodes each |
| Final baseline test | selected checkpoint, 50 episodes, test seed 20007 |
| Formal generalisation test | selected checkpoint, 50 episodes per route, test seed 20007 |

The selected checkpoints were:

| Method | Checkpoint |
|---|---|
| DQN Scratch | `dqn_step_0040000.pt` |
| PPO Scratch | `ppo_step_0042500.pt` |
| PPO Curriculum | Stage 3 `ppo_step_0020000.pt` |

The test seed separates final evaluation from checkpoint selection. Because the AirSim scene, start pose, and target are fixed within a route, different episode seeds do not create new obstacle layouts. They control software-side stochasticity and action sampling where applicable.

### 3.1 Deterministic and stochastic PPO evaluation

During deterministic evaluation, PPO selects the action with the highest actor probability. During stochastic evaluation, it samples from the actor distribution. Deterministic evaluation represents the intended deployment policy and is used throughout the formal generalisation tests. Stochastic evaluation is retained as a diagnostic of the full learned action distribution.

DQN evaluation is deterministic and selects the action with the largest Q-value.

### 3.2 Metrics

| Metric | Definition |
|---|---|
| Success rate | fraction of episodes entering the 2 m goal radius |
| Collision rate | fraction recording a new collision during the episode |
| Altitude-violation rate | fraction leaving the permitted altitude interval |
| Unsafe rate | fraction with either collision or altitude violation |
| Timeout rate | fraction reaching 150 steps without another terminal condition |
| Average reward | mean accumulated shaped return |
| Average steps | mean number of decisions before episode completion |
| Average final distance | mean Euclidean distance to the target at completion |
| Average path length | mean distance travelled along the executed trajectory |
| Average minimum depth | mean of the smallest valid depth observation in each episode |

Success, collision, and timeout are not implemented as mutually exclusive categorical labels. If the drone enters the goal radius and registers a new collision on the same step, both success and collision may equal one. Consequently, a small number of rows have outcome rates whose sum exceeds 100%.

With 50 episodes, a 50/50 observation is strong evidence of high reliability but is not proof that the underlying success probability is exactly 100%. A 95% Wilson interval for 50/50 has a lower bound of approximately 92.9%; for 0/50, the corresponding upper bound is approximately 7.1%.

## 4. Original-Route Baseline Results

### 4.1 Deterministic policies

| Method | Success | Collision | Timeout | Mean reward | Mean steps | Final distance | Path length |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN Scratch | 16% | 72% | 14% | -18.47 | 86.80 | 4.47 m | 41.66 m |
| **PPO Scratch** | **98%** | **2%** | **0%** | **141.11** | **52.28** | **2.01 m** | **38.27 m** |
| PPO Curriculum | 68% | 2% | 30% | 71.21 | 83.46 | 14.66 m | 55.17 m |

PPO Scratch completed 49 of 50 episodes successfully and provided the best combination of reliability, safety, and efficiency. DQN collided in 36 of 50 episodes, indicating that its learned goal-seeking behaviour did not reliably handle the building that blocks the direct route. PPO Curriculum was generally safe but timed out in 15 of 50 episodes. Its long mean path and large mean final distance indicate inconsistent route execution rather than a collision-dominated failure mode.

### 4.2 Stochastic PPO policies

| Method | Success | Collision | Timeout | Mean reward | Mean steps | Final distance | Path length |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO Scratch | 66% | 34% | 0% | 77.72 | 64.08 | 3.26 m | 36.55 m |
| PPO Curriculum | 46% | 38% | 16% | 41.45 | 97.12 | 4.21 m | 45.47 m |

Sampling actions substantially increased collisions for both PPO policies. The difference between deterministic and stochastic results shows that the highest-probability actions form a much more reliable route than arbitrary samples from the remaining probability mass. PPO Scratch deterministic was therefore selected as the fixed policy for extended testing.

## 5. Same-Map Route Generalisation

All generalisation routes remain inside the AirSimNH map. These experiments evaluate unseen-route generalisation within a fixed scene, not transfer to a new map or visual domain.

### 5.1 Small route perturbations

| Route | Model | Success | Collision | Timeout | Mean steps | Final distance | Path length |
|---|---|---:|---:|---:|---:|---:|---:|
| Start shifted 2 m right | DQN Scratch | 0% | 98% | 2% | 25.08 | 17.50 m | 16.66 m |
| Start shifted 2 m right | **PPO Scratch** | **100%** | **0%** | **0%** | 48.08 | 1.76 m | 35.95 m |
| Start shifted 2 m right | PPO Curriculum | 42% | 60% | 0% | 49.24 | 2.00 m | 35.08 m |
| Start 2 m left; target 2 m right | DQN Scratch | 0% | 32% | 68% | 121.82 | 8.52 m | 46.11 m |
| Start 2 m left; target 2 m right | **PPO Scratch** | **100%** | **0%** | **0%** | 50.70 | 1.76 m | 35.54 m |
| Start 2 m left; target 2 m right | PPO Curriculum | 36% | 64% | 0% | 53.56 | 2.14 m | 36.83 m |

PPO Scratch retained perfect observed performance under both perturbations, demonstrating that it did not merely replay one exact world-coordinate sequence. DQN and PPO Curriculum degraded sharply, reinforcing the selection of PPO Scratch for the larger generalisation study.

### 5.2 Extended PPO Scratch tests

| Route | Straight-line distance | Success | Collision | Timeout | Mean reward | Mean steps | Final distance | Path length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Large offset with shortened target | 25.21 m | 100% | 0% | 0% | 139.64 | 44.92 | 1.45 m | 33.06 m |
| Unseen route in another map region | 33.91 m | 100% | 0% | 0% | 154.26 | 47.04 | 1.69 m | 34.98 m |
| Target 5 m left and 12 m forward | 45.19 m | 0% | 14% | 86% | -6.56 | 132.36 | 10.03 m | 53.72 m |
| Complex unseen-route failure case | 25.83 m | 0% | 78% | 22% | -99.21 | 63.16 | 25.24 m | 37.00 m |

Route coordinates are:

| Route | Start `(x, y, z)` | Target `(x, y, z)` |
|---|---|---|
| Large offset with shortened target | `(85.413, -28.334, -3)` | `(103.756, -11.034, -3)` |
| Unseen route in another map region | `(11.179, 45.493, -3)` | `(45, 48, -3)` |
| 45.19 m combined offset | `(85.413, -15.334, -3)` | `(129.756, -24.034, -3)` |
| Complex unseen-route failure case | `(11.179, 34.493, -3)` | `(37, 34, -3)` |

## 6. Interpretation

The perfect observed performance on four modified routes demonstrates genuine but bounded transfer. In particular, success on the 33.91 m route in another region of AirSimNH shows that PPO Scratch can combine depth sensing with relative-goal information outside the exact training coordinates.

The two failed routes expose different limitations. On the 45.19 m combined-offset route, 86% of episodes timed out and the mean final distance was 10.03 m. Recorded behaviour shows successful early obstacle avoidance followed by repeated corrections in a relatively open region. This is consistent with an unstable local policy outside its training distribution, rather than absence of target information. On the complex 25.83 m route, 78% of episodes collided and the mean final distance remained 25.24 m, indicating that the learned avoidance pattern was unsuitable for the local obstacle geometry.

Straight-line distance alone does not explain performance: the model solved the unseen 33.91 m route in all 50 trials but failed every trial on the shorter 25.83 m route. Obstacle arrangement, approach direction, and similarity to the training distribution are therefore more consequential than distance by itself.

The evidence is also more consistent with a narrow training distribution than with excessive training duration. Continuing to train only on the original fixed route would probably reinforce route-specific behaviour. A more principled extension would fine-tune the selected PPO checkpoint on a distribution of valid start-target pairs, route lengths, and obstacle arrangements while retaining original-route samples to reduce catastrophic forgetting.

## 7. Reproducing the Tests

Run all commands from the project root:

```powershell
cd "C:\Users\MEOW Computer\Desktop\COMP9444"
```

The Python environment must exist at `.venv`, the three pretrained weights must be available under `pretrained\airsimnh`, and AirSimNH must be installed at `.local\airsim\AirSimNH\WindowsNoEditor\AirSimNH.exe` for the generalisation runners. Each generalisation runner automatically closes an existing AirSimNH process, launches a clean windowed instance, displays the route markers, executes exactly 50 deterministic episodes per configured model, regenerates `results` and `metrics`, and closes AirSim after completion. Existing result and metric folders for that route are replaced when the command is rerun.

### 7.1 Reproduce the formal original-route checkpoint selection and test

```powershell
.\run_checkpoint_selection.ps1
```

This runner applies the common two-stage checkpoint-selection procedure to the completed seed-7 training runs and performs the final held-out evaluation. Its machine-specific AirSim executable configuration should be checked before execution. The existing consolidated result is:

`experiments\airsimnh\validated_comparison_seed7_test_seed20007.csv`

To launch a completely new controlled training and evaluation experiment after configuring a new run tag and seed:

```powershell
.\run_comparison_experiment.ps1
```

This second command includes training and may require several hours; it is not required merely to inspect or reproduce the existing evaluation tables.

### 7.2 Small-perturbation tests for all three models

Start shifted 2 m right:

```powershell
.\generalization_tests\start_right_2m\run_all_models.ps1
```

Start shifted 2 m left and target shifted 2 m right:

```powershell
.\generalization_tests\start_left_2m_target_right_2m\run_all_models.ps1
```

Each command runs DQN Scratch, PPO Scratch, and PPO Curriculum for 50 deterministic episodes each.

### 7.3 Extended PPO Scratch tests

Large-offset shortened route:

```powershell
.\generalization_tests\large_offset_start_left_13m_target_right_8m_closer_14m\run_ppo_scratch_50.ps1
```

Successful unseen route in another AirSimNH region:

```powershell
.\generalization_tests\saved_successful_same_scene_new_route\run_ppo_scratch_50.ps1
```

Longer combined-offset route:

```powershell
.\generalization_tests\original_route_target_left_5m_forward_12m\run_ppo_scratch_50.ps1
```

Complex unseen-route failure case:

```powershell
.\generalization_tests\complex_same_scene_new_route_failure_example\run_ppo_scratch_50.ps1
```

### 7.4 Outputs

Each generalisation test directory contains:

- `route.json`: start, target, offsets, and route metadata;
- `run_*.ps1`: the one-command reproducibility runner;
- `results`: per-episode logs, step-level trajectories, and evaluation summaries;
- `metrics\summary.csv` and `metrics\summary.json`: compact formal metrics; and
- `videos`: first-person and external-view behaviour demonstrations.

Videos are qualitative illustrations only. All reported rates and averages are computed from the 50-episode CSV/JSON evaluation outputs.

## 8. Validity and Limitations

The following constraints define the scope of the conclusions:

1. All selected policies originate from one training seed. Independent training seeds would be required to estimate between-training variance.
2. All routes use one AirSimNH map. The experiments measure route generalisation within a scene, not cross-map or sim-to-real transfer.
3. Test seeds do not randomise building geometry, textures, weather, or sensor noise.
4. The extended routes were chosen as targeted success and failure cases rather than sampled randomly from a predefined route distribution. They characterise capabilities and failure boundaries, but do not estimate an average generalisation rate over all possible routes.
5. The 45.19 m experiment changes both route length and lateral target position. It should be described as a long-distance combined perturbation, not as an isolated test of distance alone.
6. Average minimum depth is derived from the smallest valid depth-image pixel and should be treated as a diagnostic, not as a calibrated physical clearance measurement.

## 9. Conclusion

Under the controlled original-route protocol, PPO Scratch is decisively stronger than both the vanilla DQN baseline and the tested curriculum design. It reaches 98% deterministic success on the held-out baseline and maintains 100% observed success across several meaningful same-map route changes. These results establish that the learned policy uses perceptual and relative-goal information and is not restricted to one exact coordinate trajectory.

Nevertheless, systematic failure on two structurally different routes demonstrates that this capability remains distribution-dependent. The policy behaves as a learned reactive navigator with limited route transfer, not as a general global planner. The current evaluation therefore supports two conclusions simultaneously: PPO Scratch is highly effective for the learned route family, and broader robustness would require training over a more diverse distribution of navigation tasks.



