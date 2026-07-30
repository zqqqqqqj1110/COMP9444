# literature review

无人机导航是一个具有挑战性的决策问题。早期代表性研究通常依赖模型驱动的运动规划与控制方法。LaValle 和 Kuffner[1]提出了随机动力学规划方法，这是一种基于采样的运动规划方法，能够在高维状态空间中进行搜索，并为具有运动和动力学约束的系统生成可行轨迹。随后，Hoffmann 等人[2]通过理论建模与实验验证，系统研究了无人机的飞行动力学和控制问题，展示了飞行研究的可行性，并为后续轨迹跟踪和导航控制方法提供了实验基础。Mellinger 和 Kumar[3]提出了最小 snap 轨迹生成方法，使四旋翼能够生成并跟踪平滑且动态可行的快速轨迹，是传统优化式无人机轨迹规划中的代表性工作。总体而言，这些传统方法对四旋翼飞行器的控制和运动规划做出了重要贡献，但它们通常依赖显式动力学模型、预定义航点、手工设计的代价函数或精确的环境表征。这限制了它们直接从高维视觉观测中学习导航行为，以及适应未知或变化环境的能力。

随着深度强化学习的发展，无人机自主导航获得了一种新的解决思路。强化学习允许智能体通过与环境交互学习行为策略，而深度神经网络则可以处理图像等高维观测信息。Mnih 等人[4]提出的 DQN 将 Q-learning 与深度神经网络结合，使智能体能够直接从高维视觉输入中学习动作价值函数，并在 Atari 控制任务中取得接近人类水平的表现。除了基于价值函数的方法，策略优化方法也是深度强化学习的重要方向。Schulman 等人[5]提出的 PPO 通过限制策略更新幅度来提高训练稳定性，同时保持实现上的简洁性和较好的样本效率。为了将这类深度强化学习方法应用到无人机导航任务中，需要一个安全、可重复且具有真实感的训练和评估环境。Shah 等人[6]提出的 AirSim 是一个基于 Unreal Engine 的高保真视觉与物理仿真平台，支持无人机动力学、传感器观测和程序化控制接口。因此，AirSim 为训练和评估基于 DQN 与 PPO 的自主无人机导航策略提供了合适的实验平台。

本项目基于 AirSim，研究深度强化学习在自主无人机导航中的应用。无人机需要从固定起点出发，在避开障碍物的同时到达指定终点。选择 DQN 和 PPO 作为对比方法，分别代表基于价值函数和基于策略优化的深度强化学习算法，以分析它们在导航任务中的训练效果、避障能力和稳定性差异。实验首先在小镇场景中进行训练，设置相同起点和终点，并设计简单、中等和较长且包含障碍物的可达路线。测试阶段先评估模型在原训练路线上的表现，再测试其在小镇中新路线上的适应能力，最后将训练好的模型迁移到新场景中，观察其跨场景泛化能力。最后通过 DQN 与 PPO 对比、多难度路线训练和跨场景测试，评估深度强化学习方法在自主无人机导航任务中的能力，局限性与鲁棒性



​	Autonomous drone navigation(ADF) is a complex task, and researches in early are mainly depend on motion Planning and control Methods based on math model. LaValle and Kuffner[1] proposed a stochastic dynamic programming method, which is a sampling-based motion planning method that can search in a high-dimensional state space and generate feasible trajectories for systems with motion and dynamic constraints. And than, Hoffmann et al. [2] systematically studied the flight dynamics and control problems of ADF through theoretical modeling and experimental verification, demonstrated the feasibility of flight research, and lay the foundation  of subsequent trajectory tracking and navigation control methods. Mellinger and Kumar[3] proposed a minimum snap trajectory generation method that enables drones to generate and track smooth and dynamically feasible fast trajectories. These are representative works in traditional optimized drone trajectory planning. However, these methods are highly depending on dynamic model and defined environment, which limits their ability to learn navigation behavior directly from high-dimensional visual observations, as well as their capacity to adapt to unknown or changing environments.

​	With the development of Deep reinforcement learning(DRL), ADF has gained a new idea for solving. RL allows agents to learn behavioral strategies by interacting with their environment, while deep neural networks(DNN) can process high-dimensional observational information such as images sent by drone currently. The DQN proposed by Mnih et al. [4] combines Q-learning with deep neural networks, enabling agents to learn action value functions directly from high-dimensional visual inputs and achieve near-human performance in Atari control tasks. In addition to value function-based methods, policy optimization methods are also an important direction of deep reinforcement learning. The PPO proposed by Schulman et al. [5] improves training stability by limiting the policy update magnitude, while maintaining simplicity in implementation and good sample efficiency. In order to apply this type of deep reinforcement learning method to UAV navigation tasks, a safe, repeatable and realistic training and evaluation environment is needed. AirSim proposed by Shah et al. [6] is a high-fidelity visual and physical simulation platform based on Unreal Engine, which supports UAV dynamics, sensor observation and programmable control interface. Therefore, AirSim provides a suitable experimental platform for training and evaluating autonomous UAV navigation strategies based on DQN and PPO.

​	This project is based on AirSim, we will do some experiments of DRL in autonomous drone navigation. The drone needs to start from a starting point and reach a designated destination while avoiding obstacles. DQN and PPO are chosen as comparison methods, representing value function-based and policy optimization-based deep reinforcement learning algorithms respectively, to analyze their training performance, obstacle avoidance capabilities, and stability differences in navigation tasks. Experiments are first conducted in a small-town scenario, with the same starting and ending points, and simple, medium, and long reachable routes containing obstacles are designed. The testing phase first evaluates the model's performance on the original training routes, then tests its adaptability on the new routes in the small town, and finally transfers the trained model to the new scenario to observe its cross-scenario generalization ability. Finally, through comparisons between DQN and PPO, training on routes of varying difficulty, and cross-scenario testing, the capabilities, limitations, and robustness of deep reinforcement learning methods in autonomous drone navigation tasks are evaluated.







[1] LaValle, S.M. and Kuffner Jr, J.J., 2001. Randomized kinodynamic planning. *The international journal of robotics research*, *20*(5), pp.378-400.

[2 ]Hoffmann, G., Waslander, S. and Tomlin, C., 2008, August. Quadrotor helicopter trajectory tracking control. In *AIAA guidance, navigation and control conference and exhibit* (p. 7410).

[3] Mellinger, D. and Kumar, V., 2011, May. Minimum snap trajectory generation and control for quadrotors. In *2011 IEEE international conference on robotics and automation* (pp. 2520-2525). Ieee.

[4] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. *nature*, *518*(7540), pp.529-533.

[5] Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. *arXiv preprint [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)*.

[6] Shah, S., Dey, D., Lovett, C. and Kapoor, A., 2017, November. Airsim: High-fidelity visual and physical simulation for autonomous vehicles. In *Field and service robotics: Results of the 11th international conference* (pp. 621-635). Cham: Springer International Publishing.