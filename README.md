# Autonomous Highway Driving with Deep Q-Networks

This repository contains the implementation and evaluation of Reinforcement Learning (RL) agents trained to drive autonomously on a crowded highway using the **[highway-env](https://github.com/eleurent/highway-env)** simulation environment. Three variants of Deep Q-Networks (DQN) are compared: **Vanilla DQN, Double DQN, and Dueling DQN**. The RL agents are benchmarked against a **heuristic baseline** and **manual control**.

---

## 📄 Project Overview

Autonomous driving is a complex, safety-critical application of AI. The goal of this project is to train a virtual agent to navigate a densely populated highway while **maximizing progress** and **avoiding collisions**. 

Key challenges addressed:

- Sparse collision events
- Trade-offs between speed and safety
- Stability and convergence of different DQN variants
- Evaluating against interpretable baselines

The project is documented in the paper [`Autonomous_Highway_Driving_in_highway_env.pdf`](./Autonomous_Highway_Driving_in_highway_env.pdf).

---

## 🏎 Environment Setup

- **Simulator:** [highway-env](https://github.com/eleurent/highway-env)
- **State representation:** Matrix \(S\) where the first row is the ego vehicle and remaining rows represent other vehicles. Each vehicle has features:
  - Presence (boolean)
  - Normalized position along x and y axes
  - Normalized velocity along x and y axes
- **Action space (discrete, 5 actions):**
  1. Change lane left  
  2. Idle  
  3. Change lane right  
  4. Accelerate  
  5. Brake
- **Episode length:** 40 steps (~23 seconds per episode)
- **Reward function:** Combination of progress reward, lane preference, and collision penalty.

---

## 📊 Baselines

Two baseline approaches are implemented:

1. **Manual Control(`manual_control.py`)**  
   - Human-like control for comparison
   - Achieves high rewards but with unsafe driving maneuvers

2. **Heuristic Baseline (`baseline.py`)**  
   - Rule-based strategy evaluating relative positions of nearby vehicles
   - Safe but conservative and suboptimal in terms of speed

---

## 🤖 Deep Q-Network Variants

1. **Vanilla DQN (`DQN.py`)**  
   - Standard DQN implementation
   - Empirically showed the best average performance in this environment

2. **Double DQN (`DoubleDQN.py`)**  
   - Separates action selection and value estimation to reduce overestimation
   - Underperformed due to overly cautious policy

3. **Dueling DQN (`DuelingDQN.py`)**  
   - Separates state value and action advantage
   - More proactive than Double DQN but less stable than Vanilla DQN

Code and weights:

- Evaluation scripts: `DQN_eval.py`, `DoubleDQN_eval.py`, `DuelingDQN_eval.py`  
- Pretrained models: `dqn_model.pth`, `double_dqn_model.pth`, `dueling_dqn_model.pth`    

---

## 📈 Evaluation

Mean cumulative rewards over 10 episodes:

| Method               | Return (Mean ± Std) |
|----------------------|------------------|
| Manual Control       | 33.05 ± 3.67     |
| Heuristic Baseline   | 18.62 ± 6.20     |
| Vanilla DQN          | 28.75 ± 0.84     |
| Double DQN           | 19.42 ± 9.16     |
| Dueling DQN          | 27.53 ± 8.97     |

Observations:

- Vanilla DQN achieves superior performance among RL methods.
- Double DQN and Dueling DQN are less stable.
- Manual control still outperforms all RL methods, but RL demonstrates safe, adaptable driving compared to heuristic baseline.

Plots and visualizations can be generated with [`plots.py`](./plots.py).

---

## 🔑 Key Considerations

- **Hyperparameter Sensitivity:** Learning rate, discount factor, and replay buffer size significantly affect performance.  
- **Reward Function:** Dominated by high-speed reward, which heavily influences the learned policy.  
- **Sparse Collisions:** Rare events are difficult for RL to learn without careful tuning.  
- **Sim-to-Real Gap:** Policies may not directly transfer to real-world driving.  
- **Scalability:** More vehicles increase state-space complexity; hierarchical or scalable models may be required.

---

## 📚 References

- Leurent, E. (2018). *An environment for autonomous driving decision-making.* [GitHub](https://github.com/eleurent/highway-env)  
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529–533  
- Van Hasselt, H., Guez, A., Silver, D. (2016). *Deep reinforcement learning with double Q-learning.* AAAI Conference  
- Wang, Z., Schaul, T., Hessel, M., Van Hasselt, H., Lanctot, M., de Freitas, N. (2016). *Dueling network architectures for deep reinforcement learning.* ICML  


