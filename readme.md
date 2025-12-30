
# Dynamic Path Planning in Unknown Environments (DDQN + LiDAR) — Reimplementation

This repository is a **course project reimplementation/adaptation** of the paper:

 **“Dynamic Path Planning of Unknown Environment Based on Deep Reinforcement Learning”** (Lei et al., 2018)

The original paper proposes a DDQN-based **local planner** using **LiDAR** to avoid obstacles in **unknown dynamic environments**, typically as part of a broader navigation stack (global path planning + local replanning). In this repo, we focus on the **RL local navigation component**, implemented and trained in a modern vectorized simulator setup (VMAS), with a set of practical adaptations for reproducibility and learning stability.

## Table of Contents

- [What’s inside](#whats-inside)
- [Key idea (paper vs. this repo)](#key-idea-paper-vs-this-repo)
- [Environment](#environment)
- [Observation & Action](#observation--action)
- [Reward](#reward)
- [Training (DDQN)](#training-ddqn)
- [Quickstart](#quickstart)
- [How to run](#how-to-run)
- [Configuration](#configuration)
- [Results & known limitations](#results--known-limitations)
- [Repo structure](#repo-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## What’s inside

- **Vectorized DDQN training** (parallel envs, shared replay buffer)
- **CNN-based Q-network** taking a 2×20×20 observation tensor
- **Curriculum learning** on start–goal distance
- **Reward shaping add-ons** to reduce sparsity when only a final goal is available
- **Evaluation + plotting utilities**, including an A* path overlay tool for map visualization

---

## Repo structure

```
.
├── README.md
├── main_train.py                # training entrypoint
├── utils_eval.py                # plotting + A* overlay + greedy rollout eval
├── agent/
│   ├── ddqn_agent.py            # CNN Q-network + helpers
│   └── params.py                # hyperparameters
└── environment/
    ├── map_layouts.py           # ASCII maps (W/U/S/.)
    └── path_planning.py         # VMAS scenario + reward/obs logic
```
---

## Key idea (paper vs. this repo)

### Paper framing (high-level)
The paper positions DDQN as a **local planner** that uses **LiDAR + local targets** (waypoints) to follow a global path while reacting to obstacles in unknown environments.

### This repo’s framing
In our implementation, we do **not** rely on global-path waypoints during training. Instead, we train the agent to navigate toward the **final goal directly**, using:
- a documented observation pipeline,
- explicit hyperparameters,
- a shaped reward (progress + safety penalties),
- and a curriculum on start distance.

This choice improves reproducibility when waypoint supervision or simulator details are missing, but also makes the learning problem **more reward-sparse** unless shaped appropriately.

---

## Environment

We use **VMAS** to simulate a 2D world with:
- a robot agent,
- walls,
- and (optional) dynamic obstacles.

Maps are encoded as **ASCII layouts** (grid-based), where:
- `W` = wall
- `U` = obstacle/object (used for map/plotting and scenario construction)
- `S` = goal marker (primarily for visualization in tooling)
- `.` = free space

The default grid used in this repo is:
- **28 (width) × 18 (height)** cells  
- **0.2 m** per cell (≈ 5.6 m × 3.6 m world)

> See `environment/map_layouts.py` for layouts and comments.

---

## Observation & Action

### Observation (CNN input)
The training loop consumes observations shaped as:

- **(num_envs, 2, 20, 20)**

The channels correspond to an “image-like” representation of:
- **LiDAR distance + angle**
- plus **goal information** embedded into the tensor (see report for design notes)

> Note: reshaping sequential LiDAR into 2D is one of the contentious/unstable design points discussed in the report.

### Action space
Discrete action space of **8 movement directions**:
- up, down, left, right,
- and the 4 diagonals.

---

## Reward

The paper’s baseline reward is short and event-driven (goal/collision/time penalty). When training directly toward a **final goal**, reward sparsity becomes a major problem.

This repo therefore uses **reward shaping add-ons**, including:
- a **progress indicator**: reward/penalty depending on whether distance to goal decreases,
- a **LiDAR-based proximity penalty** to discourage risky motion near obstacles.

See the report for the motivation and equations, and `environment/path_planning.py` for the implementation.

---

## Training (DDQN)

Training follows a standard DDQN loop:
- Q-network + target Q-network
- replay buffer sampling
- TD target uses *argmax from online network* and *value from target network*
- **soft target updates** (Polyak averaging)
- **gradient clipping** for stability
- **epsilon-greedy** exploration with exponential decay
- **curriculum learning** on start distance `L`

The implementation is vectorized with many environments running in parallel.

---

## Quickstart

### 1) Create an environment
You can use `venv` or `conda`. Example:

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
````

### 2) Install dependencies

This repo currently doesn’t ship a pinned `requirements.txt`, but imports indicate you’ll need:

* `torch`
* `vmas`
* `torchrl`
* `tensordict`
* `numpy`
* `matplotlib`
* `tqdm`

Example:

```bash
pip install torch numpy matplotlib tqdm
pip install vmas torchrl tensordict
```

> If VMAS/torchrl versions conflict with your PyTorch version, align them by installing compatible versions together.

---

## How to run

### Train

```bash
python main_train.py
```

This will:

* run training in **parallel environments**
* log training stats to `training_log.csv`
* periodically save a checkpoint (`test_ddqn.pt`)
* save the final model to `ddqn_q_network.pt`
* generate plots into a timestamped folder under `results/`

### Evaluate / visualize a path

`utils_eval.py` contains helpers to:

* load a checkpoint
* run a single environment greedily
* plot the agent path on top of the map (with an A* path for reference)

By default, running:

```bash
python utils_eval.py
```

will call:

* `eval_path_agent("test_ddqn.pt")`

You can change it to evaluate:

* `ddqn_q_network.pt`
  by editing the bottom of `utils_eval.py`.

---

## Configuration

Hyperparameters live in:

* `agent/params.py`

Key defaults (summary):

* learning rate, gamma
* epsilon schedule
* replay buffer capacity, batch size
* training start steps & frequency
* max steps/episode
* curriculum thresholds (`L_MIN`, `L_MAX`, `N1_THRESHOLD`, `N2_THRESHOLD`)

To reproduce experiments, **commit your exact hyperparameters** (and ideally add a `requirements.txt` + seed handling).

---

## Results & known limitations

The report documents:

* why the original paper is hard to reproduce precisely (missing hyperparameters and simulator details),
* instability issues in DDQN training,
* and a key concern: reshaping 1D LiDAR into 2D “images” can destroy locality assumptions of CNNs and contribute to weak gradients.

If you want to improve this project, good next steps include:

* using a 1D encoder for LiDAR (1D CNN / Transformer) instead of forced 2D reshaping,
* better goal injection (FiLM / concatenation into FC layers),
* prioritized replay / n-step returns,
* reward shaping validation & ablations,
* adding deterministic seeds and a pinned environment file.

---

## Citation

If you use this work, please cite the original paper:

**Lei, X., Zhang, Z., & Dong, P. (2018).** *Dynamic Path Planning of Unknown Environment Based on Deep Reinforcement Learning.* Journal of Robotics.

BibTeX (paper):

```bibtex
@article{lei2018dynamic,
  title={Dynamic Path Planning of Unknown Environment Based on Deep Reinforcement Learning},
  author={Lei, Xiaoyun and Zhang, Zhian and Dong, Peifang},
  journal={Journal of Robotics},
  year={2018}
}
```

You can also cite the project report:

* `INF8250AE_Report.pdf`

---

## Acknowledgements

* The original research paper by Lei et al.
* VMAS + TorchRL ecosystem for vectorized RL simulation/training
* INF8250AE course context and evaluation framework
