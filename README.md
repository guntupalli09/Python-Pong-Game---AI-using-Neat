# 🧠 Neuroevolutionary Pong with NEAT

## Overview
This project explores **neuroevolution** as an alternative to gradient-based learning by evolving neural networks to play **Pong** using **NEAT (NeuroEvolution of Augmenting Topologies)**.

The emphasis is on **learning dynamics, optimization behavior, and emergent strategy**, rather than maximizing score or visual polish.

---

## Motivation
Most modern AI systems rely on backpropagation and fixed architectures. Neuroevolution instead learns through **population dynamics and selection**, evolving both **network structure and parameters**.

Pong provides a minimal yet non-trivial environment to study how coordinated behavior emerges under evolutionary pressure.

---

## Approach

### Algorithm
- NEAT (evolves topology + weights)

### State Inputs
- Paddle position  
- Relative paddle–ball distance  
- Ball position  

### Action Space
- Stay  
- Move up  
- Move down  

### Fitness Signal
- Rally length (ball hits)  
- Optional survival time (ablation-controlled)

### Training
- Population-based self-play across generations  
- Rendering disabled during training for faster experimentation

---

## Baseline Comparison (Conceptual)
- **Random policy** → near-zero fitness  
- **Fixed reactive policy** → short rallies, brittle behavior  
- **NEAT-evolved policy** → sustained rallies and anticipatory movement  

---

## Results
Agents consistently learn to track and return the ball, with increasingly stable and anticipatory behavior emerging over generations — without any hard-coded strategy.

### Training Dynamics

![Fitness vs Generation](fitness_vs_generation.png)

*Best and mean fitness across generations, showing population-level learning rather than isolated high-performing agents.*

- Rising **mean fitness** → stable evolutionary improvement  
- Faster gains in **best fitness** → exploration + selection  
- Plateaus → convergence limits and reward shaping effects  

### Network Complexity

![Network Complexity vs Generation](complexity_vs_generation.png)

*Growth in network connectivity over generations, indicating that structural complexity increases only when beneficial for performance.*

---

## Ablation: Reward Design
An explicit ablation toggles whether **survival time** contributes to fitness.

This helps distinguish between:
- Passive survival behavior  
- Active, skillful return strategies  

The experiment highlights how **reward shaping influences emergent behavior** in evolutionary systems.

---

## Limitations
- Computationally expensive compared to gradient-based RL  
- High variance across random seeds  
- Agents occasionally exploit fitness by tracking the ball without optimal return angles, highlighting sensitivity to reward design  

---

## Future Work
- Compare NEAT against policy-gradient RL on the same environment  
- Analyze robustness under noise and partial observability  
- Study complexity–performance tradeoffs under different mutation regimes  

---

## Tech Stack
- Python  
- NEAT-Python  
- Pygame  
- Matplotlib  

---
## Why This Matters

This project focuses on how learning and intelligence emerge from simple rules, and how different optimization paradigms shape behavior — core questions in frontier AI research beyond any single algorithm.

This project prioritizes insight into learning dynamics over polished gameplay.

## Run
```bash
pip install neat-python pygame matplotlib
python main.py

---







