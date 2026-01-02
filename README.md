Neuroevolutionary Pong with NEAT
Overview

This project explores neuroevolution as an alternative to gradient-based learning by evolving neural networks to play Pong using NEAT (NeuroEvolution of Augmenting Topologies).
The emphasis is on learning dynamics, optimization behavior, and emergence of strategy, rather than maximizing game score.

Motivation

Most modern AI systems rely on backpropagation and fixed architectures. Neuroevolution instead learns through population dynamics and selection, evolving both weights and network structure.
Pong provides a minimal yet non-trivial environment to study how coordinated behavior emerges under evolutionary pressure.

Approach

Algorithm: NEAT (evolves topology + weights)

Inputs: Paddle position, relative ball distance, ball position

Actions: Stay, move up, move down

Fitness Signal: Rally length (hits) + survival time

Training: Population-based self-play across generations

Results

Agents consistently learn to track and return the ball, with increasingly stable and anticipatory behavior emerging over generations—without any hard-coded strategy.

Training Dynamics

The plot below shows best and mean fitness per generation, highlighting population-level learning rather than isolated high-performing agents.

Rising mean fitness indicates stable evolutionary improvement

Faster gains in best fitness reflect exploration and selection

Plateaus suggest convergence limits and fitness shaping effects

This provides direct insight into optimization behavior, not just final performance.

Limitations

Computationally expensive compared to gradient-based RL

High variance across random seeds

Fitness design strongly influences convergence behavior

Future Work

Compare NEAT against policy-gradient RL on the same environment

Analyze robustness under noise and partial observability

Study network complexity growth vs performance

Tech Stack

Python · NEAT-Python · Pygame · Matplotlib

Run
pip install neat-python pygame matplotlib
python main.py

Why this matters

This project reflects an interest in how learning and intelligence emerge from simple rules, and how different optimization paradigms shape behavior—core questions in AI research beyond any single algorithm.
# NEAT-Pong-Python
Using the NEAT algorithm to train an AI to play pong in Python!


https://github.com/guntupalli09/Python-Pong-Game---AI-using-Neat/assets/142756616/fd237995-30e1-4b9b-ba5f-e8251604a59a






