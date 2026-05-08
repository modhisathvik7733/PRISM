"""Crafter port — extends PRISM's world-model + recurrent-policy + PPO
recipe from BabyAI gridworld to Crafter (procedural 2D Minecraft-like).

Crafter is the standard "non-toy" benchmark for world-model RL. DreamerV3
hits ~12% geometric-mean achievement score; PPO-from-scratch hits ~5%.
This fork measures where PRISM's stack lands on the same benchmark.

All code here is additive — no module under prism/ outside this package
is modified.
"""
