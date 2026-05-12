"""Continual learning machinery for PRISM.

Modules:
- sparse_hopfield_update: Sparse Memory Finetuning pattern (Lin 2025) —
  zero gradients on Hopfield slots that didn't activate, so updates are
  localized and don't disturb other concepts.
- continual_backprop: Random reinit of dead encoder units (Sutton 2024 Nature)
  to prevent loss of plasticity in long task streams.
"""
