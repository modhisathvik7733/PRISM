"""PRISM-Lang v3.1 — pretrained AR backbone + JEPA-style latent middle.

v3.0 (prism/lang/) trained the encoder, middle, and decoder all from
scratch on bAbI. Result: 57.3% mean across 20 tasks, but the LM head
collapsed to bAbI's closed answer vocabulary so the model can't speak
free-form English.

v3.1 keeps the JEPA-middle thesis but **swaps the from-scratch AR edges
for pretrained GPT-2 weights**. The model speaks fluent English from
day 1 (GPT-2's pretraining); the latent middle is the only new piece
trained on top. Reasoning is supervised via Coconut-style continuous-
thought curriculum on GSM8K math.

Everything in this package is additive — nothing under prism/ outside
this directory is modified, including prism/lang/ which keeps the v3.0
result reproducible.
"""
