"""PRISM-Lang — AR encoder + JEPA-style latent middle + AR decoder.

The PRISM thesis (structured latent world model in the middle, weights
are not the knowledge store) ported from gridworld RL to language. AR
layers handle the language interface (tokenize → understand,
generate → speak); the middle does N steps of continuous latent
reasoning over a small set of "thought tokens" — like the Coconut paper
(arXiv 2412.06769) but with JEPA-style optional aux supervision.

Everything in this package is additive — nothing under prism/ outside
this directory is modified.
"""
