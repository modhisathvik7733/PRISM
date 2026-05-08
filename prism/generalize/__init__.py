"""Generalization fork — extends the v1.3 stack to Pickup, GoTo, Open.

Everything in this package is additive — no module here modifies code in
prism/. The fork pattern: build new helpers that compose with existing
classes by wrapping them, never by editing them.
"""
