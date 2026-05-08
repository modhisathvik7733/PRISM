"""PRISM-v4 cognitive core — Phase 1 modules.

Builds the 5 components the user prioritized (object persistence,
predictive world model, counterfactual engine, operator abstraction,
curriculum scheduler) ON TOP of the existing frozen v1.3 JEPA. None
of these modules retrain the JEPA; they probe / wrap / cluster its
existing latent representations.

If all five emergence criteria pass on BabyAI (see plan + eval_emergence
script), this validates the bigger thesis at small scale and Phase 2
(memory + curiosity) becomes worth building.
"""
