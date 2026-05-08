"""InjectingTeacher — wraps GroundedAgent's memory mode to actually solve
Pickup-v0 and Open-v0 episodes.

The hand-coded memory teacher in `GroundedAgent._memory_select` only emits
navigation actions {0, 1, 2}. That works for go-to missions because the env
auto-grants reward on adjacent+facing+forward, but Pickup needs action 3
(pickup) and Open needs action 5 (toggle). Without those, the teacher
reaches the goal then stalls until timeout, and the BC dataset never shows
the policy how to interact.

This wrapper runs the memory mode unchanged, then post-processes its output:
when the agent is adjacent + facing the goal AND the mission predicate is
'holding' or 'open', the FORWARD action is replaced with the appropriate
interaction action (3 or 5). Everything else passes through.

We intentionally don't edit `prism/agents/grounded_agent.py` — keeping the
existing v1.3 behavior bit-for-bit intact is the whole point of this fork.
"""

from __future__ import annotations

from prism.agents.grounded_agent import GroundedAgent

FORWARD = 2
PICKUP = 3
TOGGLE = 5


class InjectingTeacher:
    """Thin pass-through wrapper around `GroundedAgent`.

    Use exactly like the underlying agent: call `reset()` between episodes
    and `select_action(obs, goal_preds, allowed_actions=..., spec=...)` per
    step. The `spec` argument is the only API addition — pass the parsed
    `GoalSpec` so the wrapper knows which interaction action to inject.
    """

    def __init__(self, agent: GroundedAgent):
        self.agent = agent

    def reset(self) -> None:
        self.agent.reset()

    def select_action(
        self,
        obs,
        goal_preds,
        *,
        allowed_actions,
        spec,
    ):
        action, info = self.agent.select_action(
            obs, goal_preds, allowed_actions=allowed_actions
        )
        # Memory mode emits FORWARD when adjacent+facing the goal — fine for
        # go-to (env auto-rewards on overlap), wrong for pickup/open. Inject
        # the interaction action only at the precise moment the existing
        # teacher would have stepped onto the target.
        if action != FORWARD:
            return action, info
        adj = info.get("p_adjacent", 0.0) > 0.5
        facing = info.get("p_facing", 0.0) > 0.5
        if not (adj and facing):
            return action, info
        if spec.predicate == "holding" and PICKUP in allowed_actions:
            info["chosen"] = PICKUP
            info["branch"] = "inject_pickup"
            return PICKUP, info
        if spec.predicate == "open" and TOGGLE in allowed_actions:
            info["chosen"] = TOGGLE
            info["branch"] = "inject_toggle"
            return TOGGLE, info
        return action, info
