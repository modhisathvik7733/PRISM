"""LangGoalProvider — wraps a trained text→(color, type) classifier
into a callable that turns a mission string into a (type_id, color_id)
tuple suitable for PRISM's PPO mission_oh encoding.

Used by `scripts/ppo_train.py` and `scripts/cog_core/train_curriculum.py`
when `--goal-source lang` is set, replacing the rule-based
`goal_predicates_for_mission` parser. The rule parser is still used to
get `spec` and `allowed_actions` (mission *type*: go-to vs pickup vs
put-down); only the (color, type) of the goal object is taken from
the language model.

Stage 1.2: this is the swap that makes language drive the agent's goal
instead of a regex parser.
"""

from __future__ import annotations

from pathlib import Path

import torch

from prism.language.grounding_head import WhitespaceVocab
from prism.language.grounding_predicate_head import make_dual_head
from prism.perception.slots import NUM_COLORS, NUM_TYPES, OBJECT_TYPES


class LangGoalProvider:
    """Callable: mission_text → (type_id, color_id).

    Loads a trained TinyTransformerDualHead / BoWDualHead checkpoint plus
    its vocab and runs them on each mission. Predictions are deterministic
    (argmax over softmax) so this can be used identically in training and
    eval.
    """

    def __init__(
        self,
        lang_checkpoint: str | Path,
        vocab_checkpoint: str | Path,
        device: torch.device,
        kind: str | None = None,
    ):
        self.device = device
        self.vocab = WhitespaceVocab.load(str(vocab_checkpoint))
        ckpt = torch.load(
            str(lang_checkpoint), map_location=device, weights_only=False,
        )
        lang_kind = kind or ckpt.get("kind", "tiny_tf")
        self.model = make_dual_head(
            lang_kind, self.vocab.size, NUM_COLORS, NUM_TYPES,
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, mission: str) -> tuple[int, int]:
        """Returns (type_id, color_id) for the predicted goal object.
        type_id matches the env's int code (i.e. OBJECT_TYPES[type_idx]).
        Returns (-1, -1) if mission tokens are entirely OOV and no
        meaningful prediction is possible."""
        tokens, mask = self.vocab.encode_batch([mission])
        if not bool(mask.any().item()):
            return -1, -1
        tokens = tokens.to(self.device)
        mask = mask.to(self.device)
        out = self.model(tokens, mask)
        if isinstance(out, tuple):
            c_logits, t_logits = out
        else:
            c_logits = out[:, :NUM_COLORS]
            t_logits = out[:, NUM_COLORS:NUM_COLORS + NUM_TYPES]
        color_id = int(c_logits.argmax(-1).item())
        type_idx = int(t_logits.argmax(-1).item())
        if not (0 <= type_idx < len(OBJECT_TYPES)):
            return -1, -1
        return int(OBJECT_TYPES[type_idx]), color_id
