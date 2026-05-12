"""Unit tests for PRISM-Hybrid v5.0 components.

Run:
    cd /workspace/PRISM
    python -m pytest tests/test_hybrid_components.py -v

Or quick smoke run without pytest:
    python tests/test_hybrid_components.py
"""

from __future__ import annotations

import os
import sys

import torch

# Ensure vendor path is set up.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_concept_memory_forward():
    from prism.cog_core.concept_memory import ConceptMemory
    m = ConceptMemory(latent_dim=128, n_slots=64, slot_dim=32, n_heads=4)
    z = torch.randn(4, 128)
    out = m(z)
    assert out.shape == (4, 32), f"got {out.shape}"

    out2, attn = m(z, return_attention=True)
    assert out2.shape == (4, 32)
    assert attn.shape == (4, 64), f"got {attn.shape}"
    # Attention should sum to ~1 across slots (it's a softmax distribution).
    assert torch.allclose(attn.sum(dim=-1), torch.ones(4), atol=1e-3)
    print("✓ test_concept_memory_forward")


def test_concept_memory_save_load(tmp_path=None):
    import tempfile
    from prism.cog_core.concept_memory import ConceptMemory
    m = ConceptMemory(latent_dim=64, n_slots=32, slot_dim=16, n_heads=4)
    m.name_slot(5, "test_slot", {"color": "red"})

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        m.save(f.name)
        m2 = ConceptMemory.load(f.name, torch.device("cpu"))
    os.unlink(f.name)

    assert m2.n_slots == 32
    assert m2.slot_metadata[5]["name"] == "test_slot"
    print("✓ test_concept_memory_save_load")


def test_operator_memory_select():
    from prism.cog_core.operator_memory import OperatorMemory
    m = OperatorMemory(latent_dim=64, n_slots=16, slot_dim=16, n_heads=4)
    z = torch.randn(1, 64)
    slot, conf = m.select_operator(z)
    assert 0 <= slot < 16
    assert 0.0 <= conf <= 1.0
    print(f"✓ test_operator_memory_select (slot={slot}, conf={conf:.3f})")


def test_transformer_dynamics():
    from prism.models.transformer_dynamics import TransformerDynamics
    d = TransformerDynamics(
        concept_dim=32, n_actions=7, mission_dim=64,
        token_dim=64, n_layers=2, n_heads=4, max_seq_len=16,
    )
    B, T = 2, 8
    concepts = torch.randn(B, T, 32)
    actions = torch.randint(0, 7, (B, T))
    mission = torch.randn(B, 64)
    out = d(concepts, actions, mission)
    assert out["next_concept"].shape == (B, T, 32)
    assert out["action_logits"].shape == (B, T, 7)
    assert out["value"].shape == (B, T)
    print("✓ test_transformer_dynamics")


def test_dynamics_step():
    from prism.models.transformer_dynamics import (
        TransformerDynamics, TransformerDynamicsStep,
    )
    d = TransformerDynamics(concept_dim=32, n_actions=7, mission_dim=64,
                            token_dim=64, n_layers=2)
    step = TransformerDynamicsStep(d, buffer_size=8)
    B = 3
    buf = step.init_buffer(B, torch.device("cpu"))
    for _ in range(5):
        c = torch.randn(B, 32)
        a = torch.randint(0, 7, (B,))
        out, buf = step.step(c, a, buf, mission_emb=torch.randn(B, 64))
        assert out["action_logits"].shape == (B, 7)
    assert buf["concepts"].size(1) == 5
    print("✓ test_dynamics_step")


def test_concept_to_text():
    from prism.language.concept_to_text import ConceptToText
    m = ConceptToText(
        vocab_size=128, concept_dim=32, hidden_dim=64,
        n_layers=2, n_heads=4, max_len=16,
    )
    B, K = 2, 4
    concepts = torch.randn(B, K, 32)
    trunk = torch.randn(B, 64)
    tokens = torch.randint(0, 128, (B, 8))
    logits = m(concepts, trunk, tokens)
    assert logits.shape == (B, 8, 128)

    gen = m.generate(concepts, trunk, max_len=10)
    assert gen.dim() == 2
    assert gen.size(0) == B
    print("✓ test_concept_to_text")


def test_cycle_loss():
    from prism.cog_core.concept_memory import ConceptMemory
    from prism.language.cycle_loss import CycleConsistencyLoss
    cm = ConceptMemory(latent_dim=64, n_slots=32, slot_dim=16, n_heads=4)
    cycle = CycleConsistencyLoss(
        vocab_size=128, text_emb_dim=32, latent_dim=64,
        hidden_dim=64, n_layers=1, n_heads=4,
    )
    z = torch.randn(2, 64)
    original_concept = cm(z, return_attention=False)
    tokens = torch.randint(1, 128, (2, 10))
    out = cycle(tokens, cm, original_concept)
    assert "cycle_loss" in out
    assert out["cycle_loss"].requires_grad
    print(f"✓ test_cycle_loss (loss={out['cycle_loss'].item():.4f})")


def test_sparse_hopfield_optimizer():
    from prism.cog_core.concept_memory import ConceptMemory
    from prism.training.sparse_hopfield_update import SparseHopfieldOptimizer
    m = ConceptMemory(latent_dim=32, n_slots=16, slot_dim=16, n_heads=4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sparse = SparseHopfieldOptimizer(m, opt, threshold=0.05)

    z = torch.randn(4, 32)
    out, attn = m(z, return_attention=True)
    loss = out.sum()

    sparse.zero_grad()
    loss.backward()
    sparse.record_attention(attn)
    sparse.step()
    print("✓ test_sparse_hopfield_optimizer")


def test_hybrid_policy_step_with_value():
    """HybridPolicy must be drop-in compatible with RecurrentPolicy.step_with_value:
    accept tensor h_prev, return (logits, value, h_next_tensor)."""
    from prism.models.hybrid_policy import HybridPolicy
    policy = HybridPolicy(
        latent_in_dim=128, n_actions=7, mission_dim=24,
        hidden_dim=128, latent_proj_dim=64,
        concept_n_slots=64, concept_slot_dim=32,
        operator_n_slots=16, operator_slot_dim=32,
    )
    B = 2
    z = torch.randn(B, 128)
    prev_a = torch.tensor([-1, 3], dtype=torch.long)
    mission = torch.zeros(B, 24); mission[0, 5] = 1.0; mission[1, 10] = 1.0
    h = policy.init_hidden(B, torch.device("cpu"))

    assert isinstance(h, torch.Tensor), f"init_hidden must return a tensor, got {type(h)}"
    assert h.shape == (B, 128)

    logits, value, h_next = policy.step_with_value(z, prev_a, mission, h)
    assert logits.shape == (B, 7), f"logits shape {logits.shape}"
    assert value.shape == (B,), f"value shape {value.shape}"
    assert h_next.shape == (B, 128), f"h_next shape {h_next.shape}"
    assert isinstance(h_next, torch.Tensor), "h_next must be a tensor for PPO"
    print("✓ test_hybrid_policy_step_with_value")


def test_hybrid_policy_inspection():
    """HybridPolicy exposes active concept and operator slots for inspection."""
    from prism.models.hybrid_policy import HybridPolicy
    policy = HybridPolicy(
        latent_in_dim=128, n_actions=7, mission_dim=24,
        hidden_dim=128, latent_proj_dim=64,
        concept_n_slots=32, concept_slot_dim=32,
        operator_n_slots=8, operator_slot_dim=32,
    )
    z = torch.randn(1, 128)
    active = policy.get_active_concepts(z, threshold=0.0)
    assert len(active) >= 1, "should find at least one active concept"
    op_slot, op_conf = policy.get_active_operator(z)
    assert 0 <= op_slot < 8
    assert 0.0 <= op_conf <= 1.0
    print(f"✓ test_hybrid_policy_inspection "
          f"(active_concepts={len(active)}, op_slot={op_slot}, conf={op_conf:.3f})")


def test_hybrid_policy_no_operator():
    """HybridPolicy with use_operator_memory=False should still work."""
    from prism.models.hybrid_policy import HybridPolicy
    policy = HybridPolicy(
        latent_in_dim=128, n_actions=7, mission_dim=24,
        hidden_dim=64, latent_proj_dim=64,
        concept_n_slots=32, concept_slot_dim=32,
        use_operator_memory=False,
    )
    B = 2
    z = torch.randn(B, 128)
    prev_a = torch.tensor([-1, 0], dtype=torch.long)
    mission = torch.zeros(B, 24); mission[:, 0] = 1.0
    h = policy.init_hidden(B, torch.device("cpu"))
    logits, value, h_next = policy.step_with_value(z, prev_a, mission, h)
    assert logits.shape == (B, 7)
    print("✓ test_hybrid_policy_no_operator")


def run_all():
    print("Running PRISM-Hybrid v5.0 smoke tests...")
    test_concept_memory_forward()
    test_concept_memory_save_load()
    test_operator_memory_select()
    test_transformer_dynamics()
    test_dynamics_step()
    test_concept_to_text()
    test_cycle_loss()
    test_sparse_hopfield_optimizer()
    test_hybrid_policy_step_with_value()
    test_hybrid_policy_inspection()
    test_hybrid_policy_no_operator()
    print("\n✓ ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()
