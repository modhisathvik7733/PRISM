#!/usr/bin/env bash
# PRISM-Hybrid v5.0 — One-shot Vast.ai setup script.
#
# Usage on a fresh Vast.ai instance (PyTorch 2.x + CUDA 12.x):
#   cd /workspace/PRISM
#   bash scripts/setup_hybrid.sh
#
# This script:
#   1. Vendors the hflayers library (BSD-3-Clause) into prism/_vendor/
#   2. Installs Python dependencies
#   3. Installs Ollama and pulls phi3:mini for concept naming
#   4. Runs smoke tests to verify everything imports cleanly
#
# Expected runtime: ~5 minutes (mostly the Ollama model download).

set -e

PRISM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PRISM_ROOT"

echo "==========================================="
echo "PRISM-Hybrid v5.0 setup"
echo "Root: $PRISM_ROOT"
echo "==========================================="

# ---- 1. Activate PRISM venv if it exists ----
if [ -f "$PRISM_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PRISM_ROOT/.venv/bin/activate"
    echo "[setup] activated .venv"
else
    echo "[setup] no .venv found — run setup.sh first"
    exit 1
fi

# ---- 2. Vendor hflayers ----
VENDOR_DIR="$PRISM_ROOT/prism/_vendor"
if [ ! -d "$VENDOR_DIR/hflayers" ]; then
    echo "[setup] vendoring hflayers..."
    mkdir -p "$VENDOR_DIR"
    git clone --depth 1 https://github.com/ml-jku/hopfield-layers /tmp/hopfield-layers
    cp -r /tmp/hopfield-layers/hflayers "$VENDOR_DIR/hflayers"

    # Add __init__.py to make _vendor a package
    touch "$VENDOR_DIR/__init__.py"

    # Provenance record.
    cat > "$VENDOR_DIR/hflayers/PROVENANCE.md" <<EOF
# hflayers — vendored

Source: https://github.com/ml-jku/hopfield-layers
Commit: $(cd /tmp/hopfield-layers && git rev-parse HEAD)
Date vendored: $(date -u +%Y-%m-%d)
License: BSD-3-Clause (see LICENSE in this directory)

DO NOT MODIFY. To update, re-run scripts/setup_hybrid.sh.
EOF

    # Copy license.
    cp /tmp/hopfield-layers/LICENSE "$VENDOR_DIR/hflayers/LICENSE" 2>/dev/null || true

    rm -rf /tmp/hopfield-layers
    echo "[setup] hflayers vendored at $VENDOR_DIR/hflayers"
else
    echo "[setup] hflayers already vendored — skipping"
fi

# ---- 3. Install Python deps needed by hybrid components ----
echo "[setup] installing Python deps into active venv..."
# PRISM's venv is created via uv and may not contain pip.
# Try uv pip first (preferred — matches how PRISM was set up), then
# bootstrap pip via ensurepip, then fall back to system pip --target.
DEPS=("requests>=2.28" "numpy>=1.20")

if command -v uv &> /dev/null; then
    echo "[setup] using uv pip"
    uv pip install --quiet "${DEPS[@]}"
elif python -m pip --version &> /dev/null; then
    echo "[setup] using python -m pip"
    python -m pip install --quiet --upgrade "${DEPS[@]}"
else
    echo "[setup] bootstrapping pip via ensurepip..."
    if python -m ensurepip --upgrade --default-pip &> /dev/null; then
        python -m pip install --quiet --upgrade "${DEPS[@]}"
    else
        echo "[setup] ensurepip unavailable; installing uv to fix..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        uv pip install --quiet "${DEPS[@]}"
    fi
fi

# Sanity-check the install landed in the active venv.
if ! python -c "import requests, numpy" 2>/dev/null; then
    echo "[setup] ✗ requests/numpy not importable after install"
    echo "[setup]   Active Python: $(which python)"
    echo "[setup]   Try: uv pip install requests numpy"
    exit 1
fi
echo "[setup] ✓ requests + numpy importable"

# ---- 4. Install Ollama for local LLM (concept naming) ----
if ! command -v ollama &> /dev/null; then
    echo "[setup] installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[setup] Ollama already installed"
fi

# Start Ollama if not running.
if ! curl -fs http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[setup] starting Ollama daemon..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# Pull phi3:mini (3.8B, ~3GB VRAM, fast).
if ! ollama list 2>/dev/null | grep -q "phi3:mini"; then
    echo "[setup] pulling phi3:mini (~2GB download)..."
    ollama pull phi3:mini
else
    echo "[setup] phi3:mini already available"
fi

# ---- 5. Verify imports ----
echo "[setup] verifying imports..."
python <<'PYEOF'
import sys
sys.path.insert(0, 'prism/_vendor')
try:
    from hflayers import Hopfield, HopfieldLayer, HopfieldPooling
    from hflayers.transformer import HopfieldEncoderLayer
    print("✓ hflayers")
except Exception as e:
    print(f"✗ hflayers: {e}"); sys.exit(1)

try:
    from prism.cog_core.concept_memory import ConceptMemory
    from prism.cog_core.operator_memory import OperatorMemory
    from prism.cog_core.concept_manager import ConceptManager, OllamaLLM
    from prism.models.transformer_dynamics import TransformerDynamics
    from prism.models.hybrid_policy import HybridPolicy
    from prism.language.concept_to_text import ConceptToText
    from prism.language.cycle_loss import CycleConsistencyLoss
    from prism.training.sparse_hopfield_update import SparseHopfieldOptimizer
    from prism.training.continual_backprop import ContinualBackpropManager
    print("✓ all PRISM-Hybrid modules import cleanly")
except Exception as e:
    print(f"✗ PRISM import: {e}"); sys.exit(1)
PYEOF

# ---- 6. Verify Ollama LLM works ----
echo "[setup] testing Ollama call..."
RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "phi3:mini",
  "prompt": "Reply with just one word: ball",
  "stream": false,
  "options": {"num_predict": 5, "temperature": 0.0}
}' | python -c "import sys, json; print(json.load(sys.stdin).get('response', '').strip())")

if [ -n "$RESPONSE" ]; then
    echo "[setup] ✓ Ollama responded: '$RESPONSE'"
else
    echo "[setup] ✗ Ollama call failed"
    exit 1
fi

# ---- 7. Run unit tests ----
echo "[setup] running smoke tests..."
python tests/test_hybrid_components.py

echo ""
echo "==========================================="
echo "✓ PRISM-Hybrid v5.0 setup complete"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Train ConceptMemory:"
echo "     python -m scripts.cog_core.train_concept_memory \\"
echo "         --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\"
echo "         --rollouts runs/cog_core_phase1_factored/rollouts.npz \\"
echo "         --run-name concept_memory_v1 \\"
echo "         --use-sparse-opt --device cuda"
echo ""
echo "  2. Start ConceptManager in another tmux pane:"
echo "     python -m scripts.run_concept_manager \\"
echo "         --concept-memory-checkpoint runs/concept_memory_v1/concept_memory_final.pt \\"
echo "         --ollama-model phi3:mini --log /tmp/concept_manager.log"
echo ""
echo "  3. Train PPO with hybrid policy (after wiring into ppo_train.py)"
echo ""
