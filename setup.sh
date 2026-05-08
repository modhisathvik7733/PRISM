#!/usr/bin/env bash
# PRISM setup script — run this on a fresh Vast.ai instance.
#
# Assumes an Ubuntu 22.04 image with CUDA drivers already installed
# (most Vast.ai PyTorch templates qualify).
#
# Usage:
#   bash setup.sh                 # auto-detect CUDA and install matching torch
#   CUDA=cu128 bash setup.sh      # force a CUDA wheel (cu121 | cu124 | cu126 | cu128)
#   GPU=5090 bash setup.sh        # convenience — picks cu128 for Blackwell

set -euo pipefail

# --------------------------------------------------------------------- helpers
log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup-error]\033[0m %s\n' "$*" >&2; }

# --------------------------------------------------------------------- 1. uv
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

# --------------------------------------------------------------------- 2. python env
# Vast.ai PyTorch templates pre-activate /venv/main. If we leave VIRTUAL_ENV
# pointing there, `uv pip install` goes to /venv/main but `uv run` uses the
# project's .venv/ — they diverge and torch goes missing. Clear the inherited
# value and force everything through ./.venv.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  log "Ignoring inherited VIRTUAL_ENV=$VIRTUAL_ENV (will use ./.venv instead)"
  unset VIRTUAL_ENV
fi

log "Creating .venv with Python 3.11..."
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate
log "Active venv: $VIRTUAL_ENV"

# --------------------------------------------------------------------- 3. base deps
log "Installing base dependencies (no torch yet)..."
uv pip install -e ".[baseline,dev]"

# --------------------------------------------------------------------- 4. torch with right CUDA
# Pick the right CUDA wheel for the GPU.
#   RTX 30xx/40xx (Ampere/Ada)            → cu121 or cu124 fine
#   RTX 5060/5070/5070Ti/5080/5090 (Blackwell) → cu128 (sm_120) required
#   H100                          (Hopper)     → cu124 / cu126
CUDA="${CUDA:-}"
GPU="${GPU:-}"

if [ -z "$CUDA" ]; then
  case "$GPU" in
    5060|5070|5080|5090|blackwell) CUDA=cu128 ;;
    h100|hopper)    CUDA=cu126 ;;
    4090|4080|3090|3080|ada|ampere) CUDA=cu124 ;;
    *)
      # Auto-detect via nvidia-smi
      if command -v nvidia-smi >/dev/null 2>&1; then
        SMI_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
        log "Detected GPU: ${SMI_NAME:-unknown}"
        case "$SMI_NAME" in
          # All RTX 50-series (Blackwell, sm_120) need cu128.
          *RTX\ 50*|*5060*|*5070*|*5080*|*5090*|*B200*|*Blackwell*) CUDA=cu128 ;;
          *H100*|*H200*)              CUDA=cu126 ;;
          *4090*|*4080*|*L40*|*A100*|*3090*|*3080*) CUDA=cu124 ;;
          *)                          CUDA=cu124 ;;
        esac
      else
        err "no nvidia-smi and no CUDA hint — defaulting to cu124"
        CUDA=cu124
      fi
      ;;
  esac
fi

log "Installing torch with $CUDA wheels..."
uv pip install --index-url "https://download.pytorch.org/whl/${CUDA}" \
  "torch>=2.5" "torchvision" "torchaudio"

# --------------------------------------------------------------------- 5. sanity
log "Sanity check..."
# Use the venv's python directly — bypass `uv run` so we don't trigger any
# additional sync that could swap to a different venv.
.venv/bin/python -c "
import sys
import torch, gymnasium, minigrid
print('python:', sys.executable)
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
print('gymnasium:', gymnasium.__version__)
print('minigrid:', minigrid.__version__)
"

log "Done."
log "Activate the venv:    source .venv/bin/activate"
log "Smoke test:           .venv/bin/python -m scripts.smoke_test"
log "  (or after activate:  python -m scripts.smoke_test)"
