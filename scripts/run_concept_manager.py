"""Standalone runner for ConceptManager.

Use this to start the concept-naming agent alongside PRISM training.
Communicates with PRISM via shared queue (in-process) or via file-based
IPC (cross-process). For now, the in-process path is the primary one;
this script demonstrates the pattern.

Usage (in another tmux pane while training):
    python -m scripts.run_concept_manager \\
        --concept-memory-checkpoint runs/concept_memory_v1/concept_memory_final.pt \\
        --ollama-model phi3:mini \\
        --log /tmp/concept_manager.log
"""

from __future__ import annotations

import argparse
import time

import torch

from prism.cog_core.concept_manager import ConceptManager, OllamaLLM
from prism.cog_core.concept_memory import ConceptMemory


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--concept-memory-checkpoint", required=True)
    p.add_argument("--ollama-model", default="phi3:mini")
    p.add_argument("--ollama-endpoint",
                   default="http://localhost:11434/api/generate")
    p.add_argument("--log", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[run_concept_manager] loading: {args.concept_memory_checkpoint}")
    memory = ConceptMemory.load(args.concept_memory_checkpoint, device)

    llm = OllamaLLM(model=args.ollama_model, endpoint=args.ollama_endpoint)
    print(f"[run_concept_manager] LLM ready: {args.ollama_model}")

    manager = ConceptManager(
        concept_memory=memory,
        llm=llm,
        log_file=args.log,
    )
    manager.start()

    print("[run_concept_manager] running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
            stats = manager.get_stats()
            n_named = memory.get_named_slot_count()
            print(
                f"[run_concept_manager] named={n_named}/{memory.n_slots}, "
                f"stats={stats}"
            )
    except KeyboardInterrupt:
        print("\n[run_concept_manager] stopping...")
        manager.stop()
        manager.join(timeout=5)
        memory.save(args.concept_memory_checkpoint)
        print(f"[run_concept_manager] saved metadata to {args.concept_memory_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
