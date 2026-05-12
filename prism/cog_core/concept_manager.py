"""ConceptManager — async LLM bootstrap for naming Hopfield slots.

Architectural pattern (validated by AriGraph, Voyager, LARP, MindForge):
  - LLM is the PROPOSER, not the OWNER of concepts
  - Slots live in the ConceptMemory/OperatorMemory (your store, persistent)
  - Manager runs asynchronously in a background thread
  - When PRISM perceives a novel pattern, it flags the slot index
  - Manager calls local Ollama LLM to propose a name and properties
  - Validates the proposal (rejects hallucination) and writes to slot_metadata
  - PRISM never blocks on LLM calls

Three rules to avoid the "concepts in LLM weights" trap:
  1. LLM never holds the canonical concept — your store does
  2. LLM proposals must ground in perception or are rejected
  3. After bootstrapping, the LLM is optional (cache hits dominate)

Uses local Ollama (default: phi3:mini, ~3GB VRAM) so there are no API costs.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from queue import Empty, Queue
from typing import Any

import requests


class OllamaLLM:
    """Thin client for a local Ollama server."""

    def __init__(
        self,
        model: str = "phi3:mini",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 30.0,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def query(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        try:
            r = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}") from e


class ConceptManager(threading.Thread):
    """Background thread that names Hopfield slots via Ollama.

    Usage:
        manager = ConceptManager(concept_memory, operator_memory)
        manager.start()                                     # begin background loop
        manager.flag_unknown_concept(slot_idx, context)     # called by PRISM
        ...
        manager.stop()                                      # on shutdown

    The manager is decoupled from PRISM's training loop. It pulls from queues
    of flagged unknowns and processes them at its own pace. PRISM keeps running.
    """

    def __init__(
        self,
        concept_memory,
        operator_memory=None,
        llm: OllamaLLM | None = None,
        valid_object_types: list[str] | None = None,
        valid_colors: list[str] | None = None,
        consolidation_interval: int = 1000,
        max_queue_size: int = 1000,
        log_file: str | None = None,
    ):
        super().__init__(daemon=True)
        self.concept_memory = concept_memory
        self.operator_memory = operator_memory
        self.llm = llm or OllamaLLM()

        # BabyAI vocabularies for proposal validation.
        self.valid_object_types = valid_object_types or [
            "wall", "floor", "door", "key", "ball", "box", "goal", "lava",
            "unknown",
        ]
        self.valid_colors = valid_colors or [
            "red", "green", "blue", "purple", "yellow", "grey",
        ]

        self.concept_queue: Queue = Queue(maxsize=max_queue_size)
        self.operator_queue: Queue = Queue(maxsize=max_queue_size)

        self.consolidation_interval = consolidation_interval
        self._steps_processed = 0
        self._running = False
        self._stop_event = threading.Event()

        self.log_file = log_file
        self._stats = {
            "concepts_named": 0,
            "concepts_rejected": 0,
            "operators_named": 0,
            "llm_errors": 0,
        }

    # ---- API called by PRISM (non-blocking) ----

    def flag_unknown_concept(self, slot_idx: int, context: str) -> None:
        """Called by PRISM when a slot activates but is unnamed.

        context: human-readable description of the current scene (predicates,
                 active operator, mission goal, etc.) for the LLM to interpret.
        """
        try:
            self.concept_queue.put_nowait((slot_idx, context))
        except Exception:
            pass  # queue full — drop silently (rare)

    def flag_unknown_operator(self, slot_idx: int, context: str) -> None:
        try:
            self.operator_queue.put_nowait((slot_idx, context))
        except Exception:
            pass

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()

    # ---- Background loop ----

    def run(self) -> None:
        self._running = True
        last_consolidation_step = 0

        while self._running:
            processed_any = False

            # Drain concept queue.
            try:
                slot_idx, context = self.concept_queue.get_nowait()
                self._process_concept(slot_idx, context)
                processed_any = True
            except Empty:
                pass

            # Drain operator queue.
            try:
                slot_idx, context = self.operator_queue.get_nowait()
                self._process_operator(slot_idx, context)
                processed_any = True
            except Empty:
                pass

            # Periodic consolidation (merge similar slots, prune stale).
            self._steps_processed += 1
            if (
                self._steps_processed - last_consolidation_step
                >= self.consolidation_interval
            ):
                self._consolidate()
                last_consolidation_step = self._steps_processed

            if not processed_any:
                # Avoid busy-waiting; brief sleep when queues empty.
                self._stop_event.wait(timeout=0.5)

    # ---- Concept naming ----

    def _process_concept(self, slot_idx: int, context: str) -> None:
        # Skip if already named.
        meta = self.concept_memory.slot_metadata.get(slot_idx, {})
        if meta and not meta.get("name", "").startswith("unnamed_"):
            return

        prompt = (
            "You are observing a BabyAI gridworld scene. The agent sees the following:\n"
            f"{context}\n\n"
            "Identify the most salient object in this scene. Reply with a JSON object:\n"
            '{"object": "<one of: wall/floor/door/key/ball/box/goal>", '
            '"color": "<one of: red/green/blue/purple/yellow/grey>"}\n'
            "If the scene has no salient object, reply: {\"object\": \"unknown\", \"color\": \"none\"}"
        )

        try:
            response = self.llm.query(prompt, max_tokens=40)
            parsed = self._extract_json(response)
            obj = parsed.get("object", "unknown").lower().strip()
            color = parsed.get("color", "none").lower().strip()

            if not self._validate_concept(obj, color):
                self._stats["concepts_rejected"] += 1
                self._log(f"REJECTED slot {slot_idx}: obj={obj} color={color}")
                return

            name = f"{color}_{obj}" if color != "none" else obj
            self.concept_memory.name_slot(
                slot_idx,
                name=name,
                properties={"object_type": obj, "color": color},
            )
            self._stats["concepts_named"] += 1
            self._log(f"NAMED concept slot {slot_idx} = '{name}'")

        except Exception as e:
            self._stats["llm_errors"] += 1
            self._log(f"ERROR processing concept slot {slot_idx}: {e}")

    def _process_operator(self, slot_idx: int, context: str) -> None:
        if self.operator_memory is None:
            return
        meta = self.operator_memory.operator_metadata.get(slot_idx, {})
        if meta and not meta.get("name", "").startswith("op_"):
            return

        prompt = (
            "An agent in a BabyAI gridworld took an action with the following context:\n"
            f"{context}\n\n"
            "What is the most likely name for this action? Reply with one word: "
            "move_forward, turn_left, turn_right, pickup, drop, toggle, or done."
        )

        try:
            response = self.llm.query(prompt, max_tokens=10)
            name = response.split()[0].lower().strip(".,!?")
            valid_ops = {
                "move_forward", "turn_left", "turn_right",
                "pickup", "drop", "toggle", "done",
            }
            if name not in valid_ops:
                self._stats["concepts_rejected"] += 1
                self._log(f"REJECTED operator slot {slot_idx}: '{name}'")
                return
            self.operator_memory.name_operator(slot_idx, name)
            self._stats["operators_named"] += 1
            self._log(f"NAMED operator slot {slot_idx} = '{name}'")

        except Exception as e:
            self._stats["llm_errors"] += 1
            self._log(f"ERROR processing operator slot {slot_idx}: {e}")

    # ---- Validation ----

    def _validate_concept(self, obj: str, color: str) -> bool:
        if obj not in self.valid_object_types:
            return False
        if color != "none" and color not in self.valid_colors:
            return False
        return True

    @staticmethod
    def _extract_json(response: str) -> dict[str, Any]:
        # Find first {...} in the response.
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            return {}

    # ---- Consolidation ----

    def _consolidate(self) -> None:
        """Merge similar slots, prune stale ones.

        For now: just logs slot statistics. Future: cluster low-use slots
        with high cosine similarity into the dominant slot.
        """
        n_concepts = len(self.concept_memory.slot_metadata)
        n_named = self.concept_memory.get_named_slot_count()
        self._log(
            f"CONSOLIDATION: {n_named}/{n_concepts} concept slots named, "
            f"stats={self._stats}"
        )

    # ---- Logging ----

    def _log(self, msg: str) -> None:
        line = f"[concept_manager] {time.strftime('%H:%M:%S')} {msg}"
        print(line)
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass
