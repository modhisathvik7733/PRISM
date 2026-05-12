"""PRISM ↔ Unity WebSocket bridge — Day 1 server.

This is the Python side of the Unity bridge. It receives agent state
from Unity over a WebSocket, decides an action, and sends it back.

Day 1: the action is just random (verifies the bridge works end-to-end).
Day 2: the action will come from PRISM (state vector → substrate → action).

Protocol:
  Unity → Python (JSON per frame):
      {"agent_pos": [x, z], "target_pos": [x, z], "delta": [dx, dz],
       "step": int, "episode_done": bool}
  Python → Unity (JSON per frame):
      {"action": int}    # 0=stay, 1=N, 2=S, 3=E, 4=W
      Optional debug fields can be added.

Run:
    pip install websockets
    python unity_demo/prism_server.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys

# 5 discrete actions for the 2D top-down agent.
NUM_ACTIONS = 5
ACTION_NAMES = ["stay", "north", "south", "east", "west"]

HOST = "localhost"
PORT = 8765


async def handle_connection(websocket):
    """One Unity client → one async handler. Loop: recv state → send action."""
    print(f"[bridge] connected: {websocket.remote_address}")
    step_count = 0
    try:
        async for message in websocket:
            try:
                state = json.loads(message)
            except json.JSONDecodeError:
                print(f"[bridge] invalid JSON from Unity: {message[:80]}")
                continue

            # ---- DECIDE ACTION ----
            # Day 1: random. Day 2: this is where PRISM goes.
            action = random.randint(0, NUM_ACTIONS - 1)

            # ---- LOG OCCASIONALLY ----
            step_count += 1
            if step_count % 100 == 1:
                agent = state.get("agent_pos", [0, 0])
                target = state.get("target_pos", [0, 0])
                print(f"[bridge] step={step_count} "
                      f"agent=({agent[0]:.2f},{agent[1]:.2f}) "
                      f"target=({target[0]:.2f},{target[1]:.2f}) "
                      f"action={ACTION_NAMES[action]}")

            # ---- SEND ACTION BACK ----
            response = {"action": action}
            await websocket.send(json.dumps(response))
    except Exception as e:
        print(f"[bridge] connection error: {type(e).__name__}: {e}")
    finally:
        print(f"[bridge] disconnected after {step_count} steps")


async def main():
    try:
        import websockets
    except ImportError:
        print("[bridge] missing dependency: pip install websockets")
        sys.exit(1)

    print(f"[bridge] PRISM server listening on ws://{HOST}:{PORT}")
    print(f"[bridge] Day 1 mode: random actions. Connect from Unity and press Play.")
    async with websockets.serve(handle_connection, HOST, PORT):
        await asyncio.Future()  # serve forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[bridge] shut down by user")
