# PRISM ↔ Unity Bridge — Day 1 Setup

Goal: prove the Python ↔ Unity WebSocket bridge works end-to-end with
random actions. Once this runs, Day 2 swaps the random actions for
PRISM's policy.

## Architecture

```
   Unity (game runtime)              Python (PRISM)
   ┌──────────────────┐              ┌─────────────────────┐
   │  Agent cube      │              │                     │
   │      ▼           │              │  (Day 1) random     │
   │  PrismBridge.cs  │ ─── state ─► │  (Day 2) PRISM      │
   │  (WebSocket)     │ ◄── action ──│  decides action     │
   └──────────────────┘              └─────────────────────┘
        Unity Editor                     prism_server.py
```

State (Unity → Python) per frame:
```json
{"agent_pos": [x, z], "target_pos": [x, z], "delta": [dx, dz],
 "step": 14, "episode_done": false}
```

Action (Python → Unity):
```json
{"action": 1}   // 0=stay, 1=N, 2=S, 3=E, 4=W
```

## Setup (do once)

### 1. Python side

```bash
cd /Users/chintu/PRISM
.venv/bin/pip install websockets
```

(If you don't have the PRISM venv, use any Python 3.10+ environment.
`websockets` is the only dependency for Day 1.)

### 2. Unity side — new project

In Unity Hub:
1. Click **New project**.
2. Editor version: **Unity 6.4 (6000.4.6f1)**.
3. Template: **3D (Built-In Render Pipeline)** — simplest, no HDRP/URP setup needed.
4. Project name: `prism-unity-demo`.
5. Location: anywhere outside the PRISM repo (e.g. `~/UnityProjects/prism-unity-demo`).
6. Click **Create project**.

Wait ~1-2 minutes for Unity to import assets.

### 3. Unity side — build the demo scene

When Unity Editor opens, you'll see the default scene "SampleScene".

**Add the ground:**
- Top menu: GameObject → 3D Object → Plane.
- In the Inspector (right panel), set Transform: Position (0, 0, 0), Scale (1, 1, 1).
- The default plane is 10×10 units. Good.

**Add the agent:**
- GameObject → 3D Object → Cube.
- Rename to `Agent` (right-click in Hierarchy → Rename).
- Transform: Position (0, 0.5, 0), Scale (0.5, 0.5, 0.5).
- (Optional) Materials: drag a red material onto it so it's visible.

**Add the target:**
- GameObject → 3D Object → Sphere.
- Rename to `Target`.
- Transform: Position (3, 0.5, 3), Scale (0.5, 0.5, 0.5).
- (Optional) Drag a green material onto it.

**Position the camera (top-down view):**
- Click `Main Camera` in Hierarchy.
- Transform: Position (0, 10, 0), Rotation (90, 0, 0).
- This makes the camera look straight down at the plane.

**Add the bridge controller:**
- In Hierarchy, right-click → Create Empty.
- Rename to `BridgeManager`.

### 4. Unity side — add the C# scripts

In the Project window (bottom), navigate to Assets.
Right-click → Create → Folder, name it `Scripts`.

Right-click inside Scripts → Create → C# Script. Name it **PrismBridge**.
Double-click to open it. Replace the entire file with this:

```csharp
using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

[Serializable]
public class AgentState
{
    public float[] agent_pos;
    public float[] target_pos;
    public float[] delta;
    public int step;
    public bool episode_done;
}

[Serializable]
public class ActionMessage
{
    public int action;
}

public class PrismBridge : MonoBehaviour
{
    [Header("WebSocket")]
    public string serverUrl = "ws://localhost:8765";

    [Header("Scene refs")]
    public Transform agent;
    public Transform target;

    [Header("Agent control")]
    public float moveSpeed = 2.0f;          // units per second per move
    public float reachThreshold = 0.7f;     // target reach distance
    public float planeHalfSize = 5.0f;      // 10×10 plane, so ±5
    public float decisionInterval = 0.1f;   // seconds between decisions

    private ClientWebSocket _ws;
    private CancellationTokenSource _cts;
    private readonly ConcurrentQueue<int> _actionQueue = new();
    private float _decisionTimer = 0f;
    private int _stepCount = 0;
    private bool _episodeDone = false;
    private int _currentAction = 0; // 0=stay

    async void Start()
    {
        _cts = new CancellationTokenSource();
        _ws = new ClientWebSocket();
        try
        {
            Debug.Log($"[PrismBridge] connecting to {serverUrl} …");
            await _ws.ConnectAsync(new Uri(serverUrl), _cts.Token);
            Debug.Log("[PrismBridge] connected.");
            _ = ReceiveLoop();   // fire-and-forget receive task
        }
        catch (Exception e)
        {
            Debug.LogError($"[PrismBridge] connect failed: {e.Message}\n" +
                           "Did you start prism_server.py? (python unity_demo/prism_server.py)");
        }
    }

    void Update()
    {
        // Apply current action every frame so movement is smooth between decisions.
        ApplyAction(_currentAction);

        // Reach target?
        float dist = Vector3.Distance(agent.position, target.position);
        if (!_episodeDone && dist < reachThreshold)
        {
            _episodeDone = true;
            Debug.Log($"[PrismBridge] target reached at step {_stepCount}");
            // Reset target to a new random position for the next episode.
            target.position = new Vector3(
                UnityEngine.Random.Range(-planeHalfSize + 1, planeHalfSize - 1),
                0.5f,
                UnityEngine.Random.Range(-planeHalfSize + 1, planeHalfSize - 1)
            );
            // Reset agent to origin.
            agent.position = new Vector3(0, 0.5f, 0);
            _stepCount = 0;
        }

        // Send state at the decision interval.
        _decisionTimer += Time.deltaTime;
        if (_decisionTimer >= decisionInterval)
        {
            _decisionTimer = 0f;
            SendState();
            _stepCount++;
        }

        // Pull latest action from the queue (if any).
        while (_actionQueue.TryDequeue(out int act))
        {
            _currentAction = act;
        }

        _episodeDone = false;  // single-frame flag
    }

    private void ApplyAction(int action)
    {
        Vector3 d = Vector3.zero;
        switch (action)
        {
            case 1: d = new Vector3(0, 0, 1); break;   // N (+z)
            case 2: d = new Vector3(0, 0, -1); break;  // S
            case 3: d = new Vector3(1, 0, 0); break;   // E (+x)
            case 4: d = new Vector3(-1, 0, 0); break;  // W
            default: return;
        }
        agent.position += d * moveSpeed * Time.deltaTime;
        // Clamp to plane.
        Vector3 p = agent.position;
        p.x = Mathf.Clamp(p.x, -planeHalfSize + 0.3f, planeHalfSize - 0.3f);
        p.z = Mathf.Clamp(p.z, -planeHalfSize + 0.3f, planeHalfSize - 0.3f);
        agent.position = p;
    }

    private async void SendState()
    {
        if (_ws == null || _ws.State != WebSocketState.Open) return;

        Vector3 a = agent.position;
        Vector3 t = target.position;
        AgentState state = new AgentState
        {
            agent_pos = new[] { a.x, a.z },
            target_pos = new[] { t.x, t.z },
            delta = new[] { t.x - a.x, t.z - a.z },
            step = _stepCount,
            episode_done = _episodeDone,
        };
        string json = JsonUtility.ToJson(state);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        try
        {
            await _ws.SendAsync(new ArraySegment<byte>(bytes),
                                WebSocketMessageType.Text, true, _cts.Token);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PrismBridge] send failed: {e.Message}");
        }
    }

    private async Task ReceiveLoop()
    {
        byte[] buf = new byte[4096];
        while (_ws != null && _ws.State == WebSocketState.Open && !_cts.IsCancellationRequested)
        {
            try
            {
                WebSocketReceiveResult res = await _ws.ReceiveAsync(
                    new ArraySegment<byte>(buf), _cts.Token);
                if (res.MessageType == WebSocketMessageType.Close)
                {
                    Debug.Log("[PrismBridge] server closed connection.");
                    break;
                }
                string msg = Encoding.UTF8.GetString(buf, 0, res.Count);
                ActionMessage am = JsonUtility.FromJson<ActionMessage>(msg);
                _actionQueue.Enqueue(am.action);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[PrismBridge] recv error: {e.Message}");
                break;
            }
        }
    }

    async void OnApplicationQuit()
    {
        _cts?.Cancel();
        if (_ws != null && _ws.State == WebSocketState.Open)
        {
            try
            {
                await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure,
                                     "shutdown", CancellationToken.None);
            }
            catch { }
        }
    }
}
```

Save (Ctrl+S / Cmd+S). Back in Unity, wait a few seconds for compilation.

### 5. Unity side — wire the script to the scene

1. Click `BridgeManager` in Hierarchy.
2. In Inspector: **Add Component** → search **Prism Bridge** → click it.
3. Now drag from Hierarchy onto the Inspector slots:
   - **Agent** field: drag the `Agent` GameObject from Hierarchy.
   - **Target** field: drag the `Target` GameObject from Hierarchy.
4. Leave other settings at defaults.

### 6. Save the scene

File → Save Scene As → name it `BridgeDemo` → save in Assets.

## Run sequence

### Terminal 1 — Python server

```bash
cd /Users/chintu/PRISM
.venv/bin/python unity_demo/prism_server.py
```

Expected output:
```
[bridge] PRISM server listening on ws://localhost:8765
[bridge] Day 1 mode: random actions. Connect from Unity and press Play.
```

### Unity Editor — press Play

1. Click the **▶ Play** button at the top of Unity Editor.
2. Watch the agent (red cube) move randomly N/S/E/W.
3. When it reaches the target (green sphere), the target teleports to a new random position.

You should also see in Terminal 1:
```
[bridge] connected: ('127.0.0.1', xxxxx)
[bridge] step=1 agent=(0.00,0.00) target=(3.00,3.00) action=east
[bridge] step=101 agent=(...) target=(...) action=...
```

### If it works

Bridge is proven. Random actions sometimes find the target by chance. Day 2 swaps the random selection for PRISM — agent should start finding the target reliably.

### If it doesn't work — common issues

**"connect failed: Cannot reach server"** in Unity console:
- Python server isn't running, or is on a different port.
- Check `prism_server.py` is running in Terminal 1.

**Agent doesn't move:**
- Did you drag the `Agent` GameObject onto the PrismBridge component's Agent field?
- Did you drag the `Target` onto the Target field?

**Compile errors:**
- Did you replace the ENTIRE C# file (not just append)?
- Did you save the script before pressing Play?

**Agent moves but no console logs from Python:**
- The state-send rate is `decisionInterval = 0.1s` by default = 10 messages/sec.
- After ~10 seconds you should see the first `step=1` log line.

## What's next

**Day 2**: replace `random.randint(0, NUM_ACTIONS-1)` in `prism_server.py`
with PRISM's policy. State vector is already shaped right:
`(agent_x, agent_z, target_x, target_z, dx, dz)` — 6 floats. PRISM
substrate adapts to this via a new state-vector adapter (no JEPA needed).

**Day 3**: add a second task (e.g., a hazard zone the agent should
avoid). Use the curriculum engine to train task 1 → freeze → task 2.
Show no-forgetting comparison vs vanilla PPO.

**Day 4-7**: package as Unity SDK, build landing page, ship.
