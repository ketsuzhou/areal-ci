## Why

AReaL 2.0 (`areal/v2`) replaces the `areal/experimental/openai` proxy monolith with a
service architecture (agent_service / inference_service / training_service). To train a
pi-agent in Multica (the multi-agent collaboration environment) under v2, AReaL needs a
per-segment trajectory path: Multica drives turns through v2 inference, slices each
agent-run session into communication-bounded segments, and hands AReaL tensor references +
structure so AReaL can build the interaction DAG and train. The prior change
`sub-project-g-multica-interaction-dag` scoped this as a turn-index `DagResult` (structure
only, no tensors, no reward path) and deferred the producer. This change supersedes it with
the v2 contract: `close_segment` + per-segment tensor-ref export + `AssembledDag`, so a
session's segments become independently rewardable, independently exportable trajectories
without turn-index alignment.

**Supersedes**: `sub-project-g-multica-interaction-dag` (contract changed: turn-index
slicing -> `close_segment` + tensor-ref per segment; scores moved out of the DAG to AReaL's
judge). sub-project-g never started building (0/60 tasks), so the supersession is zero-cost.

## What Changes

- **V2 `close_segment` operation (AReaL)**: new `/rl/close_segment` on the v2
  inference_service (gateway + data_proxy + session) that snapshots the session's active
  completions into a ready trajectory **without setting a reward** (active -> ready close),
  returning a `trajectory_id`. Decouples v2's trajectory boundary from reward so each
  communication-bounded segment is its own trajectory.
- **Per-segment tensor-ref export (AReaL)**: `/export_trajectories` exports a single
  trajectory by `trajectory_id` with `remove_session=False`, storing tokens/logprobs on the
  data_proxy and returning `RTensor` refs (not bytes). The session stays live for further
  segments; refs are resolved later via `/data/<shard_id>`, `/data/batch`, and freed via
  `/data/clear`.
- **`AssembledDag` contract (Multica -> AReaL)**: at each communication event Multica calls
  `close_segment` + export and records a segment `{segment_id, agent_run_id, issue_id,
  trajectory_id, tensor_ref, closing_event, env_snapshot}`. At root-task completion Multica
  assembles `AssembledDag` = `{segments, edges, session_to_agent_run}` with typed edges
  (`DELEGATION` / `MENTION` / `COMPLETION`) and returns it via the env-dispatch polling
  endpoint (`202` in-progress / `200` + `AssembledDag` done). **No scores** (AReaL's judge
  produces them - separate change), **no turn indices** (`close_segment` replaces them),
  **no text** (stays in AReaL's capture; export is refs-only).
- **AReaL consumer (AReaL)**: client adapter polls the endpoint -> `AssembledDag`; resolves
  `tensor_ref`s to tokens/logprobs via v2 `/data/*`; builds `SuperNode(payload=tensors,
  metadata=AssembledDag)` + `ExecutionDAG` by **resolving refs** (not slicing by turn
  index). Minimal training plumbing consumes the built DAG with a placeholder reward to
  prove the path end-to-end.
- **Tensor lifecycle (AReaL)**: after training, `DELETE /data/clear` + `remove_session`
  release v2 storage.
- **Migration (Multica)**: segment / edge / env-snapshot tables keyed by `project_id` /
  `agent_run_id` / `segment_id`, carrying `trajectory_id` + `tensor_ref` (replaces
  sub-project-g's `start_turn_idx` / `end_turn_idx` columns).

## Capabilities

### New Capabilities

- `v2-segment-dag`: Per-segment v2 trajectory close/export, Multica `AssembledDag`
  assembly, and AReaL ref-resolution + `SuperNode` / `ExecutionDAG` construction from the
  assembled DAG.

### Modified Capabilities

<!-- None. sub-project-g's `interaction-dag-return` was never applied to a shipped spec;
     this change supersedes it rather than modifying an existing capability. -->

## Impact

- **AReaL v2 inference_service (Python, primary)**: new `close_segment` op in
  `areal/v2/inference_service/{gateway,data_proxy}/app.py` + `session.py`; per-trajectory
  export with `remove_session=False` + `RTensor` refs in data_proxy; `/data/*` resolve +
  `/data/clear` cleanup. `SessionData` gains a no-reward active->ready close.
- **Multica (Go, primary)**: communication-event hooks (delegation / mention / completion /
  squad briefing) call `close_segment` + export per segment; `AssembledDag` assembly +
  env-dispatch polling endpoint (replaces sub-project-g's `DagResult` return); migration
  replacing turn-index columns with `trajectory_id` + `tensor_ref`.
- **AReaL consumer (Python, secondary)**: `MulticaDagClient` polls -> `AssembledDag`;
  `SuperNodeAssembler` ref-resolve path (replaces turn-index slice); minimal training hook
  with placeholder reward.
- **Supersedes**: `sub-project-g-multica-interaction-dag` (archived).
- **Out of scope**: judge / process-reward scoring (change 2), actor-model V critic + GAE
  (change 2), tree search / branching (change 3), full sandbox snapshot/fork (Sub-project F).
