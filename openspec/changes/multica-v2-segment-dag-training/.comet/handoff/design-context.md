# Comet Design Handoff

- Change: multica-v2-segment-dag-training
- Phase: design
- Mode: compact
- Context hash: bff52d85bda76bd960f923c9bbf9de236e5b0c1c6ed03a881c14d0f2c707f2a2

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/multica-v2-segment-dag-training/proposal.md

- Source: openspec/changes/multica-v2-segment-dag-training/proposal.md
- Lines: 1-78
- SHA256: fa483adc76ee7047fa64f2d23ba7f267f2b30fabcee736ca3c4e3b7e4fae529d

```md
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
```

## openspec/changes/multica-v2-segment-dag-training/design.md

- Source: openspec/changes/multica-v2-segment-dag-training/design.md
- Lines: 1-155
- SHA256: 0c3db2836576ce1e84ead2dfd0119181fc4375c53e39fe5f29aab38c81a2df18

[TRUNCATED]

```md
## Context

AReaL 2.0 (`areal/v2`) is a service-oriented rewrite of the `areal/experimental/openai`
proxy monolith. v2's `inference_service` captures each assistant turn as an interaction in a
`SessionData` (`active_completions` + `ready_trajectories`); `set_reward` closes
active->ready and `export_trajectories` pops ready trajectories. Today v2 couples trajectory
closure to reward and exports at session granularity - insufficient for Multica, where one
agent-run session must be sliced into multiple communication-bounded segments, each
independently exportable as a trajectory and independently rewardable.

Multica is the RL environment: it drives pi-agent turns through v2 `/chat/completions`,
fires communication events (delegation / mention / completion / squad briefing), and must
hand AReaL the structure + tensor refs to build the interaction DAG (`SuperNode` /
`ExecutionDAG`) and train. The prior `sub-project-g-multica-interaction-dag` design returned
a turn-index `DagResult` (structure only) and deferred the producer; its turn-index slicing
is fragile (off-by-one between Multica's counter and AReaL's `list[Node]`) and carries no
tensor or reward path.

This change establishes the v2 segment-DAG data path: `close_segment` decouples trajectory
boundary from reward; per-segment export returns tensor refs; Multica assembles an
`AssembledDag` of refs + structure; AReaL resolves refs and builds the DAG. Scoring (judge),
V/GAE, and tree search are separate changes.

Locked architectural decisions (from the design exploration) are recorded below; open items
are deferred to task investigation or to changes 2/3.

## Goals / Non-Goals

**Goals:**

- v2 `close_segment`: no-reward active->ready close, returning `trajectory_id`.
- Per-segment export: single trajectory, `remove_session=False`, `RTensor` refs.
- `AssembledDag` contract: segments `{trajectory_id, tensor_ref, closing_event, env}`,
  typed edges, `session_to_agent_run`; no scores, no turn indices, no text.
- AReaL consumer: poll -> `AssembledDag` -> resolve refs -> `SuperNode` / `ExecutionDAG`.
- Minimal training plumbing with placeholder reward (end-to-end path proven).
- Tensor lifecycle: `/data/clear` + `remove_session` after training.

**Non-Goals:**

- Judge / process-reward scoring (change 2).
- Actor-model V critic + GAE (change 2).
- Tree search / branching / fork-from-checkpoint (change 3).
- Full sandbox snapshot/fork in env snapshots (Sub-project F).
- Text crossing to AReaL via export or `AssembledDag` (text stays in AReaL's capture).

## Decisions

### D1 - `close_segment` decouples trajectory boundary from reward

v2 today closes active->ready only via `set_reward` (which requires
`_last_reward_interaction_id`). Add `close_segment` that closes active->ready **without** a
reward, returning `trajectory_id`. One communication-bounded segment = one `close_segment`
call = one trajectory. Reward is assigned later (judge, change 2), not at close.

**Rationale:** per-segment rewards require per-segment trajectories; coupling close to
reward forces reward at boundary time. Decoupling lets Multica close on events and assign
reward independently.

### D2 - Per-segment export returns `RTensor` refs, not bytes

`/export_trajectories` exports a single trajectory by `trajectory_id` with
`remove_session=False`, stores tokens/logprobs on the data_proxy, and returns `RTensor` refs.
AReaL resolves refs via `/data/<shard_id>` / `/data/batch` when building the DAG; frees them
via `/data/clear` after training. The session stays live for further segments.

**Rationale:** keeps tensor bulk on the data_proxy (scalable) rather than in AReaL's
training process; `remove_session=False` lets one session yield many segments.

### D3 - `AssembledDag` carries structure + refs + env, NOT scores / turn-indices / text

Segment = `{segment_id, agent_run_id, issue_id, trajectory_id, tensor_ref, closing_event,
env_snapshot}`. `AssembledDag` = `{segments, edges, session_to_agent_run}` with typed edges
(`DELEGATION` fan-out / `MENTION` peer / `COMPLETION` fan-in). **No `judge_scores`** (AReaL's
judge produces them - change 2), **no `start_turn_idx` / `end_turn_idx`** (`close_segment`
defines the segment; AReaL resolves refs, not slices), **no text** (stays in AReaL's capture;
the v2 judge reads its own capture - change 2).

**Rationale:** `close_segment` + `tensor_ref` make turn indices redundant for tensor
retrieval; keeping scores out preserves the producer/consumer boundary (Multica = env +
```

Full source: openspec/changes/multica-v2-segment-dag-training/design.md

## openspec/changes/multica-v2-segment-dag-training/tasks.md

- Source: openspec/changes/multica-v2-segment-dag-training/tasks.md
- Lines: 1-190
- SHA256: ab2dd23f1f3dc28ae73b564272ee8e6f08561633d2834a8710902dd384b8175a

[TRUNCATED]

```md
## 0. Supersede sub-project-g

- [ ] 0.1 Mark `sub-project-g-multica-interaction-dag` superseded: set `archived: true` in
  its `.comet.yaml`, add a `Superseded by: multica-v2-segment-dag-training` note + reason
  (contract changed: turn-index -> close_segment + tensor_ref; scores moved to AReaL judge)
  to its `proposal.md`. Do not move/delete its files.
- [ ] 0.2 Commit: `gov(multica-v2-segment-dag): supersede sub-project-g interaction-dag`.

## 1. Investigation - v2 surface and Multica seams

- [ ] 1.1 Confirm v2 `SessionData` (`areal/v2/inference_service/data_proxy/session.py`):
  `set_reward` close path (`_mark_active_trajectory_ready_locked`,
  `_last_reward_interaction_id`), `ready_trajectories` OrderedDict, `export_trajectory` pop
  semantics. Define the no-reward close seam for `close_segment`.
- [ ] 1.2 Confirm v2 `/export_trajectories` (`data_proxy/app.py`): current multi-session
  merge + `RTensor.remotize` path; what it takes to export a single `trajectory_id` with
  `remove_session=False`.
- [ ] 1.3 Confirm v2 data_proxy `/data/<shard_id>`, `/data/batch`, `/data/clear` surface
  (add what is missing).
- [ ] 1.4 Confirm v2 gateway routes (`gateway/app.py`) to add `/rl/close_segment` and
  per-trajectory export.
- [ ] 1.5 Confirm Multica driving seam (proxy `/chat/completions` boundary) and where
  `close_segment` + export are called per communication event (delegation / mention /
  completion / squad briefing).
- [ ] 1.6 Confirm `session_id <-> agent_run_id` capture at `/rl/start_session`
  (`internal/arealrl/client.go`) and the root-task completion signal that flips the DAG
  endpoint `202` -> `200`.
- [ ] 1.7 Decide `close_segment` on empty active (error vs empty trajectory) and record in
  design.
- [ ] 1.8 Commit: `docs(v2-segment-dag): T1 v2 + multica seam confirmation`.

## 2. Migration + DB queries (Multica)

- [ ] 2.1 Add migration: `interaction_dag_segment` (segment_id, project_id, agent_run_id,
  issue_id, task_id, trajectory_id, tensor_ref, closing_event, closing_event_target,
  created_at), `interaction_dag_edge` (src_segment_id, dst_segment_id, type),
  `interaction_dag_env_snapshot` (segment_id, sandbox_ids, issue_snapshot_id, env_state).
  No `start_turn_idx` / `end_turn_idx` columns.
- [ ] 2.2 Generate / add DB query files: `CreateSegment`, `CloseSegment` (set
  trajectory_id + tensor_ref + closing_event), `AddEdge`, `CaptureEnvSnapshot`,
  `RecordSessionAgentRun`, `ListSegmentsForProject`, `ListEdgesForProject`,
  `ListEnvSnapshotsForProject`, `GetDagStatus`.
- [ ] 2.3 Ensure `DELETE /api/v1/env-dispatch/{projectID}` cascades to the new tables.
- [ ] 2.4 Commit: `feat(v2-segment-dag): migration + queries for segments edges snapshots`.

## 3. V2 close_segment (TDD)

**Files:** `areal/v2/inference_service/data_proxy/session.py`,
`areal/v2/inference_service/data_proxy/app.py`, `areal/v2/inference_service/gateway/app.py`,
tests under `areal/v2/inference_service/tests/`.

- [ ] 3.1 Failing tests: `close_segment` moves active completions into a ready trajectory
  and returns its `trajectory_id`, assigning no reward.
- [ ] 3.2 Failing tests: session stays live after `close_segment` - further turns capture
  into a new active segment that can be closed independently.
- [ ] 3.3 Failing tests: `close_segment` on a session with no active completions returns a
  typed error and produces no trajectory.
- [ ] 3.4 Failing tests: `close_segment` does not require a prior `set_reward` (no
  `_last_reward_interaction_id` dependency).
- [ ] 3.5 Implement `SessionData.close_segment()` + data_proxy `/rl/close_segment` + gateway
  route.
- [ ] 3.6 Commit: `feat(v2-segment-dag): close_segment no-reward active->ready close`.

## 4. V2 per-segment tensor-ref export + data_proxy resolve (TDD)

**Files:** `areal/v2/inference_service/data_proxy/app.py`,
`areal/v2/inference_service/gateway/app.py`, tests under
`areal/v2/inference_service/tests/`.

- [ ] 4.1 Failing tests: `/export_trajectories` by `trajectory_id` with
  `remove_session=False` returns `RTensor` refs for that segment and leaves other
  trajectories ready.
- [ ] 4.2 Failing tests: exported payload is refs-only - no message text (only `input_ids`,
  `loss_mask`, `logprobs`, `versions`, `attention_mask`).
- [ ] 4.3 Failing tests: exporting an unknown `trajectory_id` returns a typed error.
- [ ] 4.4 Failing tests: `/data/<shard_id>` and `/data/batch` resolve refs to tensor bytes;
  `DELETE /data/clear` frees shards.
- [ ] 4.5 Implement per-trajectory export + `/data/*` resolve + `/data/clear` (add missing
  endpoints).
- [ ] 4.6 Commit: `feat(v2-segment-dag): per-segment tensor-ref export + data resolve`.
```

Full source: openspec/changes/multica-v2-segment-dag-training/tasks.md

## openspec/changes/multica-v2-segment-dag-training/specs/v2-segment-dag/spec.md

- Source: openspec/changes/multica-v2-segment-dag-training/specs/v2-segment-dag/spec.md
- Lines: 1-128
- SHA256: d08cee9d11fafa153a966119c2b4efa300f74899f52354015caf1120026a51b7

[TRUNCATED]

```md
# v2-segment-dag

## ADDED Requirements

### Requirement: V2 no-reward segment close

The v2 inference_service SHALL provide a `close_segment` operation (`POST /rl/close_segment`)
that snapshots a session's active completions into a ready trajectory **without setting a
reward**, returning the new `trajectory_id`. Closing a segment MUST NOT require a prior
`set_reward` and MUST NOT assign any reward value. The session MUST remain live for further
turns and further `close_segment` calls after a close.

#### Scenario: Close produces a reward-less trajectory
- **WHEN** Multica calls `close_segment` on a session that has active completions
- **THEN** the system moves the active completions into a ready trajectory, returns its
  `trajectory_id`, and assigns no reward to any interaction in that trajectory

#### Scenario: Session stays live after close
- **WHEN** a segment is closed on a session
- **THEN** subsequent `/chat/completions` turns on the same session are captured into a new
  active segment that can be closed independently

#### Scenario: Close without active completions is rejected
- **WHEN** `close_segment` is called on a session with no active completions
- **THEN** the system returns a typed error and produces no trajectory

### Requirement: Per-segment tensor-ref export

The v2 inference_service SHALL export a single ready trajectory by `trajectory_id` via
`/export_trajectories`, storing its tokens/logprobs on the data_proxy and returning `RTensor`
references (not raw bytes). Export with `remove_session=False` MUST keep the session and its
remaining trajectories intact for further segment exports. Export MUST NOT include message
text - only tensor data (`input_ids`, `loss_mask`, `logprobs`, `versions`,
`attention_mask`).

#### Scenario: Export returns refs for one segment
- **WHEN** Multica exports a single `trajectory_id` with `remove_session=False`
- **THEN** the system returns `RTensor` refs for that segment's tensors and leaves the
  session's other trajectories ready for later export

#### Scenario: Export is refs-only, no text
- **WHEN** a trajectory is exported
- **THEN** the returned payload contains tensor refs and no message text

#### Scenario: Exporting an unknown trajectory fails
- **WHEN** Multica exports a `trajectory_id` not present in the session's ready trajectories
- **THEN** the system returns a typed error and exports nothing

### Requirement: AssembledDag contract carries structure, refs, and env only

Multica SHALL assemble an `AssembledDag` at root-task completion containing `segments`,
`edges`, and `session_to_agent_run`. Each segment MUST carry `segment_id`, `agent_run_id`,
`issue_id`, `trajectory_id`, `tensor_ref`, `closing_event`, and a ref-only `env_snapshot`.
Edges MUST carry `src_segment_id`, `dst_segment_id`, and `type` matching AReaL's `EdgeType`
(`delegation` fan-out / `mention` peer / `completion` fan-in). `AssembledDag` MUST NOT carry
scores, turn indices, or message text.

#### Scenario: Segment records refs and structure, not scores or turn indices
- **WHEN** Multica records a segment at a communication event
- **THEN** the segment carries `trajectory_id` + `tensor_ref` + `closing_event` +
  `env_snapshot` and carries no `judge_scores` and no `start_turn_idx` / `end_turn_idx`

#### Scenario: Edges match EdgeType and keep the DAG acyclic
- **WHEN** Multica assembles edges
- **THEN** each edge type is `delegation`, `mention`, or `completion`, every edge flows from
  a cause segment to an effect segment, and the resulting directed graph is acyclic

#### Scenario: AssembledDag contains no text
- **WHEN** `AssembledDag` is assembled
- **THEN** no segment or edge carries message text

### Requirement: AReaL resolves refs and builds the DAG

AReaL SHALL resolve each segment's `tensor_ref` to tokens/logprobs via the v2 data_proxy
(`/data/<shard_id>`, `/data/batch`), build a `SuperNode` whose payload is the resolved
tensors and whose metadata is the `AssembledDag` segment, and construct the `ExecutionDAG`
from the assembled edges - by ref-resolution, not by slicing a node list on turn indices.
The assembler MUST validate that the assembled DAG is acyclic and that each session's
segments cover the run without gaps.

```

Full source: openspec/changes/multica-v2-segment-dag-training/specs/v2-segment-dag/spec.md

