> **Superseded by `multica-v2-segment-dag-training`** (2026-07-08). Contract changed:
> turn-index slicing -> `close_segment` + per-segment tensor-ref export; scores moved out of
> the DAG to AReaL's judge; `AssembledDag` replaces `DagResult`. This change never started
> building (0/60 tasks); archived in favor of the v2 segment-DAG data path.

## Why

AReaL already designs and implements the **consumer** side of a multi-agent interaction
DAG (`ExecutionDAG` / `SuperNode` / `SuperNodeAssembler` in
`customized_areal/tree_search/agents/`), and
`docs/superpowers/specs/2026-06-30-supernode-multica-dag-rollout-design.md` locks the
`DagResult` contract Multica must produce at task completion (decision D14: "Multica
provides segment specs at task completion"). But the **Multica-side producer is explicitly
deferred and unimplemented** (design §10 Out of scope; risk row: "Multica-side endpoints do
not exist yet"). Without it, AReaL cannot run collaborative multi-agent DAG rollouts —
there is no source for the segments, typed edges, per-run turn indices, and env snapshots
the assembler consumes. This change implements that deferred producer: Multica converts its
own agent interactions into the DAG during execution and returns it to AReaL when the task
finishes.

## What Changes

- **Incremental interaction-DAG recording (Multica Go)**: as communication events fire
  (delegation → sub-issue, mention, completion → notify-parent, squad briefing), Multica
  records communication-bounded segments with per-run turn indices and typed edges
  (`DELEGATION` / `MENTION` / `COMPLETION`), matching AReaL's `EdgeType`. Segments are
  recorded during execution, not inferred afterward.
- **Lightweight team env snapshots per segment**: each segment records a ref-only
  `TeamEnvSnapshot` (sandbox_instance ids + issue-subtree ref). No sandbox pause/fork —
  deliberately independent of Sub-project F.
- **`DagResult` assembly + polling return endpoint**: at root-task completion Multica
  assembles `DagResult` (`session_ids`, `session_to_agent_run`, `segments`, `edges`,
  `env_snapshots`) and serves it via `GET /api/v1/env-dispatch/{projectID}/dag` — `202`
  in-progress, `200` + `DagResult` when done. Reuses the existing env-dispatch path
  (`project_id` as the root handle).
- **AReaL client adapter**: extend `MulticaEnvDispatchClient` (or add a thin
  `MulticaDagClient`) to poll the endpoint and return `DagResult`. `SuperNodeAssembler` and
  `ExecutionDAG` are unchanged consumers.
- **Migration**: new tables for interaction-DAG segments, edges, and per-segment env
  snapshots, keyed by `project_id` / `agent_run_id`.

## Capabilities

### New Capabilities

- `interaction-dag-return`: Multica records agent interactions as a communication-bounded
  segment DAG during task execution and returns the assembled `DagResult` to AReaL at task
  completion via a polling env-dispatch endpoint.

### Modified Capabilities

<!-- None. AReaL's SuperNodeAssembler / ExecutionDAG consumer is unchanged; this change
     only adds the Multica producer plus a thin AReaL client adapter. -->

## Impact

- **Multica (Go, primary)**: new service (`internal/service/interaction_dag.go`), handler
  (`GET .../env-dispatch/{projectID}/dag`), per-run turn-index tracking at the agent-run
  driving seam, env-snapshot capture at communication events, migration + generated DB
  queries. Touches `internal/service/task.go` (delegation/mention/completion hooks),
  `internal/handler/env_dispatch.go`, `internal/handler/squad_briefing.go`,
  `internal/arealrl/client.go` (session↔agent_run mapping).
- **AReaL (Python, secondary)**: thin client adapter in
  `customized_areal/tree_search/agents/` (poll endpoint → `DagResult`);
  `SuperNodeAssembler` / `ExecutionDAG` unchanged.
- **Depends on**: the locked `DagResult` / `SegmentSpec` / `EdgeSpec` / `TeamEnvSnapshot`
  contract from `2026-06-30-supernode-multica-dag-rollout-design.md`. Independent of
  Sub-project F (env snapshots are refs-only; no sandbox pause/fork).
- **Out of scope**: full sandbox snapshot/fork (F), verifier / reward backup
  (E / 2026-06-30 Phase 2), immutable branching (2026-06-26 phase1), AReaL consumer
  changes.
