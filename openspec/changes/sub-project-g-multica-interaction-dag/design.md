## Context

`customized_areal/tree_search/agents/` already implements the AReaL **consumer** of a
multi-agent interaction DAG:

- `execution_dag.py` — torch-free `ExecutionDAG` of `SuperNode`s with typed `EdgeType`
  (`DELEGATION` / `MENTION` / `COMPLETION`) and `to_records()` / `from_records()`.
- `supernode_assembler.py` — `SuperNodeAssembler.assemble(sessions_nodes, dag_result)`
  consumes a Multica-produced `DagResult`: `session_ids`, `session_to_agent_run`,
  `segments: list[SegmentSpec]` (1-based inclusive `start_turn_idx` / `end_turn_idx` +
  `closing_event` + `closing_event_target_segment`), `edges: list[EdgeSpec]`, and
  `env_snapshots: dict[segment_id, TeamEnvSnapshot]`.
- `multica_environment_protocol.md` + `2026-06-30-supernode-multica-dag-rollout-design.md`
  lock the contract (D14: "Multica provides segment specs at task completion") and
  explicitly defer the Multica-side producer (§10 Out of scope; risk: "Multica-side
  endpoints do not exist yet").

Multica already originates every edge source — delegation creates sub-issues
(`issue.parent_issue_id`), mention triggers (`internal/mention`), completion/notify-parent,
squad briefing (`handler/squad_briefing.go`) — and already calls `/rl/start_session` per
agent run (`internal/arealrl/client.go`). What is missing is recording these as
segment + edge structure with per-run turn indices during execution and returning the
assembled `DagResult` to AReaL at task completion.

Constraints: this change must produce exactly the `DagResult` AReaL already consumes (no
consumer-side changes), must reuse the existing env-dispatch path, must use a polling
return, and must stay independent of Sub-project F (env snapshots are refs-only).

## Goals / Non-Goals

**Goals:**

- Implement the deferred Multica-side producer of the locked `DagResult` contract.
- Incrementally record communication-bounded segments + typed edges + per-run turn indices
  as communication events fire during execution.
- Capture lightweight (ref-only) team env snapshots per segment.
- Assemble and return `DagResult` at root-task completion via a polling env-dispatch
  endpoint.
- Add a thin AReaL client adapter to poll the endpoint; leave `SuperNodeAssembler` /
  `ExecutionDAG` unchanged.

**Non-Goals:**

- Full sandbox snapshot/fork in env snapshots (deferred to F's pause-in-place checkpoints).
- Verifier, reward backup, GAE/critic (E / 2026-06-30 Phase 2).
- Immutable branching / fork-from-checkpoint (2026-06-26 phase1).
- Any change to `SuperNodeAssembler` / `ExecutionDAG` / the per-turn proxy interaction
  cache on the AReaL side.

## Decisions

### D1 — Incremental segment recording at communication events

Record segments + edges as communication events fire during execution, not reconstruct them
at completion.

**Rationale:** `SegmentSpec` requires 1-based inclusive `start_turn_idx` / `end_turn_idx`
whose terminal turn is the closing communication event; turn indices can only be captured
as the agent runs. The 2026-06-30 design states "Multica decides segment boundaries at
communication events during execution" and "Multica tracks turn indices as the agent runs."

**Alternative considered:** reconstruct the DAG at completion by querying sub-issues /
mentions / completions — rejected because per-run turn indices and exact segment-close
ordering are not recoverable after the fact, and the contract demands turn ranges.

### D2 — Reuse the env-dispatch path + `GET /api/v1/env-dispatch/{projectID}/dag`

Reuse `create_env_dispatch(mode="scratch")` to start the rollout (it already creates
project / issue / agent_run and calls `/rl/start_session`), and add a polling
`GET /api/v1/env-dispatch/{projectID}/dag` that returns `202` in-progress or
`200` + `DagResult` when the root task completes. `project_id` is the root handle.

**Rationale:** env-dispatch already owns the project/issue/agent_run/session machinery;
adding a result endpoint avoids a parallel `submit_root_task` / `collect_result`
orchestration path and fits the existing protocol.

**Alternative considered:** a separate `MulticaDagClient` with `submit_root_task` /
`collect_result` endpoints (assumed by the 2026-06-30 design, precedent
`/api/v1/swe-lego/issues`) — rejected for v1 to avoid duplicating orchestration; the AReaL
client can still expose a `MulticaDagClient` Protocol as a thin adapter over
env-dispatch + the new result endpoint.

### D3 — Lightweight ref-only env snapshots, independent of F

Each segment records a `TeamEnvSnapshot` containing `sandbox_ids` (current sandbox_instance
ids per team agent), `issue_snapshot_id` (issue-subtree ref at close time), and a minimal
`env_state`. No sandbox pause/fork is invoked.

**Rationale:** keeps this change unblocked and independent of F. F's pause-in-place
checkpoint machinery can later enrich these snapshots with full saved-env state.

**Alternative considered:** block on F so snapshots carry real saved sandbox state —
rejected to avoid coupling two large changes; the ref-only shape is sufficient for AReaL's
assembler today (it stamps the snapshot onto the SuperNode without interpretation).

### D4 — Polling return: `202` in-progress / `200` + `DagResult` done

The result endpoint returns `202` + a status body while the root task is running and
`200` + `DagResult` once it completes (terminal: success or failed). AReaL polls.

**Rationale:** simpler and more robust against Multica's long-running, async agent
execution than a held long-poll, SSE, or webhook. No connection-held state.

**Alternatives considered:** long-poll / SSE / webhook — rejected for v1 complexity; can be
added later behind the same endpoint.

### D5 — Turn-index via shared `interaction_id` (revised during build Task 1)

**Correction:** Multica does **not** proxy `/chat/completions`. In training mode the
sandboxed agent (`pi -p --provider areal`) routes each LLM call through `db_bridge` → a
shared Supabase table → AReaL's proxy-rollout gateway; Multica never sees the LLM
request/response directly. Consequently Multica has no native per-turn counter at the
proxy boundary, and `task_message.seq` is **not** a turn index: the in-sandbox daemon
assigns `seq` (`internal/daemon/daemon.go`) per *agent event* — `text`, `tool_use`,
`tool_result`, `thinking`, `error` each get their own `seq` — so one assistant turn
produces 3+ `task_message` rows. AReaL's `turn_idx` (`tree_store.py` `Node`) is instead
one per `/chat/completions` assistant response. `seq ≠ turn_idx`.

**Revised mechanism — shared `interaction_id`:**

1. AReaL's proxy already mints a per-response `interaction_id` (the `Node.node_id`, a UUID
   from the inference engine) and returns it in the `/chat/completions` response `id`
   (small AReaL-side change if not already exposed in the body).
2. The response transits `db_bridge` transparently to the agent. `pi`'s areal provider
   extracts the `interaction_id` and emits it in its `message_end` stream event — the
   event that is 1:1 with each `/chat/completions` response (today `pi` consumes
   `message_end` internally for usage and emits no per-turn `agent.Message`; this adds an
   `ID`/`InteractionID` field).
3. The Multica daemon captures the `interaction_id` from `message_end` and stamps it onto
   the turn's `task_message` rows (new `task_message.interaction_id` column + index; every
   event-row of a turn — text/tool_use/tool_result — shares that turn's id).
4. Multica numbers turns per session as the **1-based ordinal of `interaction_id`s in
   creation order**. A communication event (e.g. a delegation `tool_use`) carries the
   `interaction_id` of the LLM response that produced it; that id's per-session ordinal is
   the segment's `end_turn_idx`.

**Why the contract stays stable (no `SuperNodeAssembler` change):** both Multica (via
`message_end` order) and AReaL (via `list[Node]` enumerate order in
`customized_grouped_workflow.py`) define `turn_idx` as the 1-based ordinal of LLM
responses per session in execution order. Same order ⇒ same ordinals ⇒ alignment by
construction — with no Multica→AReaL query. `SegmentSpec.start_turn_idx`/`end_turn_idx`
slice `list[Node]` exactly as before; `interaction_id` is an optional audit key (AReaL may
assert `Node[turn_idx].node_id == interaction_id`).

**Rationale:** the only component bridging the `db_bridge` response (HTTP, carries
`interaction_id`) and the Multica daemon (stdout stream) is the agent (`pi`), so `pi`'s
`message_end` is the unique correct surfacing seam. Ordinal alignment removes the
off-by-one risk of independent counters.

### D6 — Segment semantics (matches 2026-06-30 D7, Option A)

A segment is all turns since the previous communication event (exclusive), up to and
including the turn that performs the next communication event. That closing-event turn is
the segment's terminal. A run's final segment with no closing communication event is a leaf
segment with `closing_event = None`. Slices of one run must densely cover `[1, len(nodes)]`
with no gaps or overlaps (the assembler validates this).

### D7 — `session_to_agent_run` captured at `/rl/start_session` time

Multica already calls `/rl/start_session` once per agent run. The recording service
captures `session_id ↔ agent_run_id` at that moment and emits it in `DagResult`.

### D8 — Edges match AReaL `EdgeType` exactly

- `DELEGATION`: parent-issue run → child sub-issue run (fan-out).
- `MENTION`: one run mentions/triggers another (peer, topology-only — does not set a causal
  parent).
- `COMPLETION`: child run completes → parent run continuation (fan-in).

Edges point cause → effect and must keep the DAG acyclic; the assembler enforces the
topological invariant and rejects cycles.

## Risks / Trade-offs

- **Turn-index drift** (off-by-one between Multica's ordinal and AReaL's `list[Node]`) →
  Mitigation: ordinals are derived from the same execution order on both sides (D5), so
  drift is structurally impossible; the assembler additionally validates dense coverage
  `[1, len(nodes)]` and raises `DAGError` on mismatch, and may assert
  `Node[turn_idx].node_id == interaction_id`. The residual risk is a missed/extra
  `message_end` event (e.g. a retried `/chat/completions` that AReaL does not cache as a
  Node) → Mitigation: `pi` emits `message_end` only for responses it commits; AReaL caches
  exactly one Node per committed response; an E2E test asserts the per-session count
  matches.
- **Concurrent fan-out delegation ordering** (planner delegates to several workers at once)
  → Mitigation: each delegation closes the planner's current segment and opens a child
  segment per worker; multiple `DELEGATION` edges are recorded deterministically. The DAG
  stays acyclic because delegation always flows parent → child.
- **Polling latency / cost** → Mitigation: AReaL polls with a reasonable backoff; the
  endpoint is a cheap status read until completion. Acceptable trade-off vs. held
  connections.
- **Env snapshot is refs-only** → lower fidelity than a real saved environment →
  Mitigation: documented as a v1 limitation; F can enrich later. AReaL stamps the snapshot
  without interpreting it, so the assembler is unaffected.
- **Segment recording failures must not break the rollout** → Mitigation: recording is
  best-effort; a recording error is logged and the segment/edge is omitted or marked, but
  the agent run continues. A `DagResult` with a missing segment is detectable by AReaL's
  assembler validation (dense-coverage check).
- **Reusing env-dispatch couples DAG-result lifetime to project cleanup** → Mitigation: the
  DAG result is read-only and derived from recorded rows; `DELETE /env-dispatch/{projectID}`
  cascades to the new segment/edge/snapshot tables so cleanup stays consistent.

## Migration Plan

1. Add migration creating `interaction_dag_segment`, `interaction_dag_edge`, and
   `interaction_dag_env_snapshot` tables (keyed by `project_id` + `agent_run_id` /
   `segment_id`), with `IF NOT EXISTS` idempotency.
2. Generate / extend DB query files (`CreateSegment`, `AddEdge`, `CaptureEnvSnapshot`,
   `ListSegmentsForProject`, `ListEdgesForProject`, `ListEnvSnapshotsForProject`,
   `GetDagStatus`).
3. Wire communication-event hooks (delegation, mention, completion, squad briefing) to the
   recording service behind a feature flag defaulting to on for trained rollouts.
4. Add the `GET /api/v1/env-dispatch/{projectID}/dag` handler.
5. Rollback: the feature flag disables recording + returns `503`/`202` from the endpoint;
   existing env-dispatch behavior is unchanged when disabled. The new tables are additive
   and safe to leave in place.

## Open Questions

- Exact seam where Multica counts an assistant turn (confirm against the agent-run driving
  code + `internal/arealrl/client.go`) — to be resolved in Task 1 investigation.
- Whether the root-task completion signal is the existing agent-run terminal status or a
  dedicated rollout-complete marker — confirm during Task 1.
- Whether `issue_snapshot_id` should be a real snapshot ref (requires F) or a pointer to the
  live issue subtree at close time — v1 uses the live issue id + close timestamp; revisit
  when F lands.
