# Brainstorm Summary

- Change: sub-project-g-multica-interaction-dag
- Date: 2026-07-07

## Confirmed Technical Approach

Implement the deferred Multica-side **producer** of the `DagResult` contract that AReaL
already consumes (`SuperNodeAssembler.assemble` in
`customized_areal/tree_search/agents/`). During a task, Multica incrementally records agent
interactions as a communication-bounded segment DAG and returns the assembled `DagResult`
at root-task completion.

Locked decisions (all user-confirmed):

- **D1 — Incremental recording at communication events.** Segments + typed edges
  (DELEGATION/MENTION/COMPLETION) + per-run turn indices are recorded as events fire
  (delegation→sub-issue, mention, completion→notify-parent, squad briefing), not
  reconstructed after completion. Reconstruct-after was rejected: per-run turn indices and
  close-ordering are unrecoverable.
- **D2 — Reuse the env-dispatch path + polling endpoint.** `create_env_dispatch(mode=scratch)`
  starts the rollout; `GET /api/v1/env-dispatch/{projectID}/dag` returns `202` in-progress
  / `200`+`DagResult` done. `project_id` is the root handle. A separate
  `submit_root_task`/`collect_result` path (assumed by the 2026-06-30 doc) is rejected for
  v1 to avoid parallel orchestration; the AReaL `MulticaDagClient` is a thin adapter over
  env-dispatch + the result endpoint.
- **D3 — Lightweight ref-only env snapshots, independent of F.** Each segment records
  `TeamEnvSnapshot` = `sandbox_ids` (sandbox_instance ids per team agent) +
  `issue_snapshot_id` + minimal `env_state`. No sandbox pause/fork. F can enrich later.
- **D4 — Polling return.** `202` in-progress / `200`+`DagResult` done; AReaL polls with
  backoff. Long-poll/SSE/webhook deferred.
- **D5 — Turn-index at the proxy driving boundary.** One assistant turn = one turn;
  Multica maintains a per-`agent_run_id` counter and stamps the closing-event turn as
  `end_turn_idx`.
- **D6 — Segment semantics (2026-06-30 D7, Option A).** Segment = turns since the previous
  comm event, up to and including the closing-event turn; leaf segment has
  `closing_event=None`; dense `[1, len(nodes)]` coverage.
- **D7 — `session_to_agent_run` captured at `/rl/start_session` time.**
- **D8 — Edges match AReaL `EdgeType` exactly** (delegation/mention/completion; acyclic).

Producer contract (unchanged AReaL consumer): `DagResult{session_ids,
session_to_agent_run, segments:list[SegmentSpec], edges:list[EdgeSpec],
env_snapshots:dict[segment_id, TeamEnvSnapshot]}`.

## Key Trade-offs and Risks

- **Turn-index drift** (off-by-one vs AReaL `list[Node]`) → assembler already validates
  ranges and raises `DAGError`; tests cover boundaries.
- **Concurrent fan-out delegation ordering** → each delegation closes planner segment +
  opens child; multiple DELEGATION edges recorded deterministically; acyclic by
  parent→child flow.
- **Polling latency/cost** → cheap status read until completion; acceptable vs held
  connections.
- **Ref-only snapshots lower fidelity** → documented v1 limitation; F enriches later;
  assembler stamps snapshot without interpreting.
- **Recording failures must not break rollout** → best-effort recording; errors logged;
  missing segments detectable by assembler dense-coverage check.
- **DAG-result lifetime coupled to project cleanup** → `DELETE /env-dispatch/{projectID}`
  cascades to new tables; read-only derived result.

## Testing Strategy

- **Multica Go (TDD):** incremental recording (delegation/mention/completion/squad
  briefing/leaf/fan-out), ref-only snapshots, `DagResult` assembly (exact shape, turn
  ranges, edge types, acyclicity, failed-rollout handling), polling endpoint
  (202/200/404/403/disabled).
- **AReaL Python (TDD):** adapter polls to 200 + returns `DagResult`; 202 retry/backoff;
  404/403 typed errors; returned `DagResult` consumed losslessly by
  `SuperNodeAssembler.assemble` on a synthetic 3-segment planner→worker→synthesizer DAG.
- **E2E (if feasible):** `mode=scratch` 3-agent team → record → poll `GET .../dag` →
  assemble losslessly. Verify env snapshots are refs-only (F-independence).

## Spec Patches

None. The delta spec `specs/interaction-dag-return/spec.md` already covers recording,
turn-index tracking, ref-only snapshots, polling return, and session mapping with
scenarios. No second requirements spec in the Design Doc.
