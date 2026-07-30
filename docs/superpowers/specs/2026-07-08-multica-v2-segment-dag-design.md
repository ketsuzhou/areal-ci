---
comet_change: multica-v2-segment-dag-training
role: technical-design
canonical_spec: openspec
archived-with: 2026-07-13-multica-v2-segment-dag-training
status: final
---

# Design: Multica v2 Segment-DAG Training (Change 1 - Data Path)

Canonical capability spec: `openspec/changes/multica-v2-segment-dag-training/specs/v2-segment-dag/spec.md`.
OpenSpec artifacts (proposal/design/tasks) are the upstream source of truth. This document
deepens the HOW (implementation approach, risks, testing, boundary conditions) and does not
restate requirements.

## Context

AReaL 2.0 (`areal/v2`) `inference_service` captures each assistant turn as an interaction in
a `SessionData` (`active_completions` + `ready_trajectories`). Today trajectory closure is
coupled to reward (`set_reward` -> `_mark_active_trajectory_ready_locked`, which requires
`_last_reward_interaction_id`), and export returns `RTensor` refs at session/trajectory
granularity. Multica needs one agent-run session sliced into multiple communication-bounded
segments, each exported as a trajectory, with reward assigned later by AReaL (judge, change 2)
- not at close time.

Code grounding (close the open-phase open questions):

- `_mark_active_trajectory_ready_locked` (`session.py:186`) is the shared active->ready
  mechanic; it already rejects empty active ("No interactions in session").
- `export_trajectory` (`session.py:310`) already accepts a `trajectory_id` and pops it.
- `ExportTrajectoriesRequest` (`session.py:64`) already has `trajectory_id` + `remove_session`.
- data_proxy `/export_trajectories` (`data_proxy/app.py:713`) already does
  `RTensor.remotize` (returns refs, not bytes).
- `/data/<shard_id>`, `/data/batch`, `/data/clear` already exist, mounted from
  `areal.infra.rpc.guard.data_blueprint` (`data_proxy/app.py:808`).
- reward-less interactions export `rewards=0.0` (`types.py:194`,
  `reward = self.reward if self.reward is not None else 0.0`).
- gateway `/export_trajectories` (`gateway/app.py:483`) revokes the session group only when
  `group_id is not None`; per-segment export omits `group_id`.

**Implication**: change 1's v2 side is smaller than originally scoped - only `close_segment`
is new; export, `/data/*`, and `/data/clear` are reused.

## Approach

`close_segment` reuses `_mark_active_trajectory_ready_locked`'s active->ready mechanic but
**skips the reward dependency**: it does not require `_last_reward_interaction_id`, does not
call `completions.set_reward`, sets `needs_online_callback=False` (Multica drives offline; no
online-ready callback), and stamps `interaction_id = completions.last_interaction_id` as the
terminal. Rejected alternatives: refactoring `set_reward` into separate set+close (larger
blast radius across the online path); setting a sentinel reward via `set_reward` (violates
D1 - reward must not be assigned at close).

## v2 Changes (AReaL)

1. `SessionData.close_segment() -> RewardResult`: under `_lock`, reject empty active
   (`ValueError`), allocate `trajectory_id = _next_trajectory_id`, build `ReadyTrajectory`
   with `needs_online_callback=False` and `interaction_id=last_interaction_id`, move
   `active_completions` -> ready, reset active to a new `InteractionCache`. Does NOT touch
   `_last_reward_interaction_id` / `_last_set_reward_time`.
2. data_proxy `POST /rl/close_segment`: session-key auth (mirror `/rl/set_reward`), call
   `session.close_segment()`, return `trajectory_id` (+ interaction_count, ready_transition).
3. gateway `POST /rl/close_segment`: mirror `/rl/set_reward` - session key, `query_router`
   by token, forward to worker.
4. Reuse `/export_trajectories` (`session_ids=[sid]`, `trajectory_id`, `remove_session=False`,
   no `group_id`), `/data/*`, `/data/clear`.

## Reward Model

`close_segment` trajectories are reward-less (`interactions.reward=None`); export yields
`rewards=0.0`. **Reward is an AReaL-side concern**: change 1 applies a placeholder (zero) at
training; change 2's judge assigns process reward and AReaL applies it at training time. v2
stays reward-agnostic for segments - no "post-close reward injection" mechanism is needed in
v2. This honors D1 (close decoupled from reward).

## Multica Changes

- Per communication event (delegation / mention / completion / squad briefing): call
  `/rl/close_segment` (session key) -> `trajectory_id`; call `/export_trajectories` with
  `session_ids=[sid]`, `trajectory_id`, `remove_session=False`, no `group_id` -> `tensor_ref`;
  record segment `{segment_id, agent_run_id, issue_id, trajectory_id, tensor_ref,
  closing_event, env_snapshot}`.
- At root-task completion: assemble `AssembledDag = {segments, edges, session_to_agent_run}`
  (typed edges `delegation` / `mention` / `completion`; no scores, no turn indices, no text)
  and serve via `GET /api/v1/env-dispatch/{projectID}/dag` (`202` in-progress / `200` done).
- Migration: `interaction_dag_segment` / `_edge` / `_env_snapshot` tables carrying
  `trajectory_id` + `tensor_ref` (no `start_turn_idx` / `end_turn_idx`).

## AReaL Consumer

- `MulticaDagClient`: poll `GET .../dag` (`202` retry with backoff, `200` -> `AssembledDag`,
  `404`/`403` typed errors).
- Resolve each segment's `tensor_ref` via `/data/<shard_id>` / `/data/batch` -> tensors.
- `SuperNodeAssembler.assemble_from_refs(dag, resolver)` (replaces turn-index slice): for each
  segment, `resolver.resolve(tensor_ref)` -> tensors; build a `SuperNode` per segment and an
  `ExecutionDAG` from the edges; validate acyclic via `topological_order()` (raises `DAGError` on
  cycle). Implementation-boundary note: `SuperNode` has no `payload` field, so resolved tensors
  attach to `metadata["tensors"]` (non-invasive); `segment_id` becomes the `node_id` so edges
  resolve directly; `task_id` is `""` because the v2 `SegmentSpec` dropped it; `session_id` is
  reverse-mapped from `session_to_agent_run`. Dense per-session coverage (gap -> `DAGError`) is
  deferred to the training-plumbing unit (U5).
- Minimal training (`segment_dag_trainer.run_segment_dag_training_step`): `get_dag` ->
  `assemble_from_refs` -> `topological_order()` -> `assemble_node_advantages` (global GAE over
  completion-ordered SuperNodes with placeholder zero reward). Change 1 exercises only the GAE
  forward path - no torch/FSDP, no judge, no critic V (those land in change 2).
  `events_from_nodes` treats an unset `value` as `0.0`, so the zero-reward SuperNodes flow
  through with zero advantages (proves the path).
- Cleanup (tensor lifecycle): `DataProxyTensorResolver.clear` (`DELETE /data/clear` with the
  consumed shard ids) + `DataProxySessionRemover.remove` (`POST /export_trajectories` with
  `remove_session=True` - the data_proxy exposes session removal only via the export endpoint).
  Cleanup is success-path only in change 1; a failed step (cycle/DAGError) propagates without
  releasing shards. Tensor-ref contract (change 1): `{"shard_id": str}` per segment; finalized
  when Multica (U6/U8) pins the export contract.

## Data Flow (single rollout)

```
Multica --/chat/completions--> v2 captures interaction (text + tensors, no reward)
[event #k]
Multica --/rl/close_segment-->            v2 -> trajectory_id
Multica --/export_trajectories([sid],traj_id,remove_session=False)--> v2 -> RTensor ref
Multica records segment_k
[completion]
Multica assembles AssembledDag; GET .../dag (202 -> 200)
AReaL polls -> AssembledDag -> /data/* resolve -> build SuperNode/DAG -> train (zero reward)
AReaL DELETE /data/clear + remove_session
```

## Boundary Conditions / Error Handling

- Empty segment (`close_segment` on no active) -> `ValueError` -> 400; no trajectory.
- Unknown `trajectory_id` export -> `KeyError` -> 400; nothing exported.
- Ref lifetime: `DELETE /data/clear` only after training resolves refs; `remove_session` only
  after the session's last segment is consumed.
- Concurrent fan-out delegation -> multiple `DELEGATION` edges, deterministic order, acyclic.
- Missing segment (gap in a run's coverage) -> assembler `DAGError`; no partial DAG.
- Per-segment export omits `group_id` to avoid the gateway revoking the session group on each
  segment.

## Testing Strategy

- v2 (`uv run pytest areal/v2/inference_service/tests/`): `close_segment` no-reward close /
  session live after close / empty-active error / no `set_reward` dependency; per-trajectory
  export with `remove_session=False` (refs-only, no text, unknown-traj error); `/data/*`
  resolve + `/data/clear`.
- Multica (scoped Go): per-event recording (delegation / mention / completion / squad / leaf /
  fan-out); `AssembledDag` assembly (no scores / turn-idx / text, acyclic); polling endpoint
  (202 / 200 / 404 / 403).
- AReaL consumer (`uv run pytest customized_areal/tree_search/tests/`): client poll;
  assembler ref-resolve + acyclic + dense coverage; 3-segment planner->worker->synthesizer
  round-trip; minimal training step; cleanup ordering.
- E2E (if feasible): `mode=scratch` 3-agent rollout -> segments -> `AssembledDag` -> resolve
  -> `ExecutionDAG` -> minimal training -> cleanup.

## Decisions Carried From Open Phase

D1 close_segment decouples boundary from reward; D2 export returns `RTensor` refs; D3
`AssembledDag` carries structure+refs+env only (no scores/turn-idx/text); D4 Multica assembles,
AReaL ref-joins; D5 text stays in AReaL capture; D6 placeholder reward for change 1; D7 reuse
env-dispatch polling, supersedes sub-project-g.

New (design phase): close_segment reuses `_mark_active_trajectory_ready_locked` minus reward
(`needs_online_callback=False`); export / `/data/*` / `/data/clear` reused as-is; reward is
AReaL-side (v2 reward-less for segments); per-segment export omits `group_id`.

## U7 Pre-Build Design Decisions (from seam trace)

Traced against live multica `server/` + areal `customized_areal/` before U7 implementation.
The plan's U7 omitted session/agent_run identity semantics; these pin them down.

- **D8 - `agent_run_id` = `task.ID` (attempt-level); no separate runs table.**
  `EnqueueAgentRun` (handler/env_dispatch.go:662) returns `util.UUIDToString(task.ID)` as the
  runID; `EnvRollout.AgentRunID = task.ID` (env_dispatch.go:717). AReal consumes `agent_run_id`
  as `SuperNode.agent_id` and as the key to reverse-lookup `session_id` from
  `session_to_agent_run` (supernode_assembler.py:159,163). The segment-table `task_id` column
  is redundant traceability - v2 `SegmentSpec` dropped it and sets `SuperNode.task_id=""`
  (supernode_assembler.py:161). Retries create a NEW `task.ID` (child via `parent_task_id`,
  agent.sql:178), so `agent_run_id` is per-attempt, not per-logical-task.

- **D9 - Fresh areal RL session per retry attempt (NOT inherited).** Today `CreateRetryTask`
  copies `p.context` (which holds `areal_proxy` = areal `SessionID`+`ProxyKey`,
  training.go:209), so the child inherits the parent's areal session and
  `maybeOpenTrainingSession`'s `hasArealProxyContext` guard (training.go:200) no-ops - no fresh
  `StartSession`, no `RecordSessionAgentRun`. A retry child's segments then carry the child's
  `task.ID` as `agent_run_id`, which is absent from `session_to_agent_run` ->
  `agent_run_to_session.get(child_id)` = None at assembly (supernode_assembler.py:163) ->
  dangling. Decision: each attempt opens its OWN areal session. Three changes: (1)
  `CreateRetryTask` strips `areal_proxy` from the child's context [keep the chat
  `session_id`/`work_dir` resume CASE-WHEN - that is the multica chat session, separate]; (2)
  `MaybeRetryFailedTask` calls `tryOpenTrainingSession(child, projectID, envID)` BEFORE
  `NotifyTaskEnqueued` (mirror enqueueMentionTask: open at task.go:614, notify at :618); (3)
  `RecordSessionAgentRun` (D10) then fires for the child. Close ordering supports it: `FailTask`
  -> `RouteTerminalTrainingTask` closes S_A (`EndSession`, training.go:424/437/445) ->
  `MaybeRetryFailedTask` creates B -> B opens fresh S_B. Only the 4 retryable reasons produce a
  child (`runtime_offline`/`runtime_recovery`/`timeout`/`codex_semantic_inactivity`,
  task.go:1725); non-retryable failures are terminal (session closed, no child). Scope: change
  (1) touches pre-existing `CreateRetryTask` (migration 055), not U7's new files - U7 in-scope
  dependency. The sweeper path (`runtime_sweeper.FailStaleTasks`, `queued_expired`) bypasses
  `FailTask` (orphaned session, no child) - pre-existing gap (training.go:274-277); the
  fresh-session design's coverage boundary is the `FailTask` path.

- **D10 - `RecordSessionAgentRun` call site.** Inside `maybeOpenTrainingSession`
  (training.go:158), immediately after `StartSession` succeeds (~line 226, after the persist at
  :220, before the slog at :227). Records `{projectID, sessionID=creds.SessionID (line 204),
  agentRunID=taskID}`. This is the single idempotent chokepoint: both the task.go `Enqueue*`
  paths (via `tryOpenTrainingSession`, task.go:510/614/750/834) and the env_dispatch path (via
  the adapter at handler/env_dispatch.go:713 -> `MaybeOpenTrainingSession`) funnel through it;
  placed after the `hasArealProxyContext` guard it fires only on real first-open. It builds the
  `session_to_agent_run: {session_id -> agent_run_id(=task.ID)}` map AReal's assembler consumes.
  Conflation trap: `maybeOpenTrainingSession` receives both `taskID` (the run == agent_run_id)
  and `agentID` (the agent). `agentRunID` = `taskID`, NOT `agentID` - AReal stores `agent_run_id`
  in a field named `agent_id` (supernode_assembler.py:159), which makes this confusion easy.

U7 open items (resolve during U7 implementation): `envID` source for a retry child's
`StartSession` (enqueue paths pass `""`; env_dispatch passes real envID; retry child is not
env-dispatched); exact `tensor_ref` shape `ExportTrajectory` returns (U6) - `CloseSegmentForEvent`
must decode it from the raw traj JSON; `envSnapshot` source for `CloseSegmentForEvent`
(refs-only, no sandbox pause/fork).

## Open Items (deferred to change 2)

- `outcome_reward` semantics (judge supernode score as outcome vs separate verifier) - affects
  reward plumbing shape in change 2.
- Judge trigger timing (completion default vs per-event config) - change 2.

## Out of Scope (change 2 / 3 / F)

Judge + process-reward scoring + actor-model V critic + GAE (change 2); tree search /
branching / `env-dispatch mode=branch` fork (change 3); full sandbox snapshot/fork
(Sub-project F).
