---
comet_change: multica-pi-diagnosis-agent
role: technical-design
canonical_spec: openspec
---

# Design: Multica Pi Diagnosis Agent (per-step process reward)

Canonical capability spec:
`openspec/changes/multica-pi-diagnosis-agent/specs/diagnosis-process-reward/spec.md`.
OpenSpec artifacts (proposal/design/tasks) are the upstream source of truth. This document
deepens the HOW (implementation approach, risks, testing, boundary conditions) and does not
restate requirements.

## Context

Per-step (process) reward today lives in **AReaL** as a flat LLM judge:
`core/judge_prompt.py` (`build_judge_instruction` / `parse_turn_scores`) builds one prompt over
the *entire* episode + gold answer, `distilling/diagnose_provider.py`
(`ExternalDiagnoseProvider.diagnose_episode`) makes one OpenAI-compatible call, and
`core/customized_grouped_workflow.py` (≈L747–787, 892–912, 1167, 1252) parses per-turn scores
and blends them with `judge_process_reward_beta` into a dense `process_reward` `r_t` (consumed
by `agents/dag_advantage.py`). Config flags live in `config.py` ≈L151–178 and
`distilling/config.py`. The judge consumes the **flattened** episode; it cannot inspect the
supernode/segment collaboration structure or the raw inter-agent conversation.

Code grounding (live seam trace, `multica/server` + areal master):

- `pi` = the Pi coding agent runtime (`pi.dev`), launched as `pi --provider areal --model
  areal-default` (`internal/service/training.go`). Precedent for **Pi-as-reviewer**:
  `internal/service/evolution_review_provider.go` `NewAgentEvolutionReviewer` /
  `EvolutionAgentReviewConfig{Provider, ExecutablePath, Model, Timeout, Backend}` launches a Pi
  agent that returns structured JSON (system prompt + `maxEvolutionReview*Bytes` budgets +
  `Timeout`). The diagnosis runner mirrors this.
- Critic reward path (scalar outcome reward): `training.go` `critic_agent_id` ("Replaces D's
  close hook", ≈L438–490), `arealSessionCloser.SetReward(ctx, proxyKey string, reward float64)`
  (≈L72) + `EndSession`; `maybeCloseTrainingSession` (≈L299) resolves reward from
  `training_dispatch` / `TRAINING_DEFAULT_REWARD`. The diagnosis agent is the **per-step
  process-reward** analogue, on a **separate channel** (see D1/D2).
- Segment DAG storage: `interaction_dag_segment` (`segment_id`, `project_id`, `agent_run_id`,
  `issue_id`, `task_id`, `trajectory_id int64`, `tensor_ref []byte`, `closing_event`,
  `closing_event_target_segment`, `created_at`) + `interaction_dag_edge` (`src_segment_id`,
  `dst_segment_id`, `type`). `InteractionDAGService.CloseSegmentForEvent`
  (`internal/service/interaction_dag.go:141`) closes the session's current segment ->
  `arealrl.CloseSegment` -> `ExportTrajectory` -> `decodeTensorRef` -> inserts the segment row.
  **No `start_turn_idx`/`end_turn_idx` columns** - the per-segment turn content is discarded
  (only `tensor_ref` kept).
- LLM messages: `task_message` (`task_id`, `seq INTEGER`, `type`, `tool`, `content`, `input`,
  `output`), indexed `(task_id, seq)` (migration 026). `task_id` = `agent_run_id` = `task.ID`
  (D8). So per-agent-run LLM turns ARE queryable in Multica, keyed by `(task_id, seq)`.
- `arealrl` client (`internal/arealrl/client.go`): `StartSession`/`SetReward(scalar)`/
  `EndSession`/`CloseSegment`/`ExportTrajectory` - all **session-scoped**, session-key or
  admin-key auth.
- `AssembledDag` (`internal/service/interaction_dag.go` `AssembleAssembledDag`; areal
  `agents/multica_dag_client.py` `SegmentSpec`/`AssembledDag`): segment-level, explicitly "no
  scores, no turn indices, no message text". Served by `GET /api/v1/env-dispatch/{project}/dag`
  (202 in-progress / 200 done / 404 / 403). AReal polls it.
- AReal consumer: `SuperNodeAssembler.assemble_from_refs` maps Multica segment specs ->
  `SuperNode` (1 segment ≈ 1 supernode); `dag_advantage.py` reads `node.process_reward` /
  `outcome_reward` / `value`.

## Approach

### Data flow

```
MULTICA                                                                  AREAL
 trained rollout ── comm event ──▶ CloseSegmentForEvent
                                   stores segment row
                                   + NEW: start_seq/end_seq (from trajectory)

 root task COMPLETE (DIAGNOSIS_AGENT_ENABLED ∧ Training ∧ INTERACTION_DAG_ENABLED)
   │
   ▼
 Diagnosis Pi agent (subprocess) + read-only tools:
   get_interaction_dag(project_id) -> segments + edges (supernode flow)
   get_segment_messages(segment_id) -> task_message sliced by (task_id, seq ∈ [start,end])
   get_task_context(task_id) -> goal/gold
 iterates supernodes, scores each LLM output [0, score_max]
   │  structured: (segment_id, seq, score, rationale)
   ▼
 interaction_dag_step_reward (NEW table)
   │
   ▼  AssembleAssembledDag includes step_rewards[]
   │
 GET /api/v1/env-dispatch/{project}/dag  ──────────────────────────────▶  poll
   200 + AssembledDag{segments, edges, step_rewards[]}                        │
                                                                              ▼
                                              SuperNodeAssembler: (segment_id, seq) -> SuperNode + Node
                                              process_reward on Node (dag_advantage consumes)
                                              [flat judge REMOVED]

 Separate, unchanged: critic scalar outcome reward ──▶ close-hook SetReward
```

### Components

1. **`internal/service/diagnosis_agent.go`** (new) - `DiagnosisAgentRunner` (Pi subprocess via
   the `NewAgentEvolutionReviewer` pattern), diagnosis system prompt (enforces score-in-
   `[0, score_max]` per LLM output + structured output, mirroring `judge_prompt.py`'s
   discipline), `parse_step_rewards` parser. `Backend` injectable; soft-fail on
   timeout/non-zero-exit/empty.
2. **`internal/service/diagnosis_tools.go`** (new) - read-only tools: `get_interaction_dag`,
   `get_segment_messages` (slices `task_message` by `(task_id, seq ∈ [start_seq, end_seq])`),
   `get_task_context`. Enforce workspace/project scoping + per-call byte/turn budgets.
3. **Schema** (new migration) - add `start_seq`/`end_seq INTEGER` to `interaction_dag_segment`
   (captured at `CloseSegmentForEvent` from the exported trajectory's turn count); new
   `interaction_dag_step_reward(segment_id, seq, score, rationale, created_at)` table.
4. **Trigger** - fire at root-task terminal in `internal/service/task.go`
   (`CompleteTask`/`captureTaskCompleted`, ≈L1357/182), **before** `/dag` returns 200-done;
   gated by `DIAGNOSIS_AGENT_ENABLED` ∧ `s.Training` ∧ `INTERACTION_DAG_ENABLED`.
5. **`AssembleAssembledDag` + `/dag`** - include `step_rewards[]` (empty when diagnosis did not
   run / soft-failed). The `/dag` 202->200 transition waits for diagnosis (bounded by timeout).
6. **AReal consumer** - `MulticaDagClient`/`SuperNodeAssembler` map `(segment_id, seq)` ->
   `SuperNode`/`Node`, write `process_reward`; remove the flat judge path (`enable_judge_process_reward`
   wiring + `judge_prompt.py` + flags).

## Decisions

### D1 - Separate channel; critic coexists (resolves open-phase Q4)

The diagnosis agent supplies **process** rewards (per-step, project-scoped, via `AssembledDag`).
The critic supplies **outcome** rewards (scalar, session-scoped, via close-hook `SetReward`).
They coexist on two channels; the diagnosis agent does **not** subsume the critic and does
**not** modify `critic-driven-training-signal` (no requirement change there - the critic path is
untouched). Coexistence is documented in the new `diagnosis-process-reward` spec. (The
open-phase proposal hedged "Modified: critic-driven-training-signal"; D1 resolves it to **no
delta** - the spec delta is the new capability only.)

### D2 - Project-scoped delivery via `AssembledDag` (resolves Q2)

The diagnosis agent is project-scoped (all sessions' segments), so session-scoped `SetReward`
is a poor fit. Per-step rewards are attached to the `AssembledDag` and served by `/dag`, which
AReal already polls. Alternatives rejected: (B) session-scoped `SetStepRewards` - splits one
project-scoped result across N sessions; (C) standalone project endpoint - redundant with
`/dag`. **Contract extension:** the `AssembledDag` (today "no scores, no turn indices") gains a
`step_rewards[]` structure - this refines the in-flight `v2-segment-dag-recording` capability
(reconcile on apply if both ship together).

### D3 - Message source: `task_message` sliced by stored turn range (resolves Q3)

`task_message` holds per-agent-run LLM turns keyed by `(task_id, seq)`. Segments lack turn-range
columns today, so `CloseSegmentForEvent` is extended to capture `start_seq`/`end_seq` from the
exported trajectory and store them on the segment row. `get_segment_messages(segment_id)` then
slices `task_message` precisely. Alternatives rejected: return the whole agent_run's messages
and let the agent infer boundaries (imprecise); fetch from areal's `tensor_ref` trajectory
(couples Multica diagnosis to areal's store).

### D4 - Keying: `(segment_id, seq)` -> aggregate to `SuperNode.process_reward`

Rewards are keyed by `(segment_id, seq)` at the **diagnosis-emission** granularity
(per-LLM-output turn). AReal maps `segment_id` -> `SuperNode`. The v2 GAE
(`events_from_nodes`) consumes a **per-segment** `SuperNode.process_reward` (one
GAE step per segment; `assemble_from_refs` builds SuperNodes with `nodes=[]`, so
there are no per-turn DAG Nodes to write to). So AReal **aggregates** the
segment's per-turn step rewards into a per-segment `SuperNode.process_reward` =
`mean(scores) / score_max` in `[0, 1]` (`score_max` served by `/dag` so AReal
does not guess Multica's scale - boundary canonicalization). Per-turn scores
remain stored in `interaction_dag_step_reward` + `/dag step_rewards[]` for future
per-token credit use. Absent/unscored segments stay `0.0` (sparse, never
fabricated). Resolved in Task 7 (user-approved aggregate-to-SuperNode option;
the design's original "seq -> Node" did not fit the v2 tensor-ref GAE path).

### D5 - Trigger sequencing

Diagnosis runs **synchronously** at root-task terminal - wired into
`RouteTerminalTrainingTask` after `maybeTriggerCheckpoint` and before the close
hook (`SetReward`/`EndSession`), so `step_rewards` are written before the task's
rewards are delivered. **Correction** (verified in Task 4): `CompleteAgentTask`
persists terminal status *before* `RouteTerminalTrainingTask`, so `/dag` sees a
terminal (200, not 202) status during the synchronous diagnosis - the rewards
land before the first done-poll that observes 200, but `/dag` does not stay 202
*during* diagnosis. A strict 202-while-diagnosis-in-progress would require a
diagnosis-in-progress flag (deferred gap). On soft-failure (`Diagnose` error),
no rewards are written and `/dag` returns 200 with an empty `step_rewards[]`
(AReal falls back to sparse `process_reward = 0.0`).

## Error handling / boundary conditions

- Diagnosis soft-failure (timeout / non-zero exit / unparseable / empty) -> log, write **no**
  step rewards, `/dag` still returns the DAG; areal uses sparse/zero `process_reward` (matches
  today's `enable_judge_process_reward=False`). **Never** fabricated defaults (project
  boundary-value rule: absence stays distinguishable).
- Rewards with no matching areal `SuperNode`/`Node` -> dropped + logged.
- Tools are read-only and workspace/project-scoped (no cross-workspace reads); per-call
  byte/turn budgets mirror `evolution_review_provider.go`.
- `ExternalDiagnoseProvider`/`diagnose_episode` is kept if `agents/critic_score.py` still uses
  it (verify before removing) - only the *judge* usage is removed.

## Testing strategy

- **Unit (Go):** `parse_step_rewards` (valid/malformed/empty); tool handlers (segment->message
  slicing, scoping, budgets); `AssembleAssembledDag` includes `step_rewards`; trigger gating;
  `start_seq`/`end_seq` captured at close.
- **Unit (Py):** `(segment_id, seq)` -> `SuperNode`/`Node` mapping; unmatched-reward drop;
  sparse fallback; judge-removal grep (`enable_judge_process_reward`/`judge_prompt`/`judge_model_name`
  resolve only to intended removals).
- **Integration (Go):** trained 3-agent `mode=scratch` rollout -> segments recorded ->
  diagnosis fires -> `step_rewards` written -> `/dag` serves them.
- **E2E (cross-repo):** rollout -> diagnosis -> `/dag` with rewards -> AReal resolve ->
  `ExecutionDag` with `process_reward` -> minimal training step -> cleanup.
- **Diagnostic (not a gate):** compare diagnosis per-step scores vs. the removed judge's scores
  on one fixed episode.

## Spec patches written back to OpenSpec

- **New** `specs/diagnosis-process-reward/spec.md` - trigger + gating, read-only tool surface,
  per-LLM-output reward output keyed by `(segment_id, seq)`, project-scoped delivery via
  `AssembledDag step_rewards[]`, soft-fail semantics, coexistence with the scalar critic.
- **Refines** in-flight `v2-segment-dag-recording` - `AssembledDag` carries `step_rewards[]`
  when diagnosis ran (reconcile on apply).
- **No delta** for `critic-driven-training-signal` (D1: critic path unchanged).

## Migration / rollout

1. Land Multica diagnosis agent + tools + trigger + schema (gated, off by default).
2. Land `AssembledDag step_rewards[]` + AReal consumer (reads per-step rewards when present,
   sparse fallback otherwise).
3. Enable `DIAGNOSIS_AGENT_ENABLED` on a test project; verify rewards land on the right
   `SuperNode`/`Node`.
4. Remove the areal flat judge + flags once diagnosis is the confirmed process-reward source.
5. Rollback: disable `DIAGNOSIS_AGENT_ENABLED`; areal falls back to sparse `process_reward`
   (judge is removed, so no automatic revert to judging - one-way step, documented).
