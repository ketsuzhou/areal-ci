## 1. Diagnosis Pi-agent runner + structured output (Multica, TDD)

**Files:** new `server/internal/service/diagnosis_agent.go` (+ `_test.go`); mirrors
`server/internal/service/evolution_review_provider.go`.

- [x] 1.1 Failing tests: `NewDiagnosisAgentRunner(DiagnosisAgentConfig{Provider,
  ExecutablePath, Model, Timeout, Backend})` launches a Pi agent with a diagnosis system
  prompt and returns structured per-step rewards (one score per LLM output).
- [x] 1.2 Failing tests: the diagnosis system prompt enforces score-in-`[0, score_max]`
  per LLM output and structured output (parsing mirrors `judge_prompt.py`'s discipline);
  unparseable output yields an empty reward set + error, not fabricated scores.
- [x] 1.3 Failing tests: timeout / non-zero exit / missing rewards are logged and surfaced
  as a soft failure (never panics); `Backend` is injectable for tests.
- [x] 1.4 Implement the runner + prompt + `parse_step_rewards` parser.
- [x] 1.5 Commit: `feat(diagnosis-agent): Pi-agent runner + per-step reward parsing`
  (multica dev a6a2ce86e; re-applied after prior-session fix did not persist).

## 2. Tool surface + per-segment turn-range capture (Multica, TDD)

**Files:** new `server/internal/service/diagnosis_tools.go` (+ `_test.go`);
`server/internal/service/interaction_dag.go` (`CloseSegmentForEvent`); new migration;
reads `task_message` (migration 026).

- [x] 2.1 Failing tests: `get_interaction_dag(project_id)` returns segments + edges
  (`segment_id`, `agent_run_id`, `closing_event`, `closing_event_target_segment`, edge
  `type`) - the supernode-granularity flow; read-only.
- [x] 2.2 Failing tests: `get_segment_messages(segment_id)` slices `task_message` by
  `(task_id, seq ∈ [start_seq, end_seq])`, respecting a per-call byte/turn budget (mirror
  `evolution_review_provider.go`'s `maxEvolutionReview*Bytes` caps).
- [x] 2.3 Failing tests: `get_task_context(task_id)` returns task goal / gold context used
  to ground "contribution to completing the task".
- [x] 2.4 Failing tests: tools reject writes (read-only) and enforce project/workspace
  scoping (no cross-workspace segment access).
- [x] 2.5 Failing tests: `CloseSegmentForEvent` captures `start_seq`/`end_seq` from the
  exported trajectory and stores them on the `interaction_dag_segment` row; a leaf segment
  covers its agent run's full turn range. (Resolves Q3.)
- [x] 2.6 Implement the tool handlers; extend `CloseSegmentForEvent` to record the turn
  range; add the `start_seq`/`end_seq` migration.
- [x] 2.7 Commit: `feat(diagnosis-agent): read-only tools + per-segment turn-range capture`.

## 3. Trigger at collaborative-task completion (Multica, TDD)

**Files:** `server/internal/service/task.go` (`CompleteTask` / `captureTaskCompleted` /
terminal path, ≈L1357/182/1567), `server/internal/service/training.go` (close-hook
ordering), tests.

- [ ] 3.1 Failing tests: at collaborative-task (root/project) terminal the diagnosis agent
  fires with the project's segment DAG, **before** `/dag` returns `200` done.
- [ ] 3.2 Failing tests: trigger is gated by `DIAGNOSIS_AGENT_ENABLED` AND `s.Training` AND
  `INTERACTION_DAG_ENABLED`; a non-trained / non-recorded task runs no diagnosis.
- [ ] 3.3 Failing tests: a diagnosis soft-failure (timeout/parse error) is logged and does
  NOT block task completion or the `/dag` done transition (best-effort, sparse-reward
  fallback).
- [ ] 3.4 Failing tests: the diagnosis agent views the *whole* project segment DAG (all
  agents), distinct from the per-agent critic terminal.
- [ ] 3.5 Implement the trigger wiring behind the flag.
- [ ] 3.6 Commit: `feat(diagnosis-agent): trigger at collaborative-task completion`.

## 4. Project-scoped reward delivery via AssembledDag (Multica, TDD)

**Files:** new migration (`interaction_dag_step_reward` table);
`server/internal/service/interaction_dag.go` (`AssembleAssembledDag`);
`server/internal/handler/env_dispatch.go` (`/dag`); areal `AssembledDag`/`SegmentSpec`
(`customized_areal/tree_search/agents/multica_dag_client.py`); tests.

- [x] 4.1 Failing tests: the diagnosis agent's per-step rewards are written to
  `interaction_dag_step_reward(segment_id, seq, score, rationale)`, keyed by
  `(segment_id, seq)`.
- [x] 4.2 Failing tests: `AssembleAssembledDag(project_id)` includes a `step_rewards[]`
  structure (one entry per scored LLM output) alongside segments + edges.
- [x] 4.3 Failing tests: `GET /api/v1/env-dispatch/{project}/dag` serves `step_rewards[]`
  on `200` done; when diagnosis did not run / soft-failed, `step_rewards[]` is empty (absent
  rewards, not zero-filled).
- [ ] 4.4 Failing tests: `/dag` stays `202` in-progress while diagnosis runs, then `200`
  done with rewards (bounded by the diagnosis timeout); soft-failure -> `200` done + empty
  `step_rewards[]`.
- [ ] 4.5 Implement the step-reward table + `AssembleAssembledDag` extension + `/dag`
  serving; extend areal `AssembledDag`/`SegmentSpec` to carry `step_rewards[]` (refines
  in-flight `v2-segment-dag-recording` - reconcile on apply).
- [ ] 4.6 Commit: `feat(diagnosis-agent): project-scoped per-step reward delivery via AssembledDag`.

> **Task 5 (plan, Go-side) split note:** 4.1-4.3 + the multica-Go portion of 4.5 (table
> already existed as migration 161; `AssembleAssembledDag` extension + `/dag` serving +
> `RecordStepRewards` upsert) are DONE - committed `11d07dbc0` on multica `dev`, SDD task
> review APPROVED (build/vet/tests exit 0, no regressions). **4.4** (202-while-diagnosis ->
> 200) is plan Task 4 (trigger) - diagnosis firing is what produces the 202 window. The
> **areal-Python portion of 4.5** (`AssembledDag`/`SegmentSpec` carry `step_rewards[]`) is
> plan Task 6 / openspec group 5. **4.6** commit landed as the Go-side
> `feat(diagnosis-agent): project-scoped step_rewards via AssembledDag + /dag` (message per
> the Task 5 brief); the areal-side commit lands with group 5.6.

## 5. AReaL consumer + flat-judge removal (Python, TDD)

**Files:** `customized_areal/tree_search/agents/multica_dag_client.py`,
`customized_areal/tree_search/agents/supernode_assembler.py`,
`customized_areal/tree_search/core/customized_grouped_workflow.py` (≈L747–787, 892–912,
1167, 1252), `customized_areal/tree_search/core/judge_prompt.py`,
`customized_areal/tree_search/config.py` (≈L151–178),
`customized_areal/tree_search/distilling/config.py`, tests.

- [ ] 5.1 Failing tests: AReal reads `step_rewards[]` from the `AssembledDag` and maps
  `(segment_id, seq)` -> `SuperNode` / `Node` -> `process_reward` on the node (consumed by
  `dag_advantage.py`).
- [ ] 5.2 Failing tests: rewards with no matching `SuperNode`/`Node` are dropped + logged
  (absence distinguishable, no default-fill).
- [ ] 5.3 Failing tests: with empty `step_rewards[]`, process reward is sparse/zero (matches
  `enable_judge_process_reward=False`), NOT a re-introduced judge.
- [ ] 5.4 Verify `critic_score.py` usage: confirm whether `ExternalDiagnoseProvider` /
  `diagnose_episode` is still needed by the critic; keep it if so (non-goal #3).
- [ ] 5.5 Remove the `enable_judge_process_reward` branch in `customized_grouped_workflow.py`
  and delete `judge_prompt.py`; remove `enable_judge_process_reward` /
  `judge_process_reward_beta` / `judge_model_name` / `judge_max_concurrency` flags.
- [ ] 5.6 Commit: `refactor(tree-search): consume multica per-step rewards, remove flat judge`.

## 6. Config + flags + spec

**Files:** `server/internal/daemon/config.go` / env, `openspec/changes/multica-pi-diagnosis-
agent/specs/diagnosis-process-reward/spec.md`.

- [ ] 6.1 Add `DIAGNOSIS_AGENT_ENABLED` (default off) + `DIAGNOSIS_AGENT_*` (path/model/
  timeout/score_max) env config, composing with `s.Training` / `INTERACTION_DAG_ENABLED`.
- [ ] 6.2 Confirm `specs/diagnosis-process-reward/spec.md` matches the implemented behavior
  (trigger/gating, tool surface, turn-range capture, per-LLM-output output, AssembledDag
  delivery, soft-fail, critic coexistence).
- [ ] 6.3 Commit: `feat(diagnosis-agent): config + spec`.

## 7. Integration + E2E

**Files:** `server/internal/service/` integration tests; cross-repo E2E.

- [ ] 7.1 Integration: a trained 3-agent `mode=scratch` rollout -> segments recorded ->
  diagnosis agent fires -> `step_rewards` written -> `/dag` serves them.
- [ ] 7.2 E2E: rollout -> diagnosis `step_rewards` -> AReal resolve -> `ExecutionDag` with
  `process_reward` on nodes -> minimal training step -> cleanup.
- [ ] 7.3 Sanity: compare diagnosis per-step rewards vs. the removed judge's scores on one
  fixed episode (diagnostic, not a gate).
- [ ] 7.4 Commit: `test(diagnosis-agent): integration + E2E`.

## 8. Sweep + docs

- [ ] 8.1 Grep sweep: `enable_judge_process_reward` / `judge_prompt` / `judge_model_name`
  resolve only to intended removals; no dangling refs.
- [ ] 8.2 Confirm `design.md` Open Questions (Q1–Q4) are resolved (Q1 per-LLM-output, Q2
  AssembledDag-attached, Q3 task_message + stored turn range, Q4 coexist) and update if any
  drifted.
- [ ] 8.3 Commit: `docs(diagnosis-agent): sweep + finalize design`.
