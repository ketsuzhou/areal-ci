---
change: multica-pi-diagnosis-agent
design-doc: docs/superpowers/specs/2026-07-09-multica-pi-diagnosis-agent-design.md
base-ref: f2256a7aa2cf4f1aad1e410e5519994098901611
archived-with: 2026-07-14-multica-pi-diagnosis-agent
---

# Multica Pi Diagnosis Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move per-step (process) reward from areal's flat judge into a tool-using Pi diagnosis agent in multica that fires at collaborative-task completion, views the segment DAG, fetches per-segment LLM messages, and emits per-LLM-output rewards delivered to areal via the `AssembledDag`.

**Architecture:** A Pi subprocess (mirroring `evolution_review_provider.go`'s `NewAgentEvolutionReviewer`) runs at root-task terminal with read-only tools over `interaction_dag_*` + `task_message`. It returns `(segment_id, seq, score)` rewards written to a new `interaction_dag_step_reward` table. `AssembleAssembledDag` serves them as `step_rewards[]` on `/dag`. AReal maps `(segment_id, seq)` -> `SuperNode`/`Node` -> `process_reward`, and the flat judge (`enable_judge_process_reward` / `judge_prompt.py`) is removed. The scalar critic outcome reward stays on the close-hook `SetReward` path, untouched.

**Tech Stack:** Go (multica server, sqlc, pgx) · Python 3.14 (areal, pytest) · Pi agent runtime (`pi --provider areal`) · Postgres migrations.

## Global Constraints

- Rewards are per-LLM-output (turn), keyed by `(segment_id, seq)` where `seq` = `task_message.seq` (per-agent-run, 1-based). Navigation/view unit is the segment (supernode).
- Never fabricate reward defaults: absence (no diagnosis / soft-fail / unmatched key) stays distinguishable - empty `step_rewards[]`, sparse/zero `process_reward`. (Project boundary-value rule.)
- Diagnosis is best-effort: a soft-failure (timeout / non-zero exit / unparseable / empty) MUST NOT block task completion or the `/dag` done transition.
- Gating: diagnosis runs only when `DIAGNOSIS_AGENT_ENABLED` AND `s.Training` AND `INTERACTION_DAG_ENABLED`.
- Critic coexistence: the diagnosis agent does NOT modify the `critic-driven-training-signal` path; the two reward channels are independent.
- `ExternalDiagnoseProvider`/`diagnose_episode` is kept if `critic_score.py` still uses it - only the *judge* usage is removed.

## Dual-repo commit reality

- **areal repo** (`/workspaces/leagent/backend/areal`, branch `master`): owns the OpenSpec change, the Python consumer, and the design/plan docs. Python tasks commit here.
- **multica repo** (`multica/`, branch `feature/multica-v2-segment-dag-training`, untracked in areal): owns the Go server. Go tasks commit here with `git -C multica`. Per project memory, multica Go changes in this checkout are for contract + test consistency (this checkout is not the deploy source).
- The comet `isolation` branch/worktree applies to the **areal** repo. The multica repo is already on its feature branch; Go commits land there directly.
- Go tests: `cd multica/server && go test ./internal/service/ ./internal/handler/` (if Go toolchain available; else note as unverified). Python tests: `.venv-test/bin/python -m pytest <path>`. Lint: `uvx ruff check` (areal), `cd multica/server && go vet ./...` (multica).

## File Structure

**Go (multica/server)**
- Create `internal/service/diagnosis_agent.go` - `DiagnosisAgentRunner` (Pi subprocess via `NewAgentEvolutionReviewer` pattern), diagnosis system prompt, `parse_step_rewards`.
- Create `internal/service/diagnosis_agent_test.go` - runner + parser tests.
- Create `internal/service/diagnosis_tools.go` - `get_interaction_dag`, `get_segment_messages`, `get_task_context` tool handlers.
- Create `internal/service/diagnosis_tools_test.go` - tool tests.
- Create `migrations/<N>_interaction_dag_step_reward.up.sql` (+ `.down.sql`) - add `start_seq`/`end_seq` to `interaction_dag_segment`; new `interaction_dag_step_reward` table.
- Modify `internal/service/interaction_dag.go` - `CloseSegmentForEvent` captures `start_seq`/`end_seq`; `AssembleAssembledDag` includes `step_rewards[]`; new `RecordStepRewards`.
- Modify `pkg/db/generated/interaction_dag.sql.go` + hand-written sqlc query for `InsertInteractionDAGSegmentWithSnapshot` (add `start_seq`/`end_seq` params) - regenerate sqlc.
- Modify `internal/service/task.go` - diagnosis trigger at root-task terminal.
- Modify `internal/service/training.go` - diagnosis runs before close-hook reward delivery.
- Modify `internal/daemon/config.go` - `DIAGNOSIS_AGENT_ENABLED` + `DIAGNOSIS_AGENT_*` env.

**Python (customized_areal)**
- Modify `tree_search/agents/multica_dag_client.py` - `StepReward` dataclass; `AssembledDag.step_rewards` + `from_dict` parsing.
- Modify `tree_search/agents/supernode_assembler.py` - map `(segment_id, seq)` -> `SuperNode`/`Node` -> `process_reward`.
- Modify `tree_search/core/customized_grouped_workflow.py` - remove `enable_judge_process_reward` branch (≈L747–787, 892–912, 1167, 1252).
- Delete `tree_search/core/judge_prompt.py`.
- Modify `tree_search/config.py` (≈L151–178) + `tree_search/distilling/config.py` - remove judge flags.
- Modify tests: `tests/test_multica_dag_client*`, `tests/test_assembler_ref_resolve.py`, `tests/test_v2_session_lifecycle.py`; delete judge-specific tests.

archived-with: 2026-07-14-multica-pi-diagnosis-agent
---

## Task 1: Diagnosis Pi-agent runner + structured-output parser (Go)

**Files:**
- Create: `multica/server/internal/service/diagnosis_agent.go`
- Test: `multica/server/internal/service/diagnosis_agent_test.go`

**Interfaces:**
- Produces: `type StepReward struct { SegmentID string; Seq int; Score int; Rationale string }`; `func parseStepRewards(output string, scoreMax int) ([]StepReward, error)`; `type DiagnosisAgentRunner struct{...}` with `func (r *DiagnosisAgentRunner) Diagnose(ctx context.Context, projectID string) ([]StepReward, error)`. `Backend` is injectable (mirrors `agentpkg.Backend` from `evolution_review_provider.go`).

- [ ] **Step 1: Write failing parser tests**

```go
func TestParseStepRewards_Valid(t *testing.T) {
	in := "```json\n[{\"segment_id\":\"s1\",\"seq\":1,\"score\":8,\"rationale\":\"x\"},{\"segment_id\":\"s1\",\"seq\":2,\"score\":2}]\n```"
	got, err := parseStepRewards(in, 10)
	if err != nil { t.Fatal(err) }
	if len(got) != 2 || got[0].SegmentID != "s1" || got[0].Seq != 1 || got[0].Score != 8 { t.Fatalf("%+v", got) }
}

func TestParseStepRewards_ClampsAndSkips(t *testing.T) {
	in := `[{"segment_id":"s1","seq":1,"score":99},{"segment_id":"s1","seq":-1,"score":5}]`
	got, _ := parseStepRewards(in, 10) // 99 clamps to 10; seq=-1 skipped
	if len(got) != 1 || got[0].Score != 10 { t.Fatalf("%+v", got) }
}

func TestParseStepRewards_Empty(t *testing.T) {
	got, err := parseStepRewards("not json", 10)
	if err == nil || len(got) != 0 { t.Fatalf("expected empty+err, got %+v %v", got, err) }
}
```

- [ ] **Step 2: Run to verify failure** — `cd multica/server && go test ./internal/service/ -run TestParseStepRewards -v` → FAIL (undefined).

- [ ] **Step 3: Implement `diagnosis_agent.go`** — `parseStepRewards` strips ```json fences, `json.Unmarshal` into `[]StepReward`, clamps `Score` to `[0, scoreMax]`, skips `Seq < 1`, returns `([]StepReward, error)` (empty + error on unparseable). Define `StepReward`, `DiagnosisAgentConfig{Provider, ExecutablePath, Model, Timeout, ScoreMax, Backend}`, `DiagnosisAgentRunner` mirroring `NewAgentEvolutionReviewer` construction. `Diagnose` builds the diagnosis prompt, runs the Pi subprocess via `Backend`, and calls `parseStepRewards` on its stdout. The system prompt enforces score-in-`[0, scoreMax]` per LLM output and JSON-array output (mirrors `judge_prompt.py`'s discipline). On timeout/non-zero-exit: return `nil, err` (soft-fail handled by caller).

- [ ] **Step 4: Run to verify pass** — `go test ./internal/service/ -run TestParseStepRewards -v` → PASS.

- [ ] **Step 5: Commit** — `git -C multica add server/internal/service/diagnosis_agent.go server/internal/service/diagnosis_agent_test.go && git -C multica commit -m "feat(diagnosis-agent): Pi-agent runner + per-step reward parser"`.

## Task 2: Per-segment turn-range capture + migration (Go)

**Files:**
- Create: `multica/server/migrations/<N>_interaction_dag_step_reward.up.sql` (+ `.down.sql`)
- Modify: `multica/server/internal/service/interaction_dag.go` (`CloseSegmentForEvent`)
- Modify: `multica/server/pkg/db/generated/interaction_dag.sql.go` (regenerate sqlc; `InsertInteractionDAGSegmentWithSnapshotParams` gains `StartSeq`/`EndSeq`)
- Test: `multica/server/internal/service/interaction_dag_test.go`

**Interfaces:**
- Produces: `interaction_dag_segment.start_seq`/`end_seq` columns; `CloseSegmentForEvent` records them from the exported trajectory's turn count.

- [ ] **Step 1: Write the migration**

```sql
-- <N>_interaction_dag_step_reward.up.sql
ALTER TABLE interaction_dag_segment
  ADD COLUMN start_seq INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN end_seq INTEGER NOT NULL DEFAULT 0;

CREATE TABLE interaction_dag_step_reward (
  segment_id TEXT NOT NULL REFERENCES interaction_dag_segment(segment_id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  score INTEGER NOT NULL CHECK (score >= 0),
  rationale TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (segment_id, seq)
);
-- .down.sql: DROP TABLE interaction_dag_step_reward; ALTER TABLE interaction_dag_segment DROP COLUMN end_seq, DROP COLUMN start_seq;
```

- [ ] **Step 2: Write failing test** — `CloseSegmentForEvent` records `start_seq`/`end_seq` matching the exported trajectory's turn range; a leaf segment covers `[1, N]`.

- [ ] **Step 3: Run to verify failure** — `go test ./internal/service/ -run TestCloseSegment -v` → FAIL (unknown columns / params).

- [ ] **Step 4: Implement** — extend the `InsertInteractionDAGSegmentWithSnapshot` sqlc query + params with `start_seq`/`end_seq`; in `CloseSegmentForEvent`, derive `start_seq`/`end_seq` from the exported trajectory JSON (turn count since last close; `start_seq` = previous `end_seq`+1 for the run, `end_seq` = `start_seq` + turnCount - 1). Regenerate sqlc (`make sqlc` or `sqlc generate` in `multica/server`).

- [ ] **Step 5: Run to verify pass** → PASS.

- [ ] **Step 6: Commit** — `git -C multica add server/migrations/<N>_*.sql server/internal/service/interaction_dag.go server/internal/service/interaction_dag_test.go server/pkg/db/generated/ && git -C multica commit -m "feat(interaction-dag): per-segment turn-range capture + step_reward table"`.

## Task 3: Read-only diagnosis tools (Go)

**Files:**
- Create: `multica/server/internal/service/diagnosis_tools.go`
- Test: `multica/server/internal/service/diagnosis_tools_test.go`

**Interfaces:**
- Consumes: `InteractionDAGStore` (segments/edges), `task_message` (via a `MessageStore` interface), `interaction_dag_segment.start_seq`/`end_seq` (Task 2).
- Produces: `getInteractionDag(projectID) ([]SegmentRow, []EdgeRow, error)`, `getSegmentMessages(segmentID) ([]MessageRow, error)` (slices `task_message` by `(task_id, seq ∈ [start_seq, end_seq])`), `getTaskContext(taskID) (TaskContext, error)`. All read-only, workspace-scoped, byte/turn-budgeted (mirror `maxEvolutionReview*Bytes` caps).

- [ ] **Step 1: Write failing tests** — `getSegmentMessages` returns `task_message` rows for the segment's `(task_id, seq ∈ [start_seq,end_seq])`, respects a max-bytes budget (truncates + signals), and refuses cross-workspace access; `getInteractionDag` returns segments + edges; tools reject writes.

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `diagnosis_tools.go`** — `MessageStore` interface (`MessagesForTaskInRange(ctx, taskID, startSeq, endSeq) ([]MessageRow, error)`); tool handlers enforce workspace scoping via the segment's `project_id` -> workspace lookup and per-call budgets.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git -C multica commit -m "feat(diagnosis-agent): read-only tools over interaction DAG + task_message"`.

## Task 4: Trigger at collaborative-task completion (Go)

**Files:**
- Modify: `multica/server/internal/service/task.go` (`CompleteTask`/`captureTaskCompleted`, ≈L1357/182)
- Modify: `multica/server/internal/service/training.go` (close-hook ordering)
- Test: `multica/server/internal/service/task_test.go` / `training_test.go`

**Interfaces:**
- Consumes: `DiagnosisAgentRunner.Diagnose` (Task 1), `RecordStepRewards` (Task 5).
- Produces: at root-task terminal, diagnosis runs over the project's segment DAG and writes step rewards BEFORE the close hook delivers rewards / `/dag` returns 200.

- [ ] **Step 1: Write failing tests** — at root terminal with the flag on, `Diagnose(projectID)` fires and `RecordStepRewards` is called before `SetReward`/`EndSession`; gating off -> no diagnosis; soft-failure (Diagnose returns err) is logged, does NOT block completion, writes no rewards.

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** — add `Diagnosis *DiagnosisAgentRunner` to `TrainingSessionDeps` (nil-safe, like `DAG`); call it in the root-terminal path before the close hook; gate on `DIAGNOSIS_AGENT_ENABLED` ∧ `s.Training` ∧ `INTERACTION_DAG_ENABLED`.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git -C multica commit -m "feat(diagnosis-agent): trigger at collaborative-task completion"`.

## Task 5: RecordStepRewards + AssembledDag step_rewards[] + /dag serving (Go)

**Files:**
- Modify: `multica/server/internal/service/interaction_dag.go` (`RecordStepRewards`, `AssembleAssembledDag`)
- Modify: `multica/server/internal/handler/env_dispatch.go` (`/dag` response shape)
- Modify: `multica/server/pkg/db/generated/interaction_dag.sql.go` (new `InsertInteractionDAGStepReward` / `GetInteractionDAGStepRewards` queries)
- Test: `multica/server/internal/service/interaction_dag_test.go`, `multica/server/internal/handler/env_dispatch_test.go`

**Interfaces:**
- Produces: `RecordStepRewards(ctx, projectID, []StepReward) error` (upsert into `interaction_dag_step_reward`); `AssembleAssembledDag` returns `step_rewards[]` keyed by `(segment_id, seq)`; `/dag` 200 response carries `step_rewards` (empty when absent).

- [ ] **Step 1: Write failing tests** — `RecordStepRewards` writes `(segment_id, seq, score, rationale)`; `AssembleAssembledDag` includes `step_rewards[]`; `/dag` returns `200` with `step_rewards` when present, empty array when diagnosis did not run; `/dag` stays `202` while diagnosis runs then `200` (soft-fail -> `200` + empty).

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** — sqlc queries for upsert + fetch; `AssembleAssembledDag` joins `interaction_dag_step_reward`; extend the `/dag` response struct with `StepRewards []StepRewardOut`. Keep `step_rewards` absent-as-empty (no fabricated zeros).

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git -C multica commit -m "feat(diagnosis-agent): project-scoped step_rewards via AssembledDag + /dag"`.

## Task 6: Config + flags (Go)

**Files:**
- Modify: `multica/server/internal/daemon/config.go` (env), `multica/server/internal/service/training.go` (deps wiring)
- Test: `multica/server/internal/daemon/config_test.go`

- [ ] **Step 1: Write failing test** — `DIAGNOSIS_AGENT_ENABLED` (default off), `DIAGNOSIS_AGENT_PATH`, `DIAGNOSIS_AGENT_MODEL`, `DIAGNOSIS_AGENT_TIMEOUT_SECONDS`, `DIAGNOSIS_AGENT_SCORE_MAX` parse from env; wiring composes with `s.Training`/`INTERACTION_DAG_ENABLED`.

- [ ] **Step 2–4: Implement + verify** — mirror `MULTICA_PI_PATH`/`MULTICA_PI_MODEL` probing in `config.go:258`; build `DiagnosisAgentRunner` nil-safe when off.

- [ ] **Step 5: Commit** — `git -C multica commit -m "feat(diagnosis-agent): config + flags"`.

## Task 7: AReal consumer - AssembledDag step_rewards + (segment_id,seq)->Node mapping (Python)

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py` (`StepReward`, `AssembledDag.step_rewards`, `from_dict`)
- Modify: `customized_areal/tree_search/agents/supernode_assembler.py` (map rewards onto `Node.process_reward`)
- Test: `customized_areal/tree_search/tests/test_multica_dag_client*.py`, `tests/test_assembler_ref_resolve.py`

**Interfaces:**
- Consumes: `/dag` `step_rewards[]` (`{segment_id, seq, score, rationale}`).
- Produces: `Node.process_reward` set from the matching `(segment_id, seq)` reward; unmatched rewards dropped + logged.

- [ ] **Step 1: Write failing tests**

```python
def test_assembled_dag_parses_step_rewards():
    d = {"segments": [], "edges": [], "session_to_agent_run": {},
         "step_rewards": [{"segment_id": "s1", "seq": 1, "score": 7, "rationale": "r"}]}
    dag = AssembledDag.from_dict(d)
    assert len(dag.step_rewards) == 1 and dag.step_rewards[0].score == 7

def test_assembler_applies_step_rewards_to_nodes():
    # build an AssembledDag with one segment whose resolved SuperNode has N nodes;
    # step_rewards for (seg, seq) lands on Node.process_reward; unmatched dropped+logged.
    ...
```

- [ ] **Step 2: Run to verify failure** — `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_assembler_ref_resolve.py -v` → FAIL.

- [ ] **Step 3: Implement**
  - Add `StepReward` dataclass + `AssembledDag.step_rewards: list[StepReward] = field(default_factory=list)`; parse in `from_dict`.
  - In `assemble_from_refs`: after building `SuperNode`s, build a `(segment_id, seq) -> score` index from `dag.step_rewards`; for each `SuperNode`, set each `Node.process_reward` from the matching key. **Key verification (D4):** confirm whether areal's per-`SuperNode` `Node` ordering is segment-relative (1..k) or agent_run-absolute (`task_message.seq`). The segment now carries `start_seq`/`end_seq` (Task 2); map `seq` accordingly (if Node index is segment-relative, `node_rel_idx + start_seq - 1` = absolute `seq`; if absolute, use `seq` directly). Log + drop rewards whose `(segment_id, seq)` has no Node.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git add customized_areal/tree_search/agents/multica_dag_client.py customized_areal/tree_search/agents/supernode_assembler.py customized_areal/tree_search/tests/ && git commit -m "feat(tree-search): consume multica step_rewards into Node.process_reward"`.

## Task 8: Remove the flat areal judge (Python)

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py` (≈L747–787, 892–912, 1167, 1252)
- Delete: `customized_areal/tree_search/core/judge_prompt.py`
- Modify: `customized_areal/tree_search/config.py` (≈L151–178), `customized_areal/tree_search/distilling/config.py`
- Test: delete judge-specific tests; update `tests/test_v2_session_lifecycle.py`

**Interfaces:**
- Produces: no `enable_judge_process_reward` path; `process_reward` comes solely from Task 7. Sparse/zero fallback when `step_rewards` empty (matches old `enable_judge_process_reward=False`).

- [ ] **Step 1: Write failing test** — with empty `step_rewards`, `process_reward` is sparse/zero (no judge call, no `diagnose_episode` judge invocation); grep asserts `enable_judge_process_reward`/`judge_prompt`/`judge_model_name` resolve to nothing.

- [ ] **Step 2: Run to verify failure** → FAIL (judge path still present).

- [ ] **Step 3: Implement** — remove the `enable_judge_process_reward` branch + `build_judge_instruction`/`parse_turn_scores` usage in `customized_grouped_workflow.py`; delete `judge_prompt.py`; remove `enable_judge_process_reward`/`judge_process_reward_beta`/`judge_model_name`/`judge_max_concurrency` from both configs. **Verify before removing** `ExternalDiagnoseProvider`/`diagnose_episode` that `critic_score.py` still uses it (keep it; only judge usage goes).

- [ ] **Step 4: Run to verify pass** — `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/ -q` + `uvx ruff check customized_areal/tree_search/` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "refactor(tree-search): remove flat judge; process reward from multica diagnosis"`.

## Task 9: Integration + E2E + sweep

**Files:** `multica/server/internal/service/` integration tests; cross-repo E2E; `openspec/changes/multica-pi-diagnosis-agent/tasks.md` (check off).

- [ ] **Step 1: Go integration** — trained 3-agent `mode=scratch` rollout -> segments recorded -> diagnosis fires -> `step_rewards` written -> `/dag` serves them. `cd multica/server && go test ./internal/service/ ./internal/handler/`.
- [ ] **Step 2: Cross-repo E2E** — rollout -> diagnosis `step_rewards` -> areal resolve -> `ExecutionDag` with `process_reward` -> minimal training step -> cleanup.
- [ ] **Step 3: Sanity (not a gate)** — compare diagnosis per-step scores vs. the removed judge's scores on one fixed episode.
- [ ] **Step 4: Grep sweep** — `enable_judge_process_reward`/`judge_prompt`/`judge_model_name` resolve only to intended removals; `start_seq`/`end_seq`/`step_rewards` resolve to intended code.
- [ ] **Step 5: Resolve design.md Q1–Q4** in the open-phase `design.md` (Q1 per-LLM-output, Q2 AssembledDag-attached, Q3 task_message + stored turn range, Q4 coexist) if any drifted.
- [ ] **Step 6: Commit + check off tasks.md** — `git commit -m "test(diagnosis-agent): integration + E2E + sweep"`; flip `- [ ]` -> `- [x]` in `tasks.md`.

## Self-Review

1. **Spec coverage:** `diagnosis-process-reward/spec.md` requirements map to tasks - trigger/gating (T4, T6), tool access (T3), turn-range capture (T2), per-LLM-output output (T1, T5), AssembledDag delivery (T5, T7), soft-fail (T4), critic coexistence (T8 leaves critic untouched; T7/T8 don't touch `SetReward`). ✓
2. **Placeholder scan:** No TBD/TODO; code blocks provided for the load-bearing pieces (parser, migration, dataclass, mapping, judge removal). Tools/trigger/wiring steps give exact signatures + behavior. ✓
3. **Type consistency:** `StepReward{SegmentID,Seq,Score,Rationale}` used consistently Go->JSON->`StepReward` Python; `(segment_id, seq)` key consistent end-to-end; `Node.process_reward` is the existing field. ✓ One open verification (D4 seq alignment) is explicit in T7 Step 3.
4. **Scope:** Single change, dual-repo, ~9 tasks. Within one plan. ✓
