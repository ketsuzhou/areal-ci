# Multi-Agent DAG RL Training in Multica — Design

**Date:** 2026-06-26 **Status:** Approved design — ready for implementation planning
**Supersedes:** `multica/specs.md` (the prior design draft; left in place for history,
but the decisions below are authoritative when they conflict)

## 1. Overview

Multica runs many agents that collaborate: a planner files sub-issues, workers pick them
up, agents mention/trigger one another, and children report back to parents. The
execution graph is therefore a **DAG of agent runs** (not a tree), with multiple agents
potentially in flight at once.

This design adds the infrastructure to train RL over that DAG:

1. **Branch the DAG** from a chosen point by snapshotting the agent's cloud sandbox,
   forking its Multica entity subtree, and replaying its transcript prefix into a fresh
   inference session.
1. **Judge task success** with a verifier agent (Multica's native `done` is
   self-reported and unreliable), producing a terminal outcome reward.
1. **Assign credit** per agent and per step, and back reward up through the DAG.

**Scope: cloud-only v1.** Daemon/local git-worktree forking is out of scope.

## 2. Locked decisions

Carried over from `multica/specs.md` §2, restated for completeness:

1. **Reward backs up over the Multica entity/execution DAG.** Nodes = agent runs on
   issues; edges = delegation (`issue.parent_issue_id`), mention-trigger, and
   completion/notification.
1. **Environment = forkable cloud sandbox behind a `ForkableEnvironment` abstraction.**
   v1 ships one new sandbox provider. Daemon/local worktree fork is dropped.
1. **Lazy snapshot.** The live sandbox is snapshotted only when a node is actually
   selected as a branch candidate.
1. **Branch point = `(task_id, seq)`** using `task_message.seq` ordering. The branch
   task replays the transcript prefix (messages up to `seq`) and drops `PriorSessionID`
   so the AReaL-served inference session starts fresh (via `db_bridge`).
1. **Multica entity subtree is forked separately.** Postgres state lives outside the
   sandbox, so it cannot ride along in a sandbox snapshot.
1. **Reward is hybrid** — a terminal verifier outcome plus per-node process signals.
   Structural backup distributes the terminal reward; process signals shape intermediate
   steps.
1. **Verifier = objective checks where available + LLM-judge fallback** over
   `issue.acceptance_criteria`.
1. **Fan-in joins: the verifier assigns per-agent credit explicitly** — no fixed
   sum/mean/max aggregation rule.

## 3. Resolutions to open risks (supersedes `multica/specs.md` §8)

The prior spec deferred four implementation-time risks. These resolutions are now
locked:

### 3.1 Lazy snapshot fidelity → snapshot-at-frontier + transcript replay

When a node is selected as a branch candidate, the live sandbox is snapshotted at
whatever frontier it has currently reached, and the transcript prefix (messages up to
`seq`) is replayed into a fresh inference session.

- **No run pausing** is introduced in v1. The AReaL runtime contract is unchanged — runs
  are not paused at decision points.
- **Known limitation:** the snapshot may be slightly ahead of `seq` if the run advanced
  past the branch point between the selection decision and the snapshot call. The
  transcript-prefix replay is the source of truth for inference state, not the sandbox
  filesystem — so this skew affects only filesystem side effects, not the model's
  visible context.
- **Why this is acceptable for v1:** the inference session is what the model sees;
  filesystem side effects are best-effort. Adding pause/resume plumbing to the AReaL
  runtime is a much larger change and is not justified for v1.

### 3.2 Point-in-time entity fidelity → reconstruct from activity log

`ForkIssueSubtree` must reconstruct overwritten `issue` fields (status, title, etc.) to
their values as of `seq` using Multica's activity log.

- **Coverage check required (Task 4):** before implementing replay logic, verify that
  the activity log captures every overwritten `issue` field. If a field is not logged,
  document it as a known gap — do not fabricate a value.
- **Append-only data is cut cleanly at `seq`:** comments, sub-issues, and `task_message`
  rows are append-only, so they are reconstructed by filtering. For `task_message`,
  filter on `seq <= branch_seq`. For comments and sub-issues (which have no `seq`),
  filter on `created_at <=` the timestamp of the `task_message` row at `branch_seq` —
  i.e. the message whose `seq == branch_seq` defines the branch point in time.
- **Per-field tests required:** the Task 4 test suite must include fixtures that
  exercise each overwritten field type through the fork, with at least one field that
  has been overwritten multiple times before `seq`.

### 3.3 Cost → lazy snapshot + concurrency semaphore

Cloud sandbox forks are metered. Lazy snapshot already bounds cost to selected branch
nodes only. Additionally:

- A **concurrency semaphore** gates fork calls on the Python side, configurable via env
  var, default sized to `group_size`. This prevents fork-storms at high `group_size`.
- **End-to-end validation runs at `group_size=2`** first; a separate scale check (higher
  `group_size`) runs only after the e2e passes.

### 3.4 Provider portability → generic Fleet endpoints dispatching to vendor

The Fleet proxy (Multica's cloud runtime) exposes **generic** snapshot + fork endpoints:

- `POST /sandboxes/{id}/snapshot` — create a snapshot of a live sandbox.
- `POST /sandboxes/{id}/fork` — create a new sandbox from a snapshot (or live sandbox,
  if the vendor supports direct fork).

The server-side handler dispatches to the underlying sandbox vendor (Daytona today, via
`_experimental_fork` / `create_snapshot`). Future vendors (Fly, raw Kubernetes) plug in
behind the same endpoints without changes on the Python side.

- **The `ForkableEnvironment` abstraction on the Python side is the only seam the
  training code sees.** The training code never imports a vendor SDK; it calls
  `ForkableEnvironment.snapshot` / `fork` / `restore` / `cleanup`.
- **Vendor SDK types stay behind the provider package** (per the backend
  `.claude/rules/backend.md` "Boundary Normalization Rule" and the `code-quality.md`
  "Sandbox-layer types stay behind the wrapper" rule).

## 4. Architecture

```
AReaL side (Python):
  ├── Agent-execution DAG store — nodes (agent runs) + edges (delegation/mention/completion)
  │   └── customized_areal/tree_search/dag/execution_dag.py  (already implemented)
  ├── Branch-candidate selection → triggers lazy sandbox snapshot
  │   └── customized_areal/tree_search/core/customized_grouped_workflow.py
  ├── Verifier agent — objective checks + LLM-judge fallback over acceptance_criteria
  │   └── customized_areal/tree_search/dag/verifier.py  (new)
  ├── DAG-aware hybrid reward backup + advantage computer
  │   └── customized_areal/tree_search/dag/backup.py, credit.py  (new)
  └── ForkableEnvironment abstraction + Fleet provider
      └── customized_areal/tree_search/dag/environment.py  (new)

Multica side (Go):
  ├── Migration: forked_from_issue_id, forked_at_seq, forked_at_task_id on issue
  │   └── server/migrations/NNN_issue_fork_provenance.up.sql  (new)
  ├── ForkIssueSubtree service with activity-log reconstruction
  │   └── internal/service/issue_fork.go  (new)
  ├── Fleet fork endpoints (snapshot + fork, vendor-dispatching)
  │   └── internal/handler/sandbox_fork.go, internal/cloudruntime/fork.go  (new)
  └── POST /api/issues/{id}/fork + DELETE cleanup handler
      └── internal/handler/issue_fork.go  (new)

db_bridge:
  └── Proxies agent LLM calls (model `areal/...`) + RL session lifecycle.
      Replaces constant set_reward(1.0) with verifier-driven reward at session end.
      Channels: rl_start_session, rl_set_reward, rl_end_session, agent_start_branch.
```

### Branching flow

```
selection(node @ task_id, seq)
   ├─ ForkableEnvironment.snapshot(live sandbox) ─► forked sandbox id
   ├─ ForkIssueSubtree(source_issue, task_id, seq) ─► forked issue id
   │      └─ reconstructs overwritten issue fields from activity log
   └─ agent_start_branch(forked sandbox + forked issue, replay msgs<=seq, no PriorSessionID)
```

## 5. Phase and task breakdown

Five phases, 15 tasks. Each task lists primary files and tests; full step-by-step detail
goes in the per-phase implementation plans at
`docs/superpowers/plans/2026-06-26-multica-dag-rl-phase{0-4}-*.md`.

### Phase 0 — Abstraction + DAG contract

- **Task 1 (done):** Agent-execution DAG model.
  `customized_areal/tree_search/dag/execution_dag.py` — `AgentRunNode`, `Edge`,
  `EdgeType`, `ExecutionDAG` with topological order, cycle detection, fork/join node
  identification. Torch-free.
- **Task 2:** `ForkableEnvironment` abstraction + Fleet sandbox provider.
  `customized_areal/tree_search/dag/environment.py` — Protocol with
  `snapshot`/`fork`/`restore`/`cleanup`; Fleet provider calls generic Fleet endpoints.
  Contract tests with a fake provider.

### Phase 1 — Multica entity fork (Go; follow `multica/CLAUDE.md`)

- **Task 3:** Migration adding `forked_from_issue_id`, `forked_at_seq`,
  `forked_at_task_id` to `issue`; sqlc queries; partial index on `forked_from_issue_id`
  for fork-tree lookups.
- **Task 4:** `ForkIssueSubtree(ctx, sourceIssueID, taskID, seq)` service in
  `internal/service/issue_fork.go`. Reconstructs overwritten `issue` fields from the
  activity log (coverage check + per-field tests required — see §3.2). Copies
  append-only data (comments, sub-issues, task_messages) cut at `seq`.
- **Task 5:** `POST /api/issues/{id}/fork` + `DELETE` cleanup handler in
  `internal/handler/issue_fork.go`. Follows the `parseUUIDOrBadRequest` / loader UUID
  convention from `multica/CLAUDE.md`. Also: Fleet fork endpoints
  (`POST /sandboxes/{id}/snapshot`, `POST /sandboxes/{id}/fork`) in
  `internal/handler/sandbox_fork.go`, dispatching to the underlying vendor (Daytona
  today).

### Phase 2 — Verifier agent (Python)

- **Task 6:** Verifier interface + objective-check runner.
  `customized_areal/tree_search/dag/verifier.py` — `Verifier` Protocol with
  `verify(run) -> VerifierResult`; objective runner executes deterministic checks (test
  pass/fail, build status, lint) when available.
- **Task 7:** LLM-judge fallback over `issue.acceptance_criteria`. Extends the verifier
  with an LLM-judge path used when objective checks are unavailable or inconclusive.
  Templates at `customized_areal/tpfc/eval_utils.py` (`verify_answer_llm_*`) and
  `customized_areal/tpfc/gaia_final_reward.py` (`compute_reward`) are the reference
  pattern.
- **Task 8:** Per-agent + per-step credit assignment.
  `customized_areal/tree_search/dag/credit.py` — assigns credit at fan-in joins
  explicitly (decision 8). No fixed sum/mean/max aggregation rule.

### Phase 3 — DAG reward backup + advantage (Python)

- **Task 9:** Extend `Node` in `customized_areal/tree_search/core/tree_store.py` with
  `process_reward: float`, DAG edge references (`parent_run_ids`, `child_run_ids`), and
  `branch_issue_id` / `branch_env_snapshot_id` provenance fields. Update `checkpoint.py`
  serialization. Fix `customized_grouped_workflow.py::select_branch_candidate` so
  cloud-env nodes are selectable (today it filters on `bool(node.branch_sandbox_id)`,
  which is wrong for cloud-env nodes that use `branch_env_snapshot_id`).
- **Task 10:** DAG-aware hybrid backup + advantage computer.
  `customized_areal/tree_search/dag/backup.py` — structural backup distributes the
  terminal reward along DAG edges; process signals shape intermediate steps. Replaces
  the linear `_backup` in `tree_store.py` for DAG runs. The existing
  `TreeAdvantageComputer` (`advantage.py`) GRPO-normalizes one reward per episode and
  broadcasts it to all turns — extended (not replaced) to consume the per-node credit
  from Task 8.

### Phase 4 — Integration + lazy branching + e2e

- **Task 11:** Bridge/session wiring — agent `custom_env` routes LLM via `db_bridge`
  (`proxy_base_url` / `proxy_api_key`); agent run → RL session (`rl_start_session` /
  `rl_set_reward` / `rl_end_session`).
- **Task 12:** Replace constant `set_reward(1.0)` with verifier-driven reward; populate
  DAG credit at session end via `rl_set_reward`.
- **Task 13:** Lazy branch on candidate selection. Triggers
  `ForkableEnvironment.snapshot` + `ForkIssueSubtree` + `agent_start_branch` when
  `select_branch_candidate` returns a node. Concurrency semaphore gates fork calls
  (§3.3).
- **Task 14:** Branch cleanup (sandbox + forked issue). Extends
  `customized_grouped_workflow.py::_cleanup_branch` to also delete the forked Multica
  issue (via the Phase 1 `DELETE` handler) and the forked sandbox.
- **Task 15:** End-to-end validation, cloud-only, `group_size=2`, then a separate scale
  check at higher `group_size`.

## 6. Testing strategy

- **Torch-free DAG model** — `customized_areal/tree_search/dag/test_execution_dag.py`
  covers topological order, cycle detection, fork/join node identification, edge
  idempotency, ancestor/descendant traversal. (Task 1 — already in place.)
- **`ForkableEnvironment` contract tests** —
  `customized_areal/tree_search/dag/test_environment.py` uses a fake provider to
  exercise snapshot/fork/restore/cleanup ordering and error paths without touching a
  real cloud vendor.
- **`ForkIssueSubtree` Go tests** — `server/internal/service/issue_fork_test.go` with
  activity-log fixtures covering each overwritten `issue` field type, including
  multi-overwrite cases. Coverage check (§3.2) is a separate test that fails the build
  if any overwritten field is not logged.
- **Verifier tests** — `customized_areal/tree_search/dag/test_verifier.py` covers
  objective-check success/failure, LLM-judge fallback (mocked), and credit assignment at
  fan-in joins.
- **DAG backup tests** — `customized_areal/tree_search/dag/test_backup.py` covers
  structural backup over a multi-node DAG, process-signal shaping, and the
  `select_branch_candidate` fix for cloud-env nodes.
- **End-to-end** — Phase 4 Task 15 runs a full branch lifecycle at `group_size=2`
  against a real Multica + cloud sandbox stack. Integration tests that require
  multi-node hardware are skipped with an explanation when unavailable (per
  `backend/areal/CLAUDE.md`).

## 7. Out of scope (v1)

- **Daemon/local git-worktree forking** — dropped; cloud-only.
- **True global consistent-cut** across concurrently running agents — v1 branches a
  single agent's lane (its sandbox + reachable issue subtree).
- **Multica UI changes** beyond hiding forked issues from default lists (optional).
- **Run pausing at decision points** — see §3.1. The AReaL runtime contract is unchanged
  in v1.
- **New vendors beyond Daytona** — the abstraction supports them, but no non-Daytona
  vendor is implemented in v1.

## 8. References

### AReaL

- Tree search module: `customized_areal/tree_search/`
- DAG model: `customized_areal/tree_search/dag/execution_dag.py` (Task 1, done)
- Node dataclass: `customized_areal/tree_search/core/tree_store.py`
- Sandbox fork pattern: `customized_areal/db_service/sandbox.py` (`clone_sandbox`,
  `bind_sandbox_to_task`)
- Workflow: `customized_areal/tree_search/core/customized_grouped_workflow.py`
  (`select_branch_candidate`, `build_branch_task`, `_cleanup_branch`)
- Advantage: `customized_areal/tree_search/core/advantage.py`
- Checkpoint: `customized_areal/tree_search/core/checkpoint.py`
- Verifier templates: `customized_areal/tpfc/eval_utils.py`,
  `customized_areal/tpfc/gaia_final_reward.py`

### Multica

- Schema: `multica/server/migrations/001_init.up.sql`, `026_task_messages.up.sql`
- Issue handler / status: `multica/server/internal/handler/issue.go`,
  `issue_child_done.go`
- Cloud runtime proxy: `multica/server/internal/cloudruntime/client.go`
- db_bridge: `multica/db_bridge/README.md`
- Conventions: `multica/CLAUDE.md`

### Project rules

- Backend boundary: `.claude/rules/backend.md` ("Boundary Normalization Rule")
- Sandbox-layer types: `.claude/rules/code-quality.md` ("Sandbox-layer types stay behind
  the wrapper")
- Multi-tenancy: `.claude/rules/multi-tenancy.md`
