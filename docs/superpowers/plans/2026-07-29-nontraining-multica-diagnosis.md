# Non-training MultiCA Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a completed non-training MultiCA dispatch run the persistent diagnosis agent, require an exact score for every assistant turn, return the enriched DAG, and attach globally normalized diagnosis rewards to AReaL training Nodes rather than SuperNodes.

**Architecture:** MultiCA gets an authenticated post-terminal diagnosis endpoint backed by the existing persistent Pi/tool-server flow. Before the agent starts, the server snapshots each segment’s actual assistant-message sequence numbers; every write and completion check uses that immutable set. The DAG response exposes the same turn mapping so AReaL can materialize one training Node per scored turn, retain per-trajectory observations on shared nodes, average them, and normalize the node means over the whole DAG.

**Tech Stack:** Go 1.22, chi, sqlc/PostgreSQL migrations, Python 3.12, httpx, pytest, AReaL tensor/data-proxy contracts.

## Global Constraints

- Do not use Comet.
- Do not modify unrelated dirty files: `multica_client.py`, `test_env_dispatch_client.py`, `dag_merge.py`, and its tests may contain user work; integrate without reverting it.
- Keep diagnosis opt-in: ordinary non-training dispatch behavior remains unchanged without `--diagnose`.
- The diagnosis writer remains the only writer of `(segment_id, seq)` records.
- Each diagnosis run freezes exact assistant `(segment_id, seq)` targets; counts and numeric ranges are insufficient.
- `SuperNode` must not own diagnosis rewards. `Node.outcome_reward` remains the verifier terminal reward.
- A globally normalized `Node.process_reward` distribution must sum to `1.0` across scored nodes, including a deterministic zero-total fallback.
- Do not add external dependencies or alter AReaL CLI config structures.

---

### Task 1: Freeze exact diagnosis targets and enforce them in MultiCA

**Files:**

- Create: `multica/server/migrations/209_interaction_dag_diagnosis_target_seqs.up.sql`
- Create: `multica/server/migrations/209_interaction_dag_diagnosis_target_seqs.down.sql`
- Modify: `multica/server/pkg/db/queries/interaction_dag.sql`
- Modify: `multica/server/internal/service/diagnosis_state.go`
- Modify: `multica/server/internal/service/diagnosis_tools.go`
- Modify: `multica/server/internal/service/diagnosis_tool_server.go`
- Modify: `multica/server/internal/service/diagnosis_agent.go`
- Test: `multica/server/internal/service/diagnosis_state_test.go`
- Test: `multica/server/internal/service/diagnosis_tool_server_test.go`
- Test: `multica/server/internal/service/diagnosis_agent_test.go`

**Interfaces:**

- Consumes: `InteractionDAGSegment{AgentRunID, StartSeq, EndSeq}`, `DiagnosisMessagePager`, and persisted `interaction_dag_step_reward` rows.
- Produces: `DiagnosisSegmentTarget{SegmentID string; AgentRunID string; AssistantSeqs []int32}` and `DiagnosisStateStore.StartSegment(ctx, runID, segmentID, assistantSeqs)`.
- Produces: `DiagnosisToolServer` validation that accepts only a frozen assistant sequence and reports every missing frozen sequence.

- [ ] **Step 1: Write failing exact-target tests**

Add a fixture with a segment whose task-message sequence numbers are non-contiguous and whose non-assistant messages are interleaved:

```go
targets := []int32{2, 7}
_, err := store.StartSegment(ctx, "run-1", "segment-1", targets)
require.NoError(t, err)

resp := doRequest(t, server, "POST", "/v1/record-step-rewards", recordStepRewardsRequest{
    SegmentID: "segment-1",
    Rewards: []stepRewardEntry{{Seq: 1, Score: 4}, {Seq: 2, Score: 5}, {Seq: 7, Score: 6}},
}, http.StatusOK)
require.Equal(t, []int{2, 7}, decodeRecordResponse(t, resp).PersistedSeqs)
require.Equal(t, []rejectedReward{{Seq: 1, Reason: "seq is not an assistant target"}}, decodeRecordResponse(t, resp).Rejected)
```

Add a completion test that proves `seq=2` and `seq=7` are required even though the target count is two; it must fail while either exact target is absent.

- [ ] **Step 2: Run the focused Go tests and verify RED**

Run:

```bash
cd multica/server && go test ./internal/service -run 'TestDiagnosis.*ExactAssistantSeq|TestDiagnosis.*NonContiguous' -count=1
```

Expected: FAIL because target sequences are not persisted or the current range-based validation accepts/rejects the wrong sequence.

- [ ] **Step 3: Add the target-set persistence and sqlc query surface**

Create a child table keyed by a diagnosis segment and target sequence:

```sql
CREATE TABLE interaction_dag_diagnosis_target_seq (
    run_id text NOT NULL,
    segment_id text NOT NULL,
    seq integer NOT NULL CHECK (seq > 0),
    PRIMARY KEY (run_id, segment_id, seq),
    FOREIGN KEY (run_id, segment_id)
      REFERENCES interaction_dag_diagnosis_segment (run_id, segment_id)
      ON DELETE CASCADE
);
```

Add sqlc queries to insert immutable targets, list targets, and count rewards whose sequence belongs to the target table. Regenerate `pkg/db/generated` with the repository’s sqlc generation command instead of editing generated Go by hand.

- [ ] **Step 4: Make state transitions and tool-server checks target-set based**

Replace count-only `StartSegment` input with frozen targets and expose them on `SegmentDiagnosisCheckpoint`:

```go
func (s *DiagnosisStateStore) StartSegment(
    ctx context.Context, runID, segmentID string, assistantSeqs []int32,
) (SegmentDiagnosisCheckpoint, error)
```

`StartSegment` must reject empty or duplicate sequences, persist them once, and be idempotent only when the existing ordered set is identical. `handleRecordStepRewards` must reject a `seq` not in the set, and `handleFinishSegment` / `CompleteSegment` must derive missing rewards by set difference, not by `1..ExpectedRewardCount`.

Change `handleGetSegmentMessages` to look up the frozen segment metadata and call the pager with that segment’s real `AgentRunID`, `StartSeq`, and `EndSeq`; never use the root task ID or count-derived bounds.

- [ ] **Step 5: Freeze targets before starting Pi and verify all completion outcomes**

In `DiagnoseOnDemand`, load the project’s assembled segments and messages before Pi execution. For each segment, select assistant message sequences from the real `[StartSeq, EndSeq]` range, call `StartSegment`, and only then build the tool server/bootstrap prompt. A segment with no assistant output completes without asking Pi to invent a reward.

After Pi returns, call `CompleteRun` only when every target set is covered; otherwise fail the run and return an error. Do not ignore `CompleteRun` errors.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
cd multica/server && go test ./internal/service -run 'TestDiagnosis.*(ExactAssistantSeq|NonContiguous|Coverage|OnDemand)' -count=1
```

Expected: PASS, including non-contiguous assistant `seq` coverage, idempotent replays, and rejection of non-target rewrites.

- [ ] **Step 7: Commit Task 1**

```bash
git add multica/server/migrations/209_interaction_dag_diagnosis_target_seqs.*.sql \
  multica/server/pkg/db/queries/interaction_dag.sql \
  multica/server/pkg/db/generated/interaction_dag_diagnosis.sql.go \
  multica/server/internal/service/diagnosis_{state,tools,tool_server,agent}.go \
  multica/server/internal/service/diagnosis_{state,tool_server,agent}_test.go
git commit -m "fix(multica): require exact diagnosis turn coverage"
```

### Task 2: Expose and run diagnosis for a terminal non-training dispatch

**Files:**

- Modify: `multica/server/cmd/server/router.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_routes.go`
- Create: `multica/server/internal/handler/env_dispatch_diagnosis.go`
- Test: `multica/server/internal/handler/env_dispatch_diagnosis_test.go`
- Modify: `multica/server/internal/service/training_config.go`
- Modify: `multica/server/internal/service/diagnosis_agent.go`
- Test: `multica/server/internal/service/diagnosis_agent_test.go`

**Interfaces:**

- Consumes: authenticated request context, `env_dispatch_run` readiness, `TrainingConfig`, `db.Queries`.
- Produces: `POST /api/v1/env-dispatch/{projectID}/diagnosis` and `POST /api/v1/env-dispatch/channels/{channelID}/diagnosis`.
- Produces: JSON `DiagnosisReport{run_id, completed_segments, total_segments, status}`; repeat calls resume or return the existing completed run.

- [ ] **Step 1: Write endpoint tests before implementation**

Use the existing env-dispatch test fixture to assert these four HTTP contracts:

```go
// terminal + diagnosis disabled -> 409 {"error":"diagnosis_unavailable"}
// in-progress root -> 409 {"error":"dispatch_not_terminal"}
// cross-workspace project -> 403 {"error":"forbidden"}
// terminal project + fake completed runner -> 200 DiagnosisReport{Status: DiagnosisRunCompleted}
```

The successful test must assert that the runner receives the project root task ID and the topologically ordered segment IDs, not merely an unordered database result.

- [ ] **Step 2: Run endpoint tests and verify RED**

Run:

```bash
cd multica/server && go test ./internal/handler -run TestEnvDispatchDiagnosis -count=1
```

Expected: FAIL because no diagnosis route/handler exists.

- [ ] **Step 3: Add one scoped diagnosis service entry point**

Add an explicit service boundary that validates ownership, terminal readiness, enabled configuration, dense DAG coverage, and target snapshot construction before invoking `DiagnoseOnDemand`:

```go
func DiagnoseEnvDispatchProject(
    ctx context.Context,
    projectID, workspaceID string,
    runner *DiagnosisAgentRunner,
    state *DiagnosisStateStore,
    cfg DiagnosisOnDemandConfig,
) (DiagnosisReport, error)
```

Define typed sentinel errors for unavailable diagnosis and a non-terminal dispatch so handlers map them without leaking internal configuration or Pi errors. Construct the runner once from `TrainingConfig` + `db.Queries`; do not couple this endpoint to `AREAL_BRIDGE_STUB_URL` or the training-session hook.

- [ ] **Step 4: Register project and channel routes**

Register:

```go
r.Post("/api/v1/env-dispatch/{projectID}/diagnosis", h.DiagnoseEnvDispatchProject)
r.Post("/api/v1/env-dispatch/channels/{channelID}/diagnosis", h.DiagnoseEnvDispatchChannel)
```

The channel facade must resolve its project with `resolveChannelProject` and delegate to the same project helper. Use the existing `GetDag` workspace / 404 / 403 distinction and `EnvDispatchService.GetDagReadiness` terminal check; never trust project IDs from request bodies.

- [ ] **Step 5: Return a stable diagnosis status and preserve idempotency**

Return `200` for completed or resumed runs and `409` for unavailable/non-terminal states. A repeated POST after a completed run must return the persisted report without launching Pi or changing rewards. Map unexpected service/database failures to `503` with a concise public error code.

- [ ] **Step 6: Run handler and service tests and verify GREEN**

Run:

```bash
cd multica/server && go test ./internal/handler ./internal/service -run 'TestEnvDispatchDiagnosis|TestDiagnoseOnDemand' -count=1
```

Expected: PASS; the existing training-root automatic diagnosis tests also remain unchanged.

- [ ] **Step 7: Commit Task 2**

```bash
git add multica/server/cmd/server/router.go \
  multica/server/internal/handler/env_dispatch{,_channel_routes,_diagnosis}.go \
  multica/server/internal/handler/env_dispatch_diagnosis_test.go \
  multica/server/internal/service/{training_config,diagnosis_agent}.go \
  multica/server/internal/service/diagnosis_agent_test.go
git commit -m "feat(multica): diagnose terminal env dispatches"
```

### Task 3: Add non-training client opt-in, polling, and enriched-DAG validation

**Files:**

- Modify: `customized_areal/tree_search/agents/multica_client.py`
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py`
- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`
- Test: `customized_areal/tree_search/tests/test_multica_dag_client.py`

**Interfaces:**

- Consumes: `EnvDispatchHandle`, the terminal ordinary DAG, and the diagnosis endpoint report.
- Produces: `MulticaEnvDispatchClient.diagnose_env_dispatch(handle) -> dict`, `wait_for_diagnosis(handle, timeout, interval) -> dict`, and CLI `--diagnose`.
- Produces: `AssembledDag.validate_diagnosis_coverage()` using exact `assistant_turn_seqs` returned for every segment.

- [ ] **Step 1: Write failing Python tests**

Add an httpx mock transport asserting the public request sequence:

```python
assert requests == [
    ("GET", "/api/v1/env-dispatch/channels/c1/dag"),
    ("POST", "/api/v1/env-dispatch/channels/c1/diagnosis"),
    ("GET", "/api/v1/env-dispatch/channels/c1/diagnosis"),
    ("GET", "/api/v1/env-dispatch/channels/c1/dag"),
]
```

Assert that `--diagnose` rejects a completed report whose final DAG omits even one known assistant target and that a run without `--diagnose` makes no diagnosis HTTP request.

- [ ] **Step 2: Run the focused Python tests and verify RED**

Run:

```bash
uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py \
  customized_areal/tree_search/tests/test_multica_dag_client.py -q
```

Expected: FAIL because the client has no diagnosis request/poll path and `AssembledDag` has no exact turn-target contract.

- [ ] **Step 3: Extend the DAG wire model without guessing turn identity**

Add `assistant_turn_seqs: list[int]` to `SegmentSpec`, emitted by MultiCA from the frozen target set. Validate that targets are positive, unique, sorted, and that every `StepReward` belongs to a known segment and known target. Keep raw scores and rationales intact; do not normalize during parsing.

- [ ] **Step 4: Implement the opt-in client path**

Add `diagnose_env_dispatch(handle)`, `get_diagnosis_status(handle)`, and
`wait_for_diagnosis(handle, timeout, interval)` using `_request`, `_headers`,
`_params`, and `_dispatch_lifecycle_prefix` so user-PAT workspace selection and
response-body redaction remain consistent.

`_debug_run` must first finish ordinary DAG polling, invoke diagnosis only when `args.diagnose`, wait for `status == "completed"`, fetch the final DAG again, call `validate_diagnosis_coverage`, and write that final payload through the existing `--dag-out` path. Error text must name the operation but redact secrets.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py \
  customized_areal/tree_search/tests/test_multica_dag_client.py -q
```

Expected: PASS, including unchanged non-diagnosis behavior and exact final-DAG coverage validation.

- [ ] **Step 6: Commit Task 3**

```bash
git add customized_areal/tree_search/agents/{multica_client,multica_dag_client}.py \
  customized_areal/tree_search/tests/test_env_dispatch_client.py \
  customized_areal/tree_search/tests/test_multica_dag_client.py
git commit -m "feat(tree-search): request non-training diagnosis"
```

### Task 4: Materialize diagnosis rewards on Nodes and normalize across one full DAG

**Files:**

- Modify: `customized_areal/tree_search/core/tree_store.py`
- Modify: `customized_areal/tree_search/agents/supernode_assembler.py`
- Modify: `customized_areal/tree_search/agents/gae.py`
- Modify: `customized_areal/tree_search/agents/dag_advantage.py`
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py`
- Modify: `customized_areal/tree_search/core/batch_convert.py`
- Test: `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`
- Test: `customized_areal/tree_search/tests/test_gae.py`
- Create: `customized_areal/tree_search/tests/test_multica_node_rewards.py`

**Interfaces:**

- Consumes: `SegmentSpec.assistant_turn_seqs`, `StepReward`, and resolved segment tensors with response spans.
- Produces: `Node.process_reward: float`, `Node.episode_scores: dict[str, float]`, and a `segment_id/seq -> Node` map.
- Produces: a Node-level ordered event sequence for GAE; SuperNode stays topology/tensor-only with `process_reward == 0.0`.

- [ ] **Step 1: Write failing reward-semantics tests**

Create three real `Node` objects sharing one `node_id` across two episodes and assert:

```python
node.episode_scores == {"episode-a": 0.8, "episode-b": 0.2}
assert node.process_reward == pytest.approx(0.5 / 1.5)
assert other.process_reward == pytest.approx(1.0 / 1.5)
assert node.outcome_reward == original_outcome_reward
assert sum(n.process_reward for n in scored_nodes) == pytest.approx(1.0)
```

Add a zero-total case asserting equal rewards over scored nodes, and a mapping case where a reward for `(segment-a, 7)` affects only that exact Node rather than another turn in the same segment.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
uv run pytest customized_areal/tree_search/tests/test_multica_node_rewards.py -q
```

Expected: FAIL because `Node` has no diagnosis fields and the assembler assigns `SuperNode.process_reward`.

- [ ] **Step 3: Add explicit Node reward state and a pure normalizer**

Extend `Node` with:

```python
process_reward: float = 0.0
episode_scores: dict[str, float] = field(default_factory=dict)
```

Implement a pure helper, for example `normalize_diagnosis_node_rewards(nodes: Iterable[Node]) -> None`, that deduplicates by `node_id`, computes each node’s arithmetic mean of `episode_scores.values()`, normalizes all positive/zero means across the entire supplied DAG, and leaves unscored Nodes at `0.0`. The zero-total fallback assigns `1 / len(scored_nodes)` to every scored node. It must never write `outcome_reward`.

- [ ] **Step 4: Materialize exact turn Nodes and remove SuperNode reward ownership**

In `assemble_from_refs`, derive response boundaries from each resolved tensor and create a `Node` for each declared `assistant_turn_seq`. Require a one-to-one correspondence; raise `DAGError` when the exported response spans and frozen assistant targets disagree. Store the exact mapping in assembler-local state, attach the generated Nodes to `SuperNode.nodes`, and keep `SuperNode.process_reward` at zero.

For every `StepReward`, write the score scaled by `score_max` into `node.episode_scores[segment.trajectory_id]`. If branch merging provides repeated observations for the same logical Node, retain each distinct episode ID. After every segment is materialized, normalize over the complete assembled DAG, not per segment or per trajectory.

- [ ] **Step 5: Make GAE and batching consume Node rewards**

Change `events_from_nodes` / `assemble_node_advantages` to consume flattened completion-ordered training Nodes and combine each `Node.process_reward` with its own `outcome_reward`. In `_supernodes_to_batched_tensor_dict`, broadcast the advantage of each materialized Node only onto that Node’s response span. Preserve SuperNode topology, edge serialization, branch backup, tensor cleanup, and existing verifier reward handling.

- [ ] **Step 6: Run Node, assembler, and GAE tests and verify GREEN**

Run:

```bash
uv run pytest customized_areal/tree_search/tests/test_multica_node_rewards.py \
  customized_areal/tree_search/tests/test_assembler_ref_resolve.py \
  customized_areal/tree_search/tests/test_gae.py -q
```

Expected: PASS; no test may expect diagnosis data in `SuperNode.process_reward` after this task.

- [ ] **Step 7: Commit Task 4**

```bash
git add customized_areal/tree_search/core/{tree_store,batch_convert,customized_grouped_workflow}.py \
  customized_areal/tree_search/agents/{supernode_assembler,gae,dag_advantage}.py \
  customized_areal/tree_search/tests/test_{multica_node_rewards,assembler_ref_resolve,gae}.py
git commit -m "feat(tree-search): normalize diagnosis rewards on nodes"
```

### Task 5: End-to-end verification and documentation alignment

**Files:**

- Modify: `docs/superpowers/specs/2026-07-29-nontraining-multica-diagnosis-design.md`
- Test: existing focused Go and Python suites from Tasks 1–4

**Interfaces:**

- Consumes: completed feature implementation and opt-in configuration.
- Produces: verified server/client contracts and an accurate operational test command.

- [ ] **Step 1: Write an integration-shaped mocked test**

Add a Python test that feeds a terminal pre-diagnosis DAG, a completed diagnosis report, and a final exact-covered DAG through `_debug_run --diagnose --dag-out`; assert that the persisted JSON is the final enriched DAG and all parsed training-node process rewards sum to one.

- [ ] **Step 2: Run the test and verify RED/GREEN in sequence**

Run the test before wiring its final integration point (RED), implement only the missing connection, then rerun it (GREEN):

```bash
uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -q
```

- [ ] **Step 3: Run all targeted regression suites**

Run:

```bash
cd multica/server && go test ./internal/service ./internal/handler -count=1
cd /workspaces/leagent/backend/areal && uv run pytest \
  customized_areal/tree_search/tests/test_env_dispatch_client.py \
  customized_areal/tree_search/tests/test_multica_dag_client.py \
  customized_areal/tree_search/tests/test_multica_node_rewards.py \
  customized_areal/tree_search/tests/test_assembler_ref_resolve.py \
  customized_areal/tree_search/tests/test_gae.py -q
```

Expected: PASS. If `uv` still points to the stale `/dfs/.../.venv/bin/python`, report that environment blocker rather than substituting a different Python runtime.

- [ ] **Step 4: Update the design record and commit verification**

Record the actual route names, frozen-target table, Node mapping rule, and commands run in the existing design document. Then:

```bash
git add docs/superpowers/specs/2026-07-29-nontraining-multica-diagnosis-design.md
git commit -m "docs: record diagnosis coverage verification"
```
