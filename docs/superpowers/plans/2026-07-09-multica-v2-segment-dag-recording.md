---
change: multica-v2-segment-dag-recording-assembly
design-doc: docs/superpowers/specs/2026-07-09-multica-v2-segment-dag-recording-design.md
base-ref: f60c86bbeda297938f641b77c97425f655c49a35
---

# Multica v2 Segment-DAG Recording + Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Finish the multica side of the v2 segment-DAG data path so a trained rollout round-trips end to end: event hooks record segments, retries open fresh areal sessions, `AssembledDag` is assembled and served at `/dag`, and AReaL resolves multi-shard `tensor_ref`s into tensors.

**Architecture:** Wire the existing `InteractionDAGService` (U7.1) into the trained-rollout event seams (D10 chokepoint + delegation/mention/completion/squad hooks), fix retry-session lifecycle (D9: strip `areal_proxy`, open fresh session before notify), add read-only `AssembleAssembledDag` + a `GET /dag` polling handler, and fix the cross-repo `tensor_ref` contract to multi-shard (Option B): multica decodes a field->shard map from the real export; AReaL fetches each shard and reassembles.

**Tech Stack:** Go 1.26 (multica server, chi router, sqlc hand-written), Python 3.12 (areal, httpx, orjson).

## Global Constraints

- **Dual-repo**: multica Go at `/workspaces/leagent/backend/areal/multica` on branch `feature/multica-v2-segment-dag-training` (U7.1 tip `157284045`); areal Python branched off `f60c86bb`.
- **Areal tests**: `.venv-test/bin/python -m pytest <path>` (project `uv`/`.venv` is broken; NEVER `uv run pytest`). Lint: `uvx ruff check <paths>`.
- **Multica Go**: scope builds/tests to touched packages. Pre-existing (NOT ours, do not gate): `go build ./...` fails at `webpush.go:180`; 16 handler `ON CONFLICT` daemon/claim failures.
- **sqlc generate is broken** in the multica repo; hand-write generated Go mirroring a sibling query (U7.1 pattern).
- **Do not trust implementer GREEN self-reports** - re-run tests yourself.
- **agent_run_id = task.ID** (the run), NOT the agent ID. AReal stores it in a field named `agent_id` (conflation trap).
- **Placeholder reward only**; judge/V/GAE are change 2. No sandbox pause/fork (F-independence).
- Conventional Commits, ~72-char subject, per-task commits.

---

## File Structure

**Multica Go** (branch `feature/multica-v2-segment-dag-training`):
- `internal/service/interaction_dag.go` - MODIFY: `decodeTensorRef` (multi-shard); ADD `AssembleAssembledDag`.
- `internal/service/interaction_dag_test.go` - MODIFY: fake `ExportTrajectory` -> real shape; ADD assembly tests.
- `pkg/db/queries/interaction_dag.sql` - ADD: `AssembleAssembledDag` queries (segments/edges/snapshots/session_runs by project).
- `pkg/db/generated/interaction_dag.sql.go` - ADD: hand-written `AssembleAssembledDag` Go (mirrors a sibling `:many` query).
- `internal/service/training.go` - MODIFY: D10 `RecordSessionAgentRun` call in `maybeOpenTrainingSession`.
- `internal/service/task.go` - MODIFY: event seams (`enqueueMentionTask`, `RouteTerminalTrainingTask`); `MaybeRetryFailedTask` open-before-notify; `CreateRetryTask` `areal_proxy` strip helper.
- `internal/handler/env_dispatch.go` - ADD: `GetDag` handler method.
- `internal/handler/env_dispatch_test.go` - ADD: `/dag` tests.
- `cmd/server/router.go` - MODIFY: register `GET /api/v1/env-dispatch/{projectID}/dag`.
- `internal/service/training_test.go` / `task_test.go` - ADD: D10/D9/seam tests.

**Areal Python** (branch off `f60c86bb`):
- `customized_areal/tree_search/agents/segment_dag_trainer.py` - MODIFY: `DataProxyTensorResolver.resolve` (multi-shard) + shard-id collection for `clear`.
- `customized_areal/tree_search/agents/multica_dag_client.py` - MODIFY: `SegmentSpec.tensor_ref` docstring/type (field->shard map).
- `customized_areal/tree_search/agents/supernode_assembler.py` - MODIFY: `assemble_from_refs` stamps the reassembled field->tensor dict.
- `customized_areal/tree_search/tests/test_segment_dag_trainer.py` (or existing) - ADD: multi-shard resolve tests.

---

## Task 1: multica `decodeTensorRef` multi-shard (contract fix, multica side)

**Files:**
- Modify: `multica/server/internal/service/interaction_dag.go:239-258` (`decodeTensorRef`)
- Modify: `multica/server/internal/service/interaction_dag_test.go` (fake `ExportTrajectory` + decode tests)

**Interfaces:**
- Consumes: U6 `ExportTrajectory` raw `json.RawMessage` = areal `traj` dict `{field: serialized RTensor}`.
- Produces: `decodeTensorRef(raw) ([]byte, error)` returning jsonb `{"<field>": {"shard_id": "...", "node_addr": "..."}, ...}`.

- [x] **Step 1: Write failing tests for multi-shard decode**

Add to `interaction_dag_test.go`. Replace the fake `ExportTrajectory` to return a real-shape RTensor-envelope traj, and assert `decodeTensorRef` extracts a field->shard map.

```go
func TestDecodeTensorRef_MultiShardEnvelope(t *testing.T) {
	// Real areal export shape: each field is a serialized RTensor dataclass whose
	// data.shard.data holds {shard_id, node_addr}.
	traj := map[string]any{
		"input_ids": map[string]any{
			"type": "dataclass", "class_path": "areal.infra.rpc.rtensor.RTensor",
			"data": map[string]any{
				"shard": map[string]any{
					"type": "dataclass", "class_path": "areal.infra.rpc.rtensor.TensorShardInfo",
					"data": map[string]any{"shard_id": "shard-input-1", "node_addr": "10.0.0.1:8000"},
				},
				"data": map[string]any{"shape": []int{1, 4}},
			},
		},
		"attention_mask": map[string]any{
			"type": "dataclass", "class_path": "areal.infra.rpc.rtensor.RTensor",
			"data": map[string]any{
				"shard": map[string]any{
					"type": "dataclass", "class_path": "areal.infra.rpc.rtensor.TensorShardInfo",
					"data": map[string]any{"shard_id": "shard-mask-1", "node_addr": "10.0.0.1:8000"},
				},
				"data": map[string]any{"shape": []int{1, 4}},
			},
		},
	}
	raw, _ := json.Marshal(traj)
	out, err := decodeTensorRef(raw)
	if err != nil {
		t.Fatalf("decodeTensorRef: %v", err)
	}
	var ref map[string]map[string]string
	if err := json.Unmarshal(out, &ref); err != nil {
		t.Fatalf("unmarshal decoded: %v", err)
	}
	if ref["input_ids"]["shard_id"] != "shard-input-1" {
		t.Fatalf("input_ids shard_id = %q", ref["input_ids"]["shard_id"])
	}
	if ref["attention_mask"]["node_addr"] != "10.0.0.1:8000" {
		t.Fatalf("attention_mask node_addr = %q", ref["attention_mask"]["node_addr"])
	}
	if _, ok := ref["shard_id"]; ok {
		t.Fatalf("must not be a single shard_id map; got %v", ref)
	}
}

func TestDecodeTensorRef_BareShardRef(t *testing.T) {
	// Tolerant: a bare {"shard_id":...} per field is also accepted.
	traj := map[string]any{
		"input_ids": map[string]any{"shard_id": "s1", "node_addr": "h:1"},
	}
	raw, _ := json.Marshal(traj)
	out, err := decodeTensorRef(raw)
	if err != nil {
		t.Fatalf("decodeTensorRef: %v", err)
	}
	var ref map[string]map[string]string
	_ = json.Unmarshal(out, &ref)
	if ref["input_ids"]["shard_id"] != "s1" {
		t.Fatalf("got %v", ref)
	}
}

func TestDecodeTensorRef_MissingShardIDIsError(t *testing.T) {
	traj := map[string]any{"input_ids": map[string]any{"data": "no shard"}}
	raw, _ := json.Marshal(traj)
	if _, err := decodeTensorRef(raw); err == nil {
		t.Fatal("expected error for field missing shard_id")
	}
}
```

Also update the existing fake `fakeArealSegmentClient.ExportTrajectory` (e.g. `interaction_dag_test.go:180,223`) to return the envelope shape so `CloseSegmentForEvent` stores a field->shard `tensor_ref`.

- [x] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable go test ./internal/service/ -run TestDecodeTensorRef -v`
Expected: FAIL (current `decodeTensorRef` returns the whole traj dict, not a field->shard map; assertions on `ref["input_ids"]["shard_id"]` fail).

- [x] **Step 3: Implement multi-shard `decodeTensorRef`**

Replace `decodeTensorRef` in `interaction_dag.go:239-258`:

```go
// decodeTensorRef extracts a field->shard map from the areal /export_trajectories
// traj dict. The real export emits one serialized RTensor per tensor field
// (input_ids, attention_mask, logprobs, loss_mask, versions); each RTensor's
// data.shard.data carries {shard_id, node_addr}. We store that field->ref map as
// the segment's tensor_ref jsonb. Absence (a field with no shard_id) is an error,
// not a default - downstream resolution would KeyError on a masked None.
func decodeTensorRef(raw json.RawMessage) ([]byte, error) {
	if len(raw) == 0 {
		return nil, errors.New("empty export payload, no tensor_ref")
	}
	var traj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &traj); err != nil {
		return nil, fmt.Errorf("decode export payload: %w", err)
	}
	if len(traj) == 0 {
		return nil, errors.New("empty export payload, no tensor fields")
	}
	out := make(map[string]map[string]string, len(traj))
	for field, fieldRaw := range traj {
		ref, err := extractShardRef(fieldRaw)
		if err != nil {
			return nil, fmt.Errorf("tensor_ref field %q: %w", field, err)
		}
		out[field] = ref
	}
	return json.Marshal(out)
}

// extractShardRef tolerates either a bare {"shard_id","node_addr"} ref or the
// full serialized RTensor dataclass envelope (data.shard.data.{shard_id,node_addr}).
func extractShardRef(fieldRaw json.RawMessage) (map[string]string, error) {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(fieldRaw, &probe); err != nil {
		return nil, fmt.Errorf("not an object: %w", err)
	}
	if sidRaw, ok := probe["shard_id"]; ok {
		var s string
		if err := json.Unmarshal(sidRaw, &s); err == nil && s != "" {
			ref := map[string]string{"shard_id": s}
			if na, ok := probe["node_addr"]; ok {
				_ = json.Unmarshal(na, &ref["node_addr"])
			}
			return ref, nil
		}
	}
	type shardInfo struct {
		ShardID  string `json:"shard_id"`
		NodeAddr string `json:"node_addr"`
	}
	type envelope struct {
		Data struct {
			Shard shardInfo `json:"shard"`
		} `json:"data"`
	}
	var env envelope
	if err := json.Unmarshal(fieldRaw, &env); err != nil {
		return nil, fmt.Errorf("no shard_id and not an RTensor envelope: %w", err)
	}
	if env.Data.Shard.ShardID == "" {
		return nil, errors.New("RTensor envelope missing shard_id")
	}
	return map[string]string{"shard_id": env.Data.Shard.ShardID, "node_addr": env.Data.Shard.NodeAddr}, nil
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable go test ./internal/service/ -run TestDecodeTensorRef -v`
Expected: PASS (3/3). Then run the full U7.1 suite to confirm the updated fake didn't regress: `go test ./internal/service/ -run TestInteractionDAG -v` -> green.

- [x] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/interaction_dag.go server/internal/service/interaction_dag_test.go
git commit -m "fix(v2-segment-dag-recording): decodeTensorRef multi-shard field->ref map (Task 1)"
```

---

## Task 2: areal `DataProxyTensorResolver` multi-shard (contract fix, areal side)

**Files:**
- Modify: `customized_areal/tree_search/agents/segment_dag_trainer.py:91-120` (`resolve`, `clear` shard collection)
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py:45` (`SegmentSpec.tensor_ref` docstring)
- Test: `customized_areal/tree_search/tests/test_segment_dag_trainer.py`

**Interfaces:**
- Consumes: `tensor_ref: dict[str, dict[str,str]]` = field->`{shard_id, node_addr}` (from Task 1).
- Produces: `resolve(tensor_ref) -> dict[str, Any]` = field->resolved tensor; `clear` receives all field shard_ids.

- [x] **Step 1: Write failing tests for multi-shard resolve**

Add to `customized_areal/tree_search/tests/test_segment_dag_trainer.py` (create if absent):

```python
import orjson
from customized_areal.tree_search.agents.segment_dag_trainer import DataProxyTensorResolver


class _FakeTransport:
    def __init__(self, routes):
        self._routes = routes
        self.requested = []

    def handle_request(self, request):
        self.requested.append((request.method, str(request.url)))
        path = str(request.url)
        if path in self._routes:
            status, body = self._routes[path]
            from httpx import Response
            return Response(status, content=body)
        from httpx import Response
        return Response(404, content=b"not found")


def _shard_body(field_value):
    # /data/<shard_id> returns serialize_value(rtensor.fetch(shard_id)) of one RTensor.
    return orjson.dumps({"type": "dataclass", "class_path": "areal.infra.rpc.rtensor.RTensor",
                         "data": {"shard": {"shard_id": "x"}, "data": {"field_value": field_value}}})


def test_resolve_multi_shard_fetches_each_field(monkeypatch):
    tensor_ref = {
        "input_ids": {"shard_id": "shard-in", "node_addr": "h:1"},
        "attention_mask": {"shard_id": "shard-am", "node_addr": "h:1"},
    }
    routes = {
        "http://x/data/shard-in": (200, _shard_body("in")),
        "http://x/data/shard-am": (200, _shard_body("am")),
    }
    transport = _FakeTransport(routes)
    resolver = DataProxyTensorResolver("http://x", _transport=transport)
    out = resolver.resolve(tensor_ref)
    assert set(out.keys()) == {"input_ids", "attention_mask"}
    assert ("GET", "http://x/data/shard-in") in transport.requested
    assert ("GET", "http://x/data/shard-am") in transport.requested


def test_resolve_missing_shard_id_raises():
    resolver = DataProxyTensorResolver("http://x", _transport=_FakeTransport({}))
    try:
        resolver.resolve({"input_ids": {"node_addr": "h:1"}})
    except KeyError:
        return
    raise AssertionError("expected KeyError for missing shard_id")


def test_resolve_404_raises_keyerror():
    resolver = DataProxyTensorResolver("http://x", _transport=_FakeTransport({}))
    try:
        resolver.resolve({"input_ids": {"shard_id": "ghost"}})
    except KeyError:
        return
    raise AssertionError("expected KeyError for 404 shard")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd /workspaces/leagent/backend/areal && .venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_segment_dag_trainer.py -v`
Expected: FAIL (current `resolve` does `tensor_ref.get("shard_id")` -> `None` -> `KeyError("tensor_ref missing 'shard_id'")` immediately; multi-shard fetch never happens).

- [x] **Step 3: Implement multi-shard `resolve`**

Replace `resolve` in `segment_dag_trainer.py:91-108`:

```python
    def resolve(self, tensor_ref: dict[str, Any]) -> dict[str, Any]:
        # tensor_ref is a field->shard map produced by Multica's decodeTensorRef:
        # {"input_ids": {"shard_id": "...", "node_addr": "..."}, ...}. The real
        # areal /export_trajectories emits one RTensor shard per tensor field, so
        # we fetch each field's shard via /data/<shard_id> and reassemble the
        # field->tensor dict the assembler stamps into metadata["tensors"].
        if not isinstance(tensor_ref, dict) or not tensor_ref:
            raise KeyError("tensor_ref missing field->shard map")
        from areal.infra.rpc.serialization import deserialize_value

        tensors: dict[str, Any] = {}
        with self._client() as client:
            for field, ref in tensor_ref.items():
                if not isinstance(ref, dict):
                    raise KeyError(f"tensor_ref field {field!r} is not a shard ref")
                shard_id = ref.get("shard_id")
                if not shard_id:
                    raise KeyError(f"tensor_ref field {field!r} missing 'shard_id'")
                resp = client.get(f"{self._base}/data/{shard_id}")
                if resp.status_code == 404:
                    raise KeyError(f"shard {shard_id} not found (field {field!r})")
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"unexpected {resp.status_code} resolving shard {shard_id}: {resp.text}"
                    )
                tensors[field] = deserialize_value(orjson.loads(resp.content))
        return tensors
```

Update the `SegmentSpec.tensor_ref` docstring in `multica_dag_client.py:45` to: `# field->shard map: {"input_ids": {"shard_id":..,"node_addr":..}, ...}`.

Update shard-id collection for `clear`: in `run_segment_dag_training_step` (same file), where shard_ids are gathered for `resolver.clear(...)`, collect from every field of each segment's `tensor_ref`:

```python
shard_ids = [
    ref["shard_id"]
    for seg in dag.segments
    for ref in (seg.tensor_ref or {}).values()
    if isinstance(ref, dict) and ref.get("shard_id")
]
resolver.clear(shard_ids)
```

(Replace the prior single-`shard_id` collection. If the prior code already iterates segments, adapt it to flatten all field refs.)

- [x] **Step 4: Run tests to verify they pass**

Run: `cd /workspaces/leagent/backend/areal && .venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_segment_dag_trainer.py -v`
Expected: PASS (3/3). Regression: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/ -k 'segment_dag or supernode or env_dispatch' -v` -> green. Lint: `uvx ruff check customized_areal/tree_search/agents/segment_dag_trainer.py customized_areal/tree_search/agents/multica_dag_client.py` -> clean.

- [x] **Step 5: Commit**

```bash
cd /workspaces/leagent/backend/areal
git add customized_areal/tree_search/agents/segment_dag_trainer.py customized_areal/tree_search/agents/multica_dag_client.py customized_areal/tree_search/tests/test_segment_dag_trainer.py
git commit -m "fix(v2-segment-dag-recording): DataProxyTensorResolver multi-shard resolve (Task 2)"
```

---

## Task 3: D10 `RecordSessionAgentRun` chokepoint (multica)

**Files:**
- Modify: `multica/server/internal/service/training.go` (`maybeOpenTrainingSession`, ~`:226`)
- Test: `multica/server/internal/service/training_test.go`

**Interfaces:**
- Consumes: `InteractionDAGService.RecordSessionAgentRun(ctx, projectID, sessionID, agentRunID, issueID string) error`; `creds.SessionID` (`training.go:204`); `taskID`; `issueID` (resolve via `GetIssue` if not in hand).
- Produces: a `session_to_agent_run` row per session-open (idempotent upsert).

- [x] **Step 1: Write failing test**

```go
func TestMaybeOpenTrainingSession_RecordsSessionAgentRun(t *testing.T) {
	svc, deps := newTrainingServiceWithDAG(t) // deps.DAG is a recording fake
	task := enqueueTrainedTask(t, svc, /*issueID=*/"iss-1", /*projectID=*/"proj-1")
	// drive maybeOpenTrainingSession as the Enqueue path does
	if err := svc.tryOpenTrainingSession(ctx, task, "proj-1", ""); err != nil {
		t.Fatalf("tryOpenTrainingSession: %v", err)
	}
	got := deps.DAG.(*recordingDAGFake).RecordedRuns()
	if len(got) != 1 || got[0].AgentRunID != task.ID.String() || got[0].SessionID == "" || got[0].IssueID != "iss-1" {
		t.Fatalf("RecordSessionAgentRun not fired correctly: %+v", got)
	}
	// idempotent: a second open attempt does not add a second row
	_ = svc.tryOpenTrainingSession(ctx, task, "proj-1", "")
	if len(deps.DAG.(*recordingDAGFake).RecordedRuns()) != 1 {
		t.Fatalf("RecordSessionAgentRun should be idempotent")
	}
}
```

(`recordingDAGFake` implements `InteractionDAGService`-shaped interface recording calls; mirror the existing fake pattern in `training_test.go`.)

- [x] **Step 2: Run test to verify it fails**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestMaybeOpenTrainingSession_RecordsSessionAgentRun -v`
Expected: FAIL (no `RecordSessionAgentRun` call; `RecordedRuns()` empty).

- [x] **Step 3: Implement D10 chokepoint**

In `maybeOpenTrainingSession` (`training.go:~226`, after the persist at `:220`, before the slog at `:227`), insert (guarded by `s.Training != nil && s.Training.DAG != nil && s.Training.DAG.Enabled()`):

```go
if s.Training != nil && s.Training.DAG != nil && s.Training.DAG.Enabled() {
	issueID := resolveIssueID(ctx, s, task) // GetIssue(task.IssueID).ID string; "" if none
	if err := s.Training.DAG.RecordSessionAgentRun(ctx, projectID, creds.SessionID, taskID, issueID); err != nil {
		slog.Warn("interaction_dag: RecordSessionAgentRun failed", "err", err, "task_id", taskID)
	}
}
```

`agentRunID = taskID` (NOT `agentID`). Add `resolveIssueID` helper (returns `task.IssueID` if already a string, else `GetIssue`). `taskID` is the arg already in `maybeOpenTrainingSession`.

- [x] **Step 4: Run test to verify it passes**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestMaybeOpenTrainingSession_RecordsSessionAgentRun -v`
Expected: PASS. Regression: `go test ./internal/service/ -run TestMaybeOpenTrainingSession -v` -> green.

- [x] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/training.go server/internal/service/training_test.go
git commit -m "feat(v2-segment-dag-recording): D10 RecordSessionAgentRun chokepoint at session open (Task 3)"
```

---

## Task 4: U7.2 event seams - delegation/mention/completion/leaf (multica)

**Files:**
- Modify: `multica/server/internal/service/task.go` (`enqueueMentionTask:581`, `RouteTerminalTrainingTask:1425`, `CompleteTask`)
- Test: `multica/server/internal/service/interaction_dag_test.go` (integration)

**Interfaces:**
- Consumes: `CloseSegmentForEvent(ctx, projectID, sessionID, proxyKey, closingEvent, envSnapshot) (segmentID, err)`; `AddEdge(ctx, projectID, srcSeg, dstSeg, edgeType)`; `SegmentIDForAgentRun(ctx, agentRunID)`.
- Produces: per-event segment + edge rows.

- [x] **Step 1: Write failing integration tests** (hermetic Postgres, U7.1 pattern). One test per seam:

```go
func TestInteractionDAG_DelegationRecordsSegmentAndEdge(t *testing.T) {
	// parent trained run opens session (D10), turns, then delegates to child.
	// Assert: parent segment closed (closing_event=delegation), delegation edge
	// parentSeg->childSeg recorded, child session opened (D10).
}

func TestInteractionDAG_MentionRecordsEdgeOnly(t *testing.T) {
	// Assert: AddEdge mention called, no CloseSegmentForEvent.
}

func TestInteractionDAG_CompletionClosesChildAndEdges(t *testing.T) {
	// Assert: child segment closed (closing_event=completion), completion edge.
}

func TestInteractionDAG_LeafRunYieldsOneLeafSegment(t *testing.T) {
	// Assert: exactly one segment, closing_event=NULL.
}

func TestInteractionDAG_ConcurrentFanOutDeterministicAcyclic(t *testing.T) {
	// Assert: N delegation edges, deterministic order, acyclic.
}
```

Drive each seam via the existing `claimTaskByRuntimeForTest`/`enqueueMentionTask` test helpers; assert against `interaction_dag_segment`/`_edge` rows directly (hermetic DB).

- [x] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestInteractionDAG_ -v`
Expected: FAIL (seams don't call `CloseSegmentForEvent`/`AddEdge` yet).

- [x] **Step 3: Implement the seams**

In `enqueueMentionTask` (`task.go:581`), after the child task is created and `tryOpenTrainingSession(child, projectID, envID)` (`:614`) but before `NotifyTaskEnqueued` (`:618`), insert the delegation hook (guarded `s.Training != nil && s.Training.DAG.Enabled()`):

```go
if s.Training != nil && s.Training.DAG != nil && s.Training.DAG.Enabled() {
    parentSeg, err := s.Training.DAG.SegmentIDForAgentRun(ctx, parentTaskID)
    if err == nil && parentSeg != "" {
        proxyKey := arealProxyKeyFromContext(parent.Context) // extract from areal_proxy
        segID, cerr := s.Training.DAG.CloseSegmentForEvent(ctx, projectID, parentSessionID, proxyKey, "delegation", leanEnvSnapshot(ctx, s, projectID))
        if cerr == nil {
            _ = s.Training.DAG.AddEdge(ctx, projectID, parentSeg, segID, "delegation")
        } else {
            slog.Warn("interaction_dag: delegation close failed", "err", cerr)
        }
    }
}
```

Mention: `AddEdge(ctx, projectID, srcSeg, dstSeg, "mention")` without `CloseSegmentForEvent`.

In `RouteTerminalTrainingTask` (`task.go:1425`), after the parent session close, insert the completion hook: `CloseSegmentForEvent(ctx, projectID, sessionID, proxyKey, "completion", leanEnvSnapshot(...))` + `AddEdge(..., "completion")` (resolve parent seg via `SegmentIDForAgentRun`).

Leaf: in `CompleteTask` for a trained run with no communication event, `CloseSegmentForEvent(ctx, projectID, sessionID, proxyKey, "", leanEnvSnapshot(...))` (closing_event="" -> NULL).

Add helpers: `arealProxyKeyFromContext(ctxJSON) string` (parse `areal_proxy.api_key` from the task context JSONB), `leanEnvSnapshot(ctx, s, projectID) map[string]any` (`sandbox_ids` from `project.env_id -> environment.sandbox_ids`, else `[]`; `env_state={}`).

- [x] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestInteractionDAG_ -v`
Expected: PASS (5/5). `go vet ./internal/service/` clean; `gofmt -l` clean.

- [x] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task.go server/internal/service/interaction_dag_test.go
git commit -m "feat(v2-segment-dag-recording): event seams delegation/mention/completion/leaf (Task 4)"
```

---

## Task 5: U7.2 gating + best-effort + squad briefing (multica)

**Files:**
- Modify: `multica/server/internal/service/task.go` (squad seam at daemon claim path)
- Modify: `multica/server/internal/service/env_dispatch.go` (squad briefing hook where `SandboxRefs` are in scope)
- Test: `multica/server/internal/service/interaction_dag_test.go`

- [x] **Step 1: Write failing tests**

```go
func TestInteractionDAG_NonTrainedRolloutRecordsNothing(t *testing.T) {
	// Assert: no segment/edge rows, no CloseSegment/Export calls (DAG.Enabled false or Training nil).
}
func TestInteractionDAG_RecordingErrorIsBestEffort(t *testing.T) {
	// Inject a failing DAG fake; Assert: run continues, error logged, no panic.
}
func TestInteractionDAG_SquadContextHandoffClosesProducerSegment(t *testing.T) {
	// Assert: the producer/parent session is closed with closing_event="squad_briefing"
	// and the eventual parent->child edge remains type="delegation". Do NOT close
	// the receiver/child session at daemon claim time.
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestInteractionDAG_ -v`
Expected: FAIL (gating not asserted; squad seam absent).

- [x] **Step 3: Implement gating + squad seam**

Gating is already in place via the `s.Training != nil && s.Training.DAG.Enabled()` guards (Task 3/4). Implement squad-context handoff at the existing parent/producer delegation seam, not at daemon claim: close the producer session with `closing_event="squad_briefing"` when the handoff is a squad-context delegation, and keep the structural edge type as `delegation` when the child later closes. Best-effort: all hooks `slog.Warn` on error and continue (already the pattern in Task 4).

- [x] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestInteractionDAG_ -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task.go server/internal/service/env_dispatch.go server/internal/service/interaction_dag_test.go
git commit -m "feat(v2-segment-dag-recording): gating + best-effort + squad briefing seam (Task 5)"
```

---

## Task 6: U7.3 `CreateRetryTask` strips `areal_proxy` (D9, multica)

**Files:**
- Modify: `multica/server/internal/service/task.go` (`createRetryTaskWithPendingWakeTransfer` / `CreateRetryTask` caller, `:1780`/`:1813`)
- Modify: `multica/server/pkg/db/queries/agent.sql:186` (or a Go post-insert context rewrite - pick lower-risk)
- Test: `multica/server/internal/service/task_test.go`

**Interfaces:**
- Consumes: `areal_proxy` key in the parent task `context` JSONB (`training.go:38-44`).
- Produces: child task `context` with `areal_proxy` removed, chat `session_id`/`work_dir` preserved.

- [x] **Step 1: Write failing test**

```go
func TestCreateRetryTask_StripsArealProxyKeepsChatSession(t *testing.T) {
	parent := enqueueTrainedTask(t, svc, "iss-1", "proj-1")
	parent.Context = mustJSON(t, map[string]any{
		"areal_proxy": map[string]any{"session_id": "S_A", "api_key": "k"},
		"session_id":  "chat-sess",
		"work_dir":    "/w",
	})
	child, err := svc.createRetryTaskWithPendingWakeTransfer(ctx, parent, retryableReason)
	if err != nil { t.Fatalf("createRetry: %v", err) }
	var cctx map[string]any
	_ = json.Unmarshal(child.Context, &cctx)
	if _, ok := cctx["areal_proxy"]; ok {
		t.Fatal("child context must NOT inherit areal_proxy")
	}
	if cctx["session_id"] != "chat-sess" || cctx["work_dir"] != "/w" {
		t.Fatalf("chat session_id/work_dir must be preserved: %v", cctx)
	}
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestCreateRetryTask_StripsArealProxy -v`
Expected: FAIL (child inherits `areal_proxy` via `agent.sql:186` `p.context` copy).

- [x] **Step 3: Implement the strip**

Lower-risk mechanism: after `CreateRetryTask` returns the child, rewrite the child's `context` JSONB to drop `areal_proxy` and persist via a targeted update (mirror the existing `MergeTaskArealProxyContext` query style - add `StripArealProxyFromTaskContext(ctx, childID)` hand-written sqlc that does `UPDATE agent_task_queue SET context = context - 'areal_proxy' WHERE id = $1`). Call it in `createRetryTaskWithPendingWakeTransfer` right after the child is created (`:1813`/`:1822`), before `broadcastTaskEvent`. (This avoids touching the `p.context` copy in `agent.sql:186`, keeping the chat `session_id`/`work_dir` CASE-WHEN at `:187-188` intact.)

```go
// after child created:
if s.Training != nil {
    if err := s.queries.StripArealProxyFromTaskContext(ctx, child.ID); err != nil {
        slog.Warn("interaction_dag: strip areal_proxy failed", "err", err, "child", child.ID)
    }
}
```

Hand-write `StripArealProxyFromTaskContext` in `generated/agent.sql.go` mirroring a sibling `:exec` update + add the query to `queries/agent.sql`.

- [x] **Step 4: Run test to verify it passes**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestCreateRetryTask_StripsArealProxy -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task.go server/pkg/db/queries/agent.sql server/pkg/db/generated/agent.sql.go server/internal/service/task_test.go
git commit -m "feat(v2-segment-dag-recording): D9 strip areal_proxy from retry child context (Task 6)"
```

---

## Task 7: U7.3 `MaybeRetryFailedTask` opens fresh session before notify (D9, multica)

**Files:**
- Modify: `multica/server/internal/service/task.go:1753` (`MaybeRetryFailedTask`, insert before `NotifyTaskEnqueued:1800`)
- Test: `multica/server/internal/service/task_test.go`

**Interfaces:**
- Consumes: `tryOpenTrainingSession(ctx, task, projectID, envID)`; `envID` from `project.env_id` via `issue_id -> GetIssue -> ProjectID -> GetProject -> EnvID`.
- Produces: child opens its own areal session; `RecordSessionAgentRun` (D10) fires for child; parent already closed by `RouteTerminalTrainingTask`.

- [x] **Step 1: Write failing tests**

```go
func TestMaybeRetryFailedTask_ChildOpensFreshSessionBeforeNotify(t *testing.T) {
	// parent trained run fails with a retryable reason; drive FailTask path.
	// Assert ordering: parent EndSession (S_A closed) BEFORE child StartSession (S_B);
	// child has its own session (RecordSessionAgentRun fires for child task.ID);
	// tryOpenTrainingSession(child) called BEFORE NotifyTaskEnqueued.
}
func TestMaybeRetryFailedTask_NonRetryableIsTerminal(t *testing.T) {
	// Assert: session closed, no child created, no tryOpenTrainingSession.
}
func TestMaybeRetryFailedTask_EnvIDFromProjectEnvID(t *testing.T) {
	// Assert: child's StartSession body env_id == project.env_id (not areal SessionID).
}
```

Use a recording fake that captures `StartSession`/`EndSession`/`NotifyTaskEnqueued` order.

- [x] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestMaybeRetryFailedTask_ -v`
Expected: FAIL (no `tryOpenTrainingSession(child)` before notify).

- [x] **Step 3: Implement open-before-notify**

In `MaybeRetryFailedTask` (`task.go:1753`), after the child is created + `areal_proxy` stripped (Task 6), before `NotifyTaskEnqueued` (`:1800`):

```go
if s.Training != nil {
    issue, _ := s.queries.GetIssue(ctx, child.IssueID)
    var projectID, envID string
    if issue.ProjectID.Valid {
        projectID = issue.ProjectID.String()
        proj, _ := s.queries.GetProject(ctx, issue.ProjectID)
        if proj.EnvID.Valid {
            envID = proj.EnvID.String()
        }
    }
    if err := s.tryOpenTrainingSession(ctx, child, projectID, envID); err != nil {
        slog.Warn("interaction_dag: retry child session open failed", "err", err, "child", child.ID)
    }
}
```

`tryOpenTrainingSession` -> `maybeOpenTrainingSession` -> `StartSession(child.ID, envID)` -> D10 `RecordSessionAgentRun(child)`. Parent already closed by `RouteTerminalTrainingTask` (`:1663` before `:1668`).

- [x] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestMaybeRetryFailedTask_ -v`
Expected: PASS (3/3).

- [x] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task.go server/internal/service/task_test.go
git commit -m "feat(v2-segment-dag-recording): D9 fresh areal session per retry before notify (Task 7)"
```

---

## Task 8: U8 `AssembleAssembledDag` + hand-written sqlc (multica)

**Files:**
- Modify: `multica/server/pkg/db/queries/interaction_dag.sql` (ADD assembly queries)
- Modify: `multica/server/pkg/db/generated/interaction_dag.sql.go` (ADD hand-written Go)
- Modify: `multica/server/internal/service/interaction_dag.go` (ADD `AssembleAssembledDag`)
- Test: `multica/server/internal/service/interaction_dag_test.go`

**Interfaces:**
- Consumes: recorded `interaction_dag_segment`/`_edge`/`_env_snapshot`/`_session_run` rows for a project.
- Produces: `AssembledDag{Segments, Edges, SessionToAgentRun}` (no scores/turn-idx/text).

- [x] **Step 1: Write failing tests**

```go
func TestAssembleAssembledDag_ProjectsRecordedRows(t *testing.T) {
	// seed segments+edges+snapshots+session_runs for proj-1; assemble.
	// Assert: segments carry trajectory_id+tensor_ref(field map)+closing_event+env_snapshot;
	// no judge_scores, no start_turn_idx/end_turn_idx, no text.
}
func TestAssembleAssembledDag_EdgesTypedAndAcyclic(t *testing.T) {
	// Assert: edge types in {delegation,mention,completion}; graph acyclic.
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestAssembleAssembledDag -v`
Expected: FAIL (`AssembleAssembledDag` undefined).

- [x] **Step 3: Implement assembly (read-only) + sqlc**

Add to `queries/interaction_dag.sql`:
```sql
-- name: ListInteractionDAGSegmentsForProject :many
SELECT segment_id, agent_run_id, issue_id, trajectory_id, tensor_ref, closing_event, closing_event_target_segment
FROM interaction_dag_segment WHERE project_id = $1 ORDER BY created_at;

-- name: ListInteractionDAGEdgesForProject :many
SELECT src_segment_id, dst_segment_id, type FROM interaction_dag_edge WHERE project_id = $1 ORDER BY id;

-- name: ListInteractionDAGSessionRunsForProject :many
SELECT session_id, agent_run_id FROM interaction_dag_session_run WHERE project_id = $1;
```
(`interaction_dag_env_snapshot` joins by segment_id in Go, or add a `:many` join query.)

Hand-write the three `:many` methods in `generated/interaction_dag.sql.go` mirroring U7.1's `ListSegmentsForProject`-style sibling (return slices of row structs). Add the row struct types if absent.

Add `AssembleAssembledDag` to `interaction_dag.go`:
```go
func (s *InteractionDAGService) AssembleAssembledDag(ctx context.Context, projectID string) (AssembledDag, error) {
	segs, err := s.q.ListInteractionDAGSegmentsForProject(ctx, projectID)
	if err != nil { return AssembledDag{}, err }
	edges, err := s.q.ListInteractionDAGEdgesForProject(ctx, projectID)
	if err != nil { return AssembledDag{}, err }
	runs, err := s.q.ListInteractionDAGSessionRunsForProject(ctx, projectID)
	if err != nil { return AssembledDag{}, err }
	// map segments (decode tensor_ref jsonb as-is; env_snapshot from a join or per-segment lookup)
	// map edges {src,dst,type}; map session_to_agent_run {session_id: agent_run_id}
	// NO scores, NO turn indices, NO text.
	return AssembledDag{Segments: ..., Edges: ..., SessionToAgentRun: ...}, nil
}
```
Define `AssembledDag`, `AssembledSegment`, `AssembledEdge` structs (mirror areal's `SegmentSpec`/`EdgeType` contract).

- [x] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/service/ -run TestAssembleAssembledDag -v`
Expected: PASS (2/2). `go build ./internal/service/ ./pkg/db/generated/` clean.

- [x] **Step 5: Commit**

```bash
cd multica && git add server/pkg/db/queries/interaction_dag.sql server/pkg/db/generated/interaction_dag.sql.go server/internal/service/interaction_dag.go server/internal/service/interaction_dag_test.go
git commit -m "feat(v2-segment-dag-recording): AssembleAssembledDag read-only assembly + sqlc (Task 8)"
```

---

## Task 9: U8 `GET /dag` handler + route (multica)

**Files:**
- Modify: `multica/server/internal/handler/env_dispatch.go` (ADD `GetDag`)
- Modify: `multica/server/cmd/server/router.go:1035` (ADD route)
- Test: `multica/server/internal/handler/env_dispatch_test.go`

**Interfaces:**
- Consumes: `InteractionDAGService.AssembleAssembledDag`; project status (in-progress vs done) for 202 vs 200; workspace gate for 403.
- Produces: `GET /api/v1/env-dispatch/{projectID}/dag` -> 202/200/404/403 + failed-status.

- [x] **Step 1: Write failing handler tests**

```go
func TestGetDag_InProgressReturns202(t *testing.T) { /* root task not complete -> 202 + status body */ }
func TestGetDag_DoneReturns200AssembledDag(t *testing.T) { /* -> 200 + AssembledDag */ }
func TestGetDag_UnknownProjectReturns404(t *testing.T) { /* -> 404 */ }
func TestGetDag_CrossWorkspaceReturns403(t *testing.T) { /* -> 403 */ }
func TestGetDag_IncompleteRolloutReturnsFailedStatus(t *testing.T) { /* segments don't densely cover -> 200 + failed status, no AssembledDag */ }
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/handler/ -run TestGetDag -v`
Expected: FAIL (no `GetDag` handler/route).

- [x] **Step 3: Implement `GetDag` + route**

Add to `env_dispatch.go`:
```go
func (h *Handler) GetDag(w http.ResponseWriter, r *http.Request) {
	projectID := chi.URLParam(r, "projectID")
	// workspace gate (mirror DeleteEnvDispatchProject): 403 cross-workspace
	if !h.canAccessProject(r, projectID) {
		writeStatus(w, http.StatusForbidden, "forbidden"); return
	}
	status, err := h.envDispatchStore.GetDagStatus(r.Context(), projectID)
	if err != nil /* unknown project */ { writeStatus(w, http.StatusNotFound, "not found"); return }
	if status == "in_progress" { writeJSON(w, http.StatusAccepted, map[string]any{"status": "in_progress"}); return }
	dag, derr := h.dagSvc.AssembleAssembledDag(r.Context(), projectID)
	if derr != nil || !denseCover(dag) {
		writeJSON(w, http.StatusOK, map[string]any{"status": "failed"}); return // D14: no partial DAG
	}
	writeJSON(w, http.StatusOK, dag)
}
```
`denseCover(dag)` checks every session in `session_to_agent_run` has segments covering the run (no gaps). Add `GetDagStatus` query (or reuse `GetDagStatus` from U7.1 if present). Register in `router.go:1035`:
```go
r.Get("/api/v1/env-dispatch/{projectID}/dag", h.GetDag)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && DATABASE_URL=... go test ./internal/handler/ -run TestGetDag -v`
Expected: PASS (5/5). `go vet ./internal/handler/` clean.

- [x] **Step 5: Commit**

```bash
cd multica && git add server/internal/handler/env_dispatch.go server/internal/handler/env_dispatch_test.go server/cmd/server/router.go
git commit -m "feat(v2-segment-dag-recording): GET /dag endpoint 202/200/404/403 + failed-status (Task 9)"
```

---

## Task 10: U10 config + regression + E2E + grep (both repos)

**Files:**
- Modify: multica `internal/service/training_config.go` (`INTERACTION_DAG_ENABLED` default), areal `multica_dag_client.py` (polling config interval/timeout/backoff)
- Test: full regression + E2E + grep sweep

- [x] **Step 1: Config defaults**

Multica: `INTERACTION_DAG_ENABLED` defaults on for trained rollouts in `LoadTrainingConfig` (mirror `AREAL_*` env reading). Areal: expose polling `interval`/`timeout`/`backoff` on `MulticaDagClient` (config-driven, with sane defaults).

- [x] **Step 2: Scoped multica regression**

Run: `cd multica/server && DATABASE_URL=... go build ./internal/handler/ ./internal/service/ ./cmd/migrate/... && go test ./internal/service/ ./internal/handler/ -run 'InteractionDAG|MaybeOpen|MaybeRetry|CreateRetry|GetDag' -v && gofmt -l server/internal/`
Expected: green (pre-existing webpush/ON CONFLICT failures excluded). Fix any `gofmt -l` output.

- [x] **Step 3: Areal regression**

Run: `cd /workspaces/leagent/backend/areal && .venv-test/bin/python -m pytest areal/v2/inference_service/tests/ customized_areal/tree_search/tests/ -k 'segment_dag or supernode or env_dispatch' -v && uvx ruff check customized_areal/tree_search/ areal/v2/inference_service/`
Expected: green (pre-existing critic/datasets/torchdata failures excluded).

- [x] **Step 4: Cross-repo E2E (if feasible)**

3-agent `mode=scratch` rollout -> `close_segment` + export per event -> poll `GET .../dag` (202 -> 200) -> AReaL resolves multi-shard tensor_refs -> `ExecutionDAG` -> minimal training -> cleanup. If hardware/services unavailable, document the skip in the commit body.

- [x] **Step 5: F-independence + grep sweep**

Verify env snapshots are refs-only (no sandbox pause/fork). Grep: `close_segment`, `AssembledDag`, `tensor_ref`, `v2-segment-dag-recording`, `env-dispatch/{projectID}/dag` resolve to intended code only; no `start_turn_idx`/`end_turn_idx` in new tables/code.

Run: `cd multica && grep -rn "start_turn_idx\|end_turn_idx" server/ ; cd /workspaces/leagent/backend/areal && grep -rn "tensor_ref" customized_areal/tree_search/ areal/v2/`
Expected: no `start_turn_idx`/`end_turn_idx`; `tensor_ref` references consistent with the field->shard map.

- [x] **Step 6: Final whole-branch review + commit**

Review both branches: `git log --oneline f60c86bb..HEAD` (areal) and `git log --oneline 157284045..HEAD` (multica). Mark READY TO MERGE / NEEDS_CHANGES.

```bash
cd /workspaces/leagent/backend/areal && git add -A && git commit -m "docs(v2-segment-dag-recording): U10 config + regression + E2E + grep (Task 10)"
```

---

## Self-Review

- **Spec coverage**: Trained-rollout recording gating (Task 4/5); retry-attempt session lifecycle D9 (Task 6/7); session-to-agent-run recording D10 (Task 3); AssembledDag assembly (Task 8); /dag 403 (Task 9); /dag failed-status (Task 9); multi-shard tensor_ref contract (Task 1/2). All spec requirements covered.
- **Placeholders**: none; helper functions (`resolveIssueID`, `arealProxyKeyFromContext`, `leanEnvSnapshot`, `denseCover`) are defined inline in their tasks.
- **Type consistency**: `decodeTensorRef` -> field->`{shard_id,node_addr}` map (Task 1) consumed by `DataProxyTensorResolver.resolve` (Task 2); `AssembledDag`/`AssembledSegment`/`AssembledEdge` (Task 8) consumed by `GetDag` (Task 9); `RecordSessionAgentRun(ctx, projectID, sessionID, agentRunID, issueID)` (Task 3) matches U7.1's 4-param signature.
- **Open impl items** (from design doc): exact `areal_proxy` strip mechanism chosen (post-insert `UPDATE context - 'areal_proxy'`, Task 6); `issue_snapshot_id` NULL (Task 4 `leanEnvSnapshot`); U10 E2E hardware-gated skip documented.
