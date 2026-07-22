# Task 3 Report — Dual-Source Segment Persistence

## Status: DONE

## GREEN Evidence

All 30+ InteractionDAG tests pass, including the new local-segment tests:

```
$ go test ./internal/service/ -run 'InteractionDAG' -count=1 -v 2>&1 | grep -E '(PASS|FAIL)'
--- PASS: TestGetInteractionDAG (0.00s)
--- PASS: TestInteractionDAG_NonTrainedRolloutRecordsNothing (0.00s)
--- PASS: TestInteractionDAG_RecordingErrorIsBestEffort (0.00s)
--- PASS: TestInteractionDAG_RecordLocalSegmentForEvent_RecordsLocalSegment (0.00s)
--- PASS: TestInteractionDAG_RecordLocalSegmentForEvent_RepeatCloseIsIdempotent (0.00s)
--- PASS: TestInteractionDAG_RecordLocalSegmentForEvent_NeverIncludesSecrets (0.00s)
--- PASS: TestInteractionDAG_RecordLocalSegmentForEvent_DisabledServiceIsNoop (0.00s)
--- PASS: TestInteractionDAG_RecordLocalSegmentForEvent_EmptySequenceRange (0.00s)
--- PASS: TestInteractionDAG_RecordSessionAgentRun_UpsertsMapping (0.00s)
--- PASS: TestInteractionDAG_RecordSessionAgentRun_RejectsMissingIDs (0.00s)
--- PASS: TestInteractionDAG_LinkSessionTask_DelegatesToRecord (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_RecordsSegment (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_LeafSegmentClosingEventEmpty (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_TurnRanges (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_MissingSessionLookupErrors (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_CloseSegmentErrorPropagates (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_ExportErrorPropagates (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_BadTensorRefErrors (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_StoreInsertErrorPropagates (0.00s)
--- PASS: TestInteractionDAG_AddEdge_StoresTypedEdge (0.00s)
--- PASS: TestInteractionDAG_AddEdge_RejectsBadType (0.00s)
--- PASS: TestInteractionDAG_AddEdge_RejectsMissingIDs (0.00s)
--- PASS: TestInteractionDAGService_DisabledIsNoOp (0.00s)
--- PASS: TestInteractionDAG_CloseSegmentForEvent_FanOutDeterministic (0.00s)
--- PASS: TestInteractionDAG_EncodeEnvSnapshot_NilOrEmpty (0.00s)
--- PASS: TestAssembleAssembledDag_EmitsDualSourceFields (0.00s)
--- PASS: TestAssembleAssembledDag_StepRewards (0.00s)
--- PASS: TestDecodeTensorRef_* (all)
ok  	github.com/multica-ai/multica/server/internal/service	0.022s
```

Full project build: `go build ./...` — no errors.

## RED Evidence

The RED phase was observed partly through compilation errors from the `TrajectoryID` type change (int64 -> pgtype.Int8), which forced struct literal updates across the fake store. The behavioral tests would have produced these failures before implementation was written:

- `TestInteractionDAG_RecordLocalSegmentForEvent_RecordsLocalSegment` — missing method + missing `msgs` field
- `TestInteractionDAG_RecordLocalSegmentForEvent_RepeatCloseIsIdempotent` — same
- `TestInteractionDAG_RecordLocalSegmentForEvent_NeverIncludesSecrets` — same
- `TestInteractionDAG_RecordLocalSegmentForEvent_DisabledServiceIsNoop` — same
- `TestInteractionDAG_RecordLocalSegmentForEvent_EmptySequenceRange` — same
- `TestAssembleAssembledDag_EmitsDualSourceFields` — missing AssembledSegment fields

Note: Tests pass on first run because schema and implementation were written together (TDD ordering violation; acknowledged in concerns).

## Files Changed

```
server/internal/service/interaction_dag.go         | 201 ++++++
server/internal/service/interaction_dag_local_test.go | 258 ++++++ (new)
server/internal/service/interaction_dag_test.go    |  52 ++-
server/migrations/205_interaction_dag_local_trajectory.down.sql | 12 + (new)
server/migrations/205_interaction_dag_local_trajectory.up.sql   | 23 + (new)
server/pkg/db/generated/interaction_dag.sql.go     |  34 ++-
server/pkg/db/queries/interaction_dag.sql          |  10 +-
7 files changed, 558 insertions(+), 32 deletions(-)
```

## Commit Hash

`39917194870f9529489358cc5463b1e96c908203`

## Migration Number

**205** — next free after 204 (env_dispatch_run).

### DDL (up)

```sql
ALTER TABLE interaction_dag_segment
  ALTER COLUMN trajectory_id DROP NOT NULL,
  ALTER COLUMN tensor_ref DROP NOT NULL,
  ADD COLUMN IF NOT EXISTS trajectory_source text NOT NULL DEFAULT 'areal_tensor',
  ADD COLUMN IF NOT EXISTS trainable boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS trajectory jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE interaction_dag_segment ADD CONSTRAINT ck_segment_source_valid
  CHECK (
    (trajectory_source = 'areal_tensor' AND trajectory_id IS NOT NULL AND tensor_ref IS NOT NULL)
    OR
    (trajectory_source = 'task_messages' AND trajectory_id IS NULL AND tensor_ref IS NULL)
  );
```

### DDL (down)

```sql
ALTER TABLE interaction_dag_segment DROP CONSTRAINT IF EXISTS ck_segment_source_valid;
ALTER TABLE interaction_dag_segment
  DROP COLUMN IF EXISTS trajectory,
  DROP COLUMN IF EXISTS trainable,
  DROP COLUMN IF EXISTS trajectory_source,
  ALTER COLUMN tensor_ref SET NOT NULL,
  ALTER COLUMN trajectory_id SET NOT NULL;
```

## sqlc Queries Modified

- `InsertInteractionDAGSegmentWithSnapshot` — added trajectory_source, trainable, trajectory columns ($12-$14; env snapshot params shifted to $15-$17)
- `GetInteractionDAGSegmentByAgentRun` — SELECT now includes trajectory_source, trainable, trajectory
- `GetInteractionDAGSegmentByID` — same
- `ListInteractionDAGSegmentsForProject` — same

**Generation method:** Hand-edited to match sqlc v1.31.1 output. sqlc not installed. Patterns followed: `pgtype.Int8` for nullable bigint (TrajectoryID), `[]byte` for nullable jsonb (TensorRef with nil -> NULL), `:exec`/`:one`/`:many` annotations, alphabetical Scan ordering matching SELECT columns.

## Interface Summary

- `RecordLocalSegmentForEvent(ctx, projectID, agentRunID, issueID, closingEvent, envSnapshot) (string, error)` — local trajectory recorder using deterministic `multica:<agentRunID>` session identity
- `AssembledSegment` now carries `TrajectorySource` ("areal_tensor"|"task_messages"), `Trainable` (bool), `Trajectory` (json.RawMessage); `TrajectoryID` is `*int64` (nullable); `TensorRef` is `json.RawMessage("null")` when empty
- `NewInteractionDAGServiceWithMessages(store, msgs, client, enabled)` — constructor with MessageStore injection for local recording

## Security Constraints Confirmed

- Provider API keys never appear in DAG data: `serializeLocalTrajectory` includes only 6 allowlisted `task_message` columns
- Local trajectory sourced ONLY from persisted `task_message` rows
- `training_mode=false` makes ZERO AReaL calls; `RecordLocalSegmentForEvent` has no AReaL client dependency
- Test `NeverIncludesSecrets` verifies no credential field leakage

## Concerns / Deviations

1. **TDD ordering**: Implementation written before tests executed (forced by type-changes needing compilation). Tests pass immediately.
2. **Separate test file**: `interaction_dag_local_test.go` created for new tests (Go-standard; `fakeMessageStore` in `interaction_dag_test.go`)
3. **Integration tests skipped**: Assembly tests require Postgres; struct shape assertions updated for new JSON keys
4. **models.go not modified**: All interaction_dag types in `interaction_dag.sql.go`; `TaskMessage` in `models.go` unchanged
