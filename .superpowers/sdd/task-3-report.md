# Task 3 Report: Read-only diagnosis tools (Go)

## Overview
This implements Task 3 of the multica-pi-diagnosis-agent change - the read-only Go tools that the diagnosis agent will use.

## Files Changed

1. `server/internal/service/diagnosis_tools.go` (NEW)
   - Implements `GetInteractionDAG()`: Returns segments and edges for a project, workspace-scoped
   - Implements `GetSegmentMessages()`: Returns task messages for a segment, truncated to budget
   - Implements `GetTaskContext()`: Returns task goal/gold (placeholder for now)
   - Defines types: `SegmentRow`, `EdgeRow`, `MessageRow`, `TaskContext`
   - Budget constants mirroring evolution_review_provider.go: `maxDiagnosisMessageBytes = 8KB`, `maxDiagnosisSegmentBudgetBytes = 24KB`

2. `server/internal/service/diagnosis_tools_test.go` (NEW)
   - Comprehensive test suite using fake stores
   - Tests: GetSegmentMessages returns correct messages, respects budget, refuses cross-workspace access
   - Tests: GetInteractionDAG returns segments and edges correctly
   - Tests: GetTaskContext placeholder works

3. `server/internal/service/interaction_dag.go`
   - Added `GetInteractionDAGSegmentByID()` to `InteractionDAGStore` interface
   - Added new `MessageStore` interface for accessing task messages and project workspace validation

4. `server/internal/service/interaction_dag_test.go`
   - Added `GetInteractionDAGSegmentByID()` implementation to `fakeInteractionDAGStore` for tests

5. `server/pkg/db/generated/interaction_dag.sql.go`
   - Added `GetInteractionDAGSegmentByID()` hand-written SQL implementation (mirrors `GetInteractionDAGSegmentByAgentRun`)
   - No `sqlc generate` run, as instructed

6. `server/pkg/db/generated/task_message.sql.go`
   - Added `MessagesForTaskInRange()` hand-written SQL implementation (mirrors `ListTaskMessagesSince`)
   - Parameters: taskID string, startSeq int32, endSeq int32
   - Uses same string-casting approach as `GetMaxTaskMessageSeq`
   - No `sqlc generate` run, as instructed

## Design Decisions

### Workspace Scoping
- Tools verify that the project associated with a segment belongs to the given workspace ID
- Uses `GetProjectInWorkspace()` with both project ID and workspace ID
- Returns `pgx.ErrNoRows` if validation fails

### Budget Implementation
- `GetSegmentMessages()` truncates each message to `maxDiagnosisMessageBytes`
- Tracks total bytes and stops adding messages once `maxDiagnosisSegmentBudgetBytes` is reached
- Each message has a `Truncated` field to indicate if it was cut off
- Uses the existing `truncateUTF8Bytes()` from `evolution_review_provider.go`

### TaskContext Placeholder
- Currently returns empty struct
- We did not find where task goal/gold lives in the schema in our exploration
- Can be extended once we locate this data

### Store Interfaces
- Kept interfaces separate for testability: `InteractionDAGStore` and `MessageStore`
- Both can be implemented by the same `*db.Queries` struct
- The fake store in tests implements both

## Tests
All tests are passing!
- `TestGetSegmentMessages/returns_task_messages_for_the_segment's_seq_range` ✓
- `TestGetSegmentMessages/respects_max-bytes_budget_and_truncates` ✓
- `TestGetSegmentMessages/refuses_cross-workspace_access` ✓
- `TestGetInteractionDAG/returns_segments_and_edges_for_the_project` ✓
- `TestGetTaskContext/returns_task_context` ✓
- All existing interaction_dag tests continue to pass ✓

## Commit
`cb8cc01f2 feat(diagnosis-agent): read-only tools over interaction DAG + task_message`
