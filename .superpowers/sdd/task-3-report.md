# Task 3 Fix Report

## Status
DONE

## Files Changed
- `server/internal/service/diagnosis_tools.go`: Implemented GetTaskContext, added per-turn budget, updated comments on budget constants
- `server/internal/service/diagnosis_tools_test.go`: Updated TestGetTaskContext with 4 subtests
- `server/internal/service/interaction_dag.go`: Added GetIssueForTask to MessageStore interface, added compile check for MessageStore
- `server/pkg/db/generated/issue.sql.go`: Added GetIssueForTask hand-written query

## GetIssueForTask Query
```sql
SELECT i.id, i.workspace_id, i.title, i.description, i.status, i.priority, i.assignee_type, i.assignee_id, i.creator_type, i.creator_id, i.parent_issue_id, i.acceptance_criteria, i.context_refs, i.position, i.due_date, i.created_at, i.updated_at, i.number, i.project_id, i.origin_type, i.origin_id, i.first_executed_at, i.start_date, i.metadata, i.forked_from_issue_id, i.forked_at_seq, i.forked_at_task_id
FROM issue i
JOIN agent_task_queue atq ON atq.issue_id = i.id
WHERE atq.id::text = $1::text
```

## TaskContext Goal/Gold Mapping
- **Goal**: Uses issue.Description if present and non-empty; otherwise uses issue.Title
- **GoldContext**: Uses issue.AcceptanceCriteria (converted to string) if present; otherwise empty string

## Workspace-Scoping Approach
1. Retrieve issue via GetIssueForTask
2. Check if issue.WorkspaceID matches the requested workspaceID
3. If not, return pgx.ErrNoRows

## Test Additions
- TestGetTaskContext now includes:
  - Returns task context with description as goal and acceptance_criteria as gold
  - Returns task context with title as goal when description is empty
  - Returns pgx.ErrNoRows for cross-workspace access
  - Returns error when task not found

## Commit Hash
1392ac75ef2b443266f725b731b37b523fd02a2e

## Test Summary
- TestGetSegmentMessages: 3/3 pass
- TestGetInteractionDAG: 1/1 pass
- TestGetTaskContext: 4/4 pass
- go vet: clean
- go build: clean

## Concerns
- The per-turn budget is implemented but not tested due to test setup constraints (segment.EndSeq was hardcoded to 5 in the test data)
