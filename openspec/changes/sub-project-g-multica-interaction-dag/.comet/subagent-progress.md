# Subagent Progress Checkpoint — sub-project-g-multica-interaction-dag

- Plan: docs/superpowers/plans/2026-07-07-interaction-dag-return.md
- OpenSpec tasks: openspec/changes/sub-project-g-multica-interaction-dag/tasks.md
- Build mode: subagent-driven-development | TDD: tdd | Isolation: branch-in-both
- areal branch: feature/20260707/sub-project-g-multica-interaction-dag (baseline commit 6bffe9f6)
- multica branch: feature/20260707/sub-project-g-multica-interaction-dag (from dev; 5 pre-existing staged files db_bridge/* + specs.md must NOT be committed by implementers — explicit-file staging only)
- base-ref (areal): 1117158676e77268810fc4f7c5a6652c4be28428

## Current task

- Plan task: "Task 1: Investigation — interaction seams and turn-index source" (plan 1.1–1.8)
- OpenSpec task: "## 1. Investigation — interaction seams and turn-index source" (tasks.md 1.1–1.8)
- Stage: D5 RESOLVED → Option B; build resuming INLINE (subagent-driven timed out: planner 120s, worker 300s)
- Implementer: coordinator (inline investigation; worker subagent timed out at 300s)
- Implementation commit: pending — D5 revision across design.md/spec.md/tasks.md/design-doc/plan (this commit)
- RED/GREEN evidence: N/A (research/design task)
- Review stages passed: —
- Review-fix round: 0/3
- Next: Task 2b (interaction_id surfacing: AReaL proxy → pi message_end → daemon stamp) → Task 2 (migration) → Task 3 (recording, TDD)

## Resolution — D5 → Option B (shared interaction_id) [2026-07-07, user-confirmed]

Verified: Multica does NOT proxy `/chat/completions`; the sandboxed agent
(`pi -p --provider areal`) routes LLM via `db_bridge` → shared Supabase → AReaL
gateway. `task_message.seq` is NOT a turn index — `daemon.go` assigns `seq.Add(1)`
per agent EVENT (text/tool_use/tool_result/thinking/error), so 1 assistant turn =
3+ task_messages. AReaL `turn_idx` (Node) = 1 per `/chat/completions`. seq ≠ turn_idx.

D5 revised to Option B: AReaL returns `interaction_id` (= Node.node_id) in the
/chat/completions response `id`; `pi` surfaces it via its `message_end` event (1:1
per LLM response; today consumed internally — adds an ID field); the daemon stamps
`task_message.interaction_id` (new column); Multica numbers turns per session =
1-based ordinal of interaction_ids in creation order; closing communication event's
interaction_id → its ordinal = end_turn_idx. Alignment by construction (both sides
= 1-based ordinal of LLM responses per session in execution order); SegmentSpec/
SuperNodeAssembler UNCHANGED. interaction_id = optional audit key.

Revised across: design.md (D5 + Risks), spec.md (SHALL req), design doc, tasks.md
(1.1 done, 2.1 migration, new §2b), plan (1.1, 2.3, Task 2b, Task 3 interfaces).
OpenSpec validate --strict: valid. Memory locked.

Build mode: INLINE (executing-plans) — subagent-driven timed out twice on this
large codebase. User confirmed G branch (feature/20260707/sub-project-g-multica-
interaction-dag); F's uncommitted change-dir deletions ride in WT untouched.

## Notes

- Task 1 is research (confirm exact hook seams + turn-index source), not TDD code. Dispatch as a focused scout; deliverable = a findings file under openspec/changes/sub-project-g-multica-interaction-dag/.comet/handoff/ or a short T1 report, committed as `docs(G): T1 interaction-DAG seam confirmation`.
- TDD implementer+review loop begins at Task 2 (migration) / Task 3 (recording service).
