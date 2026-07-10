## 1. U7.2 - Wire recorder into the trained-rollout path (TDD)

**Files:** `server/internal/service/training.go`, `server/internal/service/task.go`,
`server/internal/service/interaction_dag.go`, integration tests under
`server/internal/service/`.

- [ ] 1.1 Failing tests: `RecordSessionAgentRun` fires inside `maybeOpenTrainingSession`
  immediately after `StartSession` succeeds (D10), recording `{projectID, sessionID,
  agentRunID=taskID}`; idempotent across repeated open attempts.
- [ ] 1.2 Failing tests: at a delegation event the parent run calls `close_segment` + export,
  records a segment with `trajectory_id` + `tensor_ref` + `closing_event`, opens a child
  segment, and records a `delegation` edge.
- [ ] 1.3 Failing tests: a mention records a `mention` edge without closing a segment.
- [ ] 1.4 Failing tests: a completion calls `close_segment` + export on the child run and
  records a `completion` edge.
- [ ] 1.5 Failing tests: a squad-context handoff closes the producer/parent segment via
  `close_segment` + export with `closing_event = "squad_briefing"` while the structural edge
  remains `delegation`.
- [ ] 1.6 Failing tests: a leaf run (no communication event) yields exactly one leaf segment
  with `closing_event = None`.
- [ ] 1.7 Failing tests: concurrent fan-out delegation produces multiple `delegation` edges and
  a deterministic, acyclic segment set.
- [ ] 1.8 Failing tests: recording is gated to trained rollouts (`INTERACTION_DAG_ENABLED` AND
  `s.Training != nil`); a non-trained rollout records nothing and makes no `close_segment`/
  export calls.
- [ ] 1.9 Failing tests: a recording error is logged and the run continues (best-effort).
- [ ] 1.10 Implement the D10 chokepoint + the delegation/mention/completion/squad seams behind
  the flag.
- [ ] 1.11 Commit: `feat(v2-segment-dag-recording): wire recorder into trained-rollout path (U7.2)`.

## 2. U7.3 - Fresh areal RL session per retry (D9) (TDD)

**Files:** `server/internal/service/task.go` (`CreateRetryTask`, `MaybeRetryFailedTask`),
`server/internal/service/training.go`, tests.

- [ ] 2.1 Failing tests: `CreateRetryTask` strips `areal_proxy` from the child context while
  keeping the chat `session_id`/`work_dir` resume CASE-WHEN.
- [ ] 2.2 Failing tests: `MaybeRetryFailedTask` calls `tryOpenTrainingSession(child,
  projectID, envID)` BEFORE `NotifyTaskEnqueued` (mirror `enqueueMentionTask` ordering).
- [ ] 2.3 Failing tests: the child opens its own session (`StartSession` fires) and the child's
  `agent_run_id` (= child `task.ID`) is recorded via D10.
- [ ] 2.4 Failing tests: the parent session is closed (`EndSession`) before the child opens.
- [ ] 2.5 Failing tests: a non-retryable failure is terminal (session closed, no child).
- [ ] 2.6 Implement the retry-session lifecycle (D9).
- [ ] 2.7 Commit: `feat(v2-segment-dag-recording): fresh areal RL session per retry (U7.3, D9)`.

## 3. U8 - AssembledDag assembly + /dag endpoint (TDD)

**Files:** `server/internal/service/interaction_dag.go`,
`server/internal/handler/env_dispatch.go`, `server/internal/handler/env_dispatch_test.go`,
hand-written sqlc for `AssembleAssembledDag`.

- [ ] 3.1 Failing tests: `AssembleAssembledDag(project_id)` returns `segments`, `edges`,
  `session_to_agent_run` with each segment carrying `trajectory_id` + `tensor_ref` +
  `closing_event` + `env_snapshot` (no scores, no turn indices, no text).
- [ ] 3.2 Failing tests: edges carry `src` / `dst` / `type` = `delegation` / `mention` /
  `completion`; the assembled DAG is acyclic.
- [ ] 3.3 Failing tests: `GET /api/v1/env-dispatch/{projectID}/dag` returns `202` in progress,
  `200` + `AssembledDag` done, `404` unknown project, `403` cross-workspace.
- [ ] 3.4 Failing tests: an incomplete/failed rollout yields a `failed` status, not a partial
  `AssembledDag`; a densely-covered failed run returns `200` + `AssembledDag`.
- [ ] 3.5 Implement the read-only assembly from recorded rows + the polling handler.
- [ ] 3.6 Commit: `feat(v2-segment-dag-recording): assemble AssembledDag + /dag endpoint (U8)`.

## 4. U10 - Config + E2E + grep sweep (both repos)

- [ ] 4.1 Add `INTERACTION_DAG_ENABLED` default (on for trained rollouts) + areal polling
  config (interval, timeout, backoff) for `MulticaDagClient`.
- [ ] 4.2 Scoped multica Go build/test/vet for touched packages; `gofmt -l` clean.
- [ ] 4.3 AReaL: `.venv-test/bin/python -m pytest areal/v2/inference_service/tests/` and
  `customized_areal/tree_search/tests/ -k 'segment_dag or supernode or env_dispatch'`.
- [ ] 4.4 Cross-repo E2E if feasible: 3-agent `mode=scratch` rollout -> `close_segment` +
  export per event -> poll `GET .../dag` -> AReaL resolves refs and reconstructs the
  `ExecutionDAG` losslessly -> minimal training step -> cleanup.
- [ ] 4.5 Verify env snapshots are refs-only (no sandbox pause/fork) - F-independence.
- [ ] 4.6 grep sweep: `close_segment`, `AssembledDag`, `tensor_ref`, `v2-segment-dag-recording`,
  `env-dispatch/{projectID}/dag` resolve to intended code only; no `start_turn_idx` /
  `end_turn_idx` remain in the new tables/code.
- [ ] 4.7 Final whole-branch review -> READY TO MERGE / NEEDS_CHANGES.
- [ ] 4.8 Commit: `docs(v2-segment-dag-recording): U10 full regression + E2E + grep sweep`.

## Test runners / constraints

- AReaL tests run from `backend/areal` with `.venv-test/bin/python -m pytest <path>` (project
  `uv`/`.venv` is broken; do NOT use `uv run pytest`). Lint with `uvx ruff check <paths>`.
- Multica Go tests are scoped to touched packages; avoid repo-wide `go build ./...` (pre-existing
  `webpush.go:180` failure) and the 16 pre-existing handler `ON CONFLICT` daemon/claim failures.
- `sqlc generate` is broken in the multica repo; hand-write generated Go mirroring a sibling
  query, as in U7.1.
- Cross-repo E2E requires Multica services + the v2 inference_service; if unavailable, document
  the skipped prerequisites.
- This change uses a placeholder reward; real judge / V / GAE arrive in change 2 - do not
  implement them here.
