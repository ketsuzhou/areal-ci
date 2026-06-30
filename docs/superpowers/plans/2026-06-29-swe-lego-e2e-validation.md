# SWE-Lego × Multica × AReaL — End-to-End Validation Record

**Date:** 2026-06-30
**Plan:** `docs/superpowers/plans/2026-06-29-swe-lego-multica-areal-integration.md`
**Task 17 status:** SKIPPED (hardware unavailable) — unit tests (Tasks 1-16) are the v1 acceptance gate.

## Reason for skip

Per Task 17 Step 1 of the plan: the e2e validation requires (a) a GPU on the areal side and (b) a Fleet build-node tagged `swe-lego-build` reachable from the multica server. Neither is available in this development environment:

```
$ python -c "import torch; print('GPU available:', torch.cuda.is_available())"
GPU available: False
```

Per `backend/areal/CLAUDE.md`: "Integration tests requiring multi-node hardware are skipped with an explanation when unavailable."

## v1 acceptance gate (unit tests, all green)

### areal side (Python) — 15 tests

```
customized_areal/tree_search/tests/test_swe_lego_verifier.py            4 passed
customized_areal/tree_search/tests/test_dag_backup.py                   3 passed
customized_areal/tree_search/tests/test_advantage_per_node_credit.py    2 passed
customized_areal/tree_search/tests/test_swe_lego_issue_runner.py        4 passed
customized_areal/tree_search/tests/test_integration_multica.py          1 passed
customized_areal/tree_search/tests/test_swe_lego_anti_hacking.py         1 passed
                                                                     15 passed
```

### multica side (Go) — 14 tests

```
internal/handler  TestCreateSweLegoIssue_*                              5 passed
internal/service  TestSweLegoCacheKey*                                  3 passed
internal/service  TestSweLegoBuildScript_*                              3 passed
internal/service  TestSweLegoDockerfileTemplate                         1 passed
internal/service  TestSweLegoIssueService_*                            3 passed
                                                                      15 passed
```

(One overlap in counting: `TestCreateSweLegoIssue_*` is 5 tests in the handler package; the service package has 10 tests. Total Go: 15. Total across both sides: 30.)

## What the e2e would verify (deferred to hardware-equipped environment)

When a GPU + Fleet build-node + real multica stack is available, run Task 17 Steps 2-4:

1. **Step 2 — e2e at group_size=2**: `POST /api/v1/swe-lego/issues` returns 201 with `agent_run_ids` of length 2; build node's docker cache contains `swe-lego:<cache_key>`; two daemon-in-docker sandboxes boot; agents POST `task_message` batches; verifier produces non-`None` reward per agent; `cleanup_swe_lego_issue` deletes project + sandboxes.
2. **Step 3 — anti-cheating canary live**: `docker exec <forked-sandbox> git -C /workspace/repo log --after=<issue_date>` returns empty (confirms `git filter-repo --commit-cutoff` deleted future history).
3. **Step 4 — document the run**: record repo, base_commit, group_size, rewards, any failures. File issues as follow-up tasks.

## Deferred wiring (Task 10 → Task 17)

Task 10 (wire service deps to real queries + cloud runtime) was deferred to the e2e task because five fundamental issues prevented a clean implementation without the real stack:

1. **WorkspaceID threading** — the `sweLegoDepsAdapter` has no request context to extract `workspace_id` from.
2. **Agent runtime binding** — forking a sandbox doesn't bind it to the agent's runtime; the daemon-in-docker boot path needs a real cloud-runtime endpoint.
3. **No boot-sandbox endpoint** in the cloud-runtime proxy — the existing `cloudruntime.Request` interface has snapshot/fork/restore/cleanup but no "boot base sandbox" op.
4. **Nil handler test breaks** — the adapter's real query calls require a non-nil `*Handler.Queries` which the test harness doesn't provide.
5. **Plan's test is `t.Skip`** — the plan's Task 10 test explicitly skips without a real DB.

Real API signatures for the Task 17 implementer (documented in the plan at Task 10's deferral notes):

- `CreateProject(ctx, CreateProjectParams{WorkspaceID, Title, Status, Priority, ...})` — `project.sql.go:46`
- `DeleteProject(ctx, DeleteProjectParams{ID, WorkspaceID})` — `project.sql.go:84` (tenant guard)
- `CreateIssue(ctx, CreateIssueParams{...})` — `issue.sql.go:201`
- `SetIssueMetadataKey(ctx, SetIssueMetadataKeyParams{Key, Value []byte, ID, WorkspaceID})` — `issue.sql.go:1115`
- `TaskService.EnqueueTaskForIssue(ctx, issue db.Issue, ...)` — `task.go:432`
- `cloudruntime.Request{Method, Path, Query, Body, UserID, RequestID, Op, Headers}` → `Response{StatusCode, Header, Body}` — `client.go:42,69`

## Follow-up tasks (filed when e2e runs)

- Wire `sweLegoDepsAdapter` to real sqlc queries + cloud-runtime proxy (un-blocks Task 10's real implementation).
- Add a "boot base sandbox" op to the cloud-runtime proxy (or confirm the existing fork path suffices for daemon-in-docker boot).
- Thread `workspace_id` from the request context into the adapter.
