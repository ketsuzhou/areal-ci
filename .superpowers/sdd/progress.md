# SDD progress — env-dispatch over db_bridge

Plan: docs/superpowers/plans/2026-07-02-env-dispatch-over-db-bridge.md
Impl repo: multica/ (branch: main, per user choice b)
Controller: kiro (this session)

## Tasks
- [x] Task 1: add env_dispatch + env_dispatch_delete channels (channels.py + test)
- [x] Task 2: add rpc_env_dispatch + rpc_env_dispatch_delete to schema.sql (+ schema test)
- [x] Task 3: docs (.env.areal.example + README channel table)
- [x] Task 4: full-suite regression + assert AReaL client unchanged

## Ledger
(append "Task N: complete (commits <base7>..<head7>, review clean)" as tasks finish)

Task 1: complete (commits b34f0ac..ec3da95, review clean)
  MINOR (defer to final review): tests/test_env_dispatch_channels.py has ruff I001
  unsorted-imports on the `from _fakes import ...` group — matches committed sibling
  test_leagent_channels.py convention; not fixed to stay consistent.
Task 2: complete (commits ec3da95..1376138, review clean)
Task 3: complete (commit 392295a, review clean; README table re-padded for alignment, cells verbatim)
Task 4: complete (verification only; full db_bridge suite 169 passed/1 skipped; AReaL client unchanged)

FINAL REVIEW: READY TO MERGE (b34f0ac..392295a on multica main). No Critical/Important.
  Minor follow-ups (non-blocking): (1) ruff I001 in test_env_dispatch_channels.py (matches
  sibling convention; repo has no ruff gate); (2) add a true end-to-end AReaL-client-via-stub
  round-trip test when sub-projects B-E land (design §8).
SUB-PROJECT A COMPLETE.

# SDD progress — env-dispatch feature params (sub-project B)

Plan: docs/superpowers/plans/2026-07-02-env-dispatch-feature-params.md
Spec: docs/superpowers/specs/2026-07-02-env-dispatch-feature-params-design.md
Impl repo: multica/ (branch: main, per user choice b) + areal working tree (AReaL client, Task 6)
Controller: kiro (this session), subagent-driven
Env: Postgres reachable @ localhost:5432 (DATABASE_URL in multica/.env, db=multica); sqlc v1.31.1;
  go 1.26.1. DB-backed handler tests need `cd multica && make migrate-up` first, then
  `cd multica/server && go test ./internal/handler/...` (TestMain reads DATABASE_URL, connects,
  builds testHandler/testPool, sets testWorkspaceID/testUserID). All tasks runnable here — no deferral.

## Tasks
- [ ] Task 1: migration 141 workspace.default_self_play_env_id (+ DB column test)
- [ ] Task 2: service resume-normalize, exactly-one agent/squad, GetDefaultSelfPlayEnv dep, default-env resolve (fake-deps tests)
- [ ] Task 3: handler squad_id field + relaxed env_id/agent_id gates + GetDefaultSelfPlayEnv query/adapter (validation tests)
- [ ] Task 4: issue-path squad (SetIssueAssignee, assignee=squad + is_leader_task) (DB test)
- [ ] Task 5: chat-path squad (CreateChatTask context param + daemon briefing injection) (DB tests)
- [ ] Task 6: AReaL client optional env_id/agent_id/squad_id + resume (client tests)
- [ ] Task 7: full-suite regression + sqlc no-drift

## Ledger
(append "Task N: complete (commits <base7>..<head7>, review clean)" as tasks finish)


Task 1: complete (392295a..a136364 on multica main, review clean; TestWorkspaceDefaultSelfPlayEnvColumn PASS)
  CAVEAT (pre-existing, NOT ours): `go build ./...` fails at internal/service/webpush/webpush.go:180
  "constant 4096 overflows byte" (go 1.26 stricter). Introduced by b9e79406, unrelated to B.
  => Task builds should target specific packages (./internal/service/ ./internal/handler/ ./cmd/migrate/...)
  not `./...`; Task 7 must note this pre-existing failure rather than treat it as a regression.
  Migrator run via `go run ./cmd/migrate up` (docker/make absent) instead of `make migrate-up`.
Task 2: complete (a136364..eeb62f6 on multica main, review clean; service suite PASS, 2 Redis SKIP pre-existing)
  Built/tested scoped to ./internal/service/ to avoid the pre-existing webpush build failure.
Task 3: complete (eeb62f6..<head> on multica main, review clean; TestEnvDispatch_ 8/8 PASS incl 3 new)
  5 files: env_dispatch.go, env_dispatch_test.go, queries/workspace.sql, generated/workspace.sql.go, generated/models.go.
  GetDefaultSelfPlayEnv generated type: pgtype.UUID; adapter returns "" on !Valid.
  BLOCKER (pre-existing, NOT ours) surfaced: `cd server && sqlc generate` FAILS at
  queries/user.sql:31 "mixed positional and named parameters" (sqlc v1.31.1; user.sql last
  changed by 88ed91b). Implementer regenerated via a temp workaround (comment query, regen,
  restore). This blocks the clean-regen workflow for Tasks 4 & 5 (SetIssueAssignee, CreateChatTask
  context) and Task 7's no-drift gate. INVESTIGATING root cause before Task 4.

SQLC RESOLUTION (user chose "fix user.sql properly"): partial. Committed f0aa697 making
  UpdateUser fully positional ($2/$5 + $6::text) — this matches the committed generated code
  (which was already positional) and fixes a real source↔generated inconsistency (source had
  sqlc.narg, generated had positional; they disagreed).
  BUT DEEPER FINDING: a clean `cd server && sqlc generate` is NOT runnable in this repo. It
  rewrites 6 generated files, creates agent_skill_suggestion.sql.go + evolution.sql.go that
  REDECLARE symbols in hand-maintained *_manual.sql.go companions => BUILD FAILS. The repo's
  generated/ is partially hand-curated and diverges from sqlc v1.31.1 output. Reverted that regen.
  => REVISED APPROACH for Tasks 4 & 5: DO NOT run `sqlc generate`. Add the query to
  queries/*.sql (source-of-truth) AND hand-write the corresponding generated Go (function +
  Params struct + any models.go field) by appending to the existing generated/<file>.sql.go,
  mirroring sqlc's exact style for a sibling query in the same file. Verify via `go build` +
  the DB-backed test. Task 3's workspace.sql.go landed this way and builds/passes.
  => Task 7 no-drift gate is REPLACED by: `go build ./internal/handler/ ./internal/service/
  ./cmd/migrate/...` + full handler/service test suites (webpush ./... failure remains pre-existing).
Task 4: complete (f0aa697..86753f1 on multica main, review clean; TestEnqueueAgentRun_IssueSquad PASS)
  Hand-wrote SetIssueAssignee in generated/issue.sql.go (mirrored DeleteIssue :exec sibling). 4 files,
  no sqlc-generate churn. Adapter issue+squad branch: GetSquadInWorkspace -> SetIssueAssignee(squad) ->
  enqueue leader (squad.LeaderID) with IsLeaderTask=true; squadID=="" path unchanged (added empty-guard
  to avoid parseUUID("") panic).
Task 5: complete (86753f1..bcbe0fc on multica main, review clean; ChatSquad hint + daemon briefing tests 2/2 PASS)
  chat.sql CreateChatTask writes context via sqlc.narg('context'); hand-edited generated/chat.sql.go
  ($7 context, ForceFreshSession stays $6). daemon.go chat-path (task.ChatSessionID.Valid) injects
  buildSquadLeaderBriefing when task.Context.squad_id set & claiming agent==leader. Daemon test drives
  real ClaimTaskByRuntime via claimTaskByRuntimeForTest, asserts "Squad Operating Protocol" in claim body. 5 files.
Task 6: complete (areal master b3d646f, review clean; test_env_dispatch_client.py 6 passed)
  swe_lego_client.create_env_dispatch: env_id/agent_id now optional (None), squad_id added; payload
  includes env_id/agent_id/squad_id only when truthy; mode forwarded verbatim (resume). 2 files.
  CAVEAT: areal-root `uv run` is broken (stale .venv symlink -> /dfs/share-groups/.../AReaL-main/.venv,
  a foreign clone path). Use `python -m pytest` for areal tree_search tests. (db_bridge's own venv/uv is fine.)
Task 7: complete (verification only). Results:
  - go build ./pkg/db/generated/ = OK; go build+vet ./internal/service/ ./internal/handler/ ./cmd/migrate/... = OK.
  - service suite: ok (all pass).
  - handler suite: 798 PASS / 16 FAIL. ALL 16 failures are the PRE-EXISTING runtime-registration
    "ON CONFLICT ... no unique/exclusion constraint (42P10)" cluster (TestDaemonRegister_*/TestClaimTask_*),
    proven pre-existing by checking out base 392295a and reproducing the identical 16 failures there.
    NOT regressions from B (test-DB schema missing a runtime unique constraint; env/migration-state issue).
  - All B tests PASS: TestWorkspaceDefaultSelfPlayEnvColumn, TestEnvDispatch_* (incl RejectsBothAgentAndSquad,
    AcceptsEmptyEnvIDShape, AcceptsResumeMode), TestEnqueueAgentRun_IssueSquad, TestEnqueueAgentRun_ChatSquad,
    TestClaimTaskByRuntime_ChatSquad_InjectsLeaderBriefing.
  - AReaL client: test_env_dispatch_client.py 6 passed (python -m pytest; uv broken per Task 6 caveat).

ALL TASKS COMPLETE. multica main: 392295a..bcbe0fce (8 commits incl f0aa697 sqlc fix). areal master: b3d646f (client).
Awaiting final whole-branch review + merge decision.

FINAL REVIEW: READY TO MERGE. No Critical/Important. Spec-complete vs D1-D6; consistent across all
  dependency seams; no generated-file drift; squad/leader resolution scoped to workspace; build +
  service tests + targeted handler DB tests + gofmt/vet + areal client suite all green.
  Minor (non-blocking): (1) commit-count wording (6 feature commits + user.sql fix, no merges);
  (2) SquadID uses omitempty while AgentID doesn't (cosmetic); (3) adapter issue branch has a
  defensive zero-UUID guard made unreachable by validate()'s exactly-one rule (defense-in-depth).
SUB-PROJECT B COMPLETE. Commits local-only (not pushed): multica main 392295a..bcbe0fce,
  areal master ...b3d646f. Awaiting user decision on push.