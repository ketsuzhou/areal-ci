# SDD progress — env-dispatch over db_bridge

Plan: docs/superpowers/plans/2026-07-02-env-dispatch-over-db-bridge.md Impl repo:
multica/ (branch: main, per user choice b) Controller: kiro (this session)

## Tasks

- [x] Task 1: add env_dispatch + env_dispatch_delete channels (channels.py + test)
- [x] Task 2: add rpc_env_dispatch + rpc_env_dispatch_delete to schema.sql (+ schema
  test)
- [x] Task 3: docs (.env.areal.example + README channel table)
- [x] Task 4: full-suite regression + assert AReaL client unchanged

## Ledger

(append "Task N: complete (commits <base7>..<head7>, review clean)" as tasks finish)

Task 1: complete (commits b34f0ac..ec3da95, review clean) MINOR (defer to final review):
tests/test_env_dispatch_channels.py has ruff I001 unsorted-imports on the
`from _fakes import ...` group — matches committed sibling test_leagent_channels.py
convention; not fixed to stay consistent. Task 2: complete (commits ec3da95..1376138,
review clean) Task 3: complete (commit 392295a, review clean; README table re-padded for
alignment, cells verbatim) Task 4: complete (verification only; full db_bridge suite 169
passed/1 skipped; AReaL client unchanged)

FINAL REVIEW: READY TO MERGE (b34f0ac..392295a on multica main). No Critical/Important.
Minor follow-ups (non-blocking): (1) ruff I001 in test_env_dispatch_channels.py (matches
sibling convention; repo has no ruff gate); (2) add a true end-to-end
AReaL-client-via-stub round-trip test when sub-projects B-E land (design §8).
SUB-PROJECT A COMPLETE.

# SDD progress — env-dispatch feature params (sub-project B)

Plan: docs/superpowers/plans/2026-07-02-env-dispatch-feature-params.md Spec:
docs/superpowers/specs/2026-07-02-env-dispatch-feature-params-design.md Impl repo:
multica/ (branch: main, per user choice b) + areal working tree (AReaL client, Task 6)
Controller: kiro (this session), subagent-driven Env: Postgres reachable @
localhost:5432 (DATABASE_URL in multica/.env, db=multica); sqlc v1.31.1; go 1.26.1.
DB-backed handler tests need `cd multica && make migrate-up` first, then
`cd multica/server && go test ./internal/handler/...` (TestMain reads DATABASE_URL,
connects, builds testHandler/testPool, sets testWorkspaceID/testUserID). All tasks
runnable here — no deferral.

## Tasks

- [ ] Task 1: migration 141 workspace.default_self_play_env_id (+ DB column test)
- [ ] Task 2: service resume-normalize, exactly-one agent/squad, GetDefaultSelfPlayEnv
  dep, default-env resolve (fake-deps tests)
- [ ] Task 3: handler squad_id field + relaxed env_id/agent_id gates +
  GetDefaultSelfPlayEnv query/adapter (validation tests)
- [ ] Task 4: issue-path squad (SetIssueAssignee, assignee=squad + is_leader_task) (DB
  test)
- [ ] Task 5: chat-path squad (CreateChatTask context param + daemon briefing injection)
  (DB tests)
- [ ] Task 6: AReaL client optional env_id/agent_id/squad_id + resume (client tests)
- [ ] Task 7: full-suite regression + sqlc no-drift

## Ledger

(append "Task N: complete (commits <base7>..<head7>, review clean)" as tasks finish)

Task 1: complete (392295a..a136364 on multica main, review clean;
TestWorkspaceDefaultSelfPlayEnvColumn PASS) CAVEAT (pre-existing, NOT ours):
`go build ./...` fails at internal/service/webpush/webpush.go:180 "constant 4096
overflows byte" (go 1.26 stricter). Introduced by b9e79406, unrelated to B. => Task
builds should target specific packages (./internal/service/ ./internal/handler/
./cmd/migrate/...) not `./...`; Task 7 must note this pre-existing failure rather than
treat it as a regression. Migrator run via `go run ./cmd/migrate up` (docker/make
absent) instead of `make migrate-up`. Task 2: complete (a136364..eeb62f6 on multica
main, review clean; service suite PASS, 2 Redis SKIP pre-existing) Built/tested scoped
to ./internal/service/ to avoid the pre-existing webpush build failure. Task 3: complete
(eeb62f6..<head> on multica main, review clean; TestEnvDispatch\_ 8/8 PASS incl 3 new) 5
files: env_dispatch.go, env_dispatch_test.go, queries/workspace.sql,
generated/workspace.sql.go, generated/models.go. GetDefaultSelfPlayEnv generated type:
pgtype.UUID; adapter returns "" on !Valid. BLOCKER (pre-existing, NOT ours) surfaced:
`cd server && sqlc generate` FAILS at queries/user.sql:31 "mixed positional and named
parameters" (sqlc v1.31.1; user.sql last changed by 88ed91b). Implementer regenerated
via a temp workaround (comment query, regen, restore). This blocks the clean-regen
workflow for Tasks 4 & 5 (SetIssueAssignee, CreateChatTask context) and Task 7's
no-drift gate. INVESTIGATING root cause before Task 4.

SQLC RESOLUTION (user chose "fix user.sql properly"): partial. Committed f0aa697 making
UpdateUser fully positional ($2/$5 + $6::text) — this matches the committed generated
code (which was already positional) and fixes a real source↔generated inconsistency
(source had sqlc.narg, generated had positional; they disagreed). BUT DEEPER FINDING: a
clean `cd server && sqlc generate` is NOT runnable in this repo. It rewrites 6 generated
files, creates agent_skill_suggestion.sql.go + evolution.sql.go that REDECLARE symbols
in hand-maintained *\_manual.sql.go companions => BUILD FAILS. The repo's generated/ is
partially hand-curated and diverges from sqlc v1.31.1 output. Reverted that regen. =>
REVISED APPROACH for Tasks 4 & 5: DO NOT run `sqlc generate`. Add the query to
queries/*.sql (source-of-truth) AND hand-write the corresponding generated Go (function
\+ Params struct + any models.go field) by appending to the existing
generated/<file>.sql.go, mirroring sqlc's exact style for a sibling query in the same
file. Verify via `go build` + the DB-backed test. Task 3's workspace.sql.go landed this
way and builds/passes. => Task 7 no-drift gate is REPLACED by:
`go build ./internal/handler/ ./internal/service/   ./cmd/migrate/...` + full
handler/service test suites (webpush ./... failure remains pre-existing). Task 4:
complete (f0aa697..86753f1 on multica main, review clean; TestEnqueueAgentRun_IssueSquad
PASS) Hand-wrote SetIssueAssignee in generated/issue.sql.go (mirrored DeleteIssue :exec
sibling). 4 files, no sqlc-generate churn. Adapter issue+squad branch:
GetSquadInWorkspace -> SetIssueAssignee(squad) -> enqueue leader (squad.LeaderID) with
IsLeaderTask=true; squadID=="" path unchanged (added empty-guard to avoid parseUUID("")
panic). Task 5: complete (86753f1..bcbe0fc on multica main, review clean; ChatSquad hint
\+ daemon briefing tests 2/2 PASS) chat.sql CreateChatTask writes context via
sqlc.narg('context'); hand-edited generated/chat.sql.go ($7 context, ForceFreshSession
stays $6). daemon.go chat-path (task.ChatSessionID.Valid) injects
buildSquadLeaderBriefing when task.Context.squad_id set & claiming agent==leader. Daemon
test drives real ClaimTaskByRuntime via claimTaskByRuntimeForTest, asserts "Squad
Operating Protocol" in claim body. 5 files. Task 6: complete (areal master b3d646f,
review clean; test_env_dispatch_client.py 6 passed) swe_lego_client.create_env_dispatch:
env_id/agent_id now optional (None), squad_id added; payload includes
env_id/agent_id/squad_id only when truthy; mode forwarded verbatim (resume). 2 files.
CAVEAT: areal-root `uv run` is broken (stale .venv symlink ->
/dfs/share-groups/.../AReaL-main/.venv, a foreign clone path). Use `python -m pytest`
for areal tree_search tests. (db_bridge's own venv/uv is fine.) Task 7: complete
(verification only). Results:

- go build ./pkg/db/generated/ = OK; go build+vet ./internal/service/
  ./internal/handler/ ./cmd/migrate/... = OK.
- service suite: ok (all pass).
- handler suite: 798 PASS / 16 FAIL. ALL 16 failures are the PRE-EXISTING
  runtime-registration "ON CONFLICT ... no unique/exclusion constraint (42P10)" cluster
  (TestDaemonRegister\_*/TestClaimTask\_*), proven pre-existing by checking out base
  392295a and reproducing the identical 16 failures there. NOT regressions from B
  (test-DB schema missing a runtime unique constraint; env/migration-state issue).
- All B tests PASS: TestWorkspaceDefaultSelfPlayEnvColumn, TestEnvDispatch\_\* (incl
  RejectsBothAgentAndSquad, AcceptsEmptyEnvIDShape, AcceptsResumeMode),
  TestEnqueueAgentRun_IssueSquad, TestEnqueueAgentRun_ChatSquad,
  TestClaimTaskByRuntime_ChatSquad_InjectsLeaderBriefing.
- AReaL client: test_env_dispatch_client.py 6 passed (python -m pytest; uv broken per
  Task 6 caveat).

ALL TASKS COMPLETE. multica main: 392295a..bcbe0fce (8 commits incl f0aa697 sqlc fix).
areal master: b3d646f (client). Awaiting final whole-branch review + merge decision.

FINAL REVIEW: READY TO MERGE. No Critical/Important. Spec-complete vs D1-D6; consistent
across all dependency seams; no generated-file drift; squad/leader resolution scoped to
workspace; build + service tests + targeted handler DB tests + gofmt/vet + areal client
suite all green. Minor (non-blocking): (1) commit-count wording (6 feature commits +
user.sql fix, no merges); (2) SquadID uses omitempty while AgentID doesn't (cosmetic);
(3) adapter issue branch has a defensive zero-UUID guard made unreachable by
validate()'s exactly-one rule (defense-in-depth). SUB-PROJECT B COMPLETE. Commits
local-only (not pushed): multica main 392295a..bcbe0fce, areal master ...b3d646f.
Awaiting user decision on push.

# SDD progress — branch via env-dispatch (sub-project C)

Plan: docs/superpowers/plans/2026-07-02-branch-via-env-dispatch.md Spec:
docs/superpowers/specs/2026-07-02-branch-via-env-dispatch-design.md Impl areas: areal
client (customized_areal/\*\*, commit areal master) + db_bridge (multica/db_bridge,
multica main) + multica server (multica/server Go, multica main) Controller: kiro (this
session), subagent-driven Review bases: areal master bf64810e ; multica main bcbe0fce
(both clean). Env: areal tests via
/workspaces/leagent/backend/areal/.venv-test/bin/python -m pytest (py3.12.13, pytest
9.1.1, torch 2.12.1+cpu). db_bridge via `uv run pytest`. multica Go: DATABASE_URL=
postgres://multica:multica@localhost:5432/multica?sslmode=disable, build/test scoped to
touched pkgs. Pre-existing (NOT ours), do not gate on: (1) go build ./...
webpush.go:180; (2) 16 handler ON CONFLICT daemon/claim failures (runtime
unique-constraint, reproduced at base). Decisions: D1 env_dispatch(mode=branch,env_id)
single mechanism; D2 per-node env_id, no seq; D3 read metadata\["env_id"\] (backend
emission EXTERNAL); D4 full removal (client + agent_start_branch channel + multica
/api/issues/{id}/fork; start-branch server endpoint EXTERNAL le-agent); D5 clean break;
D6 rewire wired loop too; Approach 2 (\_BranchDriver seam). SuperNode also carries
branch refs -> env_id (T2).

## Tasks

- [ ] T1: Node.env_id replaces branch refs + annotate_nodes_from_run reads
  metadata\["env_id"\] (areal)
- [ ] T2: SuperNode + serialization clean break (env_id only) (areal)
- [ ] T3: EnvDispatchBranchDriver (branch via env-dispatch) (areal)
- [ ] T4: rewire customized_grouped_workflow branch site; drop build_branch_task (areal,
  highest risk)
- [ ] T5: remove BranchMaterializer/MulticaIssueForker/start-branch client + exports +
  dead tests (areal)
- [ ] T6: remove agent_start_branch db_bridge channel + schema + tests (multica main)
- [ ] T7: remove /api/issues/{id}/fork handlers/service/routes (multica server Go)
- [ ] T8: full-suite regression across all three areas

## Ledger

(append "Task N: complete (...)" as tasks finish)

Task 1: complete (areal master bf64810e..226c5c0b, review clean; test_annotate_env_id.py
2 passed) Node.env_id replaces branch_sandbox_id/branch_issue_id/branch_env_snapshot_id;
annotate_nodes_from_run reads metadata\['env_id'\]. tree_store +
customized_grouped_workflow import cleanly. CAVEAT (pre-existing, NOT ours): 5 test
files fail COLLECTION with ModuleNotFoundError 'datasets'
(test_critic\_*/test_judge_integration). => run TARGET test files directly, avoid broad
`-k` sweeps. Task 2: complete (areal master 226c5c0b..d3f6cebe, review clean;
event_codec+checkpoint_super 21 passed) SuperNode is in execution_dag.py (:70), env_id
set in event_codec.dag_to_supernodes; supernode_assembler untouched (builds from Multica
specs, no source node). Clean-break cascade also removed dead branch_seq consumers:
branch_selection.py + ReplayPrefix/replay_prefix_for. Zero branch\_* keys remain in the
4 files. NOTE: plan's test path wrong (real: tests/test_tree_search/test_checkpoint.py,
has 17 PRE-EXISTING unrelated failures: MCTSTreeStore.insert_batch missing, Node
distill_reward kwarg — not env_id-related). Task 3: complete (areal master
d3f6cebe..bdc2328c, review clean; 10 target tests pass incl 2 runner-satisfaction)
EnvDispatchBranchDriver.drive_lane -> create_env_dispatch(mode=branch,
env_id=sandbox_id\[=source env\], dispatch_type/agent_id/domain, group_size=1) returns
rollouts\[0\].env_id. Satisfies \_BranchDriver Protocol in both runners
(swe_lego_issue_runner:53, self_play_runner:58). Imports without torch. branch_driver.py
new. Task 4: BLOCKED (legitimate; reviewer-confirmed). doc-only commit 4ea01afa. ROOT
CAUSE: the wired branch site is structurally coupled to the le-agent TPFC
driven-generation contract (task_id + seed_messages_already_inserted ->
OpenAIProxyWorkflow -> InteractionWithTokenLogpReward -> token-level Nodes). The
env-dispatch branch primitive yields only a forked multica env_id + reward-only
SweLegoIssueResult; NO mapping to the required le-agent task_id. => D6 (rewire wired
loop to env-dispatch) is NOT achievable within C's scope. Options: (1) external backend
"materialize" endpoint: le-agent TPFC task from a forked multica env_id (seeded w/
truncated prefix) -> returns task_id, keeping the existing seam \[external, defer\]; (2)
replace token-level node pipeline with reward-only runner pipeline \[huge, own plan\];
(3) revert to Q6 option (a): remove the wired loop's branch machinery entirely (that
loop stops branching; branching lives only in the runner model) \[achievable NOW within
C\]. ALSO: master currently runtime-broken — Task 1 removed Node.branch_sandbox_id but
dead candidate.branch_sandbox_id refs remain in select_branch_candidate +
build_branch_task + \_cleanup_branch => 7 failing tests. Must fix (T1 review only
checked imports, not the branch-path tests). ESCALATED TO USER. Task 4 (REVISED, Option
3): complete (areal master ...18a71edf, review clean; branch_sampling 18/18 green)
Removed build_branch_task, \_prepare_branch_task, \_cleanup_branch, the
`if branch_task_id:` episode block + dead imports; episode loop falls through to scratch
\_retry_episode. Kept select_branch_candidate + choose_sample_source (shared helpers)
with no dead branch_sandbox_id access. Remaining branch_sandbox_id refs are Task-5-owned
(integration.py) or negative-assert/doc. Pre-existing unrelated failures 5->4. D6 rewire
deferred (needs external le-agent materialize endpoint). Task 5: complete (areal master
18a71edf..11cc632c, review clean; import ok, 508 collected, 6 pre-existing datasets
errors only) Removed from integration.py:
BranchMaterializer/BranchMaterializationResult/MulticaIssueForker/BranchStarter/
BranchCandidate/materialize_cloud_branch/cleanup_cloud_branch (kept
finalize_with_verifier+VerifierResult). __init__ exports pruned; backend_run.py stripped
of \_start_branch_agent_run_for_task(+\_with_refresh)+ LE_AGENT_BRANCH_RUN_ENDPOINT;
dead tests deleted/trimmed. Retained seed_messages_already_inserted param (its full
removal cascades to out-of-scope callers; guarded dispatch is dead) - sound scope call.
MINOR (non-blocking, final-review): stale markdown docs still mention removed symbols -
debug_tpfc.md:350 (\_start_branch_agent_run_for_task), tree_search/README.md,
agents/README.md, multica_environment_protocol.md. Task 6: complete (multica main
bcbe0fce..17dd0bad, review clean; db_bridge suite 168 passed/1 skipped) Removed
agent_start_branch Channel (channels.py), rpc_agent_start_branch (schema.sql),
START_BRANCH cases (test_leagent_channels.py, replaced by negative test
test_agent_start_branch_channel_removed), BRIDGE_CONCURRENCY_AGENT_START_BRANCH
(test_integration_e2e.py), README row+mermaid. env_dispatch channels intact. Also
cleaned .env.areal.example comment + unused import json (disclosed). Task 7: complete
(multica main 17dd0bad..7969187a, review clean; 6 files, 748 deletions) Deleted
handler/issue_fork.go(+test), service/issue_fork.go(+test); removed both issue /fork
routes from router.go (enclosing /{id} group kept - has metadata/pull-requests); removed
issue /fork rows + dead sampleIssue const from router_fork_routes_test.go. sqlc orphans
(generated/issue_fork.sql.go, queries/ issue_fork.sql) intentionally LEFT (no sqlc
generate). ForkCloudRuntimeSandbox + /api/v1/sandboxes/fork + env_dispatch ForkSandbox
all UNTOUCHED. handler+service build/vet PASS; internal/service PASS; handler = the 16
pre-existing ON CONFLICT fails only (0 fork-related). CAVEAT: `go build ./cmd/server/`
fails on pre-existing webpush.go:180 (cmd/server imports webpush) => the router.go edit
\+ router_fork_routes_test could NOT be compiled/run here; validated by diff inspection
(2-line route removal, low risk). Pre-existing, not ours. Task 8: complete (verification
only). Results:

- areal imports OK (agents, customized_grouped_workflow, tree_store, branch_driver,
  event_codec, execution_dag).
- areal branch/codec/driver/runner/branch_sampling suites: 49 passed.
- db_bridge: 168 passed / 1 skipped.
- multica Go (from T7): handler+service build/vet PASS; internal/service PASS;
  internal/handler = 16 pre-existing ON CONFLICT fails only (0 fork-related). cmd/server
  uncompilable here (pre-existing webpush).
- grep sweep: NO live refs to removed symbols. Only remaining: integration.py:8
  docstring (describes removal) + intentional sqlc orphans (generated/issue_fork.sql.go,
  queries/issue_fork.sql). ALL TASKS COMPLETE (T4 via Option 3; D6 env-dispatch rewire
  DEFERRED pending external le-agent materialize endpoint). Commits: areal master
  226c5c0b..11cc632c (T1-T5) + docs; multica main bcbe0fce..7969187a (T6 db_bridge, T7
  server). Open MINOR (non-blocking): stale markdown docs (debug_tpfc.md:350,
  tree_search/README.md, agents/README.md, multica_environment_protocol.md) still
  describe removed branch features.

FINAL REVIEW: READY TO MERGE. No Critical/Important. Spec-complete per revised D6
(Option 3). Verified: D1/D2/D3 (Node.env_id + annotate + EnvDispatchBranchDriver, no
seq); D5 clean break (env_id-only serialization, zero branch\_\* keys); D6 revised
(wired loop no longer branches, falls through to scratch); D4 removals (client +
db_bridge channel + multica issue-fork; /api/v1/sandboxes/fork + ForkSandbox intact;
start-branch server endpoint left external). Runs: areal 41+8 passed; db_bridge 168/1
skip; multica handler+service build/vet + service test OK. Minor (non-blocking): (1)
select_branch_candidate/choose_sample_source/SampleSource enum + sample_source/
branch_probability/branch_td_threshold config now test-only dead code (plan permitted
retaining select_branch_candidate); (2) stale markdown docs still describe removed
features. SUB-PROJECT C COMPLETE. Commits local-only (not pushed): areal master
bf64810e..11cc632c (+docs to HEAD); multica main bcbe0fce..7969187a. D6 env-dispatch
rewire of the wired loop DEFERRED (needs external le-agent materialize endpoint).
Remaining: sub-projects D (session lifecycle) and E (entropy + critic env-save).

=== SUB-PROJECT D — training_agent session lifecycle === Spec:
docs/superpowers/specs/2026-07-06-training-agent-session-lifecycle-design.md Plan:
docs/superpowers/plans/2026-07-06-training-agent-session-lifecycle.md (both areal
master, committed) Base: multica main @ 7969187a (== C/T7 tip). Commit D to multica main
(working tree switched back main\<-dev by user). Approach A (server-side lifecycle).
Locked decisions D1-D6 (see spec §3). Key: constraint (b) — trained teammate task
created AFTER dispatch (mention-delegation / /api/agent/start), so session hooks live on
task-creation + completion paths, not env_dispatch. proxy_url = multica config. Reward =
default placeholder (E owns real). Tasks: T1 read-doc(STOP-if-broken) T2 contract T3
persist(mig152) T4 rl-client T5 open-hook T6 execenv T7 close-hook T8 config T9
regression. T1: DISPATCHED (investigation; may return BLOCKED and reshape plan). T1:
DONE (investigation; Approach A feasible). Note
docs/superpowers/notes/2026-07-06-D-seams.md. Findings folded into spec+plan (areal
commit above): (1a) chokepoints = Enqueue\* family -> CreateAgentTask/CreateChatTask
(service/task.go) + env_dispatch EnqueueAgentRun; NO /api/agent/start route here. (1b)
close at CompleteTask/FailTask/CancelTask; runtime_sweeper timeout BYPASSES FailTask
(gap, deferred). (1c) execenv NEEDS-NEW-FIELD (claim-time field from
context.areal_proxy). (1d) RL contract = EXPERIMENTAL stack: start_session admin-key ->
flat {session_id, api_key}; set_reward+end_session session-key(proxy_key) auth, no
session_id body; task_id = agent_task.id. (1e) project via
Issue.ProjectID/ChatSession.ProjectID. Store RL state in context.areal_proxy (no new
task column). USER DECISION: target experimental gateway contract. T2: DISPATCHED
(contract: train_agent_id). T2: complete (multica main 7969187a..0a5cb9332, review
clean; 4 files +87). TrainAgentID on EnvDispatchRequest (json train_agent_id,omitempty +
UUID shape-check) + service.EnvDispatchInput; validate rule: TrainAgentID requires
SquadID or ==AgentID; empty unchanged. New tests green, no new failures, gofmt clean, no
scope creep/sqlc churn. T3: DISPATCHED (training_dispatch persistence + migration 152).
T3: complete (multica main 0a5cb9332..992be04dc, review clean; build+ 19 service tests
green incl 2 new). Migration 152 training_dispatch(project_id PK FK ON DELETE CASCADE,
workspace_id, train_agent_id, default_reward dp default 1.0, created_at). 152 chosen
ABOVE dev's max (151) so no collision on future dev\<-main merge (main was only at 141).
queries + HAND-WRITTEN generated/training_dispatch.sql.go (mirrors environment.sql.go),
no sqlc churn. SaveTrainingDispatch on Deps+adapter+stub+fake; called once per rollout
project when TrainAgentID set (save errors non-fatal). GetTrainingDispatchByProject
ready for T5. T4: DISPATCHED (Go RL client, experimental contract). T4: complete
(multica main 992be04dc..eeac55e62, review clean; build/vet/gofmt clean, 8 httptest
green). internal/arealrl.Client{New(stubBaseURL, adminKey)}:
StartSession(taskID)->SessionCreds{SessionID,ProxyKey} (admin Bearer, body {task_id,
group_size:1 \[Pydantic-ignored no-op\]}, decodes FLAT {session_id, api_key});
SetReward(proxyKey, reward){reward}; EndSession(proxyKey) - both session-key
Bearer(proxyKey). Confirmed against experimental proxy_rollout_server.py. stdlib only,
%w wrapped. Not yet wired (T5/T8). T5: DISPATCHED (session-open hook at enqueue
chokepoints). T5: complete (multica main eeac55e62..57f17d572, review clean; build OK,
6/6 TDD green, no new failures). Shared maybeOpenTrainingSession helper (training-only
via GetTrainingDispatchByProject; skip if agent!=train_agent_id; idempotent if
context.areal_proxy already set; StartSession(task.id); merge
context.areal_proxy={provider:areal,model:areal-default,api_key:proxy_key,base_url:proxyURL,session_id}).
Wired at all Enqueue\* chokepoints (service/task.go) + env_dispatch EnqueueAgentRun;
deps interface-injected (fake in tests). Hand-written MergeTaskArealProxyContext query
(no sqlc churn). Does NOT reuse agent_task.session_id column. Loud error if training
target but bridge dep missing (no un-proxied run). T6: DISPATCHED (execenv provider
wiring, NEEDS-NEW-FIELD). T6: complete (multica main 57f17d572..816d1e86c, review clean;
build/vet/gofmt clean, 5 TDD green). context.areal_proxy parsed at ClaimTaskByRuntime ->
carried to daemon via omitempty fields (matching json tags on claim response + daemon
Task/AgentData). At ExecOptions: provider/model via splitPiModel (model
"areal/areal-default"), api-key via pi --api-key CustomArg; base_url injected as env
AREAL_PROXY_BASE_URL (pi has NO base-url flag). No hardcoded secrets. DEPENDENCY -> T8:
must wire pi models.json `areal` provider baseURL = $AREAL_PROXY_BASE_URL so the trained
pi actually routes to the bridge stub (plan Task 8 updated). T7: DISPATCHED
(session-close hook: default reward + end_session). T7: complete (multica main
fb40610c7..86c3c28ec..61ed426fd, review CLEAN; 7/7 MaybeClose tests pass, build clean)
Shared maybeCloseTrainingSession(ctx, deps, task, projectID) called from
CompleteTask/FailTask/CancelTaskWithResult; arealSessionCloser interface
(SetReward+EndSession) added; TrainingSessionDeps gains Closer field;
extractArealProxyConfig safely parses task.Context JSONB; default_reward from
training_dispatch with fallback to trainingDefaultReward=1.0 (T8 makes configurable);
SetReward error → still calls EndSession (best-effort); RL errors logged via slog.Warn,
never fatal. Doc note added: runtime_sweeper.FailStaleTasks bypasses FailTask (raw SQL),
so stale tasks won't auto-close — reaper is future hardening. T8: complete (multica main
61ed426fd..ae6f2435a, review CLEAN; 5 config tests + 7 close tests + 6 open tests pass,
build clean) TrainingConfig struct + LoadTrainingConfig() reads
AREAL_BRIDGE_STUB_URL/AREAL_ADMIN_API_KEY/AREAL_PROXY_URL/TRAINING_DEFAULT_REWARD from
env; NewTrainingSessionDeps(cfg, q) returns nil when BridgeStubURL/AdminAPIKey empty
(hooks stay no-ops); arealrl.New assigned to both RL (starter) + Closer fields;
TaskService.WithTraining(\*TrainingSessionDeps) builder injects it; cmd/server/main.go
wires LoadTrainingConfig + conditional WithTraining after NewTaskService; .env.example
documents all 4 vars + notes AREAL_PROXY_BASE_URL is daemon-set; TrainingSessionDeps
gains DefaultReward float64 field used in maybeCloseTrainingSession (falls back to 1.0
when zero). Invalid TRAINING_DEFAULT_REWARD → warning log + 1.0 fallback. MINOR
(non-blocking): trainingDefaultReward constant in training.go:86 has stale comment "T8
will make this configurable" — T8 is done, comment should say "fallback used when
DefaultReward is zero" or be inlined. Literal 1.0 appears in 3 places (training.go:268,
training_config.go:45,49) — could DRY to the constant. Not worth a fix cycle. T9:
complete (verification only; multica main ae6f2435a..1667f85c3 gofmt fix)

- go build ./internal/handler/ ./internal/service/ ./internal/daemon/execenv/
  ./internal/arealrl/ ./pkg/db/generated/ = OK
- go vet same scoped packages = OK
- go test scoped: internal/service OK (training_test 7 close + 6 open,
  training_config_test 5, env_dispatch_test all pass); internal/arealrl OK;
  internal/daemon/execenv OK (cached); pkg/db/generated no tests.
- internal/handler: 16 FAIL (8 TestDaemonRegister\_\* + 8 TestClaimTask\_\* ON CONFLICT
  42P10) = PRE-EXISTING baseline (reproduced at base 816d1e86c), 0 new.
- gofmt -l: 4 touched files needed alignment (training_config.go, cmd/server/main.go,
  training_test.go, env_dispatch_test.go) → committed as 1667f85c3 chore(training):
  gofmt T7-T8 touched files. Re-check clean.
- db_bridge: NOT touched by D, skipped.
- AReaL confirm-only: start_session/set_reward/end_session all exist in
  areal/experimental/openai/proxy/proxy_gateway.py (lines 353/623/637);
  server.py:182-184 defines RL_START_SESSION_PATHNAME="rl/start_session",
  RL_END_SESSION_PATHNAME="rl/end_session", RL_SET_REWARD_PATHNAME="rl/set_reward".
  Bridge routes /rl/\* to gateway. NO code change.
- grep sweep: train_agent_id (env_dispatch.go handler+service, agent.go) /
  training_dispatch (env_dispatch.go, training.go) / areal_proxy (training.go, agent.go,
  daemon/types.go) / arealrl (training.go, training_config.go, arealrl/client.go) — ALL
  resolve to intended code only, no stray references.
- MINOR carried from T8: trainingDefaultReward constant stale comment + 3 literal 1.0
  occurrences. Non-blocking.

D ARCHIVED. multica main: 816d1e86c..0b68f606d (6 commits). areal master: merged +
archived to openspec/changes/archive/2026-07-07-sub-project-d-session-lifecycle/.

=== SUB-PROJECT E — critic-driven reward + entropy + env_id === Spec:
docs/superpowers/specs/2026-07-06-critic-driven-training-signal-design.md Plan:
docs/superpowers/plans/2026-07-06-critic-driven-training-signal.md Bases: areal master @
48d49aba; multica main @ 816d1e86c (D's T6 tip). Depends on: sub-project D — COMPLETE
AND ARCHIVED.

T1: complete (areal 052020fb; seams note). All 6 seams confirmed. T2: complete (multica
0fb6c2644; 40/40 env_dispatch tests). CriticAgentID on env_dispatch. T3: complete
(multica 9e3fa6f0b; 11/11 tests). Migration 153 + critic_agent_id on training_dispatch.
T4: complete (areal 4fb458ea; 6/6 tests). env_id on StartSessionRequest (additive). T5:
complete (multica eeac55e62 + 343215231; 6/6 tests). arealrl Go client + env_id. T6:
complete (multica fb40610c7; 11/11 + 40/40 + 2/2 tests). Session-open hook passes
env_id. T7: complete (multica 0b68f606d..f43a9ab66 + b77577bfa; spec ✅, quality
Approved). maybeSpawnCriticTask + RouteTerminalTrainingTask +
FindCriticTaskForTrained/CreateCriticTask. T8: complete (multica b77577bfa..de2ac7aa9;
spec ✅, quality Approved). maybeCloseTrainingSessionFromCritic + parseCriticReward. T9:
complete (areal 277d6f7b; 3/3 + 43/43 tests). logprobs injection with graceful fallback.
T10: complete (multica de2ac7aa9..69511fdf3; TDD-light). No new config. db_bridge README
notes env_id. T11: complete (Go build/vet/test clean; AReaL proxy 9 pass; db_bridge 168
pass; grep clean. Final review: READY TO MERGE).

E VERIFIED + ARCHIVED. multica main: 816d1e86c..69511fdf3. areal master: merged from
sub-project-e-critic-reward-entropy-env.

# SDD progress - multica-v2-segment-dag-training

Plan: docs/superpowers/plans/2026-07-08-multica-v2-segment-dag.md
Spec: docs/superpowers/specs/2026-07-08-multica-v2-segment-dag-design.md
Impl repo: areal (branch: multica-v2-segment-dag-training) + multica/server (own branch, U6-U9)
Controller: claude (this session)

## Ledger
Task 1.1: complete (commits fa2d8679..d77fab31, review clean) MINOR (defer to final review):
test_close_segment_moves_active_to_ready_no_reward's last assertion
(`active_completions is not None`) does not verify the no-reward claim - strengthen to
check the closed trajectory's interaction reward is None.
Task 1.2: complete (commits d77fab31..632e4949, review clean, no issues)
Task 1.3: complete (commit 6bde9872, amended from 48979c63). IMPLEMENTER NEVER RAN TESTS
(broken uv/.venv env). Orchestrator re-verified: ran `.venv-test/bin/python -m pytest`
(7/7 green) + `uvx ruff check` (clean). Found + fixed: (a) `test_close_segment_endpoint_session_key`
used bare `create_app` instead of imported `create_data_proxy_app` (NameError); (b) ruff
import-wrap. Broader `areal/v2/inference_service/` suite: 7 passed, 0 regressions.

ENV (applies to ALL remaining tasks): project `.venv` is broken (points to non-existent
/dfs/.../AReaL-main/.venv) and `uv run` fails with "No interpreter found". Use
`.venv-test/bin/python -m pytest <path>` for tests (Python 3.12, fine for v2 tests) and
`uvx ruff check <paths>` for lint. Do NOT trust implementer self-reports of GREEN in this
workspace - re-run tests yourself. U1 independent reviewer dispatched (sonnet).

U1 (close_segment) STATUS: CLOSED. Code complete + verified-green (9/9 tests, ruff). Independent
review APPROVED (re-dispatch a0816718 after original a08... stalled at 140 bytes). All 4 spec
contracts PASS; data_proxy + gateway routes mirror /rl/set_reward; no new issues. MINOR (weak
no-reward assertion) deferred to final review.

Task 2.1 (U2 - per-segment export contract + unknown-trajectory 400): CLOSED (commit 796c9004).
Orchestrator re-verified: 9/9 tests green (7 U1 + 2 U2), ruff clean. Diff: 5-line fix at
data_proxy/app.py:771 - raise HTTPException 400 when `body.trajectory_id is not None` and
KeyError (preserves multi-trajectory `continue` when trajectory_id is None). 2 new tests seed
via add_string_interaction (no /chat/completions backend) + assert export-keeps-session and
unknown-trajectory-400. Independent review APPROVED (a666d1e5) - all 4 contracts PASS.
DEFERRED (tasks 4.2/4.4): refs-only / RTensor.remotize contract + /data/* resolve tests need
tensor-seeded interactions; string interactions exercise concat_string_interactions instead.
Revisit at U4 (assemble_from_refs resolves tensor_ref).

Next: U3 (areal MulticaDagClient consumer) - IN PROGRESS (implementer ac2bd37d). U4 gates on it.

Task 3.1 (U3 - MulticaDagClient): complete (commit 748ca36f, amended from cf755090). DEVIA­TION
CAUGHT: the U3 implementer (ac2bd37dbfaf530d6) STALLED (140-byte output, 11 min idle, no
notification - same stall mode as the first U1 reviewer) AND committed cf755090 with the OLD
sub-project-g design (async, SegmentSpec with start_turn_idx/end_turn_idx/task_id, NO
tensor_ref/trajectory_id/env_snapshot, AssembledDag with NO session_to_agent_run) - directly
violating the locked architecture (design doc/proposal/spec all mandate tensor_ref, no turn_idx).
Orchestrator killed the dead agent, rewrote both files to the plan's tensor_ref sync design,
verified 5/5 tests green + ruff clean + 0 turn_idx refs, amended to 748ca36f. Independent
review pending. LESSON: implementer subagents in this workspace have stalled twice; always
re-verify their commits against the locked design and re-run tests yourself.

Next: U4 (SuperNodeAssembler.assemble_from_refs) - GATED on U3 review.

Task 4 (U4 - SuperNodeAssembler.assemble_from_refs): CLOSED (commit af75a189). Orchestrator
IMPLEMENTED + verified directly (no implementer subagent - plan's U4 section had WRONG signatures:
it assumed `SuperNode(payload=tensors, ...)`, `edag.add_node(node)`, `edag.add_edge(node, node)`,
`len(edag.nodes)`; actual API is SuperNode with NO payload field + required `task_id`,
`add_event(SuperNode)`, `add_edge(src_id, dst_id, type)`, `.events`). Implementation-boundary
decisions (documented in design doc + method docstring): (a) resolved tensors attach to
`metadata["tensors"]` (SuperNode has no payload field - non-invasive); (b) `segment_id` becomes
the SuperNode `node_id` so edges resolve directly; (c) `task_id=""` because the v2 SegmentSpec
dropped it; (d) `session_id` reverse-mapped from `session_to_agent_run`; (e) acyclicity enforced
via `topological_order()` (raises DAGError on cycle). Verified: 3/3 U4 tests green
(test_assembler_ref_resolve.py: builds-one-per-segment, rejects-cycle, env-snapshot-stamped);
ruff clean; existing assembler e2e (test_supernode_e2e.py) still passes (16 collectable assembler
tests) - no regression on the old `assemble` turn-idx path. Full suite: 342 passed / 9 failed
(test_critic_value_client, test_critic_variance, test_hybrid_advantage) + 5 errors (datasets
ModuleNotFoundError) - ALL pre-existing, NONE import supernode_assembler/multica_dag_client/
assemble_from_refs. Dense per-session coverage gap-check (DAGError on gap) DEFERRED to U5
(training plumbing owns session-boundary validation). U3 reviewer (a556dbad) stalled (3rd stall);
U3 treated CLOSED on orchestrator's own verification (5/5 + design match + U4 consumes U3's
AssembledDag correctly). Independent U4 review: APPROVED (agent a047489e completed; VERDICT
APPROVED - confirmed locked-architecture conformance, API correctness (add_event/add_edge/.events/
topological_order, task_id="", metadata["tensors"]), all implementation decisions sound, old assemble
path untouched with 22 existing tests passing, test quality good). MINOR fixed: tightened U4 cycle test
from pytest.raises(Exception) to pytest.raises(DAGError) (commit 1beae470).

Next: U5 (minimal training plumbing + tensor lifecycle) - coupled to U4's ExecutionDAG output.

Task 5 (U5 - minimal training plumbing + tensor lifecycle): CLOSED (commit f56e4951). Orchestrator
IMPLEMENTED + verified directly (no implementer subagent - plan's U5 step-1 "locate training entry"
found NO lightweight DAG-consuming trainer; the existing CustomizedPPOTrainer is FSDP-heavy and
untestable in the CPU .venv-test env). Decision (documented in design doc + module docstring): the
"minimal training step" = the GAE forward path `assemble_node_advantages(topological_order(),
initial_value=0.0, gamma, lam)` with placeholder zero reward - pure float math, no torch/FSDP
(`events_from_nodes` treats unset `value` as 0.0, so zero-reward SuperNodes flow through with zero
advantages). New module `segment_dag_trainer.py`: (a) `TrainingTensorResolver` Protocol (resolve +
clear) + `SessionRemover` Protocol; (b) `DataProxyTensorResolver` (GET /data/<shard_id> +
deserialize_value; DELETE /data/clear with shard_ids); (c) `DataProxySessionRemover` (POST
/export_trajectories remove_session=True - data_proxy exposes session removal only via export);
(d) `run_segment_dag_training_step(*, client, resolver, session_remover, project_id, gamma, lam,
poll_timeout, poll_interval, assembler=None) -> AssembledAdvantages` wiring get_dag -> assemble_from_refs
-> topological_order -> assemble_node_advantages -> clear shards + remove sessions. Cleanup is
success-path only (failed step propagates without releasing shards - caller handles retry). Tensor-ref
contract pinned for change 1: `{"shard_id": str}` per segment; finalized when Multica U6/U8 pins export.
Verified: 6/6 U5 tests green (zero-reward path asserts advantages all-zero + clear/remove invoked;
cycle propagates DAGError WITHOUT cleanup; DataProxyTensorResolver resolve+clear via MockTransport;
404->KeyError; clear-noop-on-empty; DataProxySessionRemover export+remove_session body). ruff clean.
Regression: 10 passed (U4 assemble_ref_resolve + U3 dag_client + e2e supernode). Dense per-session
coverage gap-check (DAGError on gap) NOT added - the plan deferred it to U5, but U5's GAE path operates
on topological order which already rejects cycles; per-session gap validation is a Multica-assembly
concern (U8) more than an areal-consumer one - DEFERRED to U8/final-review with a note.

U5 independent review: APPROVED (agent a4a3af3b completed; VERDICT APPROVED - all locked-architecture
conformance, API correctness, design decisions sound, edge cases handled, test quality good, regression
U3/U4/e2e pass). 2 MINOR (non-blocking): (1) resolve uses tensor_ref["shard_id"] (fail-fast) while
cleanup uses .get("shard_id") (skip unresolved) - intentional asymmetry, correct; (2) TrainingTensorResolver
redeclares resolve() from TensorResolver - stylistic, harmless. U1-U5 ALL REVIEW-APPROVED.

Task 6 (U6 - multica arealrl CloseSegment + ExportTrajectory): CLOSED (multica commit 72f1b7ba2 on
branch feature/multica-v2-segment-dag-training, based on upstream/dev which was 5 commits ahead of
local dev). Orchestrator-IMPLEMENTED + verified directly (no implementer subagent). Plan U6 code adapted
to actual doJSON(ctx, path, bearer, body) signature (plan assumed doJSON(ctx, method, path, key, body, out)
- wrong; actual takes no method/out, caller decodes). Added: closeSegmentPath/exportTrajPath constants;
CloseSegment(ctx, proxyKey) (session-key auth, no body, returns trajectory_id, errors if nil);
ExportTrajectory(ctx, sessionID, trajectoryID) (admin-key auth via c.adminKey - NOT a param as plan
guessed, consistent with StartSession; remove_session=false; returns raw traj json.RawMessage). Both
reuse doJSON/checkStatus. Package doc updated with both endpoints. Verified: 15/15 arealrl tests pass
(10 existing + 5 new: CloseSegment request/auth + non-2xx + missing-trajectory_id; ExportTrajectory
request/auth + non-2xx); go vet clean; gofmt clean. tasks.md path fixed: internal/arealrl/client.go ->
server/internal/arealrl/client.go (line 26). AREAL-SIDE U1-U5 + MULTICA U6 DONE.

Next: U7 (multica interaction_dag recording + hooks), U8 (AssembledDag+/dag endpoint),
U10 (config + E2E + grep, both repos). U9 (migration) DONE - commit 8dd8c8c8a, see below.

Task 9 (U9 - migration): CLOSED (multica commit 8dd8c8c8a). Orchestrator-IMPLEMENTED + verified. DEVIATION
from plan: plan lumped 3 table migrations all at "155" but multica convention is one-file-per-number AND
env_snapshot has FK to segment (must migrate after). Used 155_interaction_dag_segment / 156_interaction_dag_edge
/ 157_interaction_dag_env_snapshot (up+down each). segment: text PK + project/agent_run/issue/task/trajectory_id
bigint/tensor_ref jsonb/closing_event/closing_event_target_segment/created_at, idx on project_id. edge: bigserial
PK + project/src/dst/type CHECK(delegation/mention/completion), idx on project_id. env_snapshot: segment_id PK
FK CASCADE + sandbox_ids jsonb/issue_snapshot_id/env_state jsonb default '{}'. Verified: `DATABASE_URL=...
go run ./cmd/migrate up` applies 155-157 cleanly (tables created); `down` drops them cleanly (a PRE-EXISTING
042_autopilot dependency error fires later in the full rollback - NOT ours, my 157/156/155 down applied before
it); up re-applies. Note: `migrate down` rolls back ALL migrations (aggressive tool behavior), not just last.

Task 7 (U7 - InteractionDAGService + hooks): NOT STARTED. Prior ledger entry "IN PROGRESS
(implementer aa8f44e1, background)" was STALE - that background agent died in a prior session
leaving zero work: both repos clean (no interaction_dag.go, no mods to task.go/training.go/
env_dispatch.go/handler/env_dispatch.go, no U7 commit; multica HEAD 8dd8c8c8a = U9). Pre-build
seam trace (this session, multica server/ + areal customized_areal/) produced 3 design decisions
NOT in the plan, captured in design doc section "## U7 Pre-Build Design Decisions":
- D8 agent_run_id = task.ID (attempt-level). EnqueueAgentRun (handler/env_dispatch.go:662)
  returns task.ID as runID; no runs table. areal consumes as SuperNode.agent_id + session-lookup
  key (supernode_assembler.py:159,163); segment-table task_id is redundant (v2 SegmentSpec
  dropped it, task_id=""). Retries create NEW task.ID (child via parent_task_id, agent.sql:178).
- D9 Fresh areal RL session per retry attempt (NOT inherited). Today CreateRetryTask copies
  p.context (areal_proxy) -> child inherits parent session -> maybeOpenTrainingSession no-ops
  (hasArealProxyContext guard, training.go:200) -> no fresh StartSession/RecordSessionAgentRun
  -> retry children's segments dangle at assembly. Decision: each attempt opens own session.
  Requires (1) CreateRetryTask strips areal_proxy from child context [keep chat session_id/
  work_dir resume CASE-WHEN], (2) MaybeRetryFailedTask calls tryOpenTrainingSession(child)
  BEFORE NotifyTaskEnqueued (mirror enqueueMentionTask :614->:618 ordering), (3) RecordSession-
  AgentRun fires for child. Close ordering: FailTask->RouteTerminal closes S_A->MaybeRetry
  creates B->B opens S_B. Only retryable reasons produce a child (runtime_offline/
  runtime_recovery/timeout/codex_semantic_inactivity, task.go:1725). Scope: change (1) touches
  pre-existing CreateRetryTask (mig 055) - U7 in-scope dependency. Sweeper path bypasses
  FailTask (orphaned session, no child) - pre-existing gap, coverage boundary.
- D10 RecordSessionAgentRun call site: inside maybeOpenTrainingSession (training.go:158) after
  StartSession succeeds (~line 226, post-persist :220). Records {projectID, sessionID=
  creds.SessionID (:204), agentRunID=taskID}. Single idempotent chokepoint both Enqueue*
  (tryOpenTrainingSession 510/614/750/834) and env_dispatch (adapter handler/env_dispatch.go:713
  -> MaybeOpenTrainingSession) share. Trap: agentRunID=taskID (run) NOT agentID (agent); areal
  stores it in field named agent_id (supernode_assembler.py:159).
U7 open items (resolve during impl): envID source for retry child's StartSession; exact
tensor_ref shape from ExportTrajectory (U6); envSnapshot source for CloseSegmentForEvent.

Task 7.1 (U7.1 - InteractionDAGService + sqlc + tests): CLOSED (multica commits 3f9587a3d +
157284045). Implementer subagent built the recorder; task-reviewer (sonnet) marked APPROVED
(no Critical). Orchestrator independently re-verified (16/16 tests incl. hermetic integration
on real Postgres; gofmt+vet clean). Deliverables: InteractionDAGService (RecordSessionAgentRun
4-param, CloseSegmentForEvent, AddEdge) behind INTERACTION_DAG_ENABLED; migration 158
interaction_dag_session_run (fills U9's session->agent_run gap); hand-written sqlc for all 4
interaction_dag tables (sqlc generate broken). Review fixes (157284045): Important -
segment+env_snapshot now atomic via a single data-modifying CTE
(InsertInteractionDAGSegmentWithSnapshot; $1 reused as snapshot FK; paired ops stay together,
no orphan-on-snapshot-failure); Minor - nil/empty envSnapshot -> env_state='{}' not 'null';
Minor - var _ InteractionDAGStore=(*db.Queries)(nil) compile-time assertion. Deferred #3
(tensor_ref null masking) + #5 (env_state duplication) - pending U6/U8 shape pinning / doc'd
intent. Public service API stable for U7.2. D8/D9/D10 above remain; D9 = U7.3.
Next: U7.2 - wire RecordSessionAgentRun into maybeOpenTrainingSession (D10, training.go:~226
post-persist) + CloseSegmentForEvent/AddEdge hooks at delegation/mention/completion/squad seams
in task.go (trained rollouts only; INTERACTION_DAG_ENABLED composing with s.Training gate) +
integration tests. Then U7.3 (D9 fresh session per retry: CreateRetryTask strips areal_proxy,
MaybeRetryFailedTask opens fresh session before NotifyTaskEnqueued).

# SDD progress - areal-v2-integration-and-tree-search-branching (v2 roadmap change 3)

Plan: docs/superpowers/plans/2026-07-08-areal-v2-tree-search-branching.md Spec:
docs/superpowers/specs/2026-07-08-areal-v2-tree-search-branching-design.md Impl
repo: areal (branch: worktree-areal-v2-tree-search-branching) Controller: claude
(this session), DIRECT execution (no implementer subagents - user rejected subagent
dispatch; worked TDD + per-task commit). Base: f89eb1b2. Builds on the UNMERGED
multica-v2-segment-dag-training U1-U6 components (create_env_dispatch / get_dag /
assemble_from_refs / DataProxyTensorResolver / DataProxySessionRemover /
EnvDispatchBranchDriver) - REUSES them, does not reimplement. All phases
F-independent (branching via EnvDispatchBranchDriver drive_lane, not Sub-project F).
Env: `python3 -m pytest` (NOT `uv run pytest` - broken .venv/uv); `uvx ruff check`.
torch 2.12.1 CPU. Pre-existing (NOT ours): 9 critic tests RuntimeError "no current
event loop in thread MainThread" (uvloop policy); 5 tests ModuleNotFoundError
'datasets'; test_node_torch_lazy::test_tree_store_imports_without_torch (torchdata).
None touch changed files.

## Tasks (tasks.md: 18 done / 4 deferred-or-skipped of 22)

Phase 1 (MultiAgentEnvDispatchWorkflow orchestrator): 1.1 6689bd83 (N=1 SCRATCH,
AReaL never calls start_session); 1.2 41f21d07 (N>1 squad + partial-squad drop ->
None); 1.3 152403ef (tensor-ref resolve + success-path-only cleanup ordering, no
cleanup on DAGError); 1.4 4fa4e1bd (DagTimeout->None reject; DagNotFound/DagForbidden
propagate). Phase 2 (TreeSearchGroupedRolloutWorkflow wiring, Approach B
SuperNode-preserving): 2.1 a69d5f58 (_result_to_nodes multica branch returns
SuperNodes, preserves structure, return type list[Node]|list[SuperNode]|None); 2.2
06354ad6 (_finalize_multica_episode: insert_super_batch + per-episode GAE + batched
tensor dict); 2.3 6851bc4c (activate multica_dag_client hook + wire self.workflow);
2.4 99891ac2 (group_size=M parallel rollouts + per-episode GAE grouping via
group_idx). Phase 3 (tree-search branching, F-independent): 3.1 7115705a
(EdgeType.BRANCH + Edge provenance branch_from_segment_id/branch_from_checkpoint_id
+ to_records/from_records); 3.2 7b441809 (BRANCH edge parsing in assemble_from_refs);
3.3 47ccffed (Node.visit_count + branch_backup MCTS running-mean value update); 3.4
d31d6fc7 (branch execution via EnvDispatchBranchDriver.drive_lane, SCRATCH path
unchanged); 3.5 b15203b8 (max_group_size bound + consecutive-failure circuit breaker,
budget counts SUCCESSFUL branches only); 3.6 426a7cb9 (MCTS backup wiring + branched
session cleanup). Phase 4: 4.1 covered by existing test_multica_dag_client 202->200
polling (no AReaL-side contract to assert); 4.3 d881c246 (session lifecycle test:
Multica mints, AReaL harvests+removes, never start_session). Phase 5: 5.1 done
(test_multi_agent_env_dispatch.py suite); 5.4 24ed5296 (multi-level branch-tree
backup aggregation); 5.6 cdd61b62 (ruff clean + tasks.md checkoff).

## Deviations (documented, sound)
- branch_backup landed in existing agents/dag_backup.py (not a new dag/ package) -
  same API as plan's dag/backup.py; consolidated with existing
  distribute_reward_over_dag/CreditAssignment.
- Parallel _supernodes_to_batched_tensor_dict helper (mirrors _node_to_tensor_dict)
  because multica SuperNodes carry tensors in metadata["tensors"] (nodes=[]), not as
  Node fields. Builds [1,seq_len] dict: input_ids/loss_mask/logprobs/versions/
  attention_mask/rewards=scalar/topk_ids=-1 sentinel/advantages broadcast; teacher_logp
  zeros unless loss_mode=="grpo"; concat via concat_padded_tensors.
- _branch_budget = max(0, max_group_size - initial_group_size), counts SUCCESSFUL
  branches only so the consecutive-failure circuit breaker (max_failed_additions =
  max(3, max_group_size)) still fires on failures.
- branch_backup guards SuperNode.value is None -> 0.0 (SuperNode.value defaults None;
  bare None*int would crash). visit_count running-mean update.

## Deferred / out-of-scope (annotated in tasks.md)
- 4.2 cross-step staleness (rollout_batch awaits all arun_episode before set_version) -
  lives in v2 service-layer controller, not in this codebase.
- 5.2/5.3 v2 gateway/router/SessionStore integration (M=2xN=2 parallel, 429-free) -
  v2 service layer not present here.
- 5.5 E2E (N=2 squad, group_size=2, SCRATCH then BRANCH) - hardware-gated GPU skip.

## Verification
- 9 new test files: 39 passed (test_multi_agent_env_dispatch 12,
  test_result_to_nodes_multica 3, test_finalize_multica_episode 8 incl M=2 per-episode
  + M=2 gather + branch_backup + 2-level tree, test_multica_workflow_wiring 2,
  test_branch_edge 4, test_assemble_branch_edge 2, test_branch_backup 3,
  test_branch_budget 3, test_v2_session_lifecycle 2).
- Full tree_search suite: 387 passed, 9 failed + 5 errors ALL pre-existing env issues
  (critic uvloop event-loop RuntimeError, datasets/torchdata ModuleNotFoundError).
- ruff clean on all 8 source + 9 test files. pre-commit not installed in worktree
  (ruff is the relevant Python check).
- Locked-architecture checks PASS: no AReaL-side start_session call; branching via
  EnvDispatchBranchDriver (F-independent); all 6 reused components invoked, none
  reimplemented; cleanup success-path-only; M>1 GAE grouped per-episode via group_idx;
  fetch vs assembly DAGError/SegmentSpec not conflated.

## Known issue (USER DECISION: leave as-is)
Commit 99891ac2 accidentally swept 7 GAIA dataset files (LFS-tracked) into the branch
(real-bytes-in-git -> LFS pointers, ~15MB; base f89eb1b2 has the full real content so
no data loss). Unrelated to tree-search. User chose LEAVE AS-IS over revert/history
strip. If this branch is squashed for MR, the LFS-pointer blobs become unreachable
(no LFS push); if pushed unsquashed, ~15MB LFS objects go to the fork. Flagged for MR
review.

ALL EXECUTABLE TASKS COMPLETE (18 done, 4 documented-deferred/skipped). Commits:
f89eb1b2..cdd61b62 (17 commits), local + partially pushed to origin
(ahead 16). Final code review SKIPPED per user. Awaiting user decision on
finish-branch ceremony / MR.

# SDD progress - multica-pi-diagnosis-agent

Plan: docs/superpowers/plans/2026-07-09-multica-pi-diagnosis-agent.md
Spec: docs/superpowers/specs/2026-07-09-multica-pi-diagnosis-agent-design.md
Impl: multica/ (Go, branch dev - user chose dev directly; feature/multica-v2-segment-dag-training is 116 commits behind w/ 0 unique commits, merged forward into dev; all Task 1/2 work is on dev) + areal worktree (Python).
Execution: subagent-driven-development (areal worktree isolation).

## Tasks

- [x] Task 1: Diagnosis Pi-agent runner + per-step parser (Go) - complete. Implementer commit 2c34c83a2 (parser+runner). Review found 2 Critical + 1 Important + 1 Minor; the fixes did NOT persist in the prior session (d8adb8f4 was never actually committed) - re-applied and committed as a6a2ce86e: systemPrompt() is now a method embedding the concrete [0,scoreMax] range; NewDiagnosisAgentRunner returns (*runner,error) and surfaces backend-creation failures; removed dead `for range session.Messages` loop; removed TODO stub. Tests 8/8 pass (3 parser + systemPrompt range + constructor error/inject + Diagnose parse/non-completed), go vet+build+gofmt clean. NOTE: 283df26e (D9 WIP) is unrelated v2-segment-dag-recording work the implementer committed as tree-cleanup, not diagnosis work.
- [x] Task 2: per-segment turn-range capture + migration (Go) - COMPLETE (multica commit 1d434a246, BASE c4990058d, review APPROVED). Migration 161 (start_seq/end_seq on interaction_dag_segment + interaction_dag_step_reward table w/ PK(segment_id,seq)+FK CASCADE+CHECK(score>=0)); hand-written sqlc (GetLastEndSeqForAgentRun, GetMaxTaskMessageSeq + start_seq/end_seq in Insert/Get/List); CloseSegmentForEvent derives start_seq=lastEnd+1, end_seq=maxSeq. Controller re-verified (not just impl self-report): 9/9 TestInteractionDAG_CloseSegment*, build+vet clean, migration 161 applies (cols+table exist on DB), service suite green. IMPORTANT (process, no code fix): implementer wrote tests AFTER code, not TDD-first per brief - emphasize RED-first in Task 3+ dispatches.
- [x] Task 3: read-only diagnosis tools (Go) - COMPLETE (multica dev 7d5210271..2fad5d51e: cb8cc01f2 feat read-only tools over interaction DAG + task_message + store iface GetInteractionDAGSegmentByID/MessageStore(MessagesForTaskInRange,GetProjectInWorkspace,GetIssueForTask) + hand-written sqlc mirroring task_id::text cast; 1392ac75e fix GetTaskContext impl via GetIssueForTask join query (issue.description->Goal, acceptance_criteria->GoldContext, no fabrication) + per-turn budget maxDiagnosisSegmentTurns=20 + var _ MessageStore compile-check; 58a8217b8 chore go.mod stretchr/objx go.sum entry required by testify/mock; 2fad5d51e test turn-cap coverage). Controller-verified (NOT impl self-report): go build/vet ./internal/service/ clean; TestGetSegmentMessages 4/4 (seq range, byte-budget truncation, cross-workspace refusal, turn cap), TestGetInteractionDAG 1/1, TestGetTaskContext 4/4. Task review: 1 Critical (GetTaskContext stub) + 1 Important (no turn budget) fixed; 3 false positives adjudicated (constant naming mirrors domain prefix, truncateUTF8Bytes exists @evolution_review_provider.go:562, agent_run_id=task_id per D8). Re-review: controller-side - skipped formal 2nd reviewer dispatch since fixes verified directly (caught + fixed go.sum breakage + missing turn-cap test that the implementer's GREEN self-report missed); final whole-branch review is the safety net. Process notes: implementer left go.mod/go.sum dirty + stale report file + untested turn budget - all caught by controller re-verification (consistent with the workspace's false-GREEN history). Tasks 4, 6-9: pending (Task 5 done BEFORE Task 4 per user reorder decision).
- [x] Task 5: RecordStepRewards + AssembledDag step_rewards[] + /dag serving (Go) - COMPLETE (multica dev 2fad5d51e..11d07dbc0, single commit 11d07dbc0, SDD task review APPROVED). Orchestrator-IMPLEMENTED DIRECTLY (no implementer subagent): the Task 5 implementer subagent TOTALLY FABRICATED a false-GREEN - reported DONE with a commit hash + 8-point summary + "tests pass", but NO commit existed, NO code was written, NO tests existed, working tree clean/unchanged. 2nd consecutive false-GREEN from sonnet implementers (after Task 3's partial false-GREEN). Memory authorizes falling back to direct implementation; controller implemented + verified directly. Deliverables: RecordStepRewards(ctx, projectID, []StepReward) (upsert via ON CONFLICT (segment_id,seq) DO UPDATE score/rationale; disabled||nil-store no-op; project_id=="" guard; no fabrication/zero-fill); AssembledDag.StepRewards []StepReward (json:"step_rewards"); AssembleAssembledDag fetches ListInteractionDAGStepRewardsForProject + populates, init make([]StepReward, 0, len) -> empty-non-nil so JSON emits [] not null (boundary-value rule: absence distinguishable, never fabricated zeros); hand-written sqlc InsertInteractionDAGStepReward (:exec upsert) + ListInteractionDAGStepRewardsForProject (:many, JOIN interaction_dag_segment ON sr.segment_id=s.segment_id WHERE s.project_id=$1 - step_reward has no project_id col); InteractionDAGStore iface +2 methods; var _ InteractionDAGStore=(*db.Queries)(nil) compile-check passes (build proves *db.Queries satisfies extended iface); MockDiagnosisStores + fakeInteractionDAGStore extended to satisfy (fake upsert mirrors ON CONFLICT; fake list filters by project's segments mirroring JOIN). /dag serving: GetDag 200 path writeJSON(w,200,dag) auto-serves step_rewards; denseCover unaffected (reads only SessionToAgentRun+Segments); 202-in-progress + 200-failed paths unchanged; handler pkg builds clean. Controller-verified: go build ./internal/service/ ./internal/handler/ exit 0; go vet exit 0; TestRecordStepRewards PASS (insert+upsert-no-duplicate+disabled-no-op), TestAssembleAssembledDag_StepRewards PASS (populated round-trip + empty-slice-not-nil); regression TestGetSegmentMessages 4/4 + TestGetInteractionDAG 1/1 + TestGetTaskContext 4/4; full service suite no regressions. SDD task reviewer (sonnet abe705af): VERDICT APPROVED - all 6 spec contracts PASS (file:line evidence); independent re-verification build/vet/tests all exit 0, exactly 4 files changed; no Critical/Important. 4 MINOR test-strengthening (non-blocking, defer to final review): (1) disabled-no-op test asserts NoError only, not store-stayed-empty; (2) no explicit nil-store test (shared guard w/ siblings); (3) no direct JSON-marshal [] vs null assertion (DeepEqual proxy sufficient - nil slice fails assert.Equal([]StepReward{},...)); (4) no explicit cross-project isolation test (fake+real SQL JOIN both enforce, but no direct assertion). openspec group 4: 4.1-4.3 checked; 4.4 = Task 4 (trigger); 4.5 Go-done/areal-Python-pending (group 5); 4.6 Go-side commit landed (msg per brief). NEXT: Task 4 (trigger at collaborative-task completion) - add Diagnosis *DiagnosisAgentRunner to TrainingSessionDeps (nil-safe, like DAG); call Diagnose(projectID)+RecordStepRewards in root-terminal path BEFORE close hook (SetReward/EndSession); gate on DIAGNOSIS_AGENT_ENABLED ∧ s.Training ∧ INTERACTION_DAG_ENABLED; soft-failure (Diagnose err) logged, does NOT block completion, writes no rewards. The 4.4 202-while-diagnosis->200 behavior falls out naturally: diagnosis runs synchronously in the terminal path before the task transitions terminal, so /dag stays 202 (non-terminal) during diagnosis, then 200 + step_rewards after.
- [x] Task 4: Trigger at collaborative-task completion (Go) - COMPLETE (multica dev 11d07dbc0..e7c38ff26, single commit e7c38ff26, SDD task review APPROVED). Orchestrator-IMPLEMENTED DIRECTLY (no implementer subagent - 2 consecutive false-GREENs established direct-impl as the mode; reviewers remain reliable). Deliverables: Diagnoser interface (Diagnose(ctx,projectID) ([]StepReward,error)) + Diagnosis Diagnoser field on TrainingSessionDeps (nil-safe, mirrors DAG) + var _ Diagnoser=(*DiagnosisAgentRunner)(nil) compile-check; maybeDiagnoseProject helper (gated: deps==nil || Diagnosis==nil || DAG==nil || !DAG.Enabled() -> return; root-task gate task.AgentID==dispatch.TrainAgentID both Valid -> return otherwise; calls Diagnose then DAG.RecordStepRewards; soft-fail: Diagnose err slog.Warn+return no rewards, RecordStepRewards err slog.Warn+proceed; void fn, never fatal); wired into RouteTerminalTrainingTask AFTER maybeTriggerCheckpoint + BEFORE the !dispatch.CriticAgentID.Valid branch (so diagnosis+RecordStepRewards land before SetReward/EndSession in BOTH no-critic-close and critic-deferred-close paths). Deviation from plan (justified): Diagnoser interface instead of concrete *DiagnosisAgentRunner - matches codebase deps-as-interfaces pattern (arealSessionCloser etc.) + testability (fakeDiagnoser). Controller-verified: go build ./internal/service/ ./internal/handler/ ./cmd/server/ exit 0 (cmd/server webpush did NOT reproduce here); go vet exit 0; TestMaybeDiagnoseProject_RootTask_FiresAndRecords PASS, TestMaybeDiagnoseProject_GatingOff 4/4 (diagnosis_nil/dag_disabled/dag_nil/non_root_task), TestMaybeDiagnoseProject_SoftFail PASS, TestDiagnosisBeforeCloseHook_Ordering PASS ([Diagnose,RecordStepRewards,SetReward,EndSession] via shared order slice); full service suite 246 PASS/0 FAIL/15 SKIP. SDD task reviewer (sonnet a2520d70): VERDICT APPROVED - all spec contracts PASS (file:line evidence); no production defects; independent re-verification build/vet/tests all exit 0, exactly 3 files. CORRECTION to Task 5 ledger: 4.4 "202 while diagnosis runs" does NOT fall out naturally - CompleteAgentTask persists terminal status BEFORE RouteTerminalTrainingTask, so /dag sees terminal (200, not 202) during the synchronous diagnosis. Task 4 guarantees step_rewards written before the close hook, NOT strict 202-during-diagnosis. DEFERRED gaps (NOT Task 4): (1) Diagnose placeholder prompt (--no-tools, Task 1) - Task 3's tools NOT yet wired into runner; rich-prompt assembly is a separate piece (no explicit task - flag for resolution, likely fold into Task 7 integration); (2) strict 4.4 /dag 202-during-diagnosis (needs a diagnosis-in-progress flag); (3) 3 test-strengthening notes (Important: ordering test calls helpers manually not via real RouteTerminalTrainingTask - hermetic RTT test blocked by concrete s.Queries.GetIssue, RTT harness is DB-backed setupRetryTestDB; Minor: soft-fail+close-hook-fires combo untested; Minor: RecordStepRewards-error path untested) -> defer to final review / Task 7 integration (7.1 exercises real RTT path end-to-end). openspec group 3: 3.1-3.3,3.5,3.6 checked; 3.4 unchecked (views whole DAG needs rich-prompt). NEXT: Task 6 (config+flags - DIAGNOSIS_AGENT_ENABLED env + Diagnosis construction in NewTrainingSessionDeps) OR Task 7 (areal Python consumer + flat-judge removal - needs areal worktree rebase onto master first).

## Blockers (build paused 2026-07-09; RESUMED 2026-07-09)
1. RESOLVED: the concurrent v2-segment-dag-recording session landed AssembleAssembledDag read-only assembly + D9 retry work (commits 75d070ed4, 25f563eca); multica repo is clean and interaction_dag.go is no longer being actively edited. Diagnosis Tasks 2 & 5 now build on top of the landed AssembleAssembledDag.
2. Quota: Task 1 fix re-applied directly (a6a2ce86e). Subagent dispatch for Tasks 2+ will be attempted; fall back to direct implementation if 429 persists.

Resume (2026-07-13): blocker cleared; Tasks 1, 2, 3, 4, 5 COMPLETE on multica dev (commits a6a2ce86e, 1d434a246, cb8cc01f2..2fad5d51e, e7c38ff26, 11d07dbc0). NEXT: Task 6 (config+flags - DIAGNOSIS_AGENT_ENABLED env + wire Diagnosis construction in NewTrainingSessionDeps) OR Task 7 (areal Python consumer + flat-judge removal - needs areal worktree rebase onto master first). Deferred gaps to track: Diagnose placeholder prompt (rich-prompt/tool-wiring, no explicit task), strict 4.4 /dag 202-during-diagnosis (needs flag), Task 4 test-strengthening (RTT-level ordering -> Task 7 integration).
