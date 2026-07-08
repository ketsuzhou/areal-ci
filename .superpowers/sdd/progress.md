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
AssembledDag correctly). Independent U4 review: PENDING dispatch.

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

AREAL-SIDE UNITS COMPLETE (U1-U5). Next: multica/server branch U6-U9 (arealrl CloseSegment+
ExportTrajectory, interaction_dag recording, AssembledDag+/dag endpoint, migration 155), then U10
(config + E2E + grep sweep, both repos). Multica path correction pending (U6): proposal.md says
`internal/arealrl/client.go` but actual is `server/internal/arealrl/client.go`.
