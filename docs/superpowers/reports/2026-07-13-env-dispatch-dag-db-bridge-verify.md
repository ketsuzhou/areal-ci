# Verification Report: env-dispatch-dag-db-bridge

**Date**: 2026-07-13
**Change**: `env-dispatch-dag-db-bridge` (comet, full workflow)
**Phase**: verify (full mode)
**Scope**: two-repo - areal (`customized_areal/tree_search/`) + multica (`db_bridge/`)
**Base ref (areal)**: `853e5cbcda6929e80961ed778194c1402fb7d9ba`
**Tests**: areal 16 passed; multica db_bridge 181 passed, 1 skipped (environmental)

## Summary

| Dimension    | Status |
|--------------|--------|
| Completeness | 21/21 tasks complete; 2 modified requirements, all implemented |
| Correctness  | 10/10 spec scenarios implemented + tested |
| Coherence    | Design decisions D1-D6 followed; no spec/design-doc drift |

**Final assessment**: No CRITICAL issues, no contractual WARNINGs. Three SUGGESTION-level
notes (one pre-existing, two minor test-coverage gaps). Ready for archive with noted
improvements.

## Verification scope

This change spans two repos, verified in isolated worktrees:
- **areal** (`.../areal/.claude/worktrees/env-dispatch-dag-db-bridge`, branch
  `worktree-env-dispatch-dag-db-bridge`): `multica_dag_client.py` repoint +
  `multica_environment_protocol.md` + tests. Diff: 4 code/test/doc files + planning
  artifacts (13 files, +1543/-26 vs base ref).
- **multica** (`.../multica/.claude/worktrees/env-dispatch-dag-db-bridge`, branch
  `worktree-env-dispatch-dag-db-bridge`): `db_bridge/` channels/config/executor/relay/
  schema/stub_server/entrypoints + `.env` examples + 6 test files (8 commits, 14 files,
  +406/-26).

## Completeness

### Task completion
`openspec instructions apply` reports `total: 21, complete: 21, remaining: 0`. All tasks
checked `[x]` (grep: 21 `[x]`, 0 `[ ]`).

### Spec coverage (delta spec `specs/v2-segment-dag/spec.md`)
Two MODIFIED requirements:

1. **Polling AssembledDag return at task completion** - implemented:
   `env_dispatch_dag` channel (`channels.py:163-171`) + `rpc_env_dispatch_dag` table
   (`schema.sql:65`) + `MulticaDagClient` repoint to `AREAL_BRIDGE_STUB_URL`
   (`multica_dag_client.py:112`). `multica_api` group reconciled (D1).

2. **V2 no-reward segment close** - implemented: `rl_close_segment` channel
   (`channels.py:125-133`) + `rpc_rl_close_segment` table (`schema.sql:60`) + stub route
   auto-registration (`stub_server.py:346-348`). Gateway-group forwarding reuses existing
   `gateway_upstream_url` path; session-key auth forwarded end-to-end.

No requirement unimplemented.

## Correctness

### Requirement -> implementation mapping

| Requirement clause | Evidence |
|---|---|
| DAG fetch via `AREAL_BRIDGE_STUB_URL`, not direct multica | `multica_dag_client.py:112` (env default), `:137-139` (no auth header) |
| `202`/`200`/`404` semantics preserved end-to-end | `multica_dag_client.py:144-165`; stub generic pass-through (`stub_server.py:349` `_make_handler`) |
| Each poll = one bridged request | `multica_dag_client.py:143` (one `client.get` per iteration) |
| env-dispatch channels in `multica_api` group | `channels.py:146,154,163` |
| stub on areal host, executor on multica host | `channels.py:71,76-80` (`stub_side`/`executor_side`) |
| `close_segment` reachable via stub (no 404) | `channels.py:125-133` + `stub_server.py:346-348` (registry-driven routes) |
| session-key auth forwarded end-to-end | executor gateway path = pass-through (`executor.py`: no strip for gateway); `relay.filter_request_headers` preserves `Authorization` (`relay.py:51-57`) |
| `multica_api` executor injects upstream key + strips caller auth | `executor.py:97-105` + `relay.strip_credentials` (`relay.py:65-70`) |
| `BRIDGE_MULTICA_UPSTREAM_URL` distinct from le-agent upstream | `config.py:40,69,143,265-270` |

### Scenario coverage

| Scenario | Test |
|---|---|
| In-progress poll (202) | `test_multica_dag_client.py::test_get_dag_polls_until_200` |
| Completed poll returns AssembledDag (200) | `test_multica_dag_client.py::test_get_dag_polls_until_200` + `test_env_dispatch_dag_get_relays_path_param_and_status` |
| Unknown project rejected (404) | `test_multica_dag_client.py::test_get_dag_404_raises` + `test_env_dispatch_dag_404_passes_through` |
| DAG fetch routed through db_bridge | `test_multica_dag_client.py::test_get_dag_reads_bridge_stub_url_from_env_and_sends_no_auth` |
| Bridged poll preserves semantics (202 then 200) | `test_multica_dag_client.py::test_get_dag_polls_until_200` |
| Bridged unknown project -> DagNotFound | `test_env_dispatch_dag_404_passes_through` (bridge) + `test_get_dag_404_raises` (client) |
| Close produces reward-less trajectory | `test_close_segment_relays_session_key_and_response` (200 + trajectory_id) |
| Close without active completions rejected (typed 400) | `test_close_segment_no_active_segment_passes_through_400` (400 survives, not 502) |
| Close segment routed through db_bridge (no 404) | `test_close_segment_relays_session_key_and_response` (stub serves route) |
| Session stays live after close | (behavioral contract on AReaL gateway side; bridge transparently relays repeated calls - not bridge's concern) |

All scenarios covered with test evidence.

### Non-goal scope discipline (verified)
- `MulticaEnvDispatchClient` (`multica_client.py:53-60`) still uses direct
  `MULTICA_BASE_URL`/`MULTICA_API_KEY` - NOT repointed (correct, non-goal).
- `MulticaSweLegoProvider` (`environment.py:232-239`) and `verifier-rl` TS extension
  untouched (correct, non-goal).
- `multica_dag_client.py` has zero `MULTICA_BASE_URL`/`MULTICA_API_KEY` references (clean repoint).

### Go client "no change needed" (verified)
`multica/server/internal/arealrl/client.go` already has `closeSegmentPath = "/rl/close_segment"`
(line 69), `stubBaseURL` routing through the stub (line 76), `CloseSegment` with session-key
auth (line 174), and nullable `trajectory_id` for the no-active-segment case (line 159-160).
Confirms proposal claim - the only gap was the missing channel registration, now fixed.

## Coherence

### Design adherence (D1-D6)
- **D1** (multica side + multica_api group): `channels.py:33-34,69-80` - PASS.
- **D2** (env_dispatch_dag channel + table): `channels.py:163-171`, `schema.sql:65` - PASS.
- **D3** (pass-through-pending polling): `multica_dag_client.py:142-163` (one row per poll, client decides re-poll) - PASS.
- **D4** (auth/header handling): `config.py:40-44,143,146`, `executor.py:97-105`, `relay.py:65-70` - PASS.
- **D5** (client repoint): `multica_dag_client.py:100-109,112,132,144-168` - PASS.
- **D6** (rl_close_segment gateway channel + table): `channels.py:125-133`, `schema.sql:60`, `stub_server.py:346-348` - PASS.

### Open questions / build-phase verification (all resolved)
1. **Multica loopback port** - resolved by commit `011880f31`: multica Go server default
   `$PORT=8080`, so `BRIDGE_MULTICA_UPSTREAM_URL` defaults to `http://127.0.0.1:8080`
   (same loopback port as AReaL gateway default, different host - no collision).
2. **X-Bridge-User-Id on DAG calls** - resolved via existing mechanism: stub supports
   `X-Bridge-User-Id` header OR `BRIDGE_USER_ID` env (`stub_server.py:37,172`); executor
   forwards `bridge_user_id` (`executor.py:96`). Single-user areal stub uses env default.
3. **Schema apply** - deployment note recorded in Task 21 (apply both tables against shared
   Supabase, idempotent `schema.sql`).

### Design doc consistency / drift (comet-verify items 3, 6, 7)
- Technical design doc `docs/superpowers/specs/2026-07-13-env-dispatch-dag-db-bridge-design.md`
  locatable; frontmatter `canonical_spec: openspec` defers requirements to openspec artifacts
  and provides deeper implementation/risk/testing detail. No contradiction with change
  `design.md` (complementary, not duplicate).
- **No delta-spec-vs-design-doc drift.** Delta spec is silent on `502`/`503`/`504` re-poll and
  `BRIDGE_MULTICA_UPSTREAM_API_KEY` (implementation details); `design.md` documents them. The
  working-tree `design.md` edits (env var rename, 502/503/504 re-poll, diagram) accurately
  reflect the built implementation - they reduce drift, not create it.

### Code pattern consistency
- areal client follows conventions: `from __future__ import annotations`, dataclass DTOs,
  structured error hierarchy (`DagError`/`DagNotFound`/`DagForbidden`/`DagTimeout`), explicit
  `__all__`, `_transport` underscore-prefix for test injection (matches `segment_dag_trainer.py`).
- multica db_bridge follows conventions: `Final` env constants, `@dataclass(frozen=True, slots=True)`
  `Channel`, group-side derivation as `@property`, idempotent schema loop, generic stub handler.
- No dead code, no TODO/FIXME/XXX/HACK markers in changed files.
- Caller compatibility: all `MulticaDagClient(` constructor calls are in tests; production caller
  (`segment_dag_trainer.py:172`) receives the client as an injected dependency, so the removed
  `api_key` parameter breaks no production code.

## Dirty-worktree acceptance (comet-verify Step 1)

Two uncommitted working-tree changes in the areal worktree, accepted per user decision
("Accept & continue verify"):

1. **`.comet.yaml`**: `verify_mode: null -> full` - a verify-phase artifact produced by
   `comet-state scale`. Acceptable per Step 1 point 2 (verify-phase artifact).
2. **`design.md`**: inline doc updates reflecting build decisions - `MULTICA_UPSTREAM_API_KEY`
   -> `BRIDGE_MULTICA_UPSTREAM_API_KEY` (config-constant collision fix), `502`/`503`/`504`
   re-poll documented, diagram env var name corrected. Both verified against built code
   (`config.py:44`, `multica_dag_client.py:149`). These are accurate doc reconciliation of
   already-built implementation, not unverified implementation. Kept in working tree as the
   reference design doc for verification; folds into the final verify/finishing commit.

**Acceptance reason**: the `design.md` edits match verified code (no semantic divergence);
committing them is doc-only reconciliation, not implementation work. Non-CRITICAL.
**Impact scope**: `design.md` only (no code, test, or task changes); both files committed at
the finishing step.

## Issues by priority

### CRITICAL
_(none)_

### WARNING
_(none - all findings below are pre-existing or coverage gaps, not contractual deviations)_

### SUGGESTION (nice to fix, non-blocking)

1. **Pre-existing: httpx transport timeouts not caught in dag-client poll loop.**
   `multica_dag_client.py:143` - `resp = client.get(url)` has no `try/except`; a raw
   `httpx.ReadTimeout`/`ConnectTimeout` (from `http_timeout=10.0`) propagates uncaught
   instead of being re-polled or converted to `DagTimeout`. **Pre-existing** (the old code
   also lacked the try/except; this change only expanded the re-poll tuple from `(202)` to
   `(202, 502, 503, 504)` - an improvement). The design explicitly says timeout behavior is
   "unchanged". The documented transient manifestations (`502`/`503`/`504`, returned by the
   stub on its own timeouts) ARE handled; only raw TCP/infra stalls would crash the loop.
   Low probability on loopback. **Recommendation**: in a follow-up, wrap `client.get(url)`
   in `try/except httpx.TimeoutException` and re-poll (or raise `DagTimeout` if deadline
   exceeded) for defense-in-depth.

2. **Partial test coverage for 502/503.** Only `504` has a dedicated re-poll test
   (`test_get_dag_504_repolls_until_200`). `502`/`503` share the code path
   (`multica_dag_client.py:149` tuple membership) but are not independently tested.
   Trivially correct. **Recommendation**: parametrize the test over `(502, 503, 504)`.

3. **No combined "bridged 404" test.** The 404->`DagNotFound` mapping and the bridge-URL
   routing are tested separately but never combined in one test. Behavior is URL-independent
   so this is a coverage gap, not a correctness issue. **Recommendation**: add a test that
   drives a 404 through the bridge stub URL.

## Test results (fresh run, 2026-07-13)

```
areal-side:
  customized_areal/tree_search/tests/test_multica_dag_client.py
  customized_areal/tree_search/tests/test_segment_dag_training_path.py
  -> 16 passed in 8.55s

multica-side:
  db_bridge/tests/
  -> 181 passed, 1 skipped in 3.34s
  (skipped: test_schema_integration.py - live Postgres, BRIDGE_TEST_PG_DSN not set; environmental)
```

## Confirmation items
- All tests pass (areal + multica, fresh run). Yes.
- No hardcoded keys or security issues. Confirmed - upstream keys read from env
  (`BRIDGE_MULTICA_UPSTREAM_API_KEY`), caller credentials stripped before forwarding
  (`relay.strip_credentials`), stub is loopback-only.
