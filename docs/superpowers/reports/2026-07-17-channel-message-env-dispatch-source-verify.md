# Verification Report: channel-message-env-dispatch-source

**Date:** 2026-07-17
**Change:** channel-message-env-dispatch-source (hotfix)
**verify_mode:** light
**verify_result:** PENDING (runtime build/test/live verify deferred - see below)

## Summary

Hotfix adds `multica` migration `187_channel_message_env_dispatch_source` extending the
`channel_message_source_check` constraint to admit `source='env_dispatch'`, fixing the
env-dispatch message-dispatch HTTP 500 (`SQLSTATE 23514` / `"all rollouts failed"`).

## RED evidence (pre-fix, reproduced)

Running `python -m customized_areal.tree_search.agents.multica_client` against the live
server `http://82.157.184.89:8090` (mode=scratch, dispatch_type=message) returned:

```
create_env_dispatch failed: status=500 ... "violates check constraint
\"channel_message_source_check\" (SQLSTATE 23514)" ... "all rollouts failed"
```

Recorded in `openspec/changes/channel-message-env-dispatch-source/proposal.md`.

## Root-cause elimination (by call-site inspection)

Audited every `insertChannelMessage*` call site in `multica/server/internal/handler`:

| Call site | source value |
|-----------|--------------|
| `env_dispatch.go:803` (CreateChannelMessage) | `env_dispatch` (the bug) |
| `agent_radar_executor.go:663` | `multica` |
| `agent_radar_executor.go:800` | `multica` |
| `agent_transport.go:871` | `multica` |
| `chat_output_target.go:318` | `multica` |
| `channel.go:3310` | `lark` |
| `channel.go:5024` | `multica` |
| `issue_thread_backflow.go:142` | `multica` |
| `channel_member_system_event.go:58` | `multica` |
| `channel_unfollow_event.go:65` | `multica` |
| `wendy_handoff_dispatch.go:271` | `multica` |

Only `env_dispatch.go:803` writes a value the original CHECK `('multica','lark')` rejects.
Migration 187 admitting `'env_dispatch'` eliminates the root cause; no other caller writes
a source outside `('multica','lark','env_dispatch')`.

## Lightweight 6-check results

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | tasks.md all `[x]` | PASS | 6/6 tasks checked |
| 2 | Changed files match tasks.md | PASS | `multica/server/migrations/187_channel_message_env_dispatch_source.{up,down}.sql` + OpenSpec artifacts (multica is a nested git repo, untracked by areal) |
| 3 | Build passes | **NOT RUN in sandbox** | No `go`/`psql` available. Structural SQL validation passed (`python3 .../validate_migration_187.py`, exit 0): up admits `env_dispatch`, down rewrites rows + re-tightens, parens balanced, semicolon-terminated. **Not a real build.** Real `make migrate-up` / `go build ./server/...` / `go test` DEFERRED. |
| 4 | Related tests pass | **NOT RUN in sandbox** | No `go`/`psql` to run multica migration/handler tests. DEFERRED. |
| 5 | No obvious security issues | PASS | Constraint extension only (`CHECK source IN (...)`); no secrets, no unsafe ops, no new privileges. Down rewrites rows to the default instead of deleting (avoids orphaning quote/source_message_id references). |
| 6 | Code review | SKIPPED | `review_mode: off` (hotfix preset default) |

## Deferred runtime verification (requires a live multica environment)

This sandbox lacks `go`, `psql`, and a running multica server, so the runtime verification
cannot be executed here. On a host with Go + Postgres + the multica server:

1. `cd multica && make migrate-up` - apply migration 187, confirm no errors.
2. Re-run `python -m customized_areal.tree_search.agents.multica_client` - expect
   **HTTP 201 + EnvDispatchHandle**, not 500 / SQLSTATE 23514.
3. `go build ./server/...` and `go test ./server/internal/handler/... ./server/migrations/...`.
4. Confirm `channel_message` rows from an env-dispatch rollout now persist with
   `source='env_dispatch'`.

## Conclusion

Root cause eliminated by code inspection; migration 187 structurally validated. Runtime
build, tests, and live end-to-end green verification are DEFERRED to a live multica
environment and have NOT been run in this sandbox. Per the verification-before-completion
iron law, I do not claim the fix is verified at runtime; `verify_result` remains PENDING
until the deferred checks above run green.
