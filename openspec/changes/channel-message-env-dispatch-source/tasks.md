## 1. Add migration 187 extending channel_message_source_check

- [x] 1.1 Create `multica/server/migrations/187_channel_message_env_dispatch_source.up.sql` (drop+re-add check admitting `env_dispatch`)
- [x] 1.2 Create `multica/server/migrations/187_channel_message_env_dispatch_source.down.sql` (rewrite `env_dispatch` rows to `multica`, re-tighten check to `('multica','lark')`)
- [x] 1.3 Confirm up is idempotent (`DROP CONSTRAINT IF EXISTS`) and down is reversible (rewrites rows before re-tightening)

## 2. Verify the fix eliminates the root cause

- [x] 2.1 Confirm no other `insertChannelMessage` caller writes a `source` outside `('multica','lark','env_dispatch')` - audited all callers in multica/server/internal/handler: every other call site uses `multica` or `lark`; only `env_dispatch.go:803` writes `env_dispatch`
- [x] 2.2 End-to-end green-verify deferred: this sandbox has no `go`/`psql`/live server, so migration 187 cannot be applied and the client re-run cannot be executed here. RED evidence (500 / SQLSTATE 23514) is recorded in proposal.md; root-cause elimination confirmed by call-site inspection in 2.1. Green verify to be run on a live/dev multica server after applying migration 187 (re-run `python -m customized_areal.tree_search.agents.multica_client`, expect HTTP 201 + handle, not 500)
- [x] 2.3 Migration files match existing SQL style (4-space indent, lowercase keywords, trailing semicolons - matches migration 186); multica pre-commit has no SQL formatter; gofmt N/A (SQL files, `go` not installed in sandbox). Structural SQL validation recorded as the build check
