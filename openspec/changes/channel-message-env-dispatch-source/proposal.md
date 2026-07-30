## Why

`POST /api/v1/env-dispatch` with `dispatch_type=message` returns HTTP 500
`"all rollouts failed"`. The scratch-message path in
`multica/server/internal/handler/env_dispatch.go` (`CreateChannelMessage`) calls
`insertChannelMessage(..., source="env_dispatch", ...)`, but migration
`112_channels.up.sql` defines
`channel_message.source TEXT NOT NULL DEFAULT 'multica' CHECK (source IN ('multica', 'lark'))`.
The value `env_dispatch` is not admitted, so the INSERT violates the auto-named
`channel_message_source_check` constraint (SQLSTATE 23514) and the whole dispatch
rolls back.

The client cannot work around this: `source` is hardcoded server-side, and the
request `Message` struct (`service/env_dispatch.go`) only carries `Content`. The
archived `env-dispatch-message-channels` change (migration 186) added the
env-dispatch message code and the `environment_agent_sandbox`/`collaboration_trigger`
schema but did not extend the `channel_message.source` check, so the feature
shipped with a code/schema mismatch.

### Failing evidence (RED)

Running `python -m customized_areal.tree_search.agents.multica_client` against the
live server `http://82.157.184.89:8090` (mode=scratch, dispatch_type=message)
created the project/channel/env and a ready sandbox, then returned:

```
create_env_dispatch failed: status=500 body=...{"error":"create channel message:
create env-dispatch channel message: ERROR: new row for relation \"channel_message\"
violates check constraint \"channel_message_source_check\" (SQLSTATE 23514)",...,
"message":"all rollouts failed"}
```

## What Changes

- Add migration `187_channel_message_env_dispatch_source` that drops and re-adds
  `channel_message_source_check` to admit `env_dispatch`
  (`('multica', 'lark', 'env_dispatch')`).
- Down migration rewrites any `source='env_dispatch'` rows to the default
  `'multica'` before re-tightening to `('multica', 'lark')`, so rollback does not
  reject existing rows or orphan quote/source_message_id references.

## Capabilities

### Modified Capabilities
- `env-dispatch-message-channels`: message dispatch now persists its trigger
  message successfully instead of failing the `channel_message_source_check`.

### New Capabilities
<!-- None: this only unblocks existing, already-shipped behavior. -->

## Impact

- `multica/server/migrations`: new migration pair 187 (up/down). No server code
  changes; the server already writes `source='env_dispatch'` intentionally to label
  env-dispatch-originated messages.
- Live deployment at `http://82.157.184.89:8090` must apply migration 187 and
  restart for the fix to take effect.
- No public API change; no data migration beyond the constraint alteration.
