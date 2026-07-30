## Fix

The env-dispatch message path labels its trigger message with
`source='env_dispatch'` (`multica/server/internal/handler/env_dispatch.go`,
`CreateChannelMessage` -> `insertChannelMessage(..., source="env_dispatch", ...)`),
a value the schema rejects. The minimal fix is to extend the existing CHECK to
admit that value rather than change the code: the `env_dispatch` label is
intentional provenance (it distinguishes env-dispatch-originated messages from
`multica` UI and `lark` integration messages), so dropping it would lose signal.

## Migration 187 (up)

```sql
ALTER TABLE channel_message
    DROP CONSTRAINT IF EXISTS channel_message_source_check,
    ADD CONSTRAINT channel_message_source_check
        CHECK (source IN ('multica', 'lark', 'env_dispatch'));
```

The constraint name is the Postgres default for an unnamed column CHECK
(`<table>_<column>_check`), confirmed by the live error. `DROP IF EXISTS` makes
the up migration safe to re-run after a locally interrupted apply.

## Migration 187 (down)

```sql
UPDATE channel_message SET source = 'multica' WHERE source = 'env_dispatch';
ALTER TABLE channel_message
    DROP CONSTRAINT IF EXISTS channel_message_source_check,
    ADD CONSTRAINT channel_message_source_check
        CHECK (source IN ('multica', 'lark'));
```

Existing `env_dispatch` rows are rewritten to the default `multica` (not deleted)
so the re-tightened CHECK accepts them and no quote/source_message_id references
are orphaned.

## Why not edit migration 186

186 is already applied on the live deployment: the dispatch reached
project/channel/env/sandbox creation before failing, which requires 186's
`environment_agent_sandbox`/`collaboration_trigger` schema. Editing an applied
migration would not re-run on live; a new forward-only migration 187 is the
correct, safe fix.

## Alternatives considered

- Change the server to write `source='multica'`: passes the check but discards the
  env-dispatch provenance label the code deliberately sets. Rejected.
- Add a nullable/new source column: no need; the CHECK just needs one more value.
  Rejected as over-engineering.

## Scope confirmation (upgrade assessment)

The fix alters a DB CHECK constraint (a "database schema change" signal). Per the
comet-hotfix upgrade assessment this was surfaced as a user decision; the user
chose to continue the streamlined hotfix because the change is a minimal
constraint *extension* that makes the schema admit a value the code already
writes - no new capability, table, column, public API, or cross-module
coordination.
