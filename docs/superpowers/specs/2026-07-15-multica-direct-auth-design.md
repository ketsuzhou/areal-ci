# MultiCA Direct Authentication Design

## Goal

Let AReaL call MultiCA directly without the `db_bridge`, while preserving the
bridge for MultiCA-to-AReaL traffic. AReaL provides an explicit terminal login
flow that creates and securely reuses a MultiCA personal access token (PAT).

## Scope

- Add an explicit terminal email-code login command in AReaL.
- Persist one reusable PAT at
  `customized_areal/tree_search/agents/credentials.json`.
- Make `MulticaEnvDispatchClient` and `MulticaDagClient` call MultiCA directly
  with that PAT.
- Remove `db-bridge-executor-multica` from the self-host Compose file.
- Retain `db-bridge-stub-multica` for MultiCA-to-AReaL calls.

This change does not alter MultiCA's authentication endpoints, the
MultiCA-to-AReaL database bridge protocol, or AReaL launcher/config dataclasses.

## Authentication command

Add `customized_areal.tree_search.agents.multica_auth` with an explicit login
command:

```bash
uv run python -m customized_areal.tree_search.agents.multica_auth login \
  --base-url http://localhost:8080
```

`--base-url` falls back to `MULTICA_BASE_URL`. Login performs these steps:

1. Prompt for an email address.
2. `POST /auth/send-code` with `{"email": "..."}`.
3. Prompt for the six-digit verification code.
4. `POST /auth/verify-code` with the email and code, receiving a temporary JWT.
5. `POST /api/tokens` with the JWT and a PAT name identifying AReaL and the
   hostname. Request a 90-day expiry, matching MultiCA's CLI default.
6. Validate the returned PAT with `GET /api/me`.
7. Persist the credential only after validation succeeds.

The command reports progress and the authenticated account, but never prints or
logs the JWT, verification code, or PAT.

## Credential persistence and resolution

The credential file stores a version, normalized MultiCA base URL, and PAT. It
is written atomically and assigned owner-only `0600` permissions. The exact
credential file is added to `.gitignore`.

The saved credential is bound to its normalized base URL. A client configured
for a different server must not send the saved PAT and instead reports that the
login command must be run for that server.

Both clients resolve credentials in this order:

1. Explicit `api_key` constructor argument.
2. `MULTICA_API_KEY` environment variable.
3. PAT from `credentials.json` when its base URL matches.

Explicit and environment credentials remain available for automation and CI.
The clients never start an interactive login. Missing, malformed, mismatched,
revoked, or expired credentials produce an actionable error directing the user
to the explicit login command. HTTP 401 responses receive the same guidance.

The default path is derived relative to the authentication module, rather than
embedding an absolute workspace path, so another checkout uses its own
corresponding `customized_areal/tree_search/agents/credentials.json`.

## Direct client calls

`MulticaEnvDispatchClient` remains an asynchronous direct HTTP client. Its base
URL remains `base_url` or `MULTICA_BASE_URL`; its only behavioral change is the
saved-PAT fallback and clearer authentication failures.

`MulticaDagClient` changes from bridge-stub routing to direct MultiCA routing:

- Resolve its base URL from `base_url` or `MULTICA_BASE_URL`.
- Attach `Authorization: Bearer <PAT>` on every DAG polling request.
- Keep the existing polling, parsing, and error types.
- Treat MultiCA's `202` response as not ready and retain transient HTTP retry
  behavior where it is still applicable.
- Remove `AREAL_BRIDGE_STUB_URL` as a DAG-client fallback.

No credential value is included in exceptions or logs.

## Compose topology

Remove the complete `db-bridge-executor-multica` service from
`multica/docker-compose.selfhost.yml`, including its obsolete comments and
`BRIDGE_MULTICA_UPSTREAM_*` wiring.

Keep `db-bridge-stub-multica`: the MultiCA backend still uses it for
`/rl/*` and `/chat/completions` calls toward AReaL. Update its comments so they
describe only the retained MultiCA-to-AReaL direction and no longer reference
the removed executor.

## Failure handling

- Invalid email/code and PAT-creation failures leave any existing credential
  file unchanged.
- Invalid JSON, unsupported credential-file versions, unsafe file types, and
  base-URL mismatches fail closed without exposing credential contents.
- Writes use a temporary file plus atomic replacement; final permissions are
  `0600` even when the process umask is permissive.
- A client receiving HTTP 401 does not delete or rotate credentials
  automatically. It tells the operator to rerun login, keeping distributed
  workers non-interactive and avoiding concurrent token creation.

## Testing

Add focused tests for:

- Email-code endpoint order and payloads.
- JWT use for PAT creation and PAT validation.
- No persistence when any login step fails.
- Atomic persistence, `0600` permissions, malformed-file handling, and
  base-URL binding.
- Credential precedence: explicit argument, environment, then saved PAT.
- Bearer authentication from both clients.
- Direct DAG routing through `MULTICA_BASE_URL` and removal of the
  `AREAL_BRIDGE_STUB_URL` fallback.
- Existing DAG polling and response parsing behavior.
- Compose configuration retaining `db-bridge-stub-multica` while removing
  `db-bridge-executor-multica` and obsolete executor environment variables.

Run focused Python tests, the relevant MultiCA self-host configuration test,
format/lint checks for touched files, and `graphify update .` after code changes.

## Security properties

- PATs never transit the database bridge for AReaL-to-MultiCA requests.
- PATs are never committed, printed, or logged.
- Stored PATs are scoped to a specific MultiCA base URL and protected with
  owner-only filesystem permissions.
- Login is explicit and cannot unexpectedly prompt from a training worker.
