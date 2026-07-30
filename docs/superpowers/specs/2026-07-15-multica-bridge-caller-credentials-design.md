# Multica Bridge Caller Credentials Design

## Goal

Authenticate every AReaL-to-Multica bridge request with the caller's
`MULTICA_API_KEY`, while keeping the credential encrypted in the bridge database
and removing the executor-wide `BRIDGE_MULTICA_UPSTREAM_API_KEY` secret.

## Current behavior

`MulticaEnvDispatchClient` sends `MULTICA_API_KEY` as a bearer token when one is
configured, but the `multica_api` executor strips that token and replaces it with
`BRIDGE_MULTICA_UPSTREAM_API_KEY`. `MulticaDagClient` sends no credential and
therefore depends entirely on the executor-wide key. Request headers can be
encrypted in the database, but encryption is currently optional.

## Architecture

The caller credential becomes the only credential used for the `multica_api`
group. Both clients resolve a token from an explicit `api_key` argument first and
`MULTICA_API_KEY` second, then send it as `Authorization: Bearer <token>` to the
AReaL-side bridge stub.

The AReaL-side stub encrypts the complete `Authorization` header with the existing
Fernet-based `BRIDGE_HEADER_ENCRYPTION_KEY` before inserting the request row. The
Multica-side executor uses the same key to decrypt the header immediately before
forwarding the request. It preserves `Authorization` for `multica_api` traffic
instead of stripping and replacing it. The executor continues to remove alternate
credential headers such as `X-API-Key` and `X-Admin-API-Key` so only the supported
bearer credential reaches the Multica backend.

The AReaL stub and Multica executor must fail fast at startup when their side
serves or executes `multica_api` channels without `BRIDGE_HEADER_ENCRYPTION_KEY`.
This prevents a deployment from silently storing caller PATs in plaintext or
forwarding encrypted ciphertext as a bearer token. Both processes must receive
the same valid Fernet key.

After a request reaches a terminal state, the executor redacts its stored
`Authorization` header for every `multica_api` row, including failures. Redaction
is mandatory for this group and does not depend on the optional global
`BRIDGE_REDACT_TOKENS_AFTER_COMPLETE` setting.

## Client behavior

### `MulticaEnvDispatchClient`

- Keep `api_key` as an optional constructor argument.
- Resolve `api_key` before `MULTICA_API_KEY`, as today.
- Continue attaching the bearer token to dispatch `POST`, cleanup `DELETE`, and
  checkpoint requests.
- Continue supporting explicit `base_url` and `MULTICA_BASE_URL`; deployments
  using the bridge point either value at the AReaL-side stub.

### `MulticaDagClient`

- Add an optional keyword-only `api_key` constructor argument.
- Resolve `api_key` before `MULTICA_API_KEY`.
- Attach the bearer token to every DAG polling `GET`, including retries.
- Keep `AREAL_BRIDGE_STUB_URL` as the default base URL.

If either client has no credential, it sends no `Authorization` header. The bridge
still forwards the request, and the Multica backend remains the authority that
returns `401`. No token value may appear in exceptions or logs.

## Bridge behavior

For every `multica_api` request:

1. The stub captures the caller's `Authorization` header.
2. The stub filters hop-by-hop headers and encrypts `Authorization` before the
   database insert.
3. The executor claims the row and decrypts the header in memory.
4. The executor removes `X-API-Key` and `X-Admin-API-Key`, preserves the decrypted
   `Authorization`, and forwards the request to `BRIDGE_MULTICA_UPSTREAM_URL`.
5. The executor records the response and redacts the stored request credential.

The bridge does not validate or interpret the PAT. Multica's existing auth
middleware remains responsible for token validity, revocation, user identity, and
authorization.

## Configuration and compatibility

- Remove `BRIDGE_MULTICA_UPSTREAM_API_KEY` and
  `BridgeConfig.multica_upstream_api_key` from active configuration, examples,
  compose wiring, and documentation.
- Add `BRIDGE_HEADER_ENCRYPTION_KEY` to the Multica executor service environment
  in `multica/docker-compose.selfhost.yml`; it must be supplied by deployment
  configuration and must match the AReaL-side stub value.
- Update the AReaL and Multica example environment files to describe generating
  and sharing a Fernet key.
- `MULTICA_API_KEY` remains an AReaL caller credential and is not added to the
  Multica-side Docker Compose service.
- This is an intentional deployment-breaking security change for bridged
  `multica_api` traffic: old configurations using only
  `BRIDGE_MULTICA_UPSTREAM_API_KEY` must migrate to caller credentials plus a
  shared encryption key.

## Error handling

- Invalid or missing `BRIDGE_HEADER_ENCRYPTION_KEY` prevents the relevant bridge
  process from starting, with an error naming the required setting but never its
  value.
- A decryption failure fails the queued request without forwarding it and records
  a credential-free diagnostic.
- Missing or rejected caller credentials surface the Multica backend's HTTP
  response unchanged.
- Credential redaction is attempted after both successful forwarding and terminal
  forwarding failure. A redaction error is logged without including the header
  value and does not replace an already-recorded upstream response.

## Testing

Tests cover:

- `MulticaEnvDispatchClient` sending explicit and environment-derived bearer
  credentials to the bridge.
- `MulticaDagClient` sending explicit and environment-derived bearer credentials
  on every poll.
- Stub-side encryption ensuring the database row contains `enc:v1:` ciphertext
  and never the plaintext PAT.
- Startup rejection when either `multica_api` bridge side lacks the encryption
  key.
- Executor-side decryption and pass-through of `Authorization`, with alternate
  credential headers removed.
- Absence of `BRIDGE_MULTICA_UPSTREAM_API_KEY` injection behavior.
- Mandatory redaction for successful and failed `multica_api` rows.
- Existing non-`multica_api` authentication behavior remaining unchanged.

Focused test suites will run before the full relevant bridge and tree-search unit
tests. Because this flow is HTTP and database-queue logic, it requires no GPU or
multi-node integration environment.
