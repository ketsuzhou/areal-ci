# Multi-Port Failover for proxy-rollout ↔ db_bridge

**Date:** 2026-06-24
**Target files:**

- `areal/infra/scheduler/local.py` (AReaL)
- `areal/utils/network.py` (AReaL, pre-existing `preferred_ports` support)
- `areal/infra/rpc/guard/app.py` (AReaL, pre-existing `preferred_ports` passthrough)
- `customized_areal/.env` (AReaL)
- `db_bridge/config.py` (le-agent-dev_new)
- `db_bridge/executor.py` (le-agent-dev_new)
- `db_bridge/.env.areal.example` (le-agent-dev_new)

**Reference:** `/dfs/share-groups/letrain/zhoujie/.claude/plans/proud-riding-pie.md`

## Goal

Replace the single pinned port (`AREAL_PROXY_ROLLOUT_PORT=17727`) with a candidate
port pool + automatic failover, so that db_bridge can deterministically reach
proxy-rollout even when the primary port is occupied by a stale process, a
parallel training run, or another service on the shared host.

The previous design pinned proxy-rollout to a single port. If that port was
taken, AReaL silently fell back to a random port, and db_bridge kept
connecting to the pinned URL — a silent-fail that was hard to diagnose. This
spec upgrades both sides to a multi-candidate model with active probing and
connection-level failover.

## Approach

**Candidate pool + dual-side independent decision (coordination-free):**

- AReaL tries each candidate port in order and binds the first free one.
- db_bridge probes each candidate URL via `/health` at startup and fails over
  on connection-level errors at forward time.
- Both sides share the same candidate list (configured in their respective
  `.env` files) but never coordinate at runtime — eventual consistency is
  achieved through probing.

This avoids any runtime signaling channel between AReaL and db_bridge
(which live in separate repos and processes) while eliminating the single-port
failure mode.

## Design decisions (confirmed with user)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Candidate pool size | 5 scattered ports (`17727, 17737, 17747, 17757, 17767`, interval 10) | Scattered ports resist contiguous-block occupation by a single service |
| Probe endpoint | `GET /health` (already exists on proxy-rollout, no auth) | Zero AReaL server change needed; `proxy_rollout_server.py:216-218` returns `{"status":"ok"}` |
| Background reprobe | No — failover triggered only by forward failures | Simpler; db_bridge is request-driven (Supabase queue), so "retry on next forward" is equivalent to on-demand reprobe |

## Changes

### 1. AReaL: `areal/infra/scheduler/local.py` (lines 323-344)

The env parsing for `AREAL_PROXY_ROLLOUT_PORT` was changed from single-int to
comma-separated list. `find_free_ports` (in `areal/utils/network.py`) already
supported `preferred_ports: list[int]` and tries each in order, returning the
first free one — **zero change needed in network.py or guard/app.py**.

```python
alloc_payload: dict[str, object] = {"count": 1}
if role == "proxy-rollout":
    pinned_raw = os.environ.get("AREAL_PROXY_ROLLOUT_PORT")
    if pinned_raw:
        try:
            preferred = [
                int(p.strip()) for p in pinned_raw.split(",") if p.strip()
            ]
            if not preferred:
                logger.warning(
                    f"AREAL_PROXY_ROLLOUT_PORT={pinned_raw!r} parsed to empty list"
                )
            else:
                alloc_payload["preferred_ports"] = preferred
                logger.info(f"proxy-rollout/{idx} preferred ports: {preferred}")
        except ValueError as e:
            logger.warning(
                f"Ignoring invalid AREAL_PROXY_ROLLOUT_PORT={pinned_raw!r}: {e}"
            )
```

**Backward compatibility:** Single value `"17727"` parses to `[17727]`.

### 2. AReaL: `customized_areal/.env` (line 50)

```
AREAL_PROXY_ROLLOUT_PORT=17727,17737,17747,17757,17767
```

Comment updated to describe the candidate-pool semantics and the matching
requirement with `db_bridge/.env.areal`.

### 3. db_bridge: `db_bridge/config.py`

**New env constant:**

```python
ENV_GATEWAY_UPSTREAMS: Final = "BRIDGE_GATEWAY_UPSTREAM_URLS"
```

**New `BridgeConfig` fields:**

```python
gateway_upstream_urls: list[str] = field(default_factory=list)
active_gateway_upstream: str | None = None  # runtime mutable, updated by Executor
```

`gateway_upstream_url` (legacy single-URL field) is retained as fallback.

**New helper `_resolve_gateway_upstreams`:**

```python
def _resolve_gateway_upstreams(src: Mapping[str, str]) -> list[str]:
    """BRIDGE_GATEWAY_UPSTREAM_URLS (comma-sep) takes precedence;
    falls back to legacy BRIDGE_GATEWAY_UPSTREAM_URL as one-element list.
    Trailing slashes stripped. Always returns at least one URL."""
    raw = _get(src, ENV_GATEWAY_UPSTREAMS)
    if raw:
        urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
        if urls:
            return urls
    legacy = _get(src, ENV_GATEWAY_UPSTREAM) or _DEFAULT_GATEWAY_UPSTREAM
    return [legacy.rstrip("/")]
```

**`upstream_for_group` changed** to prefer the runtime-active URL:

```python
def upstream_for_group(self, group: Group) -> str:
    if group != "gateway":
        return self.leagent_upstream_url
    if self.active_gateway_upstream:
        return self.active_gateway_upstream
    if self.gateway_upstream_urls:
        return self.gateway_upstream_urls[0]
    return self.gateway_upstream_url
```

### 4. db_bridge: `db_bridge/executor.py`

**`connect()` — startup probe:**

After building the httpx client, if this is the areal side and no active
upstream is set yet, call `_probe_upstreams()` and pin the first URL that
returns HTTP 200 from `/health`. If all probes fail, log a warning — the
first forward will retry.

**New `_probe_upstreams`:**

```python
async def _probe_upstreams(self) -> str | None:
    """Probe each candidate gateway URL via /health, return first OK."""
    for url in self._config.gateway_upstream_urls:
        try:
            resp = await self._client.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                logger.info("upstream probe OK: %s", url)
                return url
        except Exception as exc:
            logger.warning("upstream probe failed: %s -> %s", url, exc)
    return None
```

**New `_candidate_urls`** — active URL first, remaining candidates in original
order:

```python
def _candidate_urls(self, channel: Channel) -> list[str]:
    if channel.group != "gateway":
        return [self._config.leagent_upstream_url]
    urls = self._config.gateway_upstream_urls or [self._config.gateway_upstream_url]
    active = self._config.active_gateway_upstream
    if active and active in urls:
        return [active] + [u for u in urls if u != active]
    return urls
```

**`_forward()` — failover loop:**

Replaced the single-URL forward with a loop over `_candidate_urls`. On
connection-level failure, log and continue to the next candidate. On success
(any HTTP response, including 4xx/5xx), pin the URL as active and return.
If all candidates fail, raise the last transport exception.

**Failover-triggering exceptions** (intentionally narrow):

- `httpx.ConnectError` — TCP connect refused
- `httpx.ConnectTimeout` — TCP connect timed out
- `httpx.RemoteProtocolError` — peer closed connection unexpectedly
- `httpx.ReadError` — read failure mid-response (conservatively treated as
  connection-level; service may be dying)

**Non-triggering:** HTTP 4xx/5xx responses, `httpx.ReadTimeout` (service may
be slow but is up), `httpx.HTTPStatusError` (not raised — code never calls
`raise_for_status()`).

```python
last_exc: Exception | None = None
for url in urls:
    full_url = url + req.path
    try:
        resp = await self._client.request(
            req.method, full_url, headers=headers,
            content=req.body, timeout=self._config.timeout_for(channel),
        )
        if channel.group == "gateway" and len(urls) > 1:
            self._config.active_gateway_upstream = url
        return resp
    except (httpx.ConnectError, httpx.ConnectTimeout,
            httpx.RemoteProtocolError, httpx.ReadError) as exc:
        logger.warning("upstream unreachable, failing over: %s -> %s", url, exc)
        last_exc = exc
        continue
raise last_exc
```

### 5. db_bridge: `db_bridge/.env.areal.example` (lines 14-19)

```
BRIDGE_GATEWAY_UPSTREAM_URLS=http://127.0.0.1:17727,http://127.0.0.1:17737,http://127.0.0.1:17747,http://127.0.0.1:17757,http://127.0.0.1:17767
# Legacy single-URL form still supported as fallback:
# BRIDGE_GATEWAY_UPSTREAM_URL=http://127.0.0.1:17727
```

## What was NOT changed

| File | Reason |
|------|--------|
| `areal/utils/network.py` | `find_free_ports` already supports `preferred_ports: list[int]` (lines 145-157), tries in order, returns first free |
| `areal/infra/rpc/guard/app.py` | `/alloc_ports` already accepts and passes through `preferred_ports` field |
| `areal/experimental/openai/proxy/proxy_rollout_server.py` | `GET /health` already exists (line 216-218), no auth, returns `{"status":"ok","initialized":bool}` |

## Verification

### Unit (performed during implementation)

1. **AReaL env parsing** — `AREAL_PROXY_ROLLOUT_PORT=17727,17737` → `alloc_payload["preferred_ports"] == [17727, 17737]`; single value `17727` → `[17727]` (backward compat). Verified via `py_compile` + manual trace.
2. **db_bridge config parsing** — `BRIDGE_GATEWAY_UPSTREAM_URLS=u1,u2` → `gateway_upstream_urls == [u1, u2]`; legacy `BRIDGE_GATEWAY_UPSTREAM_URL=u1` → `[u1]`; empty `URLS=` env → falls back to legacy. Verified by running `BridgeConfig.from_env()` with synthetic env.
3. **Port consistency** — AReaL `.env` ports `17727,17737,17747,17757,17767` exactly match db_bridge `.env.areal.example` ports. Verified via shell comparison.
4. **Syntax** — `py_compile` passes on `local.py`, `config.py`, `executor.py`.

### End-to-end (requires real GPU + supabase env — not run during implementation)

1. **Port-occupation resilience:** `python -m http.server 17727 &` → start AReaL → check `ss -tlnp` shows proxy-rollout bound to 17737 (next candidate) → check RolloutController log line `Proxy servers initialized. Addresses: [...]`.
2. **db_bridge startup probe:** start db_bridge executor → log should show `upstream probe OK: http://127.0.0.1:17737` (skipping the occupied 17727).
3. **Forward success:** send a test request through db_bridge → should succeed against 17737.
4. **Runtime failover:** `kill <proxy-rollout pid>` (simulating crash) → next forward should fail over to 17747 if AReaL rebound there, or exhaust candidates and record the row as failed.

### Manual checks

```bash
# AReaL side — which port did proxy-rollout actually bind?
ss -tlnp | grep -E '17727|17737|17747|17757|17767'

# db_bridge side — startup probe results
grep "upstream probe" <executor log>
```

## Risks and edge cases

1. **TOCTOU between probe and forward:** `/health` probe succeeds, but the port
   gets occupied between probe and forward. Probability is low and the forward
   failover loop covers it.
2. **All candidates exhausted:** `_forward` raises the last transport exception;
   `process_one` records the row as failed. No infinite retry — matches the
   "no background reprobe" design decision.
3. **Active stuck on 5xx:** If the active URL persistently returns HTTP 5xx
   (business error, not connection failure), no failover occurs. This is
   intentional — business errors are not port problems, switching is useless
   and would mask the real issue.
4. **Multi-rollout-worker scenario:** Each rollout worker binds its own port
   independently. db_bridge only talks to one (rank 0). If rank 0's
   proxy-rollout dies, db_bridge does not fail over to rank 1 — that requires
   the proxy gateway aggregator (`rollout_controller.py:437 start_proxy_gateway`),
   which is out of scope for this change.
5. **`active_gateway_upstream` mutability:** The field is runtime-mutable on a
   dataclass that otherwise holds config. This is a deliberate trade-off:
   keeping active state in config lets `_candidate_urls` read it directly
   without extra plumbing. If future separation of config vs. runtime state is
   needed, move `active_gateway_upstream` to an `Executor` instance attribute.

## File locations

- AReaL worktree: `/dfs/share-groups/letrain/zhoujie/AReaL-main/.claude/worktrees/multi-port-failover-v2/`
- db_bridge (in-place edits, uncommitted): `/dfs/share-groups/letrain/zhoujie/le-agent-dev_new/db_bridge/`
- Plan file: `/dfs/share-groups/letrain/zhoujie/.claude/plans/proud-riding-pie.md`
