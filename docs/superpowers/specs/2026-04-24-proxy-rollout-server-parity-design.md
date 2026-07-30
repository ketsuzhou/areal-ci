# Proxy Rollout Server Parity Design

**Date:** 2026-04-24 **Target file:**
`customized_areal/on_policy_distill/proxy/proxy_rollout_server.py` **Reference:**
`areal/experimental/openai/proxy/proxy_rollout_server.py`

## Goal

Add all missing base server functionality to the custom token-reward proxy rollout
server, so it can serve as a full-featured inference proxy in addition to its existing
token-level reward management capabilities.

## Approach

**Direct copy-port (Approach A):** Copy and adapt missing functions, endpoints, and
infrastructure from the base server into the custom server. Each function is adapted to
coexist with the existing token-reward features (custom serialization,
`TokenRewardSessionData`, custom cleanup, etc.).

## Changes

### 1. New Imports

Add the following imports (all from base server dependencies):

**stdlib:** `inspect`, `time`, `asyncio.Lock` **collections.abc:** `AsyncGenerator`
**typing:** `TYPE_CHECKING`, `Any` **Third-party:**

- `anthropic.types.message.Message`
- `fastapi.exceptions.RequestValidationError`
- `fastapi.responses.StreamingResponse`
- `litellm` Anthropic adapter + ModelResponse
- `openai` ChatCompletion, ChatCompletionChunk, CompletionCreateParams, Response,
  ResponseCreateParams

**AReaL:**

- `areal.api.cli_args.NameResolveConfig`, `OpenAIProxyConfig`
- `areal.experimental.openai.client.ArealOpenAI`
- `areal.utils.name_resolve`, `names`, `seeding`
- `areal.utils.dynamic_import.import_from_string`
- `areal.utils.hf_utils.load_hf_tokenizer`
- `areal.utils.network.find_free_ports`, `gethostip`

**From base server module:**

- `ANTHROPIC_MESSAGES_PATHNAME`, `CHAT_COMPLETIONS_PATHNAME`, `RESPONSES_PATHNAME`
- `serialize_interactions`

### 2. New Module-Level Globals

```python
_engine: InferenceEngine | None = None
_openai_client: ArealOpenAI | None = None
_adapter = AnthropicAdapter()
_allocated_ports: set[int] = set()
_port_alloc_lock = asyncio.Lock()
_last_cleanup_time: float = 0
_session_timeout_seconds: int = 3600  # Overridden by _setup_openai_client()
_experiment_name: str | None = None
_trial_name: str | None = None
_name_resolve_type: str = "nfs"
_nfs_record_root: str = "/tmp/areal/name_resolve"
_etcd3_addr: str = "localhost:2379"
```

### 3. New Helper Functions

| Function                                 | Source         | Adaptation  |
| ---------------------------------------- | -------------- | ----------- |
| `validate_json_request`                  | Base, line 133 | None needed |
| `_setup_openai_client`                   | Base, line 260 | None needed |
| `_call_client_create`                    | Base, line 522 | None needed |
| `_translate_anthropic_to_openai_request` | Base, line 690 | None needed |
| `_safe_stream_wrapper`                   | Base, line 717 | None needed |
| `cleanup_engine`                         | Base, line 914 | None needed |

### 4. New Endpoints

| Endpoint                         | Method | Purpose                                             |
| -------------------------------- | ------ | --------------------------------------------------- |
| `/health`                        | GET    | Health check with engine initialization status      |
| `/alloc_ports`                   | POST   | Allocate free ports for Slurm fork_workers          |
| `/configure`                     | POST   | Deserialize and apply config, set random seed       |
| `/set_env`                       | POST   | Set environment variables                           |
| `/create_engine`                 | POST   | Create inference engine instance                    |
| `/call`                          | POST   | Call engine methods (including `initialize`)        |
| `/{CHAT_COMPLETIONS_PATHNAME}`   | POST   | OpenAI chat completions (streaming + non-streaming) |
| `/{RESPONSES_PATHNAME}`          | POST   | OpenAI responses endpoint                           |
| `/{ANTHROPIC_MESSAGES_PATHNAME}` | POST   | Anthropic Messages API (streaming + non-streaming)  |

All inference endpoints check `_openai_client is not None` before processing and return
HTTP 500 with initialization instructions if not set up.

### 5. Lifespan Update

Update the `lifespan` context manager to call `cleanup_engine()` on shutdown:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_stale_sessions())
    logger.info(f"Token Reward Proxy Server started on {_server_host}:{_server_port}")
    yield
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    cleanup_engine()
```

### 6. Updated CLI & main()

Add CLI arguments matching the base server:

- `--experiment-name`, `--trial-name`, `--role` (for name_resolve registration)
- `--worker-index` (default -1, overridden by `SLURM_PROCID`)
- `--name-resolve-type`, `--nfs-record-root`, `--etcd3-addr`

Update `main()` to:

- Configure `name_resolve` via `name_resolve.reconfigure(NameResolveConfig(...))`
- Register server via `name_resolve.add(key, f"{host}:{port}", replace=True)`
- Support `SLURM_PROCID` for worker index override
- Use `gethostip()` when host is `0.0.0.0`
- Support `--port 0` for auto-assignment via `find_free_ports`
- Pass `log_level="warning"`, `timeout_keep_alive=300` to `uvicorn.run()`
- Call `cleanup_engine()` in `finally` block

Keep existing `--admin-api-key` argument.

### 7. Behavioral Parity Notes

- `end_session`: Already removes API keys in current custom code (no change needed).
- `grant_capacity`: Custom returns `{"message": "success", "capacity": N}`; base returns
  `{"capacity": N}`. Keep custom format (more informative, no consumers depend on exact
  parity).
- `export_trajectories`: Already removes session from cache and cleans up keys (no
  change).
- `start_session`: Custom uses async background cleanup instead of synchronous
  on-demand. Keep current approach (arguably better for throughput).

## Not in Scope

- No changes to `server.py`, `client.py`, `cache.py`, or `workflow.py`
- No changes to base server
- No refactoring of existing token-reward functionality

## Testing

- Verify `/health` returns correct initialization status before and after engine
  creation
- Verify inference endpoints work with a real engine instance
- Verify existing token-reward endpoints still work after changes
- Verify `cleanup_engine()` is called on shutdown
- Verify name_resolve registration works with CLI args
