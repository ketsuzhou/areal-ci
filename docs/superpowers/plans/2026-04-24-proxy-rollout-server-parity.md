# Proxy Rollout Server Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add all missing base server functionality (inference endpoints, engine
management, cluster integration, helpers) to the custom token-reward proxy rollout
server.

**Architecture:** Direct copy-port from
`areal/experimental/openai/proxy/proxy_rollout_server.py` into
`customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`. Each function is
adapted to coexist with existing token-reward features. No changes to other files.

**Tech Stack:** Python 3.12+ | FastAPI | OpenAI SDK | Anthropic SDK | LiteLLM | uvicorn

______________________________________________________________________

## File Structure

| File                                                               | Action | Responsibility                                     |
| ------------------------------------------------------------------ | ------ | -------------------------------------------------- |
| `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py` | Modify | Add imports, globals, helpers, endpoints, CLI args |

Only one file is modified. All additions are appended or inserted into the existing
structure.

______________________________________________________________________

### Task 1: Add new imports and globals

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py:1-54`
  (imports) and `:186-197` (globals)

- [ ] **Step 1: Add new stdlib and third-party imports**

Add to the imports section (after existing imports, before the `from areal` block):

```python
import inspect
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from anthropic.types.message import Message
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.utils import ModelResponse as LitellmModelResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParams
```

- [ ] **Step 2: Add new AReaL imports**

Add to the `from areal` block:

```python
from areal.api.cli_args import NameResolveConfig, OpenAIProxyConfig
from areal.experimental.openai.client import ArealOpenAI
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import name_resolve, names, seeding
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports, gethostip
```

- [ ] **Step 3: Add new imports from base server module**

Add to the `from areal.experimental.openai.proxy.server` import block:

```python
    ANTHROPIC_MESSAGES_PATHNAME,
    CHAT_COMPLETIONS_PATHNAME,
    RESPONSES_PATHNAME,
    serialize_interactions,
```

- [ ] **Step 4: Add TYPE_CHECKING import for InferenceEngine**

Add after the existing imports:

```python
if TYPE_CHECKING:
    from areal.api import InferenceEngine
```

- [ ] **Step 5: Add new module-level globals**

Insert after the existing globals section (after `_server_port: int = 8000`):

```python
# Engine and client (created via /create_engine and /call with method "initialize")
_engine: InferenceEngine | None = None
_openai_client: ArealOpenAI | None = None

# Port allocation tracking
_allocated_ports: set[int] = set()
_port_alloc_lock = asyncio.Lock()

# Session cleanup timing
_last_cleanup_time: float = 0
_session_timeout_seconds: int = 3600  # Overridden by _setup_openai_client()

# Name_resolve config (needed for cluster registration)
_experiment_name: str | None = None
_trial_name: str | None = None
_name_resolve_type: str = "nfs"
_nfs_record_root: str = "/tmp/areal/name_resolve"
_etcd3_addr: str = "localhost:2379"

# Anthropic request adapter
_adapter = AnthropicAdapter()
```

- [ ] **Step 6: Verify the file still parses**

Run:
`python -c "import ast; ast.parse(open('customized_areal/on_policy_distill/proxy/proxy_rollout_server.py').read())"`
Expected: No output (success)

- [ ] **Step 7: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "feat(proxy): add imports and globals for base server parity"
```

______________________________________________________________________

### Task 2: Add `validate_json_request` helper

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py` (helper
  functions section)

- [ ] **Step 1: Add `validate_json_request` function**

Insert in the Helper Functions section (after `_remove_api_keys_for_session`):

```python
async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )
```

- [ ] **Step 2: Verify parse**

Run:
`python -c "import ast; ast.parse(open('customized_areal/on_policy_distill/proxy/proxy_rollout_server.py').read())"`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "feat(proxy): add validate_json_request helper"
```

______________________________________________________________________

### Task 3: Add engine management functions and endpoints

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`

- [ ] **Step 1: Add `_setup_openai_client` function**

Insert after `validate_json_request`:

```python
def _setup_openai_client():
    """Initialize the OpenAI client from the engine configuration."""
    global _openai_client, _session_timeout_seconds, _admin_api_key
    config = _engine.config
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    openai_cfg = config.openai or OpenAIProxyConfig()
    _openai_client = ArealOpenAI(
        engine=_engine,
        tokenizer=tokenizer,
        tool_call_parser=openai_cfg.tool_call_parser,
        reasoning_parser=openai_cfg.reasoning_parser,
        engine_max_tokens=openai_cfg.engine_max_tokens,
        chat_template_type=openai_cfg.chat_template_type,
    )
    _session_timeout_seconds = openai_cfg.session_timeout_seconds
    with _lock:
        _admin_api_key = openai_cfg.admin_api_key
        if _admin_api_key == DEFAULT_ADMIN_API_KEY:
            logger.warning(
                "Using default admin API key. Change 'admin_api_key' in "
                "OpenAIProxyConfig for non-local deployments."
            )
```

- [ ] **Step 2: Add `/health` endpoint**

Insert before the Admin Endpoints section:

```python
# =============================================================================
# Health & Infrastructure Endpoints
# =============================================================================


@app.get("/health")
def health():
    return {"status": "ok", "initialized": _engine is not None}
```

- [ ] **Step 3: Add `/alloc_ports` endpoint**

```python
@app.post("/alloc_ports")
async def alloc_ports(raw_request: Request):
    """Allocate multiple free ports."""
    global _allocated_ports

    try:
        data = await raw_request.json()
        count = data.get("count")
        if count is None:
            raise HTTPException(
                status_code=400, detail="Missing 'count' field in request"
            )

        if not isinstance(count, int) or count <= 0:
            raise HTTPException(
                status_code=400, detail="'count' must be a positive integer"
            )

        async with _port_alloc_lock:
            ports = find_free_ports(count, exclude_ports=_allocated_ports)
        _allocated_ports.update(ports)

        return {"status": "success", "ports": ports, "host": _server_host}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in alloc_ports: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```

- [ ] **Step 4: Add `/configure` endpoint**

```python
@app.post("/configure")
async def configure(raw_request: Request):
    data = await raw_request.json()
    config = deserialize_value(data.get("config"))
    rank = data.get("rank", 0)
    seeding.set_random_seed(config.seed, key=f"proxy{rank}")
    return {"status": "success"}
```

- [ ] **Step 5: Add `/set_env` endpoint**

```python
@app.post("/set_env")
async def set_env(raw_request: Request):
    data = await raw_request.json()
    for key, value in data.get("env", {}).items():
        os.environ[key] = str(value)
    return {"status": "success"}
```

- [ ] **Step 6: Add `/create_engine` endpoint**

```python
@app.post("/create_engine")
async def create_engine(raw_request: Request):
    global _engine
    if _engine is not None:
        raise HTTPException(status_code=400, detail="Engine already exists")

    data = await raw_request.json()
    engine_class = import_from_string(data.get("engine"))
    init_kwargs = deserialize_value(data.get("init_kwargs", {}))
    _engine = engine_class(**init_kwargs)
    return {"status": "success"}
```

- [ ] **Step 7: Add `/call` endpoint**

```python
@app.post("/call")
async def call_engine_method(raw_request: Request):
    global _engine, _openai_client
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")

    data = await raw_request.json()
    method_name = data.get("method")
    args = deserialize_value(data.get("args", []))
    kwargs = deserialize_value(data.get("kwargs", {}))

    method = getattr(_engine, method_name)
    result = method(*args, **kwargs)

    if method_name == "initialize":
        _setup_openai_client()

    return {"status": "success", "result": serialize_value(result)}
```

- [ ] **Step 8: Verify parse**

Run:
`python -c "import ast; ast.parse(open('customized_areal/on_policy_distill/proxy/proxy_rollout_server.py').read())"`

- [ ] **Step 9: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "feat(proxy): add engine management endpoints (health, alloc_ports, configure, set_env, create_engine, call)"
```

______________________________________________________________________

### Task 4: Add inference helper functions

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`

- [ ] **Step 1: Add `_call_client_create` function**

Insert after `_setup_openai_client`:

```python
async def _call_client_create(
    create_fn,
    request: dict[str, Any] | BaseModel,
    session_id: str,
    extra_ignored_args: list[str] | None = None,
    stream: bool = False,
) -> ChatCompletion | Response | AsyncGenerator[ChatCompletionChunk, None]:
    """Common logic for chat completions and responses."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail=f"Session {session_id} already ended or expired"
            )
        session_data = _session_cache[session_id]

    session_data.update_last_access()

    sig = inspect.signature(create_fn)
    areal_client_ignored_args = ["model"] + (extra_ignored_args or [])
    areal_client_disallowed_args = ["areal_cache"]
    areal_client_allowed_args = list(
        k
        for k in sig.parameters.keys()
        if k not in areal_client_ignored_args and k not in areal_client_disallowed_args
    )

    kwargs = request.model_dump() if isinstance(request, BaseModel) else dict(request)
    dropped_args = []
    for k, v in kwargs.items():
        if k not in areal_client_allowed_args:
            dropped_args.append((k, v))

    for k, _ in dropped_args:
        del kwargs[k]

    def _is_default_value(k: str, v: Any) -> bool:
        if isinstance(request, BaseModel):
            return v == type(request).model_fields[k].default
        return False

    dropped_non_default_args = [
        (k, v)
        for k, v in dropped_args
        if k not in areal_client_ignored_args and not _is_default_value(k, v)
    ]
    if len(dropped_non_default_args):
        dropped_args_str = "\n".join(
            [f"  {k}: {v}" for k, v in dropped_non_default_args]
        )
        _warn_once(
            f"dropped unsupported non-default arguments for areal client:\n"
            f"{dropped_args_str}"
        )

    if "temperature" not in kwargs:
        kwargs["temperature"] = 1.0
        _warn_once("temperature not set in request, defaulting to 1.0")
    if "top_p" not in kwargs:
        kwargs["top_p"] = 1.0
        _warn_once("top_p not set in request, defaulting to 1.0")

    kwargs.pop("stream", None)
    if stream:
        kwargs["stream"] = True

    try:
        return await create_fn(areal_cache=session_data.completions, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
```

- [ ] **Step 2: Add `_translate_anthropic_to_openai_request` function**

```python
def _translate_anthropic_to_openai_request(anthropic_request: dict[str, Any]) -> dict:
    """Translate an Anthropic Messages API request to OpenAI format."""
    openai_request = _adapter.translate_completion_input_params(
        anthropic_request.copy()
    )
    if openai_request is None:
        raise ValueError("Failed to translate request")
    openai_request = dict(openai_request)

    if "messages" in openai_request:
        for msg in openai_request["messages"]:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                msg["content"] = "\n".join(text_parts)

    return openai_request
```

- [ ] **Step 3: Add `_safe_stream_wrapper` function**

```python
async def _safe_stream_wrapper(
    stream: AsyncGenerator,
) -> AsyncGenerator:
    """Wrap an async generator to handle client disconnection gracefully."""
    try:
        async for chunk in stream:
            yield chunk
    except asyncio.CancelledError:
        logger.info("Streaming cancelled by client disconnect")
        raise
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()
```

- [ ] **Step 4: Add `cleanup_engine` function**

Insert in the Cleanup Task section (before `_cleanup_stale_sessions` or after it):

```python
def cleanup_engine():
    """Clean up engine on shutdown."""
    global _engine
    if _engine is not None:
        try:
            _engine.destroy()
            logger.info("Engine destroyed successfully")
        except Exception as e:
            logger.error(f"Error destroying engine: {e}")
        _engine = None
```

- [ ] **Step 5: Verify parse**

Run:
`python -c "import ast; ast.parse(open('customized_areal/on_policy_distill/proxy/proxy_rollout_server.py').read())"`

- [ ] **Step 6: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "feat(proxy): add inference helper functions (call_client_create, translate_anthropic, safe_stream, cleanup_engine)"
```

______________________________________________________________________

### Task 5: Add inference endpoints

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`

- [ ] **Step 1: Add `/chat/completions` endpoint**

Insert before the Token-Level Reward Endpoints section:

```python
# =============================================================================
# OpenAI-Compatible Endpoints
# =============================================================================


@app.post(
    f"/{CHAT_COMPLETIONS_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
)
async def chat_completions(
    request: CompletionCreateParams, session_id: str = Depends(_require_session_key)
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    is_streaming = request.get("stream") is True

    if is_streaming:
        openai_stream = None
        try:
            openai_stream = await _call_client_create(
                create_fn=_openai_client.chat.completions.create,
                request=request,
                session_id=session_id,
                stream=True,
            )

            async def _openai_sse_generator(
                chunk_stream: AsyncGenerator[ChatCompletionChunk, None],
            ) -> AsyncGenerator[str, None]:
                async for chunk in chunk_stream:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            safe_stream = _safe_stream_wrapper(_openai_sse_generator(openai_stream))

            return StreamingResponse(
                safe_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as e:
            if openai_stream is not None and hasattr(openai_stream, "aclose"):
                await openai_stream.aclose()
            logger.error(f"Error setting up streaming response: {e}")
            raise HTTPException(status_code=500, detail=f"Streaming setup failed: {e}")

    return await _call_client_create(
        create_fn=_openai_client.chat.completions.create,
        request=request,
        session_id=session_id,
    )
```

- [ ] **Step 2: Add `/responses` endpoint**

```python
@app.post(
    f"/{RESPONSES_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
)
async def responses(
    request: ResponseCreateParams, session_id: str = Depends(_require_session_key)
) -> Response:
    """OpenAI-compatible responses endpoint."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )
    return await _call_client_create(
        create_fn=_openai_client.responses.create,
        request=request,
        session_id=session_id,
    )
```

- [ ] **Step 3: Add `/v1/messages` (Anthropic) endpoint**

```python
@app.post(
    f"/{ANTHROPIC_MESSAGES_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
)
async def anthropic_messages(
    raw_request: Request, session_id: str = Depends(_require_session_key)
) -> Message | StreamingResponse:
    """Anthropic Messages API compatible endpoint."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    anthropic_request = await raw_request.json()
    is_streaming = anthropic_request.get("stream", False)

    try:
        openai_request = _translate_anthropic_to_openai_request(anthropic_request)
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(
            f"Failed to convert Anthropic request to OpenAI format due to invalid input: {e}"
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid Anthropic request format: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error converting Anthropic request to OpenAI format: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during request conversion."
        )

    if is_streaming:
        openai_stream = None
        try:
            openai_stream = await _call_client_create(
                create_fn=_openai_client.chat.completions.create,
                request=openai_request,
                session_id=session_id,
                stream=True,
            )

            anthropic_sse_stream = (
                _adapter.translate_completion_output_params_streaming(
                    completion_stream=openai_stream,
                    model=anthropic_request.get("model", "default"),
                )
            )

            safe_stream = _safe_stream_wrapper(anthropic_sse_stream)

            return StreamingResponse(
                safe_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as e:
            if openai_stream is not None and hasattr(openai_stream, "aclose"):
                await openai_stream.aclose()
            logger.error(f"Error setting up streaming response: {e}")
            raise HTTPException(status_code=500, detail=f"Streaming setup failed: {e}")

    # Non-streaming
    openai_response = await _call_client_create(
        create_fn=_openai_client.chat.completions.create,
        request=openai_request,
        session_id=session_id,
        stream=False,
    )

    try:
        openai_response_dict = openai_response.model_dump()
        model_response = LitellmModelResponse(**openai_response_dict)
        anthropic_response = _adapter.translate_completion_output_params(model_response)
        if anthropic_response is None:
            raise ValueError("Failed to translate response")

        if "content" in anthropic_response and anthropic_response["content"]:
            anthropic_response["content"] = [
                block.model_dump() if hasattr(block, "model_dump") else block
                for block in anthropic_response["content"]
            ]
        return Message(**anthropic_response)
    except Exception as e:
        logger.error(f"Failed to convert OpenAI response to Anthropic format: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert response: {e}")
```

- [ ] **Step 4: Verify parse**

Run:
`python -c "import ast; ast.parse(open('customized_areal/on_policy_distill/proxy/proxy_rollout_server.py').read())"`

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "feat(proxy): add inference endpoints (chat/completions, responses, anthropic messages)"
```

______________________________________________________________________

### Task 6: Update lifespan and CLI/main()

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`

- [ ] **Step 1: Update `lifespan` to call `cleanup_engine` on shutdown**

Replace the existing `lifespan` function with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background tasks with proper lifecycle."""
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

- [ ] **Step 2: Update `main()` with cluster integration CLI args**

Replace the existing `main()` function with:

```python
def main():
    """Run the token reward proxy server."""
    parser = argparse.ArgumentParser(description="Token Reward Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: 0 = auto-assign)",
    )
    parser.add_argument(
        "--admin-api-key",
        default=DEFAULT_ADMIN_API_KEY,
        help="Admin API key for management operations",
    )
    # name_resolve config (same as base proxy_rollout_server)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--trial-name", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--worker-index", type=int, default=-1)
    parser.add_argument("--name-resolve-type", type=str, default="nfs")
    parser.add_argument(
        "--nfs-record-root", type=str, default="/tmp/areal/name_resolve"
    )
    parser.add_argument("--etcd3-addr", type=str, default="localhost:2379")
    parser.add_argument(
        "--fileroot",
        type=str,
        default=None,
        help="Root directory for log files (unused, for compatibility with rpc_server)",
    )

    args, _ = parser.parse_known_args()

    global _server_host, _server_port, _admin_api_key
    global \
        _experiment_name, \
        _trial_name, \
        _name_resolve_type, \
        _nfs_record_root, \
        _etcd3_addr
    _server_host = args.host
    if _server_host == "0.0.0.0":
        _server_host = gethostip()
    _admin_api_key = args.admin_api_key

    # Set global config for name_resolve
    _experiment_name = args.experiment_name
    _trial_name = args.trial_name
    _name_resolve_type = args.name_resolve_type
    _nfs_record_root = args.nfs_record_root
    _etcd3_addr = args.etcd3_addr

    # Get worker identity
    worker_role = args.role
    worker_index = args.worker_index

    if "SLURM_PROCID" in os.environ:
        worker_index = int(os.environ["SLURM_PROCID"])
    if worker_index == -1:
        raise ValueError("Invalid worker index. Not found from SLURM environ or args.")
    worker_id = f"{worker_role}/{worker_index}"

    # Determine port
    _server_port = args.port if args.port != 0 else find_free_ports(1)[0]
    _allocated_ports.add(_server_port)

    # Configure name_resolve and register this server
    name_resolve.reconfigure(
        NameResolveConfig(
            type=args.name_resolve_type,
            nfs_record_root=args.nfs_record_root,
            etcd3_addr=args.etcd3_addr,
        )
    )
    key = names.worker_discovery(
        args.experiment_name, args.trial_name, args.role, worker_index
    )
    name_resolve.add(key, f"{_server_host}:{_server_port}", replace=True)

    logger.info(
        f"Starting Token Reward Proxy Server on {_server_host}:{_server_port} "
        f"for worker {worker_id}"
    )

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=_server_port,
            log_level="warning",
            timeout_keep_alive=300,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Token Reward Proxy Server")
    finally:
        cleanup_engine()
        logger.info("Token Reward Proxy Server stopped.")
```

- [ ] **Step 3: Verify parse**

Run:
`python -c "import ast; ast.parse(open('customized_areal/on_policy_distill/proxy/proxy_rollout_server.py').read())"`

- [ ] **Step 4: Run pre-commit on the file**

Run:
`pre-commit run --files customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`
Expected: All checks pass (formatting, linting)

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "feat(proxy): add cluster integration, update lifespan and CLI for base server parity"
```

______________________________________________________________________

### Task 7: Smoke test and final verification

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`
  (verification only)

- [ ] **Step 1: Verify the module can be imported**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "from customized_areal.on_policy_distill.proxy.proxy_rollout_server import app; print(f'Routes: {len(app.routes)}')"`
Expected: Successfully imports and shows route count (>10 routes)

- [ ] **Step 2: Verify all expected routes are registered**

Run:
`python -c "from customized_areal.on_policy_distill.proxy.proxy_rollout_server import app; routes = [r.path for r in app.routes if hasattr(r, 'path')]; expected = ['/health', '/alloc_ports', '/configure', '/set_env', '/create_engine', '/call']; print('Missing:', [p for p in expected if p not in routes]); print('All routes:', sorted(routes))"`
Expected: No missing routes; all expected + existing RL endpoints present

- [ ] **Step 3: Run pre-commit on the full file**

Run:
`pre-commit run --files customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`
Expected: All checks pass

- [ ] **Step 4: Final commit (if any formatting fixes needed)**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "style(proxy): fix linting for proxy_rollout_server parity changes"
```
