# OpenRouter Remote Rollout Proxy Design

## Status

Approved design for supporting remote OpenAI-compatible LLMs as rollout behavior in AReaL proxy training. The three questions that were open during the initial draft have been resolved (see "Resolved Implementation Questions" at the end).

## Goal

Allow AReaL proxy users to call a remote model through OpenRouter on a per-request basis while still collecting trajectories that can train the local actor model. Local SGLang/vLLM remains the default for `model="default"`. Remote rollout is selected by using a request model name with the `remote:` prefix.

Example:

```json
{
  "model": "remote:anthropic/claude-3.5-sonnet",
  "messages": [{"role": "user", "content": "Solve this task."}]
}
```

AReaL routes this request to OpenRouter and forwards `model="anthropic/claude-3.5-sonnet"` after stripping the `remote:` prefix.

## Non-Goals

- Exact on-policy PPO using remote provider behavior logprobs.
- Remote streaming support in the first implementation.
- Remote `responses` or Anthropic `/v1/messages` support in the first implementation.
- Multi-choice `n > 1` support for remote requests.
- Training a remote provider model directly.

## Existing Context

AReaL proxy mode currently routes OpenAI-compatible calls through `ProxyRolloutServer` into `ArealOpenAI`. `ArealOpenAI` tokenizes the request with the local tokenizer, calls local `engine.agenerate()`, builds an OpenAI `ChatCompletion`, and stores an `InteractionWithTokenLogpReward` in the session `InteractionCache`. Export then serializes tensor data for PPO/GRPO training.

Remote providers can return useful text, but arbitrary remote models do not return token IDs or behavior logprobs compatible with the local trainable actor. The design therefore treats remote model outputs as off-policy behavior text and converts them into local-tokenized training samples.

## Implementation Approach

### Module Layout

- **New file** `areal/experimental/openai/proxy/remote_rollout.py` owns `RemoteRolloutClient`.
- **New file** `areal/experimental/openai/_prompt_utils.py` extracts `_extract_images_from_messages`, `_ensure_message_dict_list`, and the `apply_chat_template` re-export from `client.py` so both local and remote paths share them. `client.py` re-imports from `_prompt_utils` — no behavior change.
- **Modified** `areal/experimental/openai/proxy/proxy_rollout_server.py` adds a `_remote_client: RemoteRolloutClient | None` module-level global, constructed in `_setup_openai_client()` from the `agent_cfg`-derived tokenizer, `chat_template_type`, `engine_max_tokens`, and `recompute_enabled=agent_cfg.should_compute_prox_logp()`. The `RemoteRolloutClient` is always constructed (its OpenRouter HTTP client is lazy), regardless of `enable_remote_rollout`; that flag only controls the trainer-config warning.
- **Modified** `areal/api/cli_args.py` adds `ActorConfig.enable_remote_rollout: bool` and a warning in `__post_init__`.
- **Unchanged** `areal/experimental/openai/client.py` (apart from the `_prompt_utils` extraction), `cache.py`, `types.py`. The local generation path is untouched.

### `RemoteRolloutClient` API

```python
class RemoteRolloutClient:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        chat_template_type: str,        # must be "hf" for MVP
        engine_max_tokens: int | None = None,
        recompute_enabled: bool = False,
    ) -> None: ...

    async def create_completion(
        self,
        request: dict[str, Any],
        session_cache: InteractionCache,
    ) -> ChatCompletion: ...
```

The OpenRouter `AsyncOpenAI` client is constructed lazily on first call from `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` (env, default `https://openrouter.ai/api/v1`). Missing key fails before any network call.

### Dispatch in `chat_completions()`

```
chat_completions(request, session_id)
  ├─ resolve session_cache from _session_cache[session_id].completions
  ├─ model = request.get("model", "default")
  ├─ if model == "default":       → existing local path
  ├─ if model.startswith("remote:"): → _remote_client.create_completion(...)
  └─ else:                         → HTTPException(404)
```

`anthropic_messages()` is not modified for MVP. Remote routing is OpenAI-shaped only.

### Per-Request Lifecycle

1. **Parse & validate model**
   - `provider_model = model.removeprefix("remote:")`
   - Empty → `HTTPException(400, "remote: model name is empty")`
   - `chat_template_type != "hf"` → `HTTPException(400, "remote rollout MVP supports chat_template_type='hf' only")`
   - `stream=True` → `HTTPException(400, "remote streaming not supported in MVP")`
   - `n != 1` → `HTTPException(400, "remote n != 1 not supported")`
   - `recompute_enabled` is False and this is the first remote request → `HTTPException(500, "remote rollout requires actor.recompute_logprob=true or actor.use_decoupled_loss=true")`; the gate stays closed on failure.
   - After a successful first remote request, set an in-memory `_recompute_verified: bool` so subsequent requests skip the check.

2. **Tokenize prompt locally** (hf mode) using the shared `_prompt_utils` helpers. `apply_chat_template(tokenizer, tokenizer_messages, tools=tools_list, add_generation_prompt=True, tokenize=True, **chat_template_kwargs)`. If this fails, raise without creating a cache entry.

3. **Call OpenRouter** after local tokenization succeeds, so a tokenizer error doesn't burn an OpenRouter request. Forward `temperature`, `top_p`, `max_tokens`/`max_completion_tokens`, `stop`, `tools`, `tool_choice`, `frequency_penalty`, `metadata`. Strip `model`, `stream`, `n`, `areal_cache`, `store`. `stream=False`, `n=1` are explicit.

4. **Extract assistant output & determine cache key**
   - `choice = remote_completion.choices[0]`
   - `output_text = choice.message.content or ""`
   - `tool_calls = choice.message.tool_calls` (preserve as-is from OpenRouter)
   - `completion_id = remote_completion.id if remote_completion.id and remote_completion.id not in session_cache else f"chatcmpl-{uuid4().hex[:29]}"`
   - Empty `output_text` and no `tool_calls` → `HTTPException(400, "remote returned empty completion")` (no cache entry exists yet, no cleanup needed).

5. **Insert interaction into cache**
   - `interaction = InteractionWithTokenLogpReward(messages=deepcopy(messages_list), chat_template_type="hf")`
   - `session_cache[completion_id] = interaction`
   - From this point, every failure path does `del session_cache[completion_id]` before raising.

6. **Tokenize output locally**
   - `output_tokens = tokenizer.encode(output_text, add_special_tokens=False)`
   - Append EOS: `output_tokens = output_tokens + [tokenizer.eos_token_id]`
   - Rationale: `ModelResponse.output_tokens_without_stop` requires the last token to be EOS/PAD when `stop_reason` is `"stop"` or `"tool_calls"` (areal/api/io_struct.py:110-114). The appended EOS is stripped during training export by the existing `to_tensor_dict()` loss-mask logic.
   - `output_tokens` must be non-empty after this step.

7. **Map finish reason → stop_reason**
   - `"length"` → `"length"`
   - `"tool_calls"` → `"tool_calls"`
   - `"stop"`, `"content_filter"`, `None`, any other normal stop → `"stop"`
   - Error completions are never cached; they are raised as HTTP 502/504 before this step, so `"abort"` is not produced.

8. **Build local `ModelResponse`**
   ```python
   model_response = ModelResponse(
       input_tokens=prompt_token_ids,
       output_tokens=output_tokens,
       output_logprobs=[0.0] * len(output_tokens),       # placeholder, replaced by recompute
       output_versions=[-1] * len(output_tokens),         # "unknown rollout version"
       stop_reason=mapped_stop_reason,
       tokenizer=tokenizer,
   )
   ```
   - `output_versions=[-1]`: matches the "unknown" sentinel used in `to_tensor_dict()` for input positions (types.py:189). Real engines populate `[self._version] * len(tokens)` (inf_bridge.py:257); for off-policy remote we mark version as unknown so the trainer's recompute path replaces both logprobs and versions.

9. **Store interaction fields**
   - `interaction.completion = remote_completion` (preserve remote ID, model name, usage as-is)
   - `interaction.model_response = model_response`
   - `interaction.output_message_list = [remote_completion.choices[0].message.model_dump(exclude_none=True)]`

10. **Return** `remote_completion` to the caller. The visible response shape is OpenRouter's; we do not rewrite it.

### Cleanup Contract

The cache entry is created at step 5, after the OpenRouter call succeeds and the completion ID is known. Every failure path from step 5 onward does `del session_cache[completion_id]` before raising — this covers output tokenization failure (step 6) and ModelResponse construction failure. Failures before step 5 (validation, prompt tokenization, OpenRouter HTTP errors, empty output) raise without creating a cache entry, so no cleanup is needed.

The cache is not inserted before the OpenRouter call because remote rollout MVP has no streaming — the local path's early-insertion pattern (which guards against never-iterated generators) does not apply.

### Recompute Validation (three layers, in order)

1. **Trainer config warning** — `ActorConfig.__post_init__` in `areal/api/cli_args.py`. If `enable_remote_rollout=True` and `not self.should_compute_prox_logp()`, emit `logger.warning(...)`. Warn-not-fail here because trainer config is parsed in many contexts (training script, tests, examples) and a hard failure would break unrelated runs.
2. **First remote request safety check** — in `RemoteRolloutClient.create_completion`. If `recompute_enabled` is False, the first remote request raises `HTTPException(500, ...)`. After a successful first request, an in-memory `_recompute_verified` flag flips and subsequent requests skip the check. The proxy is one process per worker, so this is per-process state.
3. **No proxy-startup check** — the proxy does not parse `ActorConfig`; first-request check is sufficient.

## Routing

- `model="default"` uses existing local SGLang/vLLM behavior.
- `model` beginning with `remote:` uses OpenRouter.
- The provider model name is `model.removeprefix("remote:")`.
- Empty remote model names, such as `model="remote:"`, fail with a clear 400 error.
- Unknown non-default model names that do not start with `remote:` fail with a clear 400/404 error rather than silently using local inference.

Default remote provider settings:

- Base URL: `https://openrouter.ai/api/v1`
- API key environment variable: `OPENROUTER_API_KEY`
- Provider model: derived from the request model after stripping `remote:`

Optional future config can expose base URL, API key env var, request timeout, or default HTTP headers, but the MVP should work with only `OPENROUTER_API_KEY` and a `remote:` request model.

## Remote Chat Completion Flow

For `POST /chat/completions`, the proxy inspects `request.model` and dispatches: `"default"` uses the existing local `_openai_client` path; `"remote:<provider/model>"` enters the `RemoteRolloutClient` path; anything else returns a 404. The detailed per-request lifecycle (validation, OpenRouter call, retokenization, cache storage, cleanup) is in "Implementation Approach > Per-Request Lifecycle".

The cached interaction remains reward-addressable by completion ID through `/rl/set_reward`, and export continues through `/export_trajectories`.

## Tokenization And Tensor Semantics

The stored training sample uses local actor tokenization:

- `input_tokens`: prompt messages rendered through the local tokenizer/chat template.
- `output_tokens`: remote assistant text encoded by the local tokenizer, with a trailing EOS token appended so `output_tokens_without_stop` works (see Per-Request Lifecycle step 6).
- `loss_mask`: generated-token positions are trainable through existing `to_tensor_dict()` behavior.
- `logprobs`: placeholder values (`0.0`) until replaced by local recompute.
- `versions`: `[-1]` for all output positions — "unknown rollout version" sentinel matching the convention used for input positions in `to_tensor_dict()` (types.py:189). The trainer's recompute path replaces both logprobs and versions.
- `rewards`: assigned through existing reward APIs and discounting.

Remote output must produce at least one local output token. Empty output is rejected for the MVP to avoid unusable samples.

## Training Semantics

Remote rollout is off-policy relative to the local actor. The remote model supplies actions as text; the local actor supplies training logprobs by recomputation.

Required or strongly recommended trainer configuration:

```yaml
actor:
  recompute_logprob: true
```

Alternatively, when using decoupled async/off-policy training:

```yaml
actor:
  use_decoupled_loss: true
  prox_logp_method: recompute
```

When remote routing is enabled or used and neither recompute path is active, AReaL should warn or fail fast because placeholder remote logprobs would corrupt PPO ratios. `ref.compute_logp()` and KL computation remain unchanged and run on the retokenized local trajectory.

This is an off-policy RL / imitation-RL hybrid behavior. It is not exact PPO under the remote behavior distribution.

## Error Handling

- Missing `OPENROUTER_API_KEY` returns a clear proxy error before calling OpenRouter.
- OpenRouter auth/rate-limit/upstream failures return clear 4xx/5xx proxy errors and do not leave incomplete cache entries.
- Remote streaming requests fail initially with a clear unsupported message.
- Remote `n != 1` requests fail initially, matching local proxy limitations.
- Tokenization failures: prompt tokenization (before cache insert) raises without creating a cache entry; output tokenization (after cache insert) removes the incomplete cache entry and returns a clear error.
- Empty remote completions are rejected for the MVP.

## Security And Privacy

- API keys come from environment variables or secret injection and are never logged, serialized, or returned.
- Logs may include remote routing decisions, aliases, provider model names, and base URL, but not secrets.
- Documentation must warn that prompts, tool content, and conversation context are sent to OpenRouter and the selected upstream provider.

## Testing Plan

New test file: `tests/experimental/openai/test_remote_rollout.py`.

**Routing tests** (mock `_remote_client` and `_openai_client`):
1. `model="default"` → calls `_openai_client.chat.completions.create`, never touches `_remote_client`
2. `model="remote:openai/gpt-4o-mini"` → calls `_remote_client.create_completion` with stripped model
3. `model="remote:"` → 400, no remote call, no cache entry
4. `model="unknown-local-model"` → 404, no local call, no remote call, no cache entry
5. `model="remote:..."` with `stream=True` → 400
6. `model="remote:..."` with `n=2` → 400
7. `chat_template_type="concat"` + remote → 400

**Remote client unit tests** (mock OpenRouter `AsyncOpenAI`):
8. Happy path: OpenRouter returns a completion → `InteractionWithTokenLogpReward` stored with `model_response.input_tokens` = locally tokenized prompt, `output_tokens` = locally tokenized output + EOS, `output_logprobs` all zeros, `output_versions` all `-1`, `stop_reason` mapped from `finish_reason`, `interaction.completion` is the remote `ChatCompletion` (preserved ID), returned `ChatCompletion` is the remote one unmodified
9. Empty remote output → 400, no cache entry created (failure before cache insert at step 5)
10. OpenRouter 4xx → 502, no cache entry created (failure before cache insert)
11. OpenRouter network error → 504, no cache entry created (failure before cache insert)
12. Output tokenization failure (mock `tokenizer.encode` to raise) → 500, cache entry removed (failure after cache insert at step 5)
13. Missing `OPENROUTER_API_KEY` → 500 before any network call
14. Finish-reason mapping: `length`→`length`, `tool_calls`→`tool_calls`, `stop`→`stop`, `content_filter`→`stop`, `None`→`stop`

**End-to-end cache/export tests** (using the full `ProxyRolloutServer` app):
15. Remote completion → `/rl/set_reward` by remote completion ID succeeds
16. `/export_trajectories` returns tensor data with local-tokenized prompt/output and assigned reward
17. Remote completion + local completion in the same session — both stored, both exportable

**Recompute validation tests**:
18. `ActorConfig(enable_remote_rollout=True, recompute_logprob=False, use_decoupled_loss=False)` → warning logged
19. `ActorConfig(enable_remote_rollout=True, recompute_logprob=True)` → no warning
20. `ActorConfig(enable_remote_rollout=True, use_decoupled_loss=True)` → no warning
21. First remote request with recompute disabled → 500; gate stays closed on subsequent calls
22. First remote request with recompute enabled → succeeds; flag flips; subsequent calls skip the check

**Documentation example test**:
23. A doctest-style test showing `OPENROUTER_API_KEY` env var + `model="remote:anthropic/claude-3.5-sonnet"` request shape, asserting the routing decision (not a real network call).

## Resolved Implementation Questions

The three questions that were open during design have been resolved as follows.

### 1. Recompute validation placement

**Decision**: Three-layer guard. Trainer config warning in `ActorConfig.__post_init__` when `enable_remote_rollout=True` and `not should_compute_prox_logp()`. First remote request safety check in `RemoteRolloutClient.create_completion` raises `HTTPException(500)` until a successful call flips an in-memory `_recompute_verified` flag. No proxy-startup check.

**Why**: Trainer config is parsed in many contexts (training scripts, tests, examples); a hard failure there would break unrelated runs. The first-request check catches the actual misuse — a remote request without recompute — without coupling the proxy to trainer config parsing.
### 2. Completion ID preservation

**Decision**: Preserve the remote completion ID when present and unique; otherwise generate `f"chatcmpl-{uuid4().hex[:29]}"`.

**Why**: `InteractionWithTokenLogpReward.interaction_id` is read from `completion.id` (types.py:108-116), so the cached interaction must carry an ID that the caller (and `/rl/set_reward`) can reference. OpenRouter completion IDs are well-formed strings; preserving them keeps the returned `ChatCompletion` consistent with the cached interaction and lets callers set reward by the ID they actually saw. Generating a fallback handles the rare case where the upstream returns no ID.
### 3. Finish-reason mapping

**Decision**:
- `"length"` → `"length"`
- `"tool_calls"` → `"tool_calls"`
- `"stop"`, `"content_filter"`, `None`, any other normal stop → `"stop"`
- Error completions are never cached (raised as HTTP 502/504 before mapping), so `"abort"` is not produced.

**Why**: `ModelResponse.stop_reason` is `Literal["length", "stop", "tool_calls", "abort"]` (io_struct.py:69). `"length"` and `"tool_calls"` carry training signal (truncation vs. tool-call boundary); everything else collapses to `"stop"`. `"abort"` is reserved for engine-side failures and is never the result of a successful remote completion.
