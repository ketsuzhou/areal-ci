# OpenRouter Remote Rollout Proxy Design

## Status

Approved design draft for supporting remote OpenAI-compatible LLMs as rollout behavior in AReaL proxy training.

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

For `POST /chat/completions`:

1. Validate the session key and resolve `SessionData.completions` as today.
2. Inspect the request `model`.
3. If it is `default`, call existing `_openai_client.chat.completions.create()`.
4. If it starts with `remote:`, forward the request to OpenRouter with `model` rewritten to the stripped value.
5. Receive the non-streaming remote `ChatCompletion`.
6. Extract the assistant message content and tool calls from the first choice.
7. Retokenize the original prompt messages with the local tokenizer and the configured chat template.
8. Tokenize the remote assistant output with the local tokenizer.
9. Build a local `ModelResponse` containing local prompt tokens, local output tokens, placeholder output logprobs, output versions, and mapped stop reason.
10. Store an `InteractionWithTokenLogpReward` in the same interaction cache, including the remote `ChatCompletion`, input messages, output message list, and constructed `ModelResponse`.
11. Return the remote `ChatCompletion` to the caller, preserving its visible model response shape as much as possible.

The cached interaction remains reward-addressable by completion ID through `/rl/set_reward`, and export continues through `/export_trajectories`.

## Tokenization And Tensor Semantics

The stored training sample uses local actor tokenization:

- `input_tokens`: prompt messages rendered through the local tokenizer/chat template.
- `output_tokens`: remote assistant text encoded by the local tokenizer.
- `loss_mask`: generated-token positions are trainable through existing `to_tensor_dict()` behavior.
- `logprobs`: placeholder values until replaced by local recompute.
- `versions`: current or unknown rollout version marker, matching existing version conventions as closely as possible.
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
- Tokenization failures remove incomplete cache entries and return a clear error.
- Empty remote completions are rejected for the MVP.

## Security And Privacy

- API keys come from environment variables or secret injection and are never logged, serialized, or returned.
- Logs may include remote routing decisions, aliases, provider model names, and base URL, but not secrets.
- Documentation must warn that prompts, tool content, and conversation context are sent to OpenRouter and the selected upstream provider.

## Testing Plan

- Routing tests:
  - `model="default"` calls local `ArealOpenAI`.
  - `model="remote:openai/gpt-4o-mini"` calls OpenRouter with `model="openai/gpt-4o-mini"`.
  - `model="remote:"` fails.
  - Unknown non-default local model fails.
- Cache/export tests:
  - Remote completion creates an `InteractionWithTokenLogpReward`.
  - `/rl/set_reward` works by remote completion ID.
  - Export returns tensor data with local-tokenized prompt/output and assigned reward.
- Training safety tests:
  - Remote config/use warns or fails when actor logprob recompute is disabled.
  - Recompute-enabled config is accepted.
- Error cleanup tests:
  - Remote API failure does not leave incomplete interactions in the session cache.
  - Tokenization failure does not leave incomplete interactions.
- Documentation test/example:
  - Shows `OPENROUTER_API_KEY` and `model="remote:<provider/model>"` usage.

## Open Questions For Implementation

- Where to place recompute validation: proxy startup, trainer config validation, or first remote request. Prefer trainer/proxy startup warning plus first remote request safety check.
- Whether to preserve the remote completion ID or replace it with an AReaL-generated ID. Prefer preserving remote ID if present and unique, otherwise generate `chatcmpl-...`.
- How to map remote finish reasons exactly into `ModelResponse.stop_reason`. A simple mapping is `length -> length`, `tool_calls -> tool_calls`, all other normal stops -> stop, errors -> abort`.
