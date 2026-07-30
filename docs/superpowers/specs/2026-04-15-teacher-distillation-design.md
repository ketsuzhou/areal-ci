# Teacher Distillation Design Specification

## Date: 2026-04-15

## Overview

Extend the existing `on_policy_distill` proxy-based pipeline to support teacher
distillation where a teacher model evaluates the student's top-k candidate tokens at
each position and provides position-level rewards (`student_logp - teacher_logp`) for
the `grpo_distill_loss_fn`.

## Approach: Proxy-Server-Mediated Teacher

After the student generates a response through the proxy server, a
`_compute_token_rewards` function calls the remote teacher inference API to evaluate the
student's top-k candidates. The computation happens inside the workflow's
`arun_episode`, between agent completion and reward export.

### Data Flow

```
Student rollout (logprobs=top_k)
    → Proxy server records interactions with top-k student logprobs
    → _compute_token_rewards:
        1. Get student output tokens + top-k logprobs per position
        2. Call teacher API: send prefix + output tokens, get teacher logprobs
           at the student's top-k candidate token positions
        3. Compute reward = student_logp - teacher_logp per candidate
        4. Build PositionRewardInfo list
    → set_position_rewards to proxy server
    → Compute scalar reward (accuracy_reward or other)
    → Export interactions with position_rewards applied
    → Training uses grpo_distill_loss_fn + MultiCandidateFSDPEngine
```

## New Components

### 1. TeacherConfig

Add to `OnPolicyDistillConfig`:

```python
# Teacher model configuration
teacher_base_url: str = "http://localhost:8001"  # Teacher inference API URL
teacher_model_name: str = ""                      # Required: teacher model name for API
teacher_top_k: int = 10                            # Student top-k candidates per position
teacher_max_retries: int = 3                       # Retry count for teacher API calls
teacher_timeout: float = 60.0                      # Request timeout in seconds
```

### 2. TeacherClient (`core/teacher_client.py`)

Async client for the remote teacher inference API (vLLM/SGLang compatible):

```python
class TeacherClient:
    """Async client for calling the remote teacher inference API."""

    def __init__(self, config: TeacherConfig):
        self.base_url = config.teacher_base_url
        self.model_name = config.teacher_model_name
        self.top_k = config.teacher_top_k
        self.max_retries = config.teacher_max_retries
        self.timeout = config.teacher_timeout

    async def get_logprobs_for_candidates(
        self,
        input_ids: list[int],
        output_ids: list[int],
        candidate_token_ids: list[list[int]],  # [position][candidate_idx] -> token_id
        tokenizer: Any,
    ) -> list[dict[int, float]]:
        """Get teacher logprobs for specific candidate token IDs at each position.

        Uses the OpenAI-compatible completions API with logprobs to evaluate
        the teacher model on the same prefix, then extracts logprobs at the
        specific candidate token positions.

        Returns: list of {token_id: teacher_logp} dicts, one per output position.
        """
```

**Implementation approach:** Call the teacher completions API with `logprobs=top_k`
(where top_k >= max candidates across positions). The teacher returns log probabilities
for its top tokens at each position. For candidate tokens not in the teacher's top-k, we
can extract from the full log softmax (if available) or fall back to a very negative
logprob.

**Batching:** For efficiency, send the entire prefix + output as a single completions
request with `echo=True` and `logprobs` enabled, getting all positions evaluated in one
call.

### 3. \_compute_token_rewards (`core/reward_compute.py`)

Core reward computation function:

```python
async def _compute_token_rewards(
    student_output_ids: list[int],
    student_input_ids: list[int],
    student_top_k_logprobs: list[list[tuple[int, float]]],  # [position][(token_id, logp)]
    teacher_client: TeacherClient,
    top_k: int = 10,
) -> list[PositionRewardInfo]:
    """Compute position-level rewards from teacher/student logprob comparison.

    For each output position:
    1. Take student's top-k tokens as candidates
    2. Call teacher to get teacher_logp for each candidate
    3. Compute reward = student_logp - teacher_logp for each candidate
    4. Build PositionRewardInfo with candidates, rewards, and chosen_index

    Args:
        student_output_ids: Student output token IDs
        student_input_ids: Input (prefix) token IDs
        student_top_k_logprobs: Student's top-k logprobs per position.
            Each position has a list of (token_id, log_prob) tuples.
        teacher_client: TeacherClient for teacher model API calls
        top_k: Number of top candidates to consider

    Returns:
        List of PositionRewardInfo, one per output position.
    """
```

**Reward computation per position:**

For position `i` with student top-k candidates
`[(t1, s_logp_1), (t2, s_logp_2), ..., (tk, s_logp_k)]`:

1. Get teacher logprobs for tokens `{t1, t2, ..., tk}`:
   `{t1: t_logp_1, t2: t_logp_2, ...}`
1. For each candidate: `reward_j = s_logp_j - t_logp_j`
1. The `chosen_index` is the index of the actually-generated token in the candidates
   list
1. Build
   `PositionRewardInfo(position=i, candidates=[...], candidate_token_ids=[t1,...], logprobs=[s_logp_1,...], rewards=[reward_1,...], chosen_index=chosen_idx)`

**Handling missing teacher logprobs:** If a student top-k token is not in the teacher's
top-k, the teacher API won't directly return its logprob. Options:

- Use `logprobs` parameter with a high value (>= top_k) to capture more of the
  distribution
- For tokens outside the returned logprobs, assign a default very negative logprob
  (e.g., `math.log(1e-10)`)
- Future: support teacher returning full logits (requires custom inference endpoint)

### 4. Modified OnPolicyDistillAgent (`core/agent.py`)

Replace the current `_convert_to_position_rewards` (which reads from non-existent
`manager_idm.py` metadata) with `_compute_token_rewards`:

```python
class OnPolicyDistillAgent:
    def __init__(self, ..., teacher_config: TeacherConfig | None = None):
        self.teacher_client = TeacherClient(teacher_config) if teacher_config else None

    async def run(self, data, **extra_kwargs):
        # ... existing code to run agent and get completions ...

        if self.teacher_client is not None:
            # Extract student output tokens and logprobs from proxy server
            proxy_client = extra_kwargs.get("proxy_client")

            # Get the last interaction from the proxy server
            # interaction.model_response contains:
            #   - input_tokens: list[int] (input token IDs)
            #   - output_tokens: list[int] (output token IDs)
            #   - output_logprobs: list[float] (logprobs of chosen tokens)
            #   - output_top_logprobs: list[list[tuple[int, float]]]
            #       (top-k logprobs per position, NEW field to add)
            interaction = await proxy_client.get_last_interaction()
            student_output_ids = interaction.model_response.output_tokens
            student_input_ids = interaction.model_response.input_tokens
            student_top_k_logprobs = interaction.model_response.output_top_logprobs

            position_rewards = await _compute_token_rewards(
                student_output_ids=student_output_ids,
                student_input_ids=student_input_ids,
                student_top_k_logprobs=student_top_k_logprobs,
                teacher_client=self.teacher_client,
                top_k=self.teacher_config.teacher_top_k,
            )

            # Send position rewards to proxy server
            await proxy_client.set_position_rewards(completion_id, position_rewards)
```

### 5. Student Rollout Logprobs Configuration

The proxy server must be configured to return the student's top-k logprobs during
rollout. This requires:

- Setting `logprobs=top_k` in the OpenAI completions request made by the proxy server
- Extracting the top-k logprobs from the API response and storing them in the
  interaction

The AReaL proxy server already captures `output_logprobs` from completions. We need to
extend it to also capture `top_logprobs` (the top-k alternatives at each position). This
requires:

1. **Adding `output_top_logprobs` field to `ModelResponse`:** Extend the `ModelResponse`
   dataclass in `areal/experimental/openai/types.py` to include
   `output_top_logprobs: list[list[tuple[int, float]]] | None` - a list (one per output
   position) of lists of `(token_id, log_prob)` tuples.

1. **Parsing `logprobs` from completions API response:** In the proxy server's response
   handling (where it currently extracts `output_logprobs`), also extract `top_logprobs`
   from the OpenAI API response format. The OpenAI completions API returns
   `logprobs.top_logprobs` as a list of dicts mapping token strings to log
   probabilities.

1. **Token ID resolution:** Convert token strings in `top_logprobs` to token IDs using
   the tokenizer. This is needed because the OpenAI API returns token strings, but
   `PositionRewardInfo` uses `candidate_token_ids`.

1. **Config parameter:** Add `logprobs` parameter to the generation config (defaulting
   to the `teacher_top_k` value) so the proxy server requests top-k logprobs from the
   student model.

## Bug Fixes (Included in This Task)

### 1. `set_last_rewards` sends `""` instead of `None`

- File: `proxy/client.py:227`
- Fix: Change `completion_id=""` to `completion_id=None` in `set_last_rewards` and
  `set_last_position_rewards`

### 2. `_compute_logprobs_and_loss` mutates `ctx.model_inputs` without try/finally

- File: `engine/fsdp_engine.py:305-315`
- Fix: Wrap the `rolled_input_ids` override in try/finally to ensure restoration

### 3. Statistics logging gap in patched `_ppo_update`

- File: `training/actor.py`
- Fix: Add `stats_tracker.denominator()` and `stats_tracker.stat()` calls for the
  critical metrics that were dropped by the monkey-patch

### 4. Tree training path duplication

- File: `engine/fsdp_engine.py:255-277`
- Fix: Delegate to `super()._compute_logprobs_and_loss()` for the tree training path
  instead of duplicating code

### 5. Remove dead code

- Remove `_convert_to_position_rewards` metadata extraction that references non-existent
  `manager_idm.py`
- Remove unused `compute_total_loss_weight` import in `engine/fsdp_engine.py`

### 6. Add logging for teacher evaluation steps

- Log time taken for teacher API calls
- Log number of positions evaluated and candidates per position
- Log reward statistics (mean, std, min, max)

## File Changes Summary

| File                                    | Change Type | Description                                                                            |
| --------------------------------------- | ----------- | -------------------------------------------------------------------------------------- |
| `core/config.py`                        | Modify      | Add `TeacherConfig` fields to `OnPolicyDistillConfig`                                  |
| `core/teacher_client.py`                | **New**     | `TeacherClient` class for remote teacher API calls                                     |
| `core/reward_compute.py`                | **New**     | `_compute_token_rewards` function                                                      |
| `core/agent.py`                         | Modify      | Use `_compute_token_rewards` instead of metadata extraction; integrate `TeacherClient` |
| `proxy/client.py`                       | Fix         | Fix `set_last_rewards` to send `None` instead of `""`; add `get_last_interaction()`    |
| `proxy/workflow.py`                     | Modify      | Pass `teacher_client` to agent, configure `logprobs=top_k` for student rollout         |
| `proxy/server.py`                       | Modify      | Parse and store `top_logprobs` from completions API responses                          |
| `areal/experimental/openai/types.py`    | Modify      | Add `output_top_logprobs` field to `ModelResponse`                                     |
| `engine/fsdp_engine.py`                 | Fix         | Add try/finally for `rolled_input_ids` mutation; delegate tree training to super()     |
| `training/actor.py`                     | Fix         | Add back critical stats tracking in patched `_ppo_update`                              |
| `configs/config_on_policy_distill.yaml` | Modify      | Add teacher configuration section                                                      |

## Testing Strategy

1. **Unit tests for TeacherClient** - Mock the OpenAI completions API, verify logprob
   extraction
1. **Unit tests for \_compute_token_rewards** - Verify reward computation with known
   student/teacher logprobs
1. **Integration test** - End-to-end test with mock teacher API, verify position_rewards
   are correctly computed and stored
1. **Regression tests** - Verify existing proxy server, loss function, and engine tests
   still pass

## Open Questions

- **Teacher logprob coverage:** If a student top-k token is not in the teacher's
  returned logprobs, what default value to use? Current plan: `math.log(1e-10)` ≈ -23.0.
  May need to make this configurable.
- **Batching strategy:** For long sequences, should we batch multiple positions into a
  single teacher API call, or call per-position? Current plan: single call with
  `echo=True` and `logprobs=top_k`.
- **Tokenizer alignment:** If student and teacher use different tokenizers, the
  position-level alignment breaks. Current assumption: same tokenizer. May need to add
  text-level alignment as a fallback.
