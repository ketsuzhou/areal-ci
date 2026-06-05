---
name: debug-tpfc
description: Debug the TPFC agent end-to-end. Starts API + worker, runs backend_run.py, verifies final_answer matches ground truth, and iterates on code fixes until it passes. Use when debugging TPFC agent issues, running TPFC eval, verifying TPFC produces correct final_answer, or when the user asks to debug/test TPFC.
---

# Debug TPFC Agent

End-to-end debugging workflow for the TPFC (Task Planning + Function Calling) agent.

**User instruction**: $ARGUMENTS

---

## Overview

The TPFC agent loop alternates between two phases:
1. **TP (Task Planning)** — LLM generates `<subtask>` tags decomposing the query
2. **FC (Function Calling)** — LLM selects tools and executes them for the current subtask

The loop continues until the LLM outputs `<answer>...</answer>` with the final answer.

Key files:
- `backend/core/agents/builtin/tpfc.py` — TPFC builtin config (model, tools, system_prompt)
- `backend/core/tpfc/prompt.py` — `generate_tpfc_prompt()` creates system + user seed messages
- `backend/core/agents/run/service.py:_create_seed_messages()` — persists TPFC seed messages to DB
- `backend/core/agents/runtime/context/manager/manager_tpfc.py` — TPFC context manager (TP↔FC routing)
- `backend/core/agents/runtime/context/compactor/tpfc_compactor.py` — compresses tool results
- `backend/core/tpfc/backend_run.py` — entry point that starts a run via API and extracts `final_answer`
- `backend/core/tpfc/llama_tpfc_config.py` — prompt templates (`TPFC_PROMPT_PROCESSOR`, `TPFC_PROMPT_PROCESSOR_NEW`)

---

## Step 1: Start API and Worker

Start the backend services. These MUST be running before executing the test.

```bash
cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python api.py
```

```bash
cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 8 --threads 8
```

Run both in background. Wait for API to report `Application startup complete` and workers to report `Worker process is ready for action` before proceeding.

If using tmux, prefer the project's start-dev-server skill:
```bash
bash "$(git rev-parse --show-toplevel)/.agents/skills/start-dev-server/scripts/start-dev-server.sh"
```

---

## Step 2: Verify TPFC System Prompt in _create_seed_messages

Before running the test, verify that `_create_seed_messages` correctly creates the system_prompt for TPFC:

1. Read `backend/core/agents/run/service.py` around `_create_seed_messages` (line ~853)
2. Confirm the `agent_key == BuiltinId.TPFC` branch calls `generate_tpfc_prompt()` from `core/tpfc/prompt.py`
3. Confirm `generate_tpfc_prompt()` returns a messages list starting with `{"role": "system", "content": [{"type": "text", "text": ...}]}`
4. Confirm the system prompt uses `TPFC_PROMPT_PROCESSOR_NEW` (the latest template from `llama_tpfc_config.py`)
5. Verify that seed messages are persisted with `metadata={"context_type": "tp"}`

If the system_prompt is empty or missing in `tpfc.py` config (`"system_prompt": ""`), that's expected — the TPFC agent builds its system prompt via `generate_tpfc_prompt()`, not through the generic `SystemPromptBuilder`. The `system_prompt` field in the config is intentionally blank.

---

## Step 3: Run backend_run.py Test

Execute the test:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main/ && .venv/bin/python -m customized_areal.tpfc.backend_run
```


This runs the `__main__` block in `backend_run.py`, which:
1. Creates a task with the default GAIA benchmark question
2. Starts an agent run via the API (`/api/agent/start`)
3. Waits for the run to complete (SSE stream + DB polling)
4. Extracts `final_answer` from `<answer>...</answer>` tags
5. Prints `Final boxed answer: <answer>`

The ground truth for the default test case is:
- **gt**: `"Time-Parking 2: Parallel Universe"`

---

## Step 4: Evaluate Result

Compare `final_answer` against `gt`:

| Condition | Action |
|-----------|--------|
| `final_answer == gt` | Test passes. Report success. |
| `final_answer != gt` | Debug (Step 5) |
| `final_answer is None` | No `<answer>` tag found. Debug (Step 5) |
| Run timed out | Debug (Step 5) |
| Run crashed/errored | Debug (Step 5) |

---

## Step 5: Debug Loop

If `final_answer != gt`, systematically debug:

### 5a. Check API and Worker Logs

```bash
# If using tmux:
tmux capture-pane -pt leagent-dev:api | tail -200
tmux capture-pane -pt leagent-dev:agent-worker | tail -200
```

Look for:
- Auth token errors (401/403) → refresh token issue
- Model/API key errors → check env vars (`OPENROUTER_API_KEY`, `WORKSPACE_OPENAI_API_KEY`)
- Task creation failures → DB connection issue
- Worker not picking up tasks → queue mismatch

### 5b. Check Database Messages

After a run, inspect the messages stored for the task:

```python
from core.services.supabase import DBConnection
from core.tasks.messages.service import get_llm_messages

db = DBConnection()
client = await db.client
messages = await get_llm_messages(task_id, return_raw=True)
for msg in messages:
    print(f"[{msg['role']}] {str(msg.get('content',''))[:200]}")
```

Verify:
- First message is `system` with `TPFC_PROMPT_PROCESSOR_NEW` content
- Second message is `user` with `<User Query>: ...`
- TP/FC alternation follows: user→assistant(subtask)→user(fc prompt)→assistant(tool call)→tool(result)→user(observation)→assistant(subtask/answer)
- No missing or malformed messages

### 5c. Check Context Manager Flow

Read `manager_tpfc.py:build()` logic:

1. If last message has `context_type="tp"` and contains `<subtask>` → route to `_build_fc_context`
2. If last message has `context_type="fc"` (tool result) → route to `_build_tp_context`
3. Compaction in `_build_tp_context` compresses tool results via `TPFCCompactor`

Common issues:
- Wrong context_type → TP/FC routing breaks
- Compaction losing critical info → adjust compactor model or prompt
- FC context missing tool schemas → check `overwrite_schemas=True` in BuiltContext

### 5d. Check Tool Execution

Verify that the TPFC agent's builtin tools are correctly configured in `tpfc.py`:
- `sb_shell_tool`, `document_reading_tool`, `sb_files_tool`
- `audio_analysis_tool`, `video_analysis_tool`, `sb_vision_tool`
- `searching_tool`

Check that tool calls are being made and returning results. The FC phase uses `overwrite_llm_model="qwen/qwen3.5-397b-a17b"` and `WORKSPACE_OPENAI_API_KEY`/`WORKSPACE_OPENAI_API_BASE` env vars.

### 5e. Fix and Rerun

After identifying the root cause:

1. Make the code fix
2. Restart API and worker (kill existing processes and restart from Step 1)
3. Rerun the test from Step 3
4. Re-evaluate from Step 4

Repeat until `final_answer == gt`.

---

## Step 6: Report Results

When done, report:
- **Status**: PASS / FAIL (after max iterations)
- **final_answer**: The extracted answer
- **gt**: The expected ground truth
- **Match**: Whether they match
- **Iterations**: How many debug-fix-rerun cycles were needed
- **Root cause** (if debugged): What was wrong and what was fixed

---

## Environment Variables

The TPFC agent depends on these env vars (set in `backend/.env`):

| Variable | Purpose |
|----------|---------|
| `REFRESH_TOKEN` / `SUPABASE_AUTH_EMAIL` + `SUPABASE_AUTH_PASSWORD` | Supabase auth |
| `TPFC_AGENT_ID` | Agent ID (auto-created if missing) |
| `TPFC_USER_ID` | User ID for task creation |
| `LE_AGENT_API_URL` | Backend API URL (default: `http://localhost:8000`) |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | TP phase model API |
| `WORKSPACE_OPENAI_API_KEY` / `WORKSPACE_OPENAI_API_BASE` | FC phase model API |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` | Supabase connection |

---

## Step 7: Verify Proxy API Key / Base URL Override

When `run_backend` is called with explicit `api_key` and `base_url`, these must
override the credentials associated with the registered `model_name`. This test
confirms that the leagent backend's LLM client receives and uses the caller-supplied
credentials instead of the model's registered ones.

### 7a. Start API and Worker

Follow Step 1 to ensure the backend services are running. Verify with:

```bash
curl -s http://localhost:8000/health | head -1
```

If the API is not running, start it:

```bash
# Terminal 1: API
cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python api.py

# Terminal 2: Worker
cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 8 --threads 8
```

### 7b. Run the proxy override test

Run `backend_run.py` with `model_name="areal/qwen/qwen3_5-9b"` and `api_key='aaa'`.
The invalid key forces a 401 auth error from the LLM provider, proving the override
took effect (the registered model key would have succeeded).

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main/ && \
  .venv/bin/python -c "
import asyncio, os
from customized_areal.tpfc.backend_run import run_backend

result = asyncio.run(run_backend(
    task_description='What is 2+2?',
    task_file_path=[],
    gt='4',
    tags=['proxy-override-test'],
    model_name='areal/qwen/qwen3_5-9b',
    api_key='aaa',
    base_url=os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
))
print('Test complete')
" 2>&1 | tail -30
```

The run will fail with a 401 "Missing Authentication" error — this is the **expected**
proof that `api_key='aaa'` was forwarded to the LLM provider instead of the registered
model's real key.

### 7c. Verify in worker logs

Check the worker output for the proxy override evidence:

```bash
tmux capture-pane -pt leagent-dev:agent-worker | tail -200
```

Look for these patterns:

| Log evidence | Meaning |
|---|---|
| `model=areal/qwen/qwen3_5-9b` with 401 auth error | PASS — `proxy_api_key='aaa'` was used |
| `_prepare_form_data: proxy_base_url=... proxy_api_key=aaa...` | PASS — AReaL side forwarded correctly |
| Authentication error from OpenRouter (`401 - Missing Authentication`) | PASS — invalid key reached the LLM provider |
| Successful LLM call with registered model's real key | FAIL — override ignored, `proxy_api_key` not forwarded |
| Request to wrong `base_url` | FAIL — `proxy_base_url` not forwarded |

### 7d. Check debug_streams for LLM input

The leagent backend writes LLM call details to debug stream files:

```bash
ls -lt /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend/debug_streams/ | head -5
```

Find the most recent file for the `areal/qwen/qwen3_5-9b` model and inspect it:

```bash
LATEST=$(ls -t /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend/debug_streams/ | head -1)
cat "/dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend/debug_streams/$LATEST" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('model:', data.get('model'))
print('base_url:', data.get('base_url'))
api_key = data.get('api_key', data.get('apiKey', 'N/A'))
print('api_key:', api_key[:8] + '...' if isinstance(api_key, str) and len(api_key) > 8 else api_key)
"
```

Verify:
- `model` is `areal/qwen/qwen3_5-9b`
- `api_key` starts with `aaa` (not the registered model's real key)
- `base_url` matches the `OPENROUTER_BASE_URL` value

### 7e. Verify programmatically (unit tests)

Unit tests confirm the AReaL-side code paths forward proxy credentials correctly:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main/ && \
  .venv/bin/pytest tests/customized_areal/test_tpfc_backend_auth.py \
    -k "proxy_override" -v
```

This verifies:
- `_prepare_form_data(api_key='aaa', base_url='http://proxy')` produces
  `form_data['proxy_api_key'] == 'aaa'` and `form_data['proxy_base_url'] == 'http://proxy'`
- `_start_branch_agent_run_for_task(api_key='aaa', base_url='http://proxy')` sends
  `proxy_api_key='aaa'` and `proxy_base_url='http://proxy'` in the JSON body
- When `api_key` / `base_url` are `None`, the proxy fields are **absent** from the
  request (no overwrite of the registered model's credentials)
- `run_backend(api_key='aaa', base_url='http://proxy')` forwards through to form data

### Expected result

| Condition | Action |
|-----------|--------|
| Worker logs show 401 auth error with `model=areal/qwen/qwen3_5-9b` | PASS — `proxy_api_key` override works |
| Debug stream shows `api_key='aaa...'` | PASS — LLM client received the override |
| Unit tests pass | PASS — AReaL code paths correct |
| Worker calls registered model provider with its real key | FAIL — `proxy_api_key` not forwarded; file bug in leagent `llm_client.make_proxy_llm_api_call` |
| Debug stream shows registered model's real key | FAIL — override dropped somewhere in the chain |

---

## Step 8: Verify final_answer == gt with Real Model Credentials

Run `backend_run.py` with the registered model's real API key to confirm the full
TPFC pipeline produces the correct answer end-to-end.

### 8a. Start API and Worker

Follow Step 1. Verify the backend is up:

```bash
curl -s http://localhost:8000/health | head -1
```

### 8b. Run with openrouter/qwen/qwen3-vl-8b-thinking and real API key

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main/ && \
  .venv/bin/python -c "
import asyncio, os
from customized_areal.tpfc.backend_run import run_backend

task_description = (
    'The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. '
    'What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet.'
)
task_file_path = [
    '/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/dataset/gaia-benchmark/gaia/2023/validation/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx'
]
gt = 'Time-Parking 2: Parallel Universe'

result = asyncio.run(run_backend(
    task_description=task_description,
    task_file_path=task_file_path,
    gt=gt,
    tags=['e2e-test', 'real-creds'],
    model_name='openrouter/qwen/qwen3-vl-8b-thinking',
    api_key=os.environ.get('OPENROUTER_API_KEY'),
    base_url=os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
))

print(f'final_answer: {result.final_answer!r}')
print(f'gt:           {gt!r}')
print(f'match:        {result.final_answer == gt}')
"
```

This uses the real `OPENROUTER_API_KEY` so the TPFC agent can complete its full
TP→FC→answer loop. The test passes only when the extracted `final_answer` exactly
matches the ground truth.

### 8c. Evaluate result

| Condition | Action |
|-----------|--------|
| `final_answer == gt` | PASS — full pipeline works with real credentials |
| `final_answer != gt` | Partial pass — pipeline runs but answer is wrong; debug via Step 5 |
| `final_answer is None` | FAIL — no `<answer>` tag produced; check worker logs for LLM errors |
| Run times out or crashes | FAIL — check Step 5a for root cause |

---

## Key Constraints

- **System prompt is built by `generate_tpfc_prompt()`**, not `SystemPromptBuilder`. The `system_prompt` field in `TPFC_CONFIG` is intentionally `""`.
- **TP and FC phases use different models**: TP uses the agent's configured model (e.g., `areal/qwen/qwen3_5-9b` or `openrouter/qwen/qwen3-vl-8b-thinking`), FC uses `qwen/qwen3.5-397b-a17b` via workspace keys.
- **Proxy override**: `run_backend(api_key='aaa', base_url='...')` sends `proxy_api_key` and `proxy_base_url` via both the `/api/agent/start` form-data path and the `/api/agent/start-branch` JSON path. The leagent backend must forward these to the LLM client, overriding the registered model's credentials.
- **Seed messages must have `context_type: "tp"`** in metadata for the context manager to route correctly.
- **Auth tokens are shared** via `.shared_auth_token.json` for multi-process coordination.
- **Sandbox cleanup** runs automatically after each `backend_run.py` execution.
