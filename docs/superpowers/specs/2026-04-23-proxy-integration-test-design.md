# Proxy Server Integration Test Design

## Goal

Add an HTTP server integration test that verifies the on-policy distillation proxy
components (cache, server, workflow) work correctly together end-to-end. The test spins
up the real FastAPI app via `httpx.AsyncClient` and sends actual HTTP requests through
the ASGI stack, exercising real serialization, session management, and reward handling
without requiring GPU or LLM backends.

## Scope

Four test scenarios covering the critical data flows:

1. **Token rewards round-trip** — set token rewards via HTTP, export, verify
1. **Position rewards round-trip** — set position rewards via HTTP, export, verify
   derived token rewards
1. **Scalar/position reward separation** — set both scalar and position rewards, verify
   scalar reward is preserved (not overwritten)
1. **Full workflow episode** — `OpenAIProxyWorkflow.arun_episode` against the real
   server, verify tensor dict has `position_rewards` with correct `sample_index`

## Approach

Use `httpx.AsyncClient(app=app)` to send HTTP requests through the real FastAPI ASGI
stack. This tests real route handlers, Pydantic validation, session state management,
serialization/deserialization, and export logic — without subprocess, port binding, or
LLM dependencies.

### What's Real

- FastAPI `app` from `proxy_rollout_server.py` (all route handlers)
- `TokenRewardSessionData` (server-side reward storage and export)
- Pydantic request/response models and validation
- `serialize_interactions_with_position_rewards` /
  `deserialize_interactions_with_position_rewards`
- HTTP status codes and error handling

### What's Mocked

- LLM chat/completions calls (mock interactions injected directly into session data via
  `session_data.completions["id"] = mock_interaction`)
- `workflow_context` module (mock `task_id`, `get_aiohttp_session`, `get_httpx_client`)
  — only for Scenario 4
- The agent's `run()` method returns pre-defined rewards — only for Scenario 4

## Test File

`customized_areal/on_policy_distill/proxy/tests/test_server_integration.py`

## Fixtures

### `reset_server_state`

Reset the module-level globals in `proxy_rollout_server` between tests:

```python
@pytest.fixture(autouse=True)
def reset_server_state():
    import customized_areal.on_policy_distill.proxy.proxy_rollout_server as srv
    srv._session_cache.clear()
    srv._api_key_to_session.clear()
    srv._session_to_api_key.clear()
    srv._capacity = 0
    yield
    srv._session_cache.clear()
    srv._api_key_to_session.clear()
    srv._session_to_api_key.clear()
```

### `proxy_http_client`

Create an `httpx.AsyncClient` bound to the FastAPI app:

```python
@pytest.fixture
async def proxy_http_client():
    from customized_areal.on_policy_distill.proxy.proxy_rollout_server import app
    import customized_areal.on_policy_distill.proxy.proxy_rollout_server as srv
    async with httpx.AsyncClient(
        app=app, base_url="http://testserver"
    ) as client:
        yield client
```

### `admin_headers`

Authorization headers using the server's `_admin_api_key`:

```python
@pytest.fixture
def admin_headers():
    from customized_areal.on_policy_distill.proxy.proxy_rollout_server import _admin_api_key
    return {"Authorization": f"Bearer {_admin_api_key}"}
```

### `mock_interaction`

Create a mock `InteractionWithTokenLevelReward` with configurable output token count.
This is injected directly into `session_data.completions` to simulate LLM responses
without needing actual inference.

```python
@pytest.fixture
def make_mock_interaction():
    def _make(
        completion_id="comp-test",
        output_tokens=None,
        reward=None,
        messages=None,
    ):
        output_tokens = output_tokens or [100, 200, 300]
        mock_resp = Mock()
        mock_resp.output_tokens = output_tokens
        mock_resp.input_len = 5
        mock_resp.output_len = len(output_tokens)
        mock_resp.output_logprobs = [-0.5] * len(output_tokens)

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=messages or [{"role": "user", "content": "Hello"}],
            completion=Mock(id=completion_id),
            reward=reward,
        )
        interaction.interaction_id = completion_id
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        return interaction
    return _make
```

### `session_with_interaction`

Start a session, grant capacity, and inject a mock interaction:

```python
@pytest.fixture
async def session_with_interaction(proxy_http_client, admin_headers, make_mock_interaction):
    async def _setup(completion_id="comp-test", output_tokens=None):
        # Grant capacity
        await proxy_http_client.post(
            "/rl/grant_capacity", headers=admin_headers
        )
        # Start session
        resp = await proxy_http_client.post(
            "/rl/start_session",
            json={"task_id": "test-task"},
            headers=admin_headers,
        )
        data = resp.json()
        session_id = data["session_id"]
        session_api_key = data["api_key"]
        session_headers = {"Authorization": f"Bearer {session_api_key}"}

        # Inject mock interaction into session data
        from customized_areal.on_policy_distill.proxy.proxy_rollout_server import _session_cache
        session_data = _session_cache[session_id]
        interaction = make_mock_interaction(
            completion_id=completion_id,
            output_tokens=output_tokens,
        )
        session_data.completions[completion_id] = interaction

        return session_id, session_api_key, session_headers
    return _setup
```

### `live_server`

Start the FastAPI app on a real port via uvicorn in a background thread. Needed for
Scenario 4 where the workflow's `OpenAIProxyClient` uses aiohttp:

```python
@pytest.fixture
async def live_server():
    import threading
    import uvicorn
    from customized_areal.on_policy_distill.proxy.proxy_rollout_server import app

    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for server to be ready
    await asyncio.sleep(1.0)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)
```

## Test Scenarios

### Scenario 1: Token Rewards Round-Trip

```
1. Start session + inject mock interaction
2. POST /rl/set_token_rewards with {interaction_id: "comp-test", token_rewards: [0.1, 0.2, 0.3]}
3. POST /rl/end_session
4. POST /rl/export_trajectories
5. Deserialize response
6. Assert: interaction.token_rewards_list == [0.1, 0.2, 0.3]
7. Assert: interaction.reward == 0.6 (sum of token rewards)
```

### Scenario 2: Position Rewards Round-Trip

```
1. Start session + inject mock interaction (3 output tokens)
2. POST /rl/set_position_rewards with:
   - Position 0: candidates=["a","b"], rewards=[0.1, 0.5], chosen_index=1
   - Position 1: candidates=["c","d"], rewards=[0.2, 0.6], chosen_index=0
   - Position 2: candidates=["e","f","g"], rewards=[0.3, 0.7, 0.9], chosen_index=2
3. POST /rl/end_session
4. POST /rl/export_trajectories
5. Deserialize response
6. Assert: position_rewards has 3 entries with correct candidates/rewards
7. Assert: derived token_rewards_list == [0.5, 0.2, 0.9] (chosen rewards)
```

### Scenario 3: Scalar/Position Reward Separation

```
1. Start session + inject mock interaction
2. POST /rl/set_reward with {interaction_id: "comp-test", reward: 5.0}
3. POST /rl/set_position_rewards with chosen rewards summing to 1.6 (≠ 5.0)
4. POST /rl/end_session
5. POST /rl/export_trajectories
6. Deserialize response
7. Assert: interaction.reward == 5.0 (scalar preserved, NOT overwritten)
8. Assert: position_rewards attached for distillation loss
9. Assert: derived token_rewards_list == [0.5, 0.2, 0.9]
```

### Scenario 4: Full Workflow Episode

**Key design consideration**: The `OpenAIProxyWorkflow` creates an `OpenAIProxyClient`
internally, which uses `aiohttp.ClientSession` for HTTP requests. Since
`httpx.AsyncClient(app=app)` only handles httpx requests, we need to start the FastAPI
server on a real port for this scenario so aiohttp can reach it.

**Approach**: Use `uvicorn` in a background thread to serve the app on a random
available port. The test finds the port and passes it to the workflow. The thread is
stopped in the fixture teardown.

```
1. Start FastAPI app on real port via uvicorn in background thread
2. Create mock agent that:
   a. Calls proxy_client.set_reward("comp-test", 3.0)
   b. Calls proxy_client.set_position_rewards("comp-test", [...])
3. Create OpenAIProxyWorkflow with mock agent, pointing at localhost:port
4. Mock workflow_context (task_id, aiohttp session, httpx client)
5. Run workflow.arun_episode(mock_engine, {"prompt": "test"})
6. Assert: result is a tensor dict
7. Assert: "position_rewards" key in result
8. Assert: each PositionRewardInfo has correct sample_index
9. Assert: "token_rewards" tensor present in tensor dict
10. Assert: "token_reward_mask" tensor present in tensor dict
11. Stop uvicorn server
```

## Dependencies

- `httpx` (already in pyproject.toml)
- `pytest-asyncio` (already used in existing tests)
- No new dependencies needed

## Risk: Global State in proxy_rollout_server

The server module uses module-level globals (`_session_cache`, `_admin_api_key`, etc.).
The `reset_server_state` fixture clears these between tests. Tests must run sequentially
(no parallelism within this file) to avoid state conflicts.

## Out of Scope

- Trainer integration (requires GPU + distributed setup)
- LLM inference testing (requires SGLang/vLLM)
- Concurrent session stress testing
- Performance benchmarks
