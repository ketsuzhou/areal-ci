---
change: env-dispatch-agent-runtime-config
design-doc: docs/superpowers/specs/2026-07-20-env-dispatch-agent-runtime-config-design.md
base-ref: bd6c2462edfb664c696cc05a9ec791effe97d101
---

# EnvDispatch Per-Agent External Runtime Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a non-training scratch message env-dispatch agent receive an
external model runtime policy, start its isolated sandbox with that policy, and
reply without exposing its API key.

**Architecture:** Parse the nested runtime object into a typed service value,
validate it before rollout writes, and resolve it into an internal per-agent
sandbox policy that is never serialized in response DTOs. Persist canonical
policy JSON in the existing env-agent binding; both immediate leader and lazy
peer/clone provisioning decode the binding and pass only the runtime object to
the existing sandbox lifecycle.

**Tech Stack:** Go 1.22+, Chi handlers, pgx/sqlc-backed binding store, standard
library `encoding/json` and `net/url`, Python 3.12+, pytest/httpx.

## Global Constraints

- Caller runtime is accepted only for `mode=scratch` and
  `dispatch_type=message`.
- Runtime for `train_agent_id`, branch, issue dispatch, or a partial/invalid
  provider configuration fails before rollout resources are created.
- A runtime-only scratch policy resolves to sandbox template `default`.
- Branch inherits copied `sandbox_config` from the source selected by top-level
  `env_id`; branch runtime override is rejected.
- The API key must never enter `SandboxInstanceRef`, HTTP responses, errors, or
  structured logs.
- No dependency or database migration is permitted.
- Do not modify or overwrite the user's existing edits in
  `customized_areal/tree_search/agents/multica_client.py`.

---

### Task 1: Typed Request Mapping and Atomic Validation

**Files:**

- Modify: `multica/server/internal/service/env_dispatch.go`
- Modify: `multica/server/internal/service/env_dispatch_test.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Modify: `multica/server/internal/handler/env_dispatch_test.go`

**Interfaces:**

- Produces: `service.ExternalModelRuntime{BaseURL, APIKey, Model string}`.
- Produces: `service.ResolvedPerAgentSandboxPolicy{Template string, Runtime *ExternalModelRuntime}`.
- Produces: `NormalizeExternalModelRuntime(*ExternalModelRuntime) (*ExternalModelRuntime, error)`.
- Changes: `PerAgentEnvSpec` gains `Runtime *ExternalModelRuntime`.
- Changes: `ResolvePerAgentEnvSpec` returns `ResolvedPerAgentSandboxPolicy`.

- [ ] **Step 1: Add failing service validation cases**

Extend `TestEnvDispatchPerAgentEnvSpecs_ShapeValidation` and add a table-driven
test covering runtime-only scratch success, whitespace normalization, partial
runtime, relative/non-HTTP URL, branch runtime, issue runtime, and training-target
runtime. Use only a sentinel test key:

```go
validRuntime := &ExternalModelRuntime{
    BaseURL: " https://provider.invalid/v1 ",
    APIKey:  " synthetic-secret-for-tests ",
    Model:   " model-a ",
}

in := EnvDispatchInput{
    Mode: EnvModeScratch, Domain: EnvDomainSelfPlay,
    DispatchType: EnvDispatchMessage, AgentID: "agent-a",
    PerAgentEnvSpecs: []PerAgentEnvSpec{{
        AgentID: "agent-a", Runtime: validRuntime,
    }},
}
```

For every invalid case, invoke `Dispatch` with `newFakeEnvDispatchDeps()` and
assert its env/project/channel/sandbox/task call collections remain empty.

- [ ] **Step 2: Run the focused service tests and verify RED**

Run from `multica/server`:

```bash
go test ./internal/service -run 'TestEnvDispatchPerAgentEnvSpecs_(ShapeValidation|RuntimeValidation)' -count=1
```

Expected: compile failures for the new runtime types/field, or assertion failures
because runtime-only entries and mode/training restrictions are not implemented.

- [ ] **Step 3: Implement typed normalization and validation**

Add the service types and a normalization function using `net/url` and
`strings.TrimSpace`:

```go
type ExternalModelRuntime struct {
    BaseURL string `json:"base_url"`
    APIKey  string `json:"api_key"`
    Model   string `json:"model"`
}

type ResolvedPerAgentSandboxPolicy struct {
    Template string
    Runtime  *ExternalModelRuntime
}

func NormalizeExternalModelRuntime(in *ExternalModelRuntime) (*ExternalModelRuntime, error) {
    if in == nil {
        return nil, nil
    }
    out := &ExternalModelRuntime{
        BaseURL: strings.TrimSpace(in.BaseURL),
        APIKey:  strings.TrimSpace(in.APIKey),
        Model:   strings.TrimSpace(in.Model),
    }
    if out.BaseURL == "" || out.APIKey == "" || out.Model == "" {
        return nil, fmt.Errorf("base_url, api_key, and model are required")
    }
    parsed, err := url.Parse(out.BaseURL)
    if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
        return nil, fmt.Errorf("base_url must be an absolute HTTP(S) URL")
    }
    return out, nil
}
```

Change shape validation to receive the full `EnvDispatchInput`. Treat a runtime
object as a valid scratch policy even without template/base env, but keep the
template/base mutual exclusion. Validate runtime mode/domain and reject the map
entry matching `TrainAgentID`. Error messages may name fields and agent IDs but
must not format runtime values.

- [ ] **Step 4: Add failing handler mapping and non-disclosure tests**

Extend `TestEnvDispatch_ParsesPerAgentEnv` with a runtime-only JSON request and
assert it advances beyond the old “needs a template or base_env_id” error. Add a
direct `mapPerAgentEnvSpecs` test that asserts all three typed values. Extend
`TestMapRollouts_IncludesSandboxRefs` with a sentinel key in a separate internal
policy and assert marshaled rollout JSON does not contain the sentinel.

- [ ] **Step 5: Implement handler request mapping**

Add a handler request type and map it explicitly:

```go
type ExternalModelRuntimeRequest struct {
    BaseURL string `json:"base_url"`
    APIKey  string `json:"api_key"`
    Model   string `json:"model"`
}

type PerAgentEnvRequest struct {
    Template  string                       `json:"template,omitempty"`
    BaseEnvID string                       `json:"base_env_id,omitempty"`
    Runtime   *ExternalModelRuntimeRequest `json:"runtime,omitempty"`
}
```

`mapPerAgentEnvSpecs` must allocate a new service runtime value rather than
retaining handler-layer pointers.

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

```bash
cd multica/server
gofmt -w internal/service/env_dispatch.go internal/service/env_dispatch_test.go internal/handler/env_dispatch.go internal/handler/env_dispatch_test.go
go test ./internal/service ./internal/handler -run 'EnvDispatch.*(PerAgent|Runtime)|MapRollouts' -count=1
```

Expected: PASS; DB-dependent handler tests may report their existing explicit
skip when the test database is unavailable.

- [ ] **Step 7: Commit the nested-repository Task 1 change**

Load `commit-conventions` before committing, then:

```bash
git -C multica add server/internal/service/env_dispatch.go server/internal/service/env_dispatch_test.go server/internal/handler/env_dispatch.go server/internal/handler/env_dispatch_test.go
git -C multica commit -m "feat(env-dispatch): validate per-agent model runtime"
```

### Task 2: Secret-Safe Binding Policy Persistence

**Files:**

- Create: `multica/server/internal/handler/env_dispatch_channel_policy.go`
- Create: `multica/server/internal/handler/env_dispatch_channel_policy_test.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
- Modify: `multica/server/internal/service/env_dispatch_test.go`

**Interfaces:**

- Consumes: `service.ResolvedPerAgentSandboxPolicy` and
  `NormalizeExternalModelRuntime` from Task 1.
- Produces: `envDispatchSandboxConfig{Template string, Runtime *service.ExternalModelRuntime}`.
- Produces: `marshalEnvDispatchSandboxConfig(service.ResolvedPerAgentSandboxPolicy) (json.RawMessage, error)`.
- Produces: `decodeEnvDispatchSandboxConfig(json.RawMessage) (envDispatchSandboxConfig, error)`.
- Changes: `CreateEnvDispatchChannel` receives
  `map[string]ResolvedPerAgentSandboxPolicy`.

- [ ] **Step 1: Write failing pure policy codec tests**

Create tests that marshal a resolved policy, assert canonical trimmed values and
template `default`, decode it back, and reject malformed/partial stored JSON.
Assert `json.Marshal(service.SandboxInstanceRef{})` cannot contain the sentinel
secret because runtime policy is not stored on that DTO.

```go
policy := service.ResolvedPerAgentSandboxPolicy{
    Template: "default",
    Runtime: &service.ExternalModelRuntime{
        BaseURL: "https://provider.invalid/v1",
        APIKey: "synthetic-secret-for-tests",
        Model: "model-a",
    },
}
```

- [ ] **Step 2: Run codec tests and verify RED**

```bash
cd multica/server
go test ./internal/handler -run 'TestEnvDispatchSandboxConfig' -count=1
```

Expected: compile failure because the policy codec does not exist.

- [ ] **Step 3: Implement the policy codec**

Create the focused handler file. Use `json.Decoder`/`json.Unmarshal`, normalize
runtime through the service helper, default an empty template to `default`, and
return errors without embedding the raw JSON or field values:

```go
type envDispatchSandboxConfig struct {
    Template string                        `json:"template"`
    Runtime  *service.ExternalModelRuntime `json:"runtime,omitempty"`
}
```

Malformed stored policy must be an error. Do not retain the current ignored
`json.Unmarshal` error pattern.

- [ ] **Step 4: Write failing binding propagation tests**

Update the service fake to capture the exact resolved policy map passed to
`CreateEnvDispatchChannel`. Dispatch a scratch squad with two different runtime
objects and assert per-agent isolation and the default template. Add a handler
adapter test, using the existing DB test skip convention where required, that
reads `environment_agent_sandbox.sandbox_config` and checks canonical JSON.

- [ ] **Step 5: Implement resolved-policy channel creation**

Change `ResolvePerAgentEnvSpec` to return a policy rather than a public sandbox
ref. Resolve an explicit base env as before; select `default` when runtime is the
only input. Change the dependency interface/fakes and scratch channel creation
to pass policy maps. In `CreateEnvDispatchChannel`, call the codec for each
configured member and persist `{}` for members without an override.

Never assign runtime policy to `SandboxInstanceRef.RuntimeMetadata`.

- [ ] **Step 6: Run Task 2 tests and verify GREEN**

```bash
cd multica/server
gofmt -w internal/handler/env_dispatch_channel_policy.go internal/handler/env_dispatch_channel_policy_test.go internal/handler/env_dispatch.go internal/service/env_dispatch.go internal/service/env_dispatch_test.go
go test ./internal/service ./internal/handler -run 'EnvDispatch.*(Policy|PerAgent|Binding)|TestEnvDispatchSandboxConfig' -count=1
```

Expected: PASS or existing explicit DB skips only.

- [ ] **Step 7: Commit the nested-repository Task 2 change**

```bash
git -C multica add server/internal/handler/env_dispatch_channel_policy.go server/internal/handler/env_dispatch_channel_policy_test.go server/internal/handler/env_dispatch.go server/internal/service/env_dispatch.go server/internal/service/env_dispatch_test.go
git -C multica commit -m "feat(env-dispatch): persist per-agent runtime policy"
```

### Task 3: Create and Clone Provisioning Consume Stored Runtime

**Files:**

- Modify: `multica/server/internal/handler/env_dispatch_channel_policy.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_policy_test.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_provision.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_store_test.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_copy.go` only if a
  failing inheritance test reveals that existing verbatim `sandbox_config` copy
  is insufficient.

**Interfaces:**

- Consumes: strict binding codec from Task 2.
- Produces: `envDispatchSandboxConfig.createInput(workspaceID, daemonID string) (service.CreateSandboxInstanceInput, error)`.
- Preserves: existing binding single-flight and branch `sandbox_config` copy.

- [ ] **Step 1: Write failing create-input tests**

Given canonical config, assert the helper returns `Template: "default"`,
`DaemonEnabled: true`, `RuntimeEnv["MULTICA_DAEMON_ID"]`, and runtime JSON exactly
equal to the three-key object. Add malformed JSON and partial runtime cases that
must fail without including the sentinel key in `err.Error()`.

- [ ] **Step 2: Run provisioning tests and verify RED**

```bash
cd multica/server
go test ./internal/handler -run 'TestEnvDispatchSandboxConfig(CreateInput|RejectsSecretDisclosure)' -count=1
```

Expected: compile failure because `createInput` is absent.

- [ ] **Step 3: Implement strict create-input construction**

Marshal only `config.Runtime` into `CreateSandboxInstanceInput.Runtime`:

```go
runtimeJSON := json.RawMessage(nil)
if config.Runtime != nil {
    encoded, err := json.Marshal(config.Runtime)
    if err != nil {
        return service.CreateSandboxInstanceInput{}, fmt.Errorf("encode sandbox runtime policy: %w", err)
    }
    runtimeJSON = encoded
}
return service.CreateSandboxInstanceInput{
    WorkspaceID: workspaceID,
    Template: config.Template,
    DaemonEnabled: true,
    Runtime: runtimeJSON,
    RuntimeEnv: map[string]string{"MULTICA_DAEMON_ID": daemonID},
}, nil
```

Replace the permissive local anonymous config decode in
`provisionEnvDispatchAgent` with the strict codec/helper. On failure, use the
existing compensation path and `markFailed`; never continue with `default` or
the agent's shared runtime.

- [ ] **Step 4: Write failing leader, peer, concurrency, and branch tests**

Use the existing service provisioning fake to assert leader `SandboxConfig`
identifies the stored policy. Extend store/copy tests to prove:

- concurrent claims still have one winner;
- a pending peer retains its runtime policy after the original request;
- channel copy preserves `sandbox_config` byte-for-byte and source sandbox ID;
- clone `CreatePayload` contains the inherited runtime object;
- errors and serialized status/rollout responses omit the sentinel key.

Keep DB-dependent cases guarded by the repository's existing database skip.

- [ ] **Step 5: Make the minimal production adjustments for GREEN**

Ensure the initial leader call passes the binding-owned configuration (the
provisioner already falls back to `binding.SandboxConfig` for `{}`). Preserve
the current claim/status transitions and existing branch copy SQL. Modify copy
code only if the RED test proves a real gap.

- [ ] **Step 6: Run Task 3 tests and verify GREEN**

```bash
cd multica/server
gofmt -w internal/handler/env_dispatch_channel_policy.go internal/handler/env_dispatch_channel_policy_test.go internal/handler/env_dispatch_channel_provision.go internal/handler/env_dispatch_channel_store_test.go
go test ./internal/handler ./internal/service -run 'EnvDispatch.*(Provision|FirstMention|Branch|Binding|Runtime)|TestEnvDispatchSandboxConfig' -count=1
```

Expected: PASS or existing explicit DB skips only.

- [ ] **Step 7: Commit the nested-repository Task 3 change**

```bash
git -C multica add server/internal/handler/env_dispatch_channel_policy.go server/internal/handler/env_dispatch_channel_policy_test.go server/internal/handler/env_dispatch_channel_provision.go server/internal/handler/env_dispatch_channel_store_test.go
git -C multica diff --name-only --cached
git -C multica commit -m "fix(env-dispatch): configure isolated sandbox inference"
```

### Task 4: Client Contract, Documentation, and Verification

**Files:**

- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`
- Modify: `customized_areal/tree_search/README.md`
- Do not modify: `customized_areal/tree_search/agents/multica_client.py` unless the
  new serialization test fails for a reason not caused by the user's existing
  uncommitted edits.
- Modify: `openspec/changes/env-dispatch-agent-runtime-config/tasks.md`

**Interfaces:**

- Consumes: existing `MulticaEnvDispatchClient.create_env_dispatch(per_agent_env=...)` generic serialization.
- Produces: a secret-free runtime-only scratch request example.

- [ ] **Step 1: Add the client serialization regression test**

Extend the existing test with nested runtime data and assert exact pass-through:

```python
runtime_policy = {
    "agent-1": {
        "runtime": {
            "base_url": "https://provider.invalid/v1",
            "api_key": "synthetic-secret-for-tests",
            "model": "model-a",
        }
    }
}
```

Call `create_env_dispatch(..., per_agent_env=runtime_policy)` and assert the
request body equals `runtime_policy`.

- [ ] **Step 2: Run the client test and verify current behavior**

```bash
uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -k per_agent_env -q
```

Expected: PASS without production client changes. If it fails, inspect the
existing user diff before making the smallest compatible adjustment.

- [ ] **Step 3: Add a secret-free README example**

Document scratch runtime-only semantics and branch inheritance. Use
`https://provider.example/v1`, `YOUR_ROTATED_API_KEY`, and `provider-model`; do
not include any live endpoint credential.

- [ ] **Step 4: Run complete targeted verification**

```bash
cd multica/server
go test ./internal/service ./internal/handler -count=1
cd ../../..
uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -q
openspec validate env-dispatch-agent-runtime-config --strict
graphify update .
```

Expected: all available tests PASS; database integration tests may only skip via
their existing explicit availability guard. If `go` is unavailable, record the
toolchain blocker and run the same commands in the deployed build environment
before completion.

- [ ] **Step 5: Run deployed end-to-end verification after redeploy**

Use a fresh non-training agent and a newly rotated key supplied outside source
control. POST scratch message env-dispatch with a runtime-only per-agent entry,
wait for the sandbox daemon and agent reply, then GET the channel DAG. Assert:

- one isolated binding runtime is used;
- the agent posts a non-empty reply;
- response/error payloads do not contain the provider key;
- DAG nodes and edges reference existing IDs and the graph is acyclic.

- [ ] **Step 6: Check off OpenSpec tasks and commit exact outer-repository files**

Load `commit-conventions`, then stage only files owned by this change. Exclude
the user's pre-existing client edit and unrelated reports/changes:

```bash
git add docs/superpowers/specs/2026-07-20-env-dispatch-agent-runtime-config-design.md docs/superpowers/plans/2026-07-20-env-dispatch-agent-runtime-config.md customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/README.md openspec/changes/env-dispatch-agent-runtime-config
git diff --cached --name-only
git commit -m "feat(env-dispatch): configure external sandbox models"
```

Do not mark deployed verification complete until the redeployed service has
actually returned an agent reply and a valid DAG.
