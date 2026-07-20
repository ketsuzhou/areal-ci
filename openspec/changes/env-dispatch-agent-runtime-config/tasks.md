## 1. Request Contract and Validation Tests

- [ ] 1.1 Add handler tests that parse
  `per_agent_env.<agent_id>.runtime` and map `base_url`, `api_key`, and `model`
  to the service input without exposing them in the response.
- [ ] 1.2 Add service tests for complete runtime configuration, missing/blank
  fields, invalid URL schemes, runtime on `train_agent_id`, and no-resource
  side effects on validation failure.
- [ ] 1.3 Add backward-compatibility tests for per-agent environment entries that
  omit `runtime`.

## 2. Runtime Policy Propagation Tests

- [ ] 2.1 Add channel-binding tests proving each agent's canonical runtime policy
  is persisted with its template and remains isolated from other squad members.
- [ ] 2.2 Add provisioning tests proving immediate leader create and lazy peer
  first-mention create pass the stored runtime object to the sandbox lifecycle.
- [ ] 2.3 Add clone/retry and concurrent first-mention tests proving one canonical
  runtime policy is preserved by the existing single-flight path.
- [ ] 2.4 Add failure-path assertions that responses, errors, and structured logs do
  not contain the synthetic API key.

## 3. Server Implementation

- [ ] 3.1 Add typed external runtime request/service structures and atomic
  validation for complete values, absolute HTTP(S) URL, and non-training target.
- [ ] 3.2 Carry the validated runtime policy through per-agent environment
  resolution and serialize it into the existing env-agent sandbox binding
  configuration.
- [ ] 3.3 Decode the stored policy in `provisionEnvDispatchAgent` and populate
  `CreateSandboxInstanceInput.Runtime` for create and clone payloads.
- [ ] 3.4 Audit env-dispatch response/error/log paths and prevent runtime secret
  disclosure without changing response schemas.

## 4. Client Compatibility and Documentation

- [ ] 4.1 Verify the standalone MultiCA client's generic `per_agent_env`
  serialization carries the nested runtime object unchanged; add a focused test
  or minimal client adjustment only if needed.
- [ ] 4.2 Add a secret-free request example documenting per-agent external model
  configuration and the current rejection of caller runtime for
  `train_agent_id`.

## 5. Verification

- [ ] 5.1 Run targeted Go handler/service/provisioning tests and the relevant
  Python client tests; record any unavailable toolchain or integration suite.
- [ ] 5.2 Run repository formatting/lint checks appropriate to the changed files
  and update the code graph with `graphify update .`.
- [ ] 5.3 After deployment, dispatch to a fresh non-training agent with a rotated
  external provider credential, confirm the agent replies in the channel, and
  validate that the returned channel DAG is structurally valid.
