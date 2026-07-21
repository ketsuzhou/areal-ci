# Task 1 Report: Required training-mode request contract

## Files changed (4, all in allowed scope)

- `multica/server/internal/handler/env_dispatch.go` - added `EnvDispatchRequest.TrainingMode *bool` field, nil-rejection (HTTP 400), and dereferenced pass-through to `EnvDispatchInput.TrainingMode bool`.
- `multica/server/internal/service/env_dispatch.go` - added `EnvDispatchInput.TrainingMode bool` field and two validation rules in `validate()`.
- `multica/server/internal/handler/env_dispatch_test.go` - added `TestEnvDispatch_TrainingMode` (5 subtests); updated all existing POST env-dispatch test bodies to include `training_mode`.
- `multica/server/internal/service/env_dispatch_test.go` - added `TestEnvDispatchValidate_TrainingMode` (5 cases); updated existing tests that set `TrainAgentID` to also set `TrainingMode: true`.

## RED verification

Command:
```
cd /workspaces/leagent/backend/areal/multica/server && go test ./internal/handler/... ./internal/service/... -run 'EnvDispatch.*TrainingMode' -count=1
```

Result: **FAIL**

- `internal/service`: build failed - `EnvDispatchInput has no field or method TrainingMode` (5 compilation errors). This proves the training-mode contract did not exist yet.
- `internal/handler`: `ok` - handler tests are skipped in this environment (no Postgres reachable; `TestMain` in `handler_test.go` calls `os.Exit(0)`). The handler test file compiled correctly but could not run. This is the existing behavior for ALL handler tests in this environment (not specific to this task).

## GREEN verification

Command (same as RED):
```
cd /workspaces/leagent/backend/areal/multica/server && go test ./internal/handler/... ./internal/service/... -run 'EnvDispatch.*TrainingMode' -count=1 -v
```

Result: **PASS**
- `internal/service`: `TestEnvDispatchValidate_TrainingMode` - PASS (0.00s). All 5 cases passed: false+train_agent_id rejected, false+critic_agent_id rejected, true without train_agent_id rejected, true+train_agent_id accepted, false without IDs accepted.
- `internal/handler`: `ok` (tests skipped - no DB; compiles cleanly via `go vet`).

Full service test suite (all tests, not just TrainingMode): `ok` - no regressions.

## Commit

- Hash: `f124119f34c063892020639a97dcb62621dfc48a`
- Branch: `feature/20260721/env-dispatch-nontraining-dag`
- Message: `feat(env-dispatch): require explicit training_mode request contract`
- 4 files changed, 187 insertions(+), 28 deletions(-)
- Not pushed.

## Implementation summary

### Handler (`env_dispatch.go`)
- `EnvDispatchRequest.TrainingMode *bool` with json tag `training_mode` - pointer distinguishes omitted (nil) from explicit false.
- After JSON decode: `if req.TrainingMode == nil { writeError(w, 400, "training_mode is required"); return }` - rejects before constructing `EnvDispatchInput`.
- Service call: `TrainingMode: *req.TrainingMode` - passes the dereferenced bool.

### Service (`env_dispatch.go`)
- `EnvDispatchInput.TrainingMode bool` - non-pointer; the handler guarantees it is always set.
- In `validate()`, placed before the existing train_agent_id block:
  - `if !in.TrainingMode && (in.TrainAgentID != "" || in.CriticAgentID != "")` -> `"validation_failed: training_mode=false forbids train_agent_id and critic_agent_id"`
  - `if in.TrainingMode && in.TrainAgentID == ""` -> `"validation_failed: training_mode=true requires train_agent_id"`

### Test updates to existing tests
Existing tests that set `TrainAgentID` without `TrainingMode` were updated to set `TrainingMode: true` (the zero value `false` would now trigger the new "false forbids training IDs" rule). Updated tests:
- `TestValidate_TrainAgentID` (3 sub-cases)
- `TestEnvDispatchInput_Validate_CriticAgentID` (8 table cases; one expected error changed from "critic_agent_id requires train_agent_id" to "training_mode=true requires train_agent_id" because the new rule fires first)
- `TestDispatch_ScratchMessage_TrainingSessionLinkedAfterEnqueue`
- `TestDispatch_PersistsTrainingDispatchWhenTrainAgentSet`
- `TestEnvDispatchTrainedRolloutCreatesSandboxInstanceRefs`
- `TestEnvDispatchSandboxInstanceBranchCreatesFreshFromTemplate`
- `TestEnvDispatchPerAgentEnvSpecsAssignDistinctSandboxRefs`
- `TestEnvDispatchPerAgentEnvSpecsPartialSquadUsesDefaults`
- `TestEnvDispatch_PersistsCriticAgentID`
- `TestEnvDispatch_NoCritic_PersistsNull`
- All existing handler POST env-dispatch test bodies: added `"training_mode":false` or `"training_mode":true` (matching whether `train_agent_id` is present).

## Self-review

- **Contract correct**: `TrainingMode *bool` in handler (pointer for omitted detection), `TrainingMode bool` in service (dereferenced). Matches the brief exactly.
- **Validation rules correct**: false forbids train_agent_id AND critic_agent_id; true requires train_agent_id. Matches the brief's exact conditions.
- **No backward-compat inference**: training_mode is required; nil is rejected at the handler. No inference from train_agent_id.
- **No AReaL call gating**: per the plan, Task 1 only enforces the validation contract. The zero-call behavior for training_mode=false is realized in later tasks.
- **Secrets**: no API keys or secrets in DAG data, responses, errors, or logs. (Not relevant to Task 1 but verified.)
- **Scope**: only the 4 allowed files modified. No dependencies added.
- **Existing tests**: all updated to accommodate the new required field. Full service test suite passes with no regressions.
- **Handler tests**: skipped in this environment (no DB) but compile correctly. Will run in CI.
- **Formatting**: `gofmt` clean. `go vet` clean.

## Risk-signal self-report

- **Cross-module / cross-subsystem coordinated change**: NO - handler and service are in the same env-dispatch subsystem.
- **Security-sensitive surface (auth, authorization, crypto, SQL, external input handling, secrets/credentials)**: YES - external input handling (HTTP request validation on a public API endpoint).
- **Concurrency, locks, shared mutable state**: NO.
- **Data or schema migration**: NO.
- **Public API contract or external interface change**: YES - `training_mode` is now a required field on `POST /api/v1/env-dispatch`. This is a breaking change for existing callers who do not send `training_mode`.
- **Single-task diff exceeds 200 lines**: YES (marginally) - 187 insertions + 28 deletions = 215 total changed lines. However, ~80 of those are mechanical updates to existing tests (adding `TrainingMode: true` or `"training_mode":false` to existing test inputs). The actual new logic is ~31 lines of production code + ~90 lines of new tests.
