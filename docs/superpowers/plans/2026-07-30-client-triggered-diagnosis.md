# Client-Triggered Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an authenticated client POST the sole trigger for diagnosis and
remove both diagnosis enablement flags plus the legacy training trigger.

**Architecture:** The non-training diagnosis handlers retain ownership,
terminal-DAG, topology, runner-construction, and idempotency checks but stop
consulting feature flags. Training dependency construction no longer exposes a
diagnoser, and terminal training routing no longer calls diagnosis. The AReaL
client keeps `--diagnose` as its sole opt-in action.

**Tech Stack:** Go 1.26, Chi HTTP handlers, sqlc/pgx, Go `testing`, Python
3.12, pytest/httpx, GitHub Actions.

## Global Constraints

- Do not add dependencies or change public route paths.
- Do not install or run the local Python environment; use online CI for Python
  verification.
- Delete both `DIAGNOSIS_AGENT_ENABLED` and
  `DIAGNOSIS_AGENT_ON_DEMAND_ENABLED`; do not replace them with another server
  admission flag.
- Retain the remaining Pi runtime settings and all authorization, readiness,
  idempotency, exact-turn coverage, and DAG normalization behavior.
- Never trigger diagnosis from a training event; only the authenticated
  diagnosis POST may create a diagnosis run.

**Execution order:** Execute Task 2 before Task 1. Task 2 removes the legacy
training dependency that currently references `DiagnosisAgentEnabled`; this
keeps every intermediate commit compilable while Task 1 subsequently deletes
the configuration field and route gate. Execute Tasks 3 and 4 afterward.

---

### Task 1: Remove configuration gates from the non-training diagnosis route

**Files:**

- Modify: `multica/server/internal/service/training_config.go:25-145`
- Modify: `multica/server/internal/handler/env_dispatch_diagnosis.go:124-128`
- Modify: `multica/server/internal/service/training_config_test.go:203-255`
- Test: `multica/server/internal/handler/env_dispatch_diagnosis_test.go`

**Interfaces:**

- Consumes: `service.LoadTrainingConfig()` for Pi path, model, timeout, and
  score maximum.
- Produces: `POST /api/v1/env-dispatch/{projectID}/diagnosis` and its channel
  facade execute after their existing non-configuration validation passes.

- [ ] **Step 1: Write failing configuration and handler tests**

  Replace flag-oriented expectations with a runtime-settings-only test:

  ```go
  func TestLoadTrainingConfig_DiagnosisRuntimeSettings(t *testing.T) {
      clearTrainingEnv(t)
      t.Setenv("DIAGNOSIS_AGENT_PATH", "/usr/local/bin/pi")
      t.Setenv("DIAGNOSIS_AGENT_MODEL", "anthropic/claude-sonnet-5")
      t.Setenv("DIAGNOSIS_AGENT_TIMEOUT_SECONDS", "120")
      t.Setenv("DIAGNOSIS_AGENT_SCORE_MAX", "20")

      cfg := LoadTrainingConfig()

      assert.Equal(t, "/usr/local/bin/pi", cfg.DiagnosisAgentPath)
      assert.Equal(t, "anthropic/claude-sonnet-5", cfg.DiagnosisAgentModel)
      assert.Equal(t, 120*time.Second, cfg.DiagnosisAgentTimeout)
      assert.Equal(t, 20, cfg.DiagnosisAgentScoreMax)
  }
  ```

  Add a handler-level regression using the existing database-backed handler
  fixture: leave both removed variables unset, arrange a terminal dense DAG,
  and assert the request does not return the former `409
  {"error":"diagnosis_unavailable"}` before it reaches the runner seam.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run:

  ```bash
  go test ./internal/service ./internal/handler -run 'TestLoadTrainingConfig_DiagnosisRuntimeSettings|TestDiagnoseEnvDispatch.*WithoutEnablementFlags' -count=1
  ```

  Expected: FAIL because `TrainingConfig` still exposes the removed booleans or
  the handler still rejects unset enablement flags.

- [ ] **Step 3: Remove the configuration surface and handler gate**

  In `TrainingConfig`, delete only these fields and constants:

  ```go
  DiagnosisAgentEnabled        bool
  DiagnosisAgentOnDemandEnabled bool
  diagnosisAgentEnabledEnv
  diagnosisAgentOnDemandEnabledEnv
  ```

  Delete their `os.Getenv`/`strconv.ParseBool` blocks. Retain the path, model,
  timeout, score-max, and paging settings unchanged.

  In `diagnoseEnvDispatchProject`, remove:

  ```go
  if !cfg.InteractionDAGEnabled || !cfg.DiagnosisAgentEnabled || !cfg.DiagnosisAgentOnDemandEnabled {
      writeJSON(w, http.StatusConflict, map[string]any{"error": "diagnosis_unavailable"})
      return
  }
  ```

  Keep `cfg := service.LoadTrainingConfig()` because the handler uses its Pi
  runtime settings below the readiness checks.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run the Task 1 command again.

  Expected: PASS. A valid terminal DAG advances past admission without either
  removed variable; invalid ownership/readiness/DAG cases retain their current
  response codes.

- [ ] **Step 5: Commit Task 1**

  ```bash
  git add server/internal/service/training_config.go \
    server/internal/service/training_config_test.go \
    server/internal/handler/env_dispatch_diagnosis.go \
    server/internal/handler/env_dispatch_diagnosis_test.go
  git commit -m "refactor(diagnosis): remove enablement gates"
  ```

### Task 2: Remove diagnosis from the training execution path

**Files:**

- Modify: `multica/server/internal/service/training.go:91-95,114-139,609-660,694-700`
- Modify: `multica/server/internal/service/training_config.go:142-232`
- Modify: `multica/server/internal/service/training_test.go:601-786`
- Modify: `multica/server/internal/service/training_config_test.go:257-304`

**Interfaces:**

- Consumes: `TrainingSessionDeps` used by terminal training routing.
- Produces: terminal training only closes/critics as before; it has no
  `Diagnoser`, no `maybeDiagnoseProject`, and cannot persist diagnosis rewards.

- [ ] **Step 1: Write the failing training-dependency test**

  Replace the former enabled/disabled wiring test with:

  Convert `TestDiagnosisBeforeCloseHook_Ordering` into a red regression for
  the actual terminal route. Reuse its fake closer and task context, but invoke
  `RouteTerminalTrainingTask` through the existing task-service test fixture
  and assert only the close/critic order. The pre-change route still invokes
  `maybeDiagnoseProject`, so the observed order includes `Diagnose` and
  `RecordStepRewards` and fails the new expectation.

  Keep `TestNewTrainingSessionDeps_DAGWiredInProductionPath` as the dependency
  construction regression; it uses the existing `q := db.New(nil)` fixture and
  must continue to assert that DAG recording remains configured after the
  diagnosis collaborator is removed.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run:

  ```bash
  go test ./internal/service -run 'TestNewTrainingSessionDeps_DoesNotWireDiagnosis|TestMaybeDiagnoseProject|TestDiagnosisBeforeCloseHook' -count=1
  ```

  Expected: FAIL because the terminal route still invokes
  `maybeDiagnoseProject` before the close/critic route.

- [ ] **Step 3: Delete the complete legacy trigger surface**

  Remove all of the following together:

  ```go
  type Diagnoser interface { ... }
  TrainingSessionDeps.Diagnosis
  buildDiagnoser(...)
  maybeDiagnoseProject(...)
  maybeDiagnoseProject(ctx, s.Training, task, projectID, dispatch)
  ```

  Delete the fake diagnoser, diagnosis dependency helper, diagnosis-before-close
  ordering helper, and all `TestMaybeDiagnoseProject_*` /
  `TestDiagnosisBeforeCloseHook_Ordering` cases. Update comments on terminal
  routing so they describe only critic and session-close sequencing.

- [ ] **Step 4: Run focused service tests and verify GREEN**

  Run:

  ```bash
  go test ./internal/service -count=1
  ```

  Expected: PASS with no references to `Diagnoser`, `maybeDiagnoseProject`, or
  `DiagnosisAgentEnabled` in production or test code.

- [ ] **Step 5: Commit Task 2**

  ```bash
  git add server/internal/service/training.go \
    server/internal/service/training_config.go \
    server/internal/service/training_test.go \
    server/internal/service/training_config_test.go
  git commit -m "refactor(training): remove automatic diagnosis"
  ```

### Task 3: Preserve explicit client opt-in and verify through online CI

**Files:**

- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`
- Modify: `.github/workflows/nontraining-multica-diagnosis.yml`
- Test: `customized_areal/tree_search/tests/test_multica_dag_client.py`
- Test: `customized_areal/tree_search/tests/test_multica_node_rewards.py`

**Interfaces:**

- Consumes: `MulticaEnvDispatchClient.diagnose_env_dispatch(handle)` and the
  `--diagnose` debug option.
- Produces: a diagnosis POST only when explicitly requested, followed by exact
  turn coverage validation and global node-reward normalization.

- [ ] **Step 1: Write the failing client opt-in regression**

  Add a mock-transport test that runs the debug/client path without diagnosis
  selected and records every request:

  ```python
  def test_debug_flow_without_diagnose_never_posts_diagnosis():
      seen: list[tuple[str, str]] = []
      # Return a completed DAG to its GET poll and fail the test on POST /diagnosis.
      ...
      assert ("POST", "/api/v1/env-dispatch/proj-1/diagnosis") not in seen
  ```

  Keep the existing positive test asserting `diagnose_env_dispatch()` posts the
  dispatch-scoped endpoint and rejects non-completed reports.

- [ ] **Step 2: Run the targeted test in online CI and verify RED**

  Push the test-only commit to the isolated CI branch and inspect the
  `Non-training MultiCA diagnosis` check. Expected: FAIL until the test drives
  the debug path correctly, or until the current client behavior is verified
  by the new assertion.

- [ ] **Step 3: Make the minimal client adjustment, only if RED proves one is needed**

  Preserve the existing conditional boundary:

  ```python
  if args.diagnose:
      client.diagnose_env_dispatch(handle)
  ```

  Do not add environment-variable checks or make diagnosis automatic. If the
  regression is already green, keep production client code unchanged and commit
  only the test.

- [ ] **Step 4: Run online contract CI and verify GREEN**

  The workflow must run:

  ```bash
  python -m pytest -q \
    customized_areal/tree_search/tests/test_env_dispatch_client.py \
    customized_areal/tree_search/tests/test_multica_dag_client.py \
    customized_areal/tree_search/tests/test_multica_node_rewards.py
  ```

  Expected: all selected tests pass. Record the GitHub Actions run URL; do not
  substitute a local Python environment.

- [ ] **Step 5: Commit Task 3**

  ```bash
  git add customized_areal/tree_search/tests/test_env_dispatch_client.py \
    .github/workflows/nontraining-multica-diagnosis.yml
  git commit -m "test(tree-search): verify explicit diagnosis opt-in"
  ```

### Task 4: Final integration verification and deployment configuration cleanup

**Files:**

- Modify: deployment environment documentation that lists diagnosis variables,
  if one references either removed name.
- Test: `multica/server/internal/service/training_config_test.go`
- Test: `multica/server/internal/handler/env_dispatch_diagnosis_test.go`

**Interfaces:**

- Consumes: deployed Pi runtime settings and an authenticated client POST.
- Produces: no documented or deployed dependency on either removed enablement
  variable.

- [ ] **Step 1: Search for the removed variables**

  Run:

  ```bash
  rg -n 'DIAGNOSIS_AGENT_ENABLED|DIAGNOSIS_AGENT_ON_DEMAND_ENABLED|DiagnosisAgentEnabled|DiagnosisAgentOnDemandEnabled|maybeDiagnoseProject|\bDiagnoser\b' .
  ```

  Expected: no production, test, workflow, compose, or documentation matches.

- [ ] **Step 2: Run complete MultiCA affected suites**

  Run:

  ```bash
  go test ./internal/service -count=1
  go test ./internal/handler -count=1
  ```

  Expected: both commands exit 0.

- [ ] **Step 3: Verify online deployment and client-triggered E2E**

  After CI deploys the merged MultiCA change, use an authenticated CI fixture to
  create a non-training dispatch, wait for terminal DAG readiness, POST
  `/diagnosis`, and assert a completed report plus exact `step_rewards` for all
  assistant turns. Do not set either removed variable; pass only the Pi runtime
  settings required by the deployment.

- [ ] **Step 4: Record deployment verification evidence**

  Attach the successful CI run URL, the deployment revision, and the E2E
  report's exact assistant-turn count and `step_rewards` count to the pull
  request. Do not claim E2E success if no authenticated fixture exists.
