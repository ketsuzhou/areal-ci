# Client-Triggered Diagnosis Design

## Goal

Make an authenticated client POST to a terminal environment-dispatch DAG the
only way to run diagnosis. Neither training nor server startup may initiate a
diagnosis run implicitly.

## Decision

Remove `DIAGNOSIS_AGENT_ENABLED` and
`DIAGNOSIS_AGENT_ON_DEMAND_ENABLED` completely. The backend must no longer
parse, expose, or gate on either environment variable.

The existing client call to `POST /api/v1/env-dispatch/{project_id}/diagnosis`
(or its channel facade) is the opt-in action. A normal non-training dispatch
does not make this request unless the caller explicitly selects diagnosis.

## Server Behavior

The diagnosis handlers continue to enforce all non-configuration checks:

- authenticated user and workspace/project ownership;
- terminal environment-dispatch state;
- non-empty, dense, acyclic assembled DAG;
- a resolvable root task;
- existing completed diagnosis idempotency.

After those checks, the handler creates the Pi diagnosis runner using the
remaining runtime settings (`DIAGNOSIS_AGENT_PATH`,
`DIAGNOSIS_AGENT_MODEL`, `DIAGNOSIS_AGENT_TIMEOUT_SECONDS`, and
`DIAGNOSIS_AGENT_SCORE_MAX`). Pi failures continue to return
`diagnosis_failed`; unavailable runtime setup continues to return
`diagnosis_unavailable`.

## Training Compatibility

The legacy training-session diagnosis integration is removed. In particular,
`NewTrainingSessionDeps` must not build or attach a diagnoser, and no training
event may invoke a diagnosis agent. Training reward behavior remains otherwise
unchanged.

## Client Behavior

`MulticaEnvDispatchClient.diagnose_env_dispatch()` remains the explicit API.
The debug command retains `--diagnose`; omitting it must only poll and return
the normal assembled DAG. Supplying it posts diagnosis after normal DAG
readiness and returns the scored DAG/report.

## Verification

Tests must prove that a terminal, valid DAG reaches the diagnosis runner with
no feature flags set; the prior flag-disabled rejection tests are removed or
rewritten. Tests must also prove that training dependency construction has no
diagnoser and that the client invokes diagnosis only when requested. Existing
authentication, readiness, idempotency, coverage, and global node-reward
normalization tests remain required.
