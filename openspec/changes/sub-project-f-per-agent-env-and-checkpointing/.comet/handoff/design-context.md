# Comet Design Handoff

- Change: sub-project-f-per-agent-env-and-checkpointing
- Phase: design
- Mode: compact
- Context hash: da6e37ca5874df46bd4fd61bf5bccf3d2ff825be2eda826d1c65d82174820e2c

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic,
source-traceable context pack, not an agent-authored summary.

## openspec/changes/sub-project-f-per-agent-env-and-checkpointing/proposal.md

- Source: openspec/changes/sub-project-f-per-agent-env-and-checkpointing/proposal.md
- Lines: 1-113
- SHA256: c3b1b99504467f8dc55b150316bae256f301ba2beef47beb11c4a1548c6418b7

\[TRUNCATED\]

```md
## Why

E closes the RL training loop with a terminal critic reward but explicitly
defers richer branching to F ("per-interaction reward is out of scope —
sub-project F or later"). Today the env-dispatch path can only branch from a
segment's terminal turn (SuperNode.env_id frontier), so RL training explores
alternatives only at trajectory endpoints — not at the trainable decisions
along the way (delegations, mentions, tool calls, file changes). F enables
branching from ANY trainable decision point by eagerly checkpointing
(env_id + per-issue subtree DB snapshot + full sandbox snapshot) at structural
DAG events and entropy-gated tool calls. This dramatically increases trajectory
diversity for RL training and enables debugging/reproducibility via full state
capture. F also adds per-agent env customization so squad members can run in
isolated sandboxes — cleaner credit assignment and support for heterogeneous
base environments (e.g., coder in a Python image, tester in a Node image).

## What Changes

- **Per-agent / per-group env customization (multica + AReaL client)**:
  env-dispatch contract extended to accept per-agent env specs. Each agent in a
  squad can be assigned its own sandbox from its own base env; the squad shares
  the multica entity subtree (issues / tasks / messages). Different rollout
  groups can use different base envs. AReaL client passes per-agent env specs.
- **Event-triggered state checkpointing (multica)**: eagerly save
  (env_id + per-issue subtree DB snapshot + full sandbox snapshot) at specific
  trainable decision points. Non-trainable events (claiming, cascade
  cancellations, sweeper timeouts, autopilot retry, sandbox lifecycle, RL
  session management) do NOT trigger checkpoints — they carry no policy signal.
- **Checkpoint triggers — always (5 structural decisions)**: delegation
  (`EnqueueTaskForIssue`/`EnqueueTaskForMention`/`EnqueueTaskForSquadLeader`),
  mention (`EnqueueTaskForMention`), completion (`CompleteTask`, LLM-driven
  path only — not sweeper/autopilot), failure (`FailTask`, LLM-driven path
  only), squad leader briefing generation. These are the DAG-growth and
  terminal-decision points where alternative policy choices are most
  consequential.
- **Checkpoint triggers — entropy-gated (all tool calls, incl. file changes)**:
  use E's captured logprobs to compute entropy per tool-call decision;
  checkpoint only if entropy exceeds a configurable threshold. High entropy =
  the LLM was uncertain which tool to pick = a meaningful branch point. Low
  entropy = skip (the LLM was confident; little training value in branching).
  Fallback: if logprobs are unavailable (E's graceful degradation path),
  entropy-gated checkpoints are skipped — always-checkpoint events still fire.
- **Branch-from-checkpoint (multica + AReaL)**: new operation that restores
  the sandbox snapshot + DB subtree and creates a new
  `env_dispatch(mode=branch-from-checkpoint)`. Extends C's `mode=branch` to
  accept a checkpoint as the branch source (instead of only a live env_id).
  AReaL client gains a branch-from-checkpoint API.
- **Cost posture**: full eager snapshots, accept the cost (simplest design).
  Copy-on-write / lazy sandbox / per-rollout configurability are explicitly
  deferred to future hardening — F ships the simple eager path first and
  optimizes only if cost becomes a real bottleneck.

## Capabilities

### New Capabilities

- `env-checkpointing`: Event-triggered state checkpointing for RL training on
  the env-dispatch path. Eagerly captures (env_id + per-issue subtree DB
  snapshot + full sandbox snapshot) at trainable decision points: 5 structural
  events always (delegation, mention, completion, failure, squad leader
  briefing); all tool calls entropy-gated via E's logprobs. Enables branching
  from any checkpoint for trajectory diversity and debugging/reproducibility.
  Non-trainable events (claiming, cascades, sweeper, autopilot, sandbox
  lifecycle, RL session management) are explicitly excluded.

- `per-agent-env-customization`: Per-agent and per-group sandbox assignment in
  env-dispatch. Squad members can run in isolated sandboxes from different
  base envs while sharing the multica entity subtree. Extends the env-dispatch
  contract with per-agent env specs, enabling heterogeneous base environments
  within a single rollout and cleaner per-agent credit assignment.

### Modified Capabilities

- `critic-driven-training-signal` (from E, not yet archived): F's
  entropy-gated checkpointing consumes E's logprobs. If logprobs are
  unavailable (E's graceful degradation path), entropy-gated checkpoints are
  skipped — always-checkpoint events still fire. No contract change to E; F
  depends on E's logprobs being available for the entropy-gated path.

- `training-session-lifecycle` (from D, not yet archived): F's checkpoints
```

Full source: openspec/changes/sub-project-f-per-agent-env-and-checkpointing/proposal.md

## openspec/changes/sub-project-f-per-agent-env-and-checkpointing/design.md

- Source: openspec/changes/sub-project-f-per-agent-env-and-checkpointing/design.md
- Lines: 1-334
- SHA256: c09e6917a83ba0cdc4aa008d9c5ccb48511f54279df896384b701b5ff85b4939

\[TRUNCATED\]

```md
# Sub-project F — per-agent env customization + event-triggered env checkpointing (design)

Status: DRAFT (2026-07-06)
Date: 2026-07-06
Repos: multica `main` (primary); AReaL `master` (client + entropy helper)

## 1. Goal

Enable branching from ANY trainable decision point on the env-dispatch path
(not just terminal turns) by eagerly checkpointing (env_id + per-issue subtree
DB snapshot + full sandbox snapshot) at structural DAG events and entropy-gated
tool calls. Additionally, allow per-agent / per-group sandbox customization so
squad members can run in isolated sandboxes from different base envs. F
dramatically increases trajectory diversity for RL training and enables
debugging/reproducibility via full state capture.

## 2. Actors / flow

```

env_dispatch (per-agent envs: {coder: python-env, tester: node-env, ...}) │ ▼
\[multica\] dispatch squad: each member gets its own sandbox from its base env │ shared
multica entity subtree (issues / tasks / messages) ▼ trained member task created → D's
open hook → start_session(env_id) │ ▼ trained agent works → LLM traffic via AReaL proxy
(E captures logprobs) │ │ ┌── always-checkpoint events ──────────────────────────────┐ │
│ delegation / mention / completion / failure / squad │ │ │ leader briefing →
checkpoint(env_id, DB subtree, │ │ │ sandbox snapshot, event_ref) │ │
└──────────────────────────────────────────────────────────┘ │ │ ┌── entropy-gated tool
calls ──────────────────────────────┐ │ │ each tool-call decision: compute entropy from
E's │ │ │ logprobs; if entropy > threshold → checkpoint │ │ │ (includes file-change tool
calls) │ │ └──────────────────────────────────────────────────────────┘ │ ▼ AReaL
selects a checkpoint as branch candidate │ ▼ F's branch-from-checkpoint \[multica\]
restore sandbox snapshot + DB subtree → new env_dispatch (mode=branch-from-checkpoint,
checkpoint_id=...) │ ▼ branched rollout begins (new RL session via D's open hook)

```

## 3. Decisions (locked)

### F1 — Two pillars, one change

Per-agent env customization and event-triggered checkpointing are distinct
features but ship together as F. They share the env-dispatch contract surface
and the multica handler/service layer. A rollout may use either or both.

### F2 — Checkpoint trigger taxonomy

Three tiers, mapped to whether the triggering event reflects an LLM policy
decision (trainable) or a code rule (not trainable):

| Tier | Events | Why |
|------|--------|-----|
| Always | delegation, mention, completion (LLM path), failure (LLM path), squad leader briefing | Structural DAG-growth + terminal-decision points. Rare, high-impact. Alternative policy choices here are the most consequential. |
| Entropy-gated | all tool calls (incl. file changes) | Frequent, low-stakes individually. Entropy from E's logprobs gates: high entropy = LLM was uncertain = good branch point. |
| Never | claiming, cascade cancellations, sweeper timeouts, autopilot retry, sandbox lifecycle, RL session management | Code-driven, no policy signal. Checkpointing wastes storage and complicates the trajectory graph. |

### F3 — Per-agent env customization granularity

Both per-agent (within a squad) AND per-group (across squads). The env-dispatch
request accepts an optional `per_agent_envs: {<agent_id>: <base_env_spec>}`
map. When omitted, today's behavior (all lanes fork from one base env) is
preserved.

### F4 — Checkpoint content
```

Full source: openspec/changes/sub-project-f-per-agent-env-and-checkpointing/design.md

## openspec/changes/sub-project-f-per-agent-env-and-checkpointing/tasks.md

- Source: openspec/changes/sub-project-f-per-agent-env-and-checkpointing/tasks.md
- Lines: 1-302
- SHA256: 2fff7c65abbf700de9aea410aae236d7b2c4e52cfb676552ad4fd45bb64ab065

\[TRUNCATED\]

```md
# Tasks — sub-project-f-per-agent-env-and-checkpointing

## Task 1: Investigation — seams confirmation

**Files**: `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`,
`internal/service/task.go` (delegation/mention/completion/failure seams),
daemon/pi tool-call boundary, Fleet sandbox snapshot API, AReaL proxy logprob capture point.

- [ ] Confirm per-agent env dispatch seam: `EnvDispatchRequest` / `service.EnvDispatchInput`
  struct shape, validation site, dispatch loop.
- [ ] Confirm always-event hook seams: exact call sites for
  `EnqueueTaskForIssue`/`EnqueueTaskForMention`/`EnqueueTaskForSquadLeader`
  (delegation + mention), `CompleteTask` (LLM path vs sweeper/autopilot
  discriminator), `FailTask` (same discriminator), squad leader briefing
  generation site.
- [ ] Confirm tool-call boundary in pi/daemon: where the daemon receives
  tool-call completion + the AReaL proxy logprob capture point (E's seam).
- [ ] Confirm Fleet sandbox snapshot API: create snapshot, fork/restore
  snapshot (used by C's lazy-snapshot path).
- [ ] Confirm DB subtree serialization approach: issue + sub-issues + tasks +
  messages + comments query shape.
- [ ] Confirm AReaL client seam: `swe_lego_client.py` `create_env_dispatch`
  signature + where E will attach logprobs.
- [ ] Document findings in design doc updates (T1 section or inline).
- [ ] Commit: `docs(F): T1 investigation — seam confirmation`.

## Task 2: Per-agent env contract (multica handler + service) (TDD)

**Files**: `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`,
`internal/service/env_dispatch_test.go`.

- [ ] Failing tests:
  - `TestEnvDispatch_PerAgentEnvs_Valid` — per_agent_envs map with valid
    agent_id → base_env_id → each agent gets its own sandbox.
  - `TestEnvDispatch_PerAgentEnvs_UnknownAgent` — unknown agent_id → 400.
  - `TestEnvDispatch_PerAgentEnvs_UnknownBaseEnv` — unknown base_env_id → 400.
  - `TestEnvDispatch_PerAgentEnvs_MixedWithTopLevel` — per_agent_envs +
    top-level env_id for same agent → 400.
  - `TestEnvDispatch_PerAgentEnvs_Empty` — empty per_agent_envs → today's
    behavior (backward compat).
  - `TestEnvDispatch_PerAgentEnvs_PartialSquad` — some agents have custom
    envs, others use default → validated.
- [ ] Add `per_agent_envs map[string]string` to `EnvDispatchRequest` (handler)
  and `EnvDispatchInput` (service).
- [ ] Handler validation: resolve each agent_id → real agent in workspace;
  resolve each base_env_id → real env; reject mixing with top-level env_id
  for the same agent.
- [ ] Service: when per_agent_envs is set, dispatch each agent to its own
  sandbox from its specified base env. Squad shares one multica entity subtree.
- [ ] Run: `go test ./internal/service/ -run 'EnvDispatch_PerAgent'`.
- [ ] Commit: `feat(env-dispatch): per-agent env customization contract`.

## Task 3: Per-agent env persistence + AReaL client (TDD)

**Files**: multica migration NNN, `internal/service/training_dispatch.go`,
`internal/service/training_dispatch_test.go`,
`customized_areal/tree_search/agents/swe_lego_client.py`.

- [ ] Failing tests (multica):
  - `TestCreateTrainingDispatch_WithPerAgentEnvs` — per_agent_envs persisted
    and round-tripped.
  - `TestGetTrainingDispatchByProject_WithPerAgentEnvs` — returned with
    per_agent_envs.
  - `TestCreateTrainingDispatch_NoPerAgentEnvs` — field is null when not set
    (backward compat).
- [ ] Migration: `ALTER TABLE training_dispatch ADD COLUMN per_agent_envs
    JSONB NULL`. Hand-write generated Go (mirror sibling — do NOT run
    `sqlc generate` repo-wide).
- [ ] Extend `CreateTrainingDispatch` / `GetTrainingDispatchByProject` queries
  to handle `per_agent_envs`.
- [ ] Failing tests (AReaL):
  - `test_create_env_dispatch_with_per_agent_envs` — per_agent_envs dict
    passed in request body.
  - `test_create_env_dispatch_without_per_agent_envs` — field omitted when
    empty.
- [ ] Extend `create_env_dispatch` to accept optional `per_agent_envs` dict.
- [ ] Run: `go test ./internal/service/ -run 'TrainingDispatch.*PerAgent'` +
  `uv run pytest customized_areal/tree_search/tests/ -k 'per_agent_env'`.
- [ ] Commit: `feat(env-dispatch): persist per-agent envs + AReaL client`.

```

Full source: openspec/changes/sub-project-f-per-agent-env-and-checkpointing/tasks.md
