# Comet Design Handoff

- Change: multica-pi-diagnosis-agent
- Phase: design
- Mode: compact
- Context hash: 13e8460723ed0fd4e17e08dafcce38ac8e4f03f5c0e7c4c83b890bbbdeefc580

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/multica-pi-diagnosis-agent/proposal.md

- Source: openspec/changes/multica-pi-diagnosis-agent/proposal.md
- Lines: 1-80
- SHA256: b08e3bc5c05e7bdac6aaba3ef9e62a3bda3ce1a4a6473511e7e15f84ebf4878b

```md
## Why

Per-step (process) reward — crediting each LLM turn for its contribution to completing the
task — is today computed in **AReaL** by a flat LLM judge: `core/judge_prompt.py` feeds the
entire episode + gold answer to one model call and parses per-turn scores, wired through
`enable_judge_process_reward` in `core/customized_grouped_workflow.py`. That judge sees only
the **flattened** episode; it cannot inspect the **supernode / segment structure** of the
multi-agent collaboration (delegation / completion / mention edges, per-segment turn ranges)
or the raw inter-agent conversation, both of which live in **Multica** (`interaction_dag_*`).

This change moves per-step process reward into Multica as a **diagnosis agent**: a tool-using
**Pi** agent that fires after a collaborative task completes, views the full segment DAG
(= the supernode-granularity collaboration flow), calls tools to fetch each segment's LLM
messages, and emits a reward per LLM output. AReaL consumes those per-step rewards and its
flat judge path is removed. This **supersedes the "judge in areal" decision** recorded in
change `multica-v2-segment-dag-training` (process reward now lives in Multica, not AReaL).

## What Changes

- **New Multica diagnosis agent (Pi subprocess)**: a post-task Pi agent following the
  `evolution_review_provider.go` `NewAgentEvolutionReviewer` execution pattern — given the
  segment DAG + a tool set at collaborative-task completion, it iterates over
  supernodes/segments, fetches their LLM messages, and returns structured per-LLM-output
  rewards (each = that turn's contribution to completing the task).
- **Tool surface over the interaction DAG**: tools query `interaction_dag_segment` /
  `interaction_dag_edge` (the segment DAG = supernode flow) and the LLM messages within each
  segment, plus task goal / gold context, so each reward is grounded in the actual
  collaboration rather than a flattened transcript.
- **Trigger at collaborative-task completion**: the diagnosis agent fires when a multi-agent
  collaborative task reaches completion — wired into the `task.go` completion / terminal path,
  gated by a flag, composing with the existing `s.Training` / `INTERACTION_DAG_ENABLED` gates.
  Distinct from the per-agent critic terminal (it views the *whole* project's segment DAG).
- **Per-step reward handoff to AReaL**: extend the close-hook reward delivery (today
  `SetReward(proxy_key, scalar)`) to carry per-step process rewards keyed so AReaL attaches
  each to the right `SuperNode` / turn. **BREAKING** to the areal-side reward source.
- **Remove the AReaL flat judge**: delete the `enable_judge_process_reward` path in
  `core/customized_grouped_workflow.py` (the `diagnose_episode` + `judge_prompt` + `beta`
  wiring) and `core/judge_prompt.py`; remove the `enable_judge_process_reward` /
  `judge_process_reward_beta` / `judge_model_name` / `judge_max_concurrency` config flags.
  AReaL instead consumes Multica-produced per-step rewards.
- **Non-removal**: `distilling/diagnose_provider.py` (`ExternalDiagnoseProvider` /
  `diagnose_episode`) stays if still used by the critic path (`agents/critic_score.py`) — only
  the *judge* usage is removed, not the shared provider.

## Capabilities

### New Capabilities

- `diagnosis-process-reward`: a Multica-side Pi diagnosis agent that fires at collaborative-task
  completion, views the segment (supernode) DAG, fetches per-segment LLM messages via tools, and
  emits a per-LLM-output process reward (contribution to task completion), handed to AReaL as
  the process-reward signal.

### Modified Capabilities

- `critic-driven-training-signal`: the close-hook reward delivery is generalized from a single
  scalar to also carry per-step process rewards produced by the diagnosis agent. Exact
  coexistence with the scalar critic outcome reward (separate parallel path vs. diagnosis
  subsumes the critic) is decision **D1** in `design.md`; the spec delta is finalized there.

## Impact

- **Multica server (Go, primary)**: new diagnosis-agent service package (Pi runner + diagnosis
  system prompt + structured per-step reward output, mirroring
  `internal/service/evolution_review_provider.go`); tool handlers over
  `internal/service/interaction_dag.go` (+ the LLM-message store); trigger wiring in
  `internal/service/task.go` (`CompleteTask` / terminal path) and `internal/service/training.go`
  (close hook); `arealrl` client reward-delivery extension (`SetReward` scalar -> per-step).
- **AReaL (Python, secondary)**: remove `enable_judge_process_reward` wiring in
  `customized_areal/tree_search/core/customized_grouped_workflow.py` (≈L747–787, 892–912, 1167,
  1252) and `customized_areal/tree_search/core/judge_prompt.py`; remove judge config flags in
  `customized_areal/tree_search/config.py` (≈L151–178) and
  `customized_areal/tree_search/distilling/config.py`; consume Multica per-step rewards and map
  them onto `SuperNode` / turn (`process_reward`).
- **Contract / breaking**: the process-reward source moves areal -> multica; the
  reward-delivery API gains a per-step shape. Supersedes "judge in areal" (change
  `multica-v2-segment-dag-training`).
- **Out of scope**: outcome reward / critic scalar mechanism (unless D1 subsumes it); tree
  search / branching; full sandbox snapshot/fork; the shared `ExternalDiagnoseProvider` if the
  critic still uses it.
```

## openspec/changes/multica-pi-diagnosis-agent/design.md

- Source: openspec/changes/multica-pi-diagnosis-agent/design.md
- Lines: 1-200
- SHA256: 2651f95124b035a74f3a569af739ddb8d645871162c48249c9d1656e4f074773

[TRUNCATED]

```md
## Context

Per-step process reward today lives in **AReaL** as a flat LLM judge: `core/judge_prompt.py`
(`build_judge_instruction` / `parse_turn_scores`) builds a prompt over the *entire* episode +
gold answer, `distilling/diagnose_provider.py` (`ExternalDiagnoseProvider.diagnose_episode`)
makes one OpenAI-compatible call, and `core/customized_grouped_workflow.py` (≈L747–787,
892–912, 1167, 1252) parses per-turn scores and blends them with `judge_process_reward_beta`
into a dense `process_reward` `r_t`. Config: `enable_judge_process_reward` etc. in
`config.py` ≈L151–178 and `distilling/config.py`.

The judge's blind spot: it consumes the **flattened** episode. The **supernode / segment
structure** - the actual multi-agent collaboration topology - is assembled in AReaL from
**Multica segment specs** (`SuperNodeAssembler`; 1 Multica segment ≈ 1 `SuperNode`), and the
raw inter-agent conversation + per-segment LLM messages live in **Multica**
(`interaction_dag_segment` / `interaction_dag_edge` tables in
`internal/service/interaction_dag.go`). A flat judge cannot iteratively inspect that
structure or the raw messages.

Two Multica precedents make an agent-based rewarder natural:

1. `internal/service/evolution_review_provider.go` already launches a **Pi agent** as a
   reviewer (`NewAgentEvolutionReviewer` / `EvolutionAgentReviewConfig{Provider,
   ExecutablePath, Model, Timeout, Backend}`) returning structured JSON - the execution
   template for the diagnosis agent.
2. `internal/service/training.go` already has a **critic-agent reward path**
   (`critic_agent_id`; "Replaces D's close hook", ≈L438–490): an agent runs at the trained
   task's terminal, its result is parsed for `{"reward": <float>}`, and
   `arealSessionCloser.SetReward(proxy_key, reward)` + `EndSession` fire on the critic's
   terminal. The diagnosis agent is the **per-step / process-reward** analogue of this
   scalar critic.

`pi` = the Pi coding agent (`pi.dev`), a Multica runtime launched as `pi --provider areal
--model areal-default` (`training.go`). `training.go`'s `arealProxyConfig` carries the
per-session `APIKey`/`BaseURL`/`SessionID` the close hook reuses.

Canonical capability spec (to be written in the specs phase):
`openspec/changes/multica-pi-diagnosis-agent/specs/diagnosis-process-reward/spec.md`.

## Goals / Non-Goals

**Goals**:

- A Multica Pi diagnosis agent that fires at collaborative-task completion, views the segment
  (supernode) DAG, and fetches per-segment LLM messages via tools.
- Emits a **per-LLM-output** process reward (each = contribution to completing the task).
- Deliver those per-step rewards to AReaL keyed to `SuperNode` / turn, replacing the flat
  areal judge as the process-reward source.
- Remove the areal judge path (`enable_judge_process_reward` wiring + `judge_prompt.py` +
  flags).

**Non-Goals**:

- Outcome reward / the scalar critic (`critic-driven-training-signal`) - kept as-is unless D1
  chooses to subsume it.
- Tree search / branching; full sandbox snapshot/fork.
- Removing the shared `ExternalDiagnoseProvider` / `diagnose_episode` if the critic
  (`critic_score.py`) still uses it.
- Training the diagnosis agent itself (it is a fixed judge, not a training target - mirrors
  the critic).

## Decisions

### D1 - Diagnosis agent is a NEW parallel path, not an extension of the critic

The critic (`critic-driven-training-signal`) is a **squad member** auto-spawned at the
**trained agent's** terminal, receiving only that agent's output and producing a **scalar**
outcome reward. The diagnosis agent differs on every axis: it is **not** a squad member, it
fires at **collaborative-task / project** completion (views the *whole* segment DAG across
all agents), it has **tool access** to raw per-segment messages, and it produces
**per-step** rewards. Extending the critic to emit per-step rewards would force tool access
+ whole-DAG view onto a squad-member agent that by spec receives one agent's output.

Decision: diagnosis agent is a **new, parallel** post-task path. The scalar critic stays for
outcome reward; the diagnosis agent supplies process reward. They coexist at the close hook.
Alternative considered: diagnosis subsumes the critic (one agent emits both scalar + per-step)
- rejected for now (different trigger scope and inputs); revisit if D6's delivery merge makes
  it natural.

### D2 - Execution model: tool-using Pi agent subprocess (confirmed)

```

Full source: openspec/changes/multica-pi-diagnosis-agent/design.md

## openspec/changes/multica-pi-diagnosis-agent/tasks.md

- Source: openspec/changes/multica-pi-diagnosis-agent/tasks.md
- Lines: 1-135
- SHA256: 2cd0ad3996a8f0e7e3f3a8c4fbf2c89557fd8851504e6f7ff66c781c7e008cf1

[TRUNCATED]

```md
## 1. Diagnosis Pi-agent runner + structured output (Multica, TDD)

**Files:** new `server/internal/service/diagnosis_agent.go` (+ `_test.go`); mirrors
`server/internal/service/evolution_review_provider.go`.

- [ ] 1.1 Failing tests: `NewDiagnosisAgentRunner(DiagnosisAgentConfig{Provider,
  ExecutablePath, Model, Timeout, Backend})` launches a Pi agent with a diagnosis system
  prompt and returns structured per-step rewards (one score per LLM output).
- [ ] 1.2 Failing tests: the diagnosis system prompt enforces score-in-`[0, score_max]`
  per LLM output and structured output (parsing mirrors `judge_prompt.py`'s discipline);
  unparseable output yields an empty reward set + error, not fabricated scores.
- [ ] 1.3 Failing tests: timeout / non-zero exit / missing rewards are logged and surfaced
  as a soft failure (never panics); `Backend` is injectable for tests.
- [ ] 1.4 Implement the runner + prompt + `parse_step_rewards` parser.
- [ ] 1.5 Commit: `feat(diagnosis-agent): Pi-agent runner + per-step reward parsing`.

## 2. Tool surface + per-segment turn-range capture (Multica, TDD)

**Files:** new `server/internal/service/diagnosis_tools.go` (+ `_test.go`);
`server/internal/service/interaction_dag.go` (`CloseSegmentForEvent`); new migration;
reads `task_message` (migration 026).

- [ ] 2.1 Failing tests: `get_interaction_dag(project_id)` returns segments + edges
  (`segment_id`, `agent_run_id`, `closing_event`, `closing_event_target_segment`, edge
  `type`) - the supernode-granularity flow; read-only.
- [ ] 2.2 Failing tests: `get_segment_messages(segment_id)` slices `task_message` by
  `(task_id, seq ∈ [start_seq, end_seq])`, respecting a per-call byte/turn budget (mirror
  `evolution_review_provider.go`'s `maxEvolutionReview*Bytes` caps).
- [ ] 2.3 Failing tests: `get_task_context(task_id)` returns task goal / gold context used
  to ground "contribution to completing the task".
- [ ] 2.4 Failing tests: tools reject writes (read-only) and enforce project/workspace
  scoping (no cross-workspace segment access).
- [ ] 2.5 Failing tests: `CloseSegmentForEvent` captures `start_seq`/`end_seq` from the
  exported trajectory and stores them on the `interaction_dag_segment` row; a leaf segment
  covers its agent run's full turn range. (Resolves Q3.)
- [ ] 2.6 Implement the tool handlers; extend `CloseSegmentForEvent` to record the turn
  range; add the `start_seq`/`end_seq` migration.
- [ ] 2.7 Commit: `feat(diagnosis-agent): read-only tools + per-segment turn-range capture`.

## 3. Trigger at collaborative-task completion (Multica, TDD)

**Files:** `server/internal/service/task.go` (`CompleteTask` / `captureTaskCompleted` /
terminal path, ≈L1357/182/1567), `server/internal/service/training.go` (close-hook
ordering), tests.

- [ ] 3.1 Failing tests: at collaborative-task (root/project) terminal the diagnosis agent
  fires with the project's segment DAG, **before** `/dag` returns `200` done.
- [ ] 3.2 Failing tests: trigger is gated by `DIAGNOSIS_AGENT_ENABLED` AND `s.Training` AND
  `INTERACTION_DAG_ENABLED`; a non-trained / non-recorded task runs no diagnosis.
- [ ] 3.3 Failing tests: a diagnosis soft-failure (timeout/parse error) is logged and does
  NOT block task completion or the `/dag` done transition (best-effort, sparse-reward
  fallback).
- [ ] 3.4 Failing tests: the diagnosis agent views the *whole* project segment DAG (all
  agents), distinct from the per-agent critic terminal.
- [ ] 3.5 Implement the trigger wiring behind the flag.
- [ ] 3.6 Commit: `feat(diagnosis-agent): trigger at collaborative-task completion`.

## 4. Project-scoped reward delivery via AssembledDag (Multica, TDD)

**Files:** new migration (`interaction_dag_step_reward` table);
`server/internal/service/interaction_dag.go` (`AssembleAssembledDag`);
`server/internal/handler/env_dispatch.go` (`/dag`); areal `AssembledDag`/`SegmentSpec`
(`customized_areal/tree_search/agents/multica_dag_client.py`); tests.

- [ ] 4.1 Failing tests: the diagnosis agent's per-step rewards are written to
  `interaction_dag_step_reward(segment_id, seq, score, rationale)`, keyed by
  `(segment_id, seq)`.
- [ ] 4.2 Failing tests: `AssembleAssembledDag(project_id)` includes a `step_rewards[]`
  structure (one entry per scored LLM output) alongside segments + edges.
- [ ] 4.3 Failing tests: `GET /api/v1/env-dispatch/{project}/dag` serves `step_rewards[]`
  on `200` done; when diagnosis did not run / soft-failed, `step_rewards[]` is empty (absent
  rewards, not zero-filled).
- [ ] 4.4 Failing tests: `/dag` stays `202` in-progress while diagnosis runs, then `200`
  done with rewards (bounded by the diagnosis timeout); soft-failure -> `200` done + empty
  `step_rewards[]`.
- [ ] 4.5 Implement the step-reward table + `AssembleAssembledDag` extension + `/dag`
  serving; extend areal `AssembledDag`/`SegmentSpec` to carry `step_rewards[]` (refines
  in-flight `v2-segment-dag-recording` - reconcile on apply).
- [ ] 4.6 Commit: `feat(diagnosis-agent): project-scoped per-step reward delivery via AssembledDag`.

```

Full source: openspec/changes/multica-pi-diagnosis-agent/tasks.md

## openspec/changes/multica-pi-diagnosis-agent/specs/diagnosis-process-reward/spec.md

- Source: openspec/changes/multica-pi-diagnosis-agent/specs/diagnosis-process-reward/spec.md
- Lines: 1-134
- SHA256: 8d87bef77baa417cef061c029bed6f5b35a5dc78768f0b3ff55855bc8a9f0604

[TRUNCATED]

```md
# diagnosis-process-reward

## ADDED Requirements

### Requirement: Diagnosis agent trigger and gating

Multica SHALL run a Pi-based diagnosis agent when a multi-agent collaborative task reaches
completion (the root/project task terminal), gated by `DIAGNOSIS_AGENT_ENABLED` AND
`s.Training` AND `INTERACTION_DAG_ENABLED`. The diagnosis agent SHALL fire before the
`/dag` endpoint returns a `200` done status for the project, so per-step rewards are present on
the first done-poll. The diagnosis agent views the project's entire segment DAG (all agents),
distinct from the per-agent critic terminal. Non-trained rollouts, or projects with the flag
off, SHALL NOT run the diagnosis agent and SHALL incur no diagnosis overhead.

#### Scenario: Diagnosis fires at project completion before /dag done

- **WHEN** a trained collaborative task's root task reaches a terminal state AND
  `DIAGNOSIS_AGENT_ENABLED` is on AND `INTERACTION_DAG_ENABLED` is on
- **THEN** the diagnosis agent runs over the project's segment DAG and the `/dag` endpoint
  does not return `200` done until the diagnosis agent has finished or its timeout elapsed

#### Scenario: Gating off - no diagnosis

- **WHEN** `DIAGNOSIS_AGENT_ENABLED` is off, or the project is not a trained rollout, or
  `INTERACTION_DAG_ENABLED` is off
- **THEN** no diagnosis agent runs and no per-step rewards are produced

### Requirement: Diagnosis agent read-only tool access

The diagnosis agent SHALL be a tool-using Pi agent subprocess (following the
`evolution_review_provider.go` `NewAgentEvolutionReviewer` execution pattern) with read-only
tools over Multica storage: `get_interaction_dag(project_id)` returning segments + edges (the
supernode-granularity collaboration flow), `get_segment_messages(segment_id)` returning the LLM
messages/turns within one segment, and `get_task_context(task_id)` returning task goal/gold
context. Tools SHALL enforce workspace/project scoping (no cross-workspace reads) and per-call
byte/turn budgets. Tools SHALL reject writes.

#### Scenario: Agent fetches a segment's messages

- **WHEN** the diagnosis agent calls `get_segment_messages(segment_id)` for a segment in the
  project it is diagnosing
- **THEN** it receives the LLM messages/turns for that segment sliced by the segment's
  `start_seq`..`end_seq` range, scoped to the project's workspace

#### Scenario: Cross-workspace tool access refused

- **WHEN** a tool is invoked for a segment outside the diagnosing project's workspace
- **THEN** the tool returns no data and the access is refused

### Requirement: Per-segment turn-range capture

`CloseSegmentForEvent` SHALL capture the closed segment's turn range (`start_seq`, `end_seq`)
from the exported trajectory and store it on the `interaction_dag_segment` row, so
`get_segment_messages` can slice `task_message` precisely by `(task_id, seq)`. A leaf segment
SHALL cover its agent run's full turn range.

#### Scenario: Segment records its turn range at close

- **WHEN** a communication event closes a segment during a trained rollout
- **THEN** the segment row records `start_seq` and `end_seq` corresponding to the turns in that
  segment's exported trajectory

### Requirement: Per-LLM-output reward output

The diagnosis agent SHALL emit one reward per LLM output (turn), scored as an integer in
`[0, score_max]` reflecting that turn's contribution to completing the task, in a structured
output parsed by Multica (mirroring `judge_prompt.py`'s score discipline). Each reward SHALL be
keyed by `(segment_id, seq)` so AReal can attach it to the corresponding `SuperNode`/`Node`.
Unparseable or partial output SHALL yield no rewards for the affected turns (not fabricated
scores).

#### Scenario: Structured per-step rewards parsed

- **WHEN** the diagnosis agent returns structured output with per-turn scores
- **THEN** Multica parses one `(segment_id, seq, score)` reward per scored LLM output and
  writes each to `interaction_dag_step_reward`

#### Scenario: Unparseable output yields no rewards

- **WHEN** the diagnosis agent's output is unparseable or empty
```

Full source: openspec/changes/multica-pi-diagnosis-agent/specs/diagnosis-process-reward/spec.md

