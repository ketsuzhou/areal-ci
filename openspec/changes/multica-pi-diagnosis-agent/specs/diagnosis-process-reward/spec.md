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
keyed by `(segment_id, seq)` so AReal can attach it to the corresponding
`SuperNode` (per-segment `process_reward`, aggregated from the segment's per-turn
scores; the v2 GAE consumes one reward per segment).
Unparseable or partial output SHALL yield no rewards for the affected turns (not fabricated
scores).

#### Scenario: Structured per-step rewards parsed

- **WHEN** the diagnosis agent returns structured output with per-turn scores
- **THEN** Multica parses one `(segment_id, seq, score)` reward per scored LLM output and
  writes each to `interaction_dag_step_reward`

#### Scenario: Unparseable output yields no rewards

- **WHEN** the diagnosis agent's output is unparseable or empty
- **THEN** no rewards are written for the affected turns and no scores are fabricated

### Requirement: Project-scoped reward delivery via AssembledDag

Per-step rewards SHALL be delivered to AReal project-scoped, attached to the `AssembledDag` and
served by `GET /api/v1/env-dispatch/{project}/dag` as a `step_rewards[]` structure keyed by
`(segment_id, seq)`. This extends the `AssembledDag` contract (previously "no scores, no turn
indices") and refines the in-flight `v2-segment-dag-recording` capability. When the diagnosis
agent did not run or soft-failed, `step_rewards[]` SHALL be empty (absent rewards, not
zero-filled).

#### Scenario: /dag serves step rewards

- **WHEN** AReal polls `/dag` for a project whose diagnosis agent produced per-step rewards
- **THEN** the `200` `AssembledDag` response includes a `step_rewards[]` with one entry per
  scored LLM output, keyed by `(segment_id, seq)`

#### Scenario: No diagnosis - empty step_rewards

- **WHEN** AReal polls `/dag` for a project where the diagnosis agent did not run or soft-failed
- **THEN** the `AssembledDag` response carries an empty `step_rewards[]` and no fabricated
  rewards

### Requirement: Soft-failure does not block completion

A diagnosis soft-failure (timeout, non-zero exit, parse error, or empty output) SHALL be logged
and SHALL NOT block task completion or the `/dag` done transition. On soft-failure, `/dag`
SHALL return `200` done with empty `step_rewards[]`, and AReal SHALL fall back to sparse/zero
`process_reward` (matching `enable_judge_process_reward=False`).

#### Scenario: Diagnosis timeout - sparse fallback

- **WHEN** the diagnosis agent exceeds its timeout
- **THEN** the failure is logged, `/dag` returns `200` done with empty `step_rewards[]`, and
  AReal uses sparse `process_reward`

### Requirement: Coexistence with the scalar critic outcome reward

The diagnosis agent's per-step process rewards (project-scoped, via `AssembledDag`) and the
critic's scalar outcome reward (session-scoped, via the close-hook `SetReward`) SHALL coexist
as two independent reward channels. The diagnosis agent SHALL NOT replace or modify the
`critic-driven-training-signal` path; a project MAY have either, both, or neither configured.

#### Scenario: Both diagnosis and critic configured

- **WHEN** a project has `DIAGNOSIS_AGENT_ENABLED` on and a `critic_agent_id` set
- **THEN** AReal receives per-step process rewards via `AssembledDag step_rewards[]` AND the
  scalar outcome reward via the close-hook `SetReward`, independently

#### Scenario: Diagnosis without critic

- **WHEN** a project has `DIAGNOSIS_AGENT_ENABLED` on and no `critic_agent_id`
- **THEN** AReal receives per-step process rewards via `AssembledDag step_rewards[]` and the
  close hook uses `TRAINING_DEFAULT_REWARD` for the outcome reward, unchanged
