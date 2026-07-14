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
