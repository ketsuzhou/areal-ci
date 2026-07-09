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

Launch a real Pi runtime agent via the `NewAgentEvolutionReviewer` pattern
(`EvolutionAgentReviewConfig{Provider, ExecutablePath, Model, Timeout, Backend}`), with a
diagnosis system prompt + tool set; it iterates (list segments -> fetch messages -> score)
and returns structured per-step rewards. Chosen over an OpenAI-compatible single completion
(`OpenAICompatibleEvolutionReviewer` / `ExternalDiagnoseProvider`) because the user requires
**tool access** to iteratively inspect each supernode's messages rather than scoring a
flattened context in one shot. The diagnosis system prompt mirrors `judge_prompt.py`'s
score-in-`[0, score_max]` discipline and the structured-XML-output contract so the same
parsing assumptions apply.

### D3 - Trigger: collaborative-task / project completion, not per-agent terminal

The diagnosis agent fires when the **multi-agent collaborative task** completes (the root /
project task reaching terminal), so it can view the full segment DAG for the project. This is
wired into `task.go`'s completion path (`CompleteTask` / `captureTaskCompleted` /
`EventTaskCompleted`, ≈L1357/182/1567), gated by a new `DIAGNOSIS_AGENT_ENABLED` flag
composing with `s.Training` and `INTERACTION_DAG_ENABLED`. It runs **before** the close hook
delivers rewards, so the per-step rewards are available to the handoff (D6). Distinct from
the critic, which fires at the *trained agent's* terminal.

### D4 - Tool surface (over the interaction DAG + message store)

The Pi agent is given tools that query Multica storage:

- `get_interaction_dag(project_id)` -> segments + edges (the supernode-granularity flow:
  `segment_id`, `agent_run_id`, `closing_event`, `closing_event_target_segment`, edge
  `type` = delegation/completion/mention).
- `get_segment_messages(segment_id)` -> the LLM messages / turns within one segment (the
  per-LLM-output units to score).
- `get_task_context(task_id)` -> task goal / gold answer (so "contribution to completing the
  task" is grounded).
- (optional) `get_inter_agent_conversation(...)` -> raw inter-agent messages, if not already
  covered by segment messages + edges.

Exact tool names + whether messages are directly queryable (vs. reconstructed from a
transcript store) is finalized in the specs/design phase. Tools are read-only.

### D5 - Reward output schema & keying

Output: one reward per LLM output (turn), score in `[0, score_max]` (mirroring
`judge_prompt.py`). Keying must let AReaL attach each reward to the right `SuperNode` /
turn. Candidate keys: `(segment_id, turn_idx)` (segment_id ≈ supernode; turn_idx within the
segment's `start_turn_idx`..`end_turn_idx` range) or a stable `node_id`. Decision leans
`(segment_id, turn_idx)` since segments are Multica-native and `SuperNodeAssembler` already
maps segment -> `SuperNode`; the exact areal-side node identity is confirmed when the
consumer is written (D7). Rewards with no matching areal node are dropped + logged (absence
stays distinguishable - no fabricated defaults, per the project's boundary-value rule).

### D6 - Reward handoff to AReaL: extend the close-hook delivery to per-step

Today `arealSessionCloser.SetReward(ctx, proxyKey string, reward float64)` carries one
scalar. Decision: extend the reward-delivery contract to carry per-step rewards
(e.g. `SetStepRewards(proxyKey, []StepReward{Key, Score})` or a per-step field on the
existing call), implemented in the `arealrl` bridge client + AReaL's bridge endpoint.
Alternatives considered: (a) attach per-step rewards to the `AssembledDag` / segment export
(`ExportTrajectory`) so AReaL reads them at DAG-poll time - attractive (co-located with the
DAG) but couples reward to the read path; (b) a new standalone endpoint. Chosen mechanism
finalized in specs; the close hook remains the trigger point. **This is the spec-level
modification to `critic-driven-training-signal`** (scalar -> scalar-or-per-step delivery).

### D7 - AReaL consumes per-step rewards; flat judge removed

Remove the `enable_judge_process_reward` branch in `customized_grouped_workflow.py`
(≈L747–787, 892–912, 1167, 1252) and `core/judge_prompt.py`; drop the judge config flags.
AReaL instead reads per-step rewards delivered by D6, maps `(segment_id, turn_idx)` ->
`SuperNode` / turn, and writes them as `process_reward` on the node (the same field
`dag_advantage.py` consumes). Fallback when no per-step rewards arrive: sparse/zero process
reward (matching today's `enable_judge_process_reward=False` behavior) - **not** a
re-introduced judge. `ExternalDiagnoseProvider` / `diagnose_episode` is kept if
`critic_score.py` still uses it (verify before removing).

### D8 - Granularity: per-LLM-output reward, supernode(segment)-structured navigation

Restating the user's intent: the agent **navigates/views** at supernode (segment) granularity
and the **reward output** is one per LLM output (turn). This matches the existing judge's
per-turn granularity but adds supernode-structured tool access. If one-reward-per-supernode
was intended instead, D5/D7 keying simplifies - flagged as Open Question Q1.

## Risks / Trade-offs

- [Pi subprocess latency/cost at every task completion] -> Mitigation: gate behind
  `DIAGNOSIS_AGENT_ENABLED`; bound turn budget + timeout (mirror
  `evolution_review_provider.go`'s `Timeout`); best-effort (a diagnosis failure must not
  block task completion - fall back to sparse reward, log).
- [Reward keying drift between Multica segment/turn and areal SuperNode/node] -> Mitigation:
  D5 keys on Multica-native `(segment_id, turn_idx)`; areal maps via the existing
  segment->SuperNode mapping; unmatched rewards dropped + logged, never defaulted.
- [Tool access to raw LLM messages = a new read surface over potentially large transcripts]
  -> Mitigation: read-only tools; per-call byte/turn budgets (mirror
  `evolution_review_provider.go`'s `maxEvolutionReview*Bytes` caps).
- [Removing the areal judge is breaking] -> Mitigation: land Multica delivery + areal
  consumer first, then remove the judge in a follow-up task so there is always a
  process-reward source; keep `ExternalDiagnoseProvider` for the critic.
- [Diagnosis agent reward quality unvalidated] -> Mitigation: E2E compares diagnosis
  per-step rewards against the removed judge's scores on a fixed episode as a sanity check
  (not a correctness gate).

## Open Questions

- **Q1** (granularity): per-LLM-output (assumed, D8) vs. one-reward-per-supernode? Confirm
  with requester.
- **Q2** (delivery mechanism): close-hook `SetStepRewards` (D6) vs. attaching rewards to
  `AssembledDag`/`ExportTrajectory`? Decide in specs phase.
- **Q3** (message store): are per-segment LLM messages directly queryable today, or must
  the tool reconstruct them from a transcript/activity store? Needs a read of the message
  storage path.
- **Q4** (critic coexistence): does the project want diagnosis + scalar critic both feeding
  training (D1 default), or should diagnosis eventually subsume the critic?

## Migration Plan

1. Land Multica diagnosis agent + tools + trigger (gated, off by default).
2. Land per-step reward delivery (D6) + AReaL consumer (D7), reading per-step rewards when
   present, sparse fallback otherwise.
3. Enable `DIAGNOSIS_AGENT_ENABLED` on a test project; verify per-step rewards land on the
   right `SuperNode`/turn.
4. Remove the areal judge path + flags once the diagnosis agent is the confirmed
   process-reward source.
5. Rollback: disable `DIAGNOSIS_AGENT_ENABLED`; areal falls back to sparse process reward
  (judge is removed, so no automatic revert to judging - document this as a one-way step).
