## Why

E closes the RL training loop with a terminal critic reward but explicitly defers richer
branching to F ("per-interaction reward is out of scope — sub-project F or later").
Today the env-dispatch path can only branch from a segment's terminal turn
(SuperNode.env_id frontier), so RL training explores alternatives only at trajectory
endpoints — not at the trainable decisions along the way (delegations, mentions, tool
calls, file changes). F enables branching from ANY trainable decision point by eagerly
checkpointing (env_id + per-issue subtree DB snapshot + full sandbox snapshot) at
structural DAG events and entropy-gated tool calls. This dramatically increases
trajectory diversity for RL training and enables debugging/reproducibility via full
state capture. F also adds per-agent env customization so squad members can run in
isolated sandboxes — cleaner credit assignment and support for heterogeneous base
environments (e.g., coder in a Python image, tester in a Node image).

## What Changes

- **Per-agent / per-group env customization (multica + AReaL client)**: env-dispatch
  contract extended to accept per-agent env specs. Each agent in a squad can be assigned
  its own sandbox from its own base env; the squad shares the multica entity subtree
  (issues / tasks / messages). Different rollout groups can use different base envs.
  AReaL client passes per-agent env specs.
- **Event-triggered state checkpointing (multica)**: eagerly save (env_id + per-issue
  subtree DB snapshot + full sandbox snapshot) at specific trainable decision points.
  Non-trainable events (claiming, cascade cancellations, sweeper timeouts, autopilot
  retry, sandbox lifecycle, RL session management) do NOT trigger checkpoints — they
  carry no policy signal.
- **Checkpoint triggers — always (5 structural decisions)**: delegation
  (`EnqueueTaskForIssue`/`EnqueueTaskForMention`/`EnqueueTaskForSquadLeader`), mention
  (`EnqueueTaskForMention`), completion (`CompleteTask`, LLM-driven path only — not
  sweeper/autopilot), failure (`FailTask`, LLM-driven path only), squad leader briefing
  generation. These are the DAG-growth and terminal-decision points where alternative
  policy choices are most consequential.
- **Checkpoint triggers — entropy-gated (all tool calls, incl. file changes)**: use E's
  captured logprobs to compute entropy per tool-call decision; checkpoint only if
  entropy exceeds a configurable threshold. High entropy = the LLM was uncertain which
  tool to pick = a meaningful branch point. Low entropy = skip (the LLM was confident;
  little training value in branching). Fallback: if logprobs are unavailable (E's
  graceful degradation path), entropy-gated checkpoints are skipped — always-checkpoint
  events still fire.
- **Branch-from-checkpoint (multica + AReaL)**: new operation that restores the sandbox
  snapshot + DB subtree and creates a new `env_dispatch(mode=branch-from-checkpoint)`.
  Extends C's `mode=branch` to accept a checkpoint as the branch source (instead of only
  a live env_id). AReaL client gains a branch-from-checkpoint API.
- **Cost posture**: full eager snapshots, accept the cost (simplest design).
  Copy-on-write / lazy sandbox / per-rollout configurability are explicitly deferred to
  future hardening — F ships the simple eager path first and optimizes only if cost
  becomes a real bottleneck.

## Capabilities

### New Capabilities

- `env-checkpointing`: Event-triggered state checkpointing for RL training on the
  env-dispatch path. Eagerly captures (env_id + per-issue subtree DB snapshot + full
  sandbox snapshot) at trainable decision points: 5 structural events always
  (delegation, mention, completion, failure, squad leader briefing); all tool calls
  entropy-gated via E's logprobs. Enables branching from any checkpoint for trajectory
  diversity and debugging/reproducibility. Non-trainable events (claiming, cascades,
  sweeper, autopilot, sandbox lifecycle, RL session management) are explicitly excluded.

- `per-agent-env-customization`: Per-agent and per-group sandbox assignment in
  env-dispatch. Squad members can run in isolated sandboxes from different base envs
  while sharing the multica entity subtree. Extends the env-dispatch contract with
  per-agent env specs, enabling heterogeneous base environments within a single rollout
  and cleaner per-agent credit assignment.

### Modified Capabilities

- `critic-driven-training-signal` (from E, not yet archived): F's entropy-gated
  checkpointing consumes E's logprobs. If logprobs are unavailable (E's graceful
  degradation path), entropy-gated checkpoints are skipped — always-checkpoint events
  still fire. No contract change to E; F depends on E's logprobs being available for the
  entropy-gated path.

- `training-session-lifecycle` (from D, not yet archived): F's checkpoints coexist with
  D's RL sessions. Checkpoints are taken at trainable events within a session's
  lifetime; branch-from-checkpoint creates a new session (via D's open hook) for the
  branched trajectory. No contract change to D; F layers on top of D's session
  lifecycle.

## Impact

- **multica (Go, primary)**: env-dispatch contract extension (per-agent env specs),
  checkpoint trigger hooks (5 always + entropy-gated tool calls), checkpoint storage
  (new table + sandbox snapshot lifecycle), branch-from-checkpoint operation. Touches
  `internal/handler/env_dispatch.go`
  - `internal/service/env_dispatch.go` (per-agent envs), `internal/service/ task.go`
    (checkpoint triggers on transitions), new `internal/service/env_checkpoint.go`
    (checkpoint lifecycle), new `internal/handler/env_checkpoint.go`
    (branch-from-checkpoint API), new migration (env_checkpoint table + per-agent env
    columns on env_dispatch).
- **AReaL (Python)**: client gains per-agent env specs + branch-from-checkpoint API; new
  entropy computation helper consumes E's logprobs. Touches
  `customized_areal/tree_search/agents/swe_lego_client.py` (client), new
  `customized_areal/tree_search/agents/env_checkpoint.py` (entropy + checkpoint client)
  or extension of existing modules.
- **db_bridge**: existing `/api/v1/env-dispatch` carries per-agent env specs
  transparently in the JSON body; new `/api/v1/env-checkpoint` channel for checkpoint
  lifecycle (create / list / branch-from).
- **Depends on**: Sub-project E (logprobs for entropy gating). E is at build phase, not
  yet implemented. F's entropy-gated checkpoints require E's logprobs; F's
  always-checkpoint events and per-agent env customization are independent of E and
  could ship first if needed.
- **Out of scope**: D6 env-dispatch rewire of the wired loop (stays deferred — F is
  env-dispatch only); tree_search wired loop integration; cost optimization (COW, lazy
  sandbox, per-rollout configurability); per- interaction reward (separate concern — F
  is about branching, not reward signal richness); critic agent training; multi-critic
  ensembles.
