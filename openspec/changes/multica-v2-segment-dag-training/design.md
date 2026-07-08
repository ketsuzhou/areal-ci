## Context

AReaL 2.0 (`areal/v2`) is a service-oriented rewrite of the `areal/experimental/openai`
proxy monolith. v2's `inference_service` captures each assistant turn as an interaction in a
`SessionData` (`active_completions` + `ready_trajectories`); `set_reward` closes
active->ready and `export_trajectories` pops ready trajectories. Today v2 couples trajectory
closure to reward and exports at session granularity - insufficient for Multica, where one
agent-run session must be sliced into multiple communication-bounded segments, each
independently exportable as a trajectory and independently rewardable.

Multica is the RL environment: it drives pi-agent turns through v2 `/chat/completions`,
fires communication events (delegation / mention / completion / squad briefing), and must
hand AReaL the structure + tensor refs to build the interaction DAG (`SuperNode` /
`ExecutionDAG`) and train. The prior `sub-project-g-multica-interaction-dag` design returned
a turn-index `DagResult` (structure only) and deferred the producer; its turn-index slicing
is fragile (off-by-one between Multica's counter and AReaL's `list[Node]`) and carries no
tensor or reward path.

This change establishes the v2 segment-DAG data path: `close_segment` decouples trajectory
boundary from reward; per-segment export returns tensor refs; Multica assembles an
`AssembledDag` of refs + structure; AReaL resolves refs and builds the DAG. Scoring (judge),
V/GAE, and tree search are separate changes.

Locked architectural decisions (from the design exploration) are recorded below; open items
are deferred to task investigation or to changes 2/3.

## Goals / Non-Goals

**Goals:**

- v2 `close_segment`: no-reward active->ready close, returning `trajectory_id`.
- Per-segment export: single trajectory, `remove_session=False`, `RTensor` refs.
- `AssembledDag` contract: segments `{trajectory_id, tensor_ref, closing_event, env}`,
  typed edges, `session_to_agent_run`; no scores, no turn indices, no text.
- AReaL consumer: poll -> `AssembledDag` -> resolve refs -> `SuperNode` / `ExecutionDAG`.
- Minimal training plumbing with placeholder reward (end-to-end path proven).
- Tensor lifecycle: `/data/clear` + `remove_session` after training.

**Non-Goals:**

- Judge / process-reward scoring (change 2).
- Actor-model V critic + GAE (change 2).
- Tree search / branching / fork-from-checkpoint (change 3).
- Full sandbox snapshot/fork in env snapshots (Sub-project F).
- Text crossing to AReaL via export or `AssembledDag` (text stays in AReaL's capture).

## Decisions

### D1 - `close_segment` decouples trajectory boundary from reward

v2 today closes active->ready only via `set_reward` (which requires
`_last_reward_interaction_id`). Add `close_segment` that closes active->ready **without** a
reward, returning `trajectory_id`. One communication-bounded segment = one `close_segment`
call = one trajectory. Reward is assigned later (judge, change 2), not at close.

**Rationale:** per-segment rewards require per-segment trajectories; coupling close to
reward forces reward at boundary time. Decoupling lets Multica close on events and assign
reward independently.

### D2 - Per-segment export returns `RTensor` refs, not bytes

`/export_trajectories` exports a single trajectory by `trajectory_id` with
`remove_session=False`, stores tokens/logprobs on the data_proxy, and returns `RTensor` refs.
AReaL resolves refs via `/data/<shard_id>` / `/data/batch` when building the DAG; frees them
via `/data/clear` after training. The session stays live for further segments.

**Rationale:** keeps tensor bulk on the data_proxy (scalable) rather than in AReaL's
training process; `remove_session=False` lets one session yield many segments.

### D3 - `AssembledDag` carries structure + refs + env, NOT scores / turn-indices / text

Segment = `{segment_id, agent_run_id, issue_id, trajectory_id, tensor_ref, closing_event,
env_snapshot}`. `AssembledDag` = `{segments, edges, session_to_agent_run}` with typed edges
(`DELEGATION` fan-out / `MENTION` peer / `COMPLETION` fan-in). **No `judge_scores`** (AReaL's
judge produces them - change 2), **no `start_turn_idx` / `end_turn_idx`** (`close_segment`
defines the segment; AReaL resolves refs, not slices), **no text** (stays in AReaL's capture;
the v2 judge reads its own capture - change 2).

**Rationale:** `close_segment` + `tensor_ref` make turn indices redundant for tensor
retrieval; keeping scores out preserves the producer/consumer boundary (Multica = env +
structure, AReaL = scoring + training); text isolation holds because AReaL's capture already
stores `messages`.

### D4 - Multica assembles the DAG; AReaL consumes (ref-join, not turn-slice)

Multica records segments + edges at communication events (calling `close_segment` + export
each time) and assembles `AssembledDag` at root-task completion, returning it via the
env-dispatch polling endpoint (`202` / `200`). AReaL's `SuperNodeAssembler` is modified from
turn-index slicing to **ref-resolution**: each segment's `tensor_ref` -> tensors -> SuperNode
payload; edges -> `ExecutionDAG`; `session_to_agent_run` -> run mapping.

**Rationale:** Multica is the driver and owns event boundaries; assembling there avoids
AReaL re-deriving structure. Ref-join removes the turn-index alignment fragility of
sub-project-g.

### D5 - Text stays in AReaL's capture; Multica sends no text

v2 capture stores each interaction's `messages` (text) in `SessionData`. Export returns
refs-only (no text). Multica sends no text to AReaL. The v2 judge (change 2) reads AReaL's
own capture for text.

**Rationale:** text isolation is preserved (Multica never sends text); AReaL's judge has
text from its own capture without a text-relay path.

### D6 - Placeholder reward for the data-path change

Change 1 proves the path end-to-end with a placeholder reward (terminal `set_reward` or
zero). Real process rewards (judge) + V + GAE arrive in change 2.

**Rationale:** keeps the data-path change independently testable (rollout -> segments ->
`AssembledDag` -> DAG build -> train-with-placeholder) without coupling to the judge.

### D7 - Reuse env-dispatch polling return; supersedes sub-project-g's `DagResult`

Reuse `create_env_dispatch` to start the rollout and the
`GET .../env-dispatch/{projectID}/dag` polling endpoint, but return `AssembledDag` (refs +
structure) instead of sub-project-g's turn-index `DagResult`. sub-project-g is superseded
(never started; 0/60 tasks).

## Risks / Trade-offs

- **`close_segment` + export per event adds v2 round-trips** (close + export per segment)
  -> Mitigation: cheap local calls within v2; export is ref-creation, not tensor copy.
- **Ref lifetime vs. training timing** (refs freed before training consumes them)
  -> Mitigation: `/data/clear` only after training; `remove_session=False` until the
  session's last segment is exported.
- **`AssembledDag` acyclicity / dense coverage** (missing or mis-ordered segments)
  -> Mitigation: AReaL's assembler validates topological order and that every session's
  segments cover the run without gaps; a missing segment is detectable.
- **Migration replaces turn-index columns** (sub-project-g's schema not yet built)
  -> Mitigation: sub-project-g never started; the new migration is additive from scratch.
- **Per-segment export requires v2 `SessionData` to hold multiple ready trajectories**
  -> Mitigation: `SessionData.ready_trajectories` is already an `OrderedDict`; export pops by
  `trajectory_id`. Confirm in task investigation.

## Open Questions

- `close_segment` on a session with no active completion (empty segment) - error or empty
  trajectory? Resolve in task investigation.
- Whether `/export_trajectories` already supports per-trajectory + `remove_session=False` or
  needs extension (v2 data_proxy's current export pops ready and revokes the session group).
- Exact v2 data_proxy `/data/<shard_id>`, `/data/batch`, `/data/clear` surface - confirm
  against current v2 (some may need adding).
- `outcome_reward` semantics: does the judge's supernode score serve as outcome, or is a
  separate verifier reward retained? (change 2, but affects the reward plumbing shape).
- Judge trigger timing: completion-only (change 2 default) vs per-event (future config).

## Future (out of this change)

- **Change 2 - Judge + V + GAE**: pi-agent judge in AReaL (reads own capture) -> per-node /
  per-supernode process reward; actor-model critic -> V; GAE; real training. (V via option
  (a): AReaL runs the actor-model critic for V; the pi-agent judge gives process reward;
  GAE combines; completion-time trigger.)
- **Change 3 - Tree search**: `MCTSTreeStore` in AReaL; judge-driven branch selection;
  Multica `env-dispatch mode=branch` fork; multi-branch rollout + training.
