# Design - areal-v2-integration-and-tree-search-branching

## 1. Context

Builds on `multica-v2-segment-dag-training` (v2 per-segment data path: `close_segment`,
`AssembledDag`, `session_to_agent_run`, ref-resolution into `SuperNode`/`ExecutionDAG`).
This change adds (a) multi-agent squad coordination and (b) tree-search branching on the
v2 online training path, plus (c) v2 online integration. Reference design doc:
`docs/superpowers/specs/2026-07-08-multica-v2-segment-dag-design.md`.

## 2. v2 online training path (the substrate)

The v2 `inference_service` controller has two modes (`controller.py:786`):

```
external_mode = (config.api_url is not None)
  offline  : controller drives an agent (_run_offline)
  online   : external agents drive; controller waits for ready trajectories
```

Online mode (`InferenceServiceWorkflow(agent=None)`, `controller/workflow.py:230`):

```
arun_episode -> _run_online -> controller.wait_for_online_trajectory()  [blocks on future]
                                     ^ future resolved by callback server
                                     |
   data_proxy POSTs /callback/online_ready {session_id, trajectory_id}  (controller.py:904)
                                     |
   _export_interactions([session_id], trajectory_id) -> traj dict       (remove_session=True)
   return traj  -> PPO trainer
```

The controller is **trajectory-source-agnostic**: `wait_for_online_trajectory` consumes
any ready trajectory regardless of SCRATCH vs BRANCH origin. This is the key enabler -
branching is "Multica produces more ready trajectories," needing no special AReaL-side
branch harvest path.

Training loop: PPO trainer -> `controller.rollout_batch(N)` -> collect N trajectories ->
`/ppo/actor/compute_advantages` -> `/update_weights` (training_service) ->
`controller.set_version(v+1)`.

## 3. Key decisions

### D1 - Multica owns `/rl/start_session` (online / external-user path)

**Decision**: Multica's agents call `/rl/start_session` on the v2 gateway themselves;
AReaL does **not** call `rl_session.start(agent_run_id=...)` (that line in
`self_play_runner.py:100` is the legacy AReaL-orchestrated pattern, retired for this
path).

**Why**: aligns with the v2-segment-dag contract where Multica owns
`session_to_agent_run` and delivers it in the `AssembledDag`. `StartSessionRequest`
carries only `{task_id, api_key?, group_size}` - no `agent_run_id` - so the binding is
Multica-side by construction.

**Alternative rejected**: AReaL orchestrates (`self_play_runner` pattern). Would require
AReaL to drive the squad and own the binding, inverting the v2 contract and re-introducing
the AReaL<->Multica coupling the v2 track removed.

### D2 - Approach A: base workflow is multi-agent; grouped workflow stays group-level

**Decision**: a new multi-agent `RolloutWorkflow` is the `self.workflow` wrapped by
`TreeSearchGroupedRolloutWorkflow`. One `self.workflow.arun_episode` call = one task =
N agents = N sessions = one `AssembledDag`. The grouped workflow stays group-level:
`asyncio.gather` over M `_run_fresh_episode` calls (shape unchanged, `:1496`), each
returning a multi-agent `AssembledDag`.

**Why**: matches the code's stated intent (`customized_grouped_workflow.py:19` "env-dispatch
runner model"; `:95` "multi-agent coordinator Phase 1b/2 not yet wired"). Keeps multica
squad/edge/env-dispatch coordination out of the grouped workflow.

**Alternative rejected (B)**: grouped workflow calls `self.workflow.arun_episode` N times
per group and assembles the squad itself. Leaks squad/edge/session_to_agent_run
coordination into the grouped workflow; balloons `_wrap_leaf_super`.

### D3 - v2 mint-on-demand; no pre-grant / no capacity ratchet

**Decision**: sessions are minted dynamically via `start_session(group_size=N)`. No
pre-instantiation of M x N proxy workers, no ready-worker queue, no `grant_capacity`
pre-grant.

**Why**: v2 `SessionStore.start_session` (`data_proxy/session.py:423`) neither checks nor
decrements `_capacity` - minting is unbounded at the store. The legacy
`areal/experimental/openai/proxy/` model (the `grant_capacity` ratchet, ready-worker queue,
3600s `_OnlineAgent` block, `end_session` non-recycle) **does not apply to v2**. The v2
gateway just `query_router` -> forward -> mint -> register-in-router (`gateway/app.py:274`).

**Consequence for branching**: a branch = another `start_session(group_size=k)` call,
on demand. No B-branch-budget, no spare cancellation, no deadlock (the Option-A
pre-grant concerns from the legacy analysis are moot on v2).

### D4 - Staleness gates trainer batch admission, not session minting

**Decision**: the staleness manager (`controller.py:1051` `get_capacity` ->
`staleness_manager.get_capacity()`) bounds how many `arun_episode`s the trainer submits
per batch (`max_concurrent_rollouts`, `max_head_offpolicyness`). It is **not** in the
gateway's `start_session` path (`gateway/app.py:274` forwards without a staleness check).

**Consequence**: Multica can mint sessions freely; if it mints faster than the trainer
admits, ready trajectories queue in `_completed_online_results` (`controller.py:920`) and
drain in later batches - natural backpressure. **The staleness boundary is the trainer's
batch admission.** Branch-produced trajectories are stale only if they leak across a
weight update; the batch boundary prevents that.

**Open risk (see §6)**: because Multica's direct `start_session` bypasses the staleness
manager, a branch minted late in a batch could produce data relative to a soon-to-be-old
policy version. Acceptable within a single training step (same policy version); must
confirm branches don't cross step boundaries.

### D5 - `BRANCH` edge type; branch = new session from a closed segment's checkpoint

**Decision**: the `AssembledDag` gains a `BRANCH` edge type carrying provenance
(`branch_from_segment_id`, `branch_from_checkpoint_id`). A branch opens a new
`start_session` for the branched agent(s) from a closed segment's checkpoint (Sub-project
F's checkpoint-fork primitive), runs, closes new segments, and emits `BRANCH` edges
linking them to the parent segment. `max_group_size` bounds total branch count per query.

**Why**: v2-segment-dag's `close_segment` makes each segment its own trajectory; branching
from a closed segment's checkpoint = new trajectory from that point. `BRANCH` edge
preserves tree provenance for advantage backup.

**Alternative rejected**: extend/reuse the parent session. Contradicts v2-segment-dag's
per-segment-trajectory model and loses branch provenance.

## 4. Data flow (end-to-end)

```
PPO trainer (training_service)
  │  rollout_batch(M)  -> submit M arun_episode tasks
  ▼
TreeSearchGroupedRolloutWorkflow.arun_episode                        (customized_grouped_workflow.py:1457)
  │  _arun_episode_fixed: asyncio.gather(M x _run_fresh_episode)     (:1496)
  │     each _run_fresh_episode -> self.workflow.arun_episode         (:1371)
  ▼
MultiAgentEnvDispatchWorkflow.arun_episode  (NEW base workflow, = self.workflow)
  │  1. create_env_dispatch -> Multica creates N-agent squad (N agent_runs)
  │  2. /rl/start_session(group_size=N) -> v2 gateway mints N sessions, returns N x {sid, api_key}
  │     (Multica records session_to_agent_run; AReaL never sees agent_run_id)
  │  3. distribute api_keys; N agents run task (online, collaborative)
  │  4. comm events: close_segment + export per segment (v2-segment-dag)
  │     [branch: new start_session(group_size=k) from closed-segment checkpoint; BRANCH edge]
  │  5. poll env-dispatch endpoint -> AssembledDag{segments, edges(+BRANCH), session_to_agent_run}
  │  return AssembledDag
  ▼
_result_to_nodes(AssembledDag)  (generalized)                        (:929)
  │  per segment: turn-Nodes, one episode_id per agent-run
  │  -> multi-agent SuperNode (N agent-runs + typed edges)           (_wrap_leaf_super -> general builder)
  ▼
_finalize_episode: aggregate multi-agent SuperNodes; per-agent-run advantage/credit
  │
  ▼ trajectories (tokens/logprobs/rewards) via InferenceServiceWorkflow online harvest
PPO trainer: compute_advantages -> update_weights -> set_version(v+1)
```

Harvest side (online): each of the M x N sessions, when its trajectory is ready
(`set_reward` / `close_segment`), callbacks `/callback/online_ready`; the controller
resolves a waiting `arun_episode` (or queues). The multi-agent base workflow awaits all
N sessions' ready trajectories for one task before assembling the `AssembledDag`.

## 5. Multi-agent wiring touchpoints (Approach A)

| Touchpoint | File:line | Change |
|---|---|---|
| `self.workflow.arun_episode` call | `customized_grouped_workflow.py:1371` | returns `AssembledDag` (not interaction dict); call shape unchanged |
| `_result_to_nodes` | `:929` | branch on result type: single-agent dict (unchanged) vs `AssembledDag` (per-segment Nodes, per-agent-run `episode_id`) |
| `_wrap_leaf_super` | `:91` | generalize to multi-agent `SuperNode` builder (N runs + typed edges); leaf = N=1 |
| `multica_dag_client` hook | `:682`/`:852` | activated (used by base workflow to poll `AssembledDag`) |
| `_arun_episode_fixed` gather | `:1496` | shape unchanged (M parallel calls) |
| `_finalize_episode` | `:1754` | per-agent-run credit/advantage (already the `SuperNode`/`ExecutionDAG` model) |

The multi-agent base workflow (new, `customized_areal/tree_search/agents/`) owns:
`create_env_dispatch`, `start_session(group_size=N)`, credential distribution,
`AssembledDag` polling, ready-trajectory synchronization across N sessions.

## 6. Open questions / risks

- **R1 - `BRANCH` edge semantics**: exact shape of branch provenance and how
  advantage backup distributes reward across `BRANCH` edges (vs the existing
  `DELEGATION`/`MENTION`/`COMPLETION` backup). Needs a concrete backup rule.
- **R2 - Staleness on Multica's direct `start_session`**: confirm branches consume
  sessions within the same training step (no cross-weight-update leak). May require a
  trainer-side admission signal to Multica, or accept the batch-boundary guarantee.
- **R3 - `AssembledDag` delta to `v2-segment-dag`**: the `BRANCH` edge extends the
  shipped `AssembledDag` contract. Once `multica-v2-segment-dag-training` archives,
  capture `BRANCH` as a delta to the `v2-segment-dag` spec (not a new capability) - or
  keep it under `v2-multi-agent-tree-search` if v2-segment-dag ships first without it.
- **R4 - Branch checkpoint source**: branch-from-segment-checkpoint depends on
  Sub-project F's checkpoint-fork primitive. If F is not ready, scope branching to
  SCRATCH-only initially and defer BRANCH.
- **R5 - M x N session scaling**: dynamic minting is unbounded at the store; real bound
  is inference-engine (SGLang/vLLM) throughput. Confirm staleness manager's
  `max_concurrent_rollouts` is the operative backpressure, not engine OOM.

## 7. Out of scope

- Judge / process-reward scoring (v2 change 2a); actor-model V critic + GAE (v2 change 2b).
- Full sandbox snapshot/fork internals (Sub-project F).
- Legacy `areal/experimental/openai/proxy/` path (superseded; its capacity model does not
  apply to v2).
- `agent_versions` immutability, RLS, multi-tenancy (unchanged).
