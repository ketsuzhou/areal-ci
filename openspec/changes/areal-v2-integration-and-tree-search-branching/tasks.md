# Tasks - areal-v2-integration-and-tree-search-branching

Phased. Each task lists primary files. Check off on completion. Depends on
`multica-v2-segment-dag-training` (AssembledDag / close_segment / session_to_agent_run)
having landed; Sub-project F (checkpoint-fork) for branch primitives.

## Phase 1 - Multi-agent env-dispatch base workflow

- [ ] 1.1 Define `MultiAgentEnvDispatchWorkflow(RolloutWorkflow)` skeleton
      (`customized_areal/tree_search/agents/multi_agent_env_dispatch.py`): `__init__`
      (gateway_addr, admin_api_key, multica_dag_client, group_size, timeout, discount,
      export_style); `arun_episode(engine, data)` returning an `AssembledDag`.
- [ ] 1.2 `create_env_dispatch` call: drive Multica to create the N-agent squad for the
      task; obtain N `agent_run_id`s + `env_id`/`project_id` per rollout. Reuse the
      `_MulticaClient` protocol shape from `self_play_runner.py:37`.
- [ ] 1.3 `start_session(group_size=N)` against the v2 gateway
      (`/rl/start_session`, `controller/workflow.py:67` `_start_session` is the reference
      client): mint N sessions, receive N x `{session_id, session_api_key}`. Confirm
      Multica records `session_to_agent_run` (AReaL does not).
- [ ] 1.4 Credential distribution: return the N `api_key`s to Multica (or expose via the
      env-dispatch response) so each agent runs `chat/completions` with its own key. Define
      the AReaL->Multica handoff contract.
- [ ] 1.5 Ready-trajectory synchronization: await all N sessions' ready trajectories for
      one task (each session callbacks `/callback/online_ready` on the controller; the base
      workflow correlates by `group_id` / `session_to_agent_run`) before assembling.
- [ ] 1.6 `AssembledDag` polling: poll the env-dispatch endpoint (202 in-progress / 200 +
      `AssembledDag` done) via `multica_dag_client`; return the `AssembledDag`.
- [ ] 1.7 Export + cleanup: per-session `/export_trajectories` (`remove_session=True`)
      for tensor refs after the DAG is assembled; `DELETE /data/clear` + `remove_session`
      post-training (v2-segment-dag lifecycle).

## Phase 2 - `TreeSearchGroupedRolloutWorkflow` multi-agent wiring (Approach A)

- [ ] 2.1 Generalize `_result_to_nodes` (`customized_grouped_workflow.py:929`): branch on
      result type. `AssembledDag` path -> per-segment turn-Nodes via
      `interactions_dict_to_nodes`, one `episode_id` per `agent_run_id` (not per group).
      Single-agent dict path unchanged.
- [ ] 2.2 Replace `_wrap_leaf_super` (`:91`) with a multi-agent `SuperNode` builder: N
      agent-runs (`agent_id`, `issue_id` populated from `AssembledDag` segments) + typed
      edges (`DELEGATION`/`MENTION`/`COMPLETION`/`BRANCH`). Leaf SuperNode = degenerate
      N=1 case (preserve single-agent tests).
- [ ] 2.3 Activate `multica_dag_client` hook (`:682`/`:852`): wire it into the
      `MultiAgentEnvDispatchWorkflow` constructed as `self.workflow` for the multica path.
- [ ] 2.4 `_finalize_episode` (`:1754`): per-agent-run credit assignment at fan-in joins
      (decision 8 of the master design); aggregate multi-agent SuperNodes. Verify
      `TreeAdvantageComputer` consumes per-node credit across the DAG.
- [ ] 2.5 Wire `MultiAgentEnvDispatchWorkflow` as `self.workflow` in the grouped workflow
      constructor for the v2/multica config path; keep `OpenAIProxyWorkflow`/
      `InferenceServiceWorkflow` for non-multica paths.
- [ ] 2.6 `group_size=M` verification: confirm `asyncio.gather` over M
      `_run_fresh_episode` (`:1496`) still produces M multi-agent SuperNodes (one per
      squad rollout); cached-episode path (`load_untrained_episodes`) handles multi-agent
      SuperNodes.

## Phase 3 - Tree-search branching

- [ ] 3.1 `BRANCH` edge type: add to `EdgeType` (`execution_dag.py`) and the `AssembledDag`
      edge contract; carry provenance (`branch_from_segment_id`,
      `branch_from_checkpoint_id`).
- [ ] 3.2 Branch-from-segment-checkpoint: when `select_branch_candidate`
      (`customized_grouped_workflow.py:233`) returns a node, open a new
      `start_session(group_size=k)` for the branched agent(s) from the closed segment's
      checkpoint (Sub-project F fork primitive); run; close new segments.
- [ ] 3.3 `BRANCH` edge emission: Multica records `BRANCH` edges linking branched segments
      to the parent segment in the `AssembledDag`; `select_branch_candidate` /
      `choose_sample_source` (SCRATCH/BRANCH/MIXED, `:195`) extended to the multi-agent
      setting.
- [ ] 3.4 Branch advantage backup: define the backup rule across `BRANCH` edges (R1);
      extend `backup.py` structural backup to distribute reward along `BRANCH` provenance.
- [ ] 3.5 `max_group_size` bound: ensure total branch count per query <=
      `max_group_size - initial_group_size` (`:654`); consecutive-failure circuit breaker
      (`:28`) applies to multi-agent branches.
- [ ] 3.6 Branch cleanup: extend `_cleanup_branch` to release branched sessions
      (`remove_session`) + forked env/issue subtree (Sub-project F).

## Phase 4 - v2 online training integration

- [ ] 4.1 Online mode wiring: construct `InferenceServiceWorkflow(controller, agent=None,
      ...)` for the v2 online path so `arun_episode` -> `_run_online`
      (`controller/workflow.py:230`) -> `wait_for_online_trajectory`.
- [ ] 4.2 Controller external_mode: launch inference_service with `api_url` set so
      `external_mode` (`controller.py:786`) starts the callback server
      (`_start_online_callback_server`, `:789`); confirm `/callback/online_ready` reaches
      `wait_for_online_trajectory`.
- [ ] 4.3 Staleness admission: confirm the staleness manager (`controller.py:1051`) gates
      `rollout_batch` admission, not `start_session`; verify backpressure via
      `_completed_online_results` (`:920`) when Multica mints faster than the trainer
      admits (R2/R5).
- [ ] 4.4 Cross-step staleness guarantee (R2): confirm branch sessions are minted and
      harvested within one training step; add a guard or test that no ready trajectory
      leaks across `set_version` (`:1013`).
- [ ] 4.5 Session lifecycle: M x N sessions minted via `start_session(group_size=N)`;
      harvested via `/export_trajectories` (`remove_session=True`); `close_segment` per
      segment. No pre-grant / no capacity ratchet (D3).

## Phase 5 - Tests and end-to-end

- [ ] 5.1 `MultiAgentEnvDispatchWorkflow` unit tests: N-session minting,
      `AssembledDag` polling, ready-trajectory synchronization, with a fake Multica
      client + fake v2 gateway (no real inference engine).
- [ ] 5.2 `_result_to_nodes` multi-agent branch tests: `AssembledDag` -> per-agent-run
      Nodes + multi-agent SuperNode with typed edges; leaf N=1 parity with single-agent.
- [ ] 5.3 `BRANCH` edge tests: branch provenance recorded; backup rule distributes
      reward across `BRANCH` edges; `max_group_size` bound enforced.
- [ ] 5.4 Staleness/backpressure test: Multica mints faster than trainer admits ->
      `_completed_online_results` queues; no cross-step leak.
- [ ] 5.5 Concurrency test: M x N parallel sessions on the v2 gateway; router routes by
      `session_key` -> worker; no 429 from `SessionStore` (minting unbounded).
- [ ] 5.6 End-to-end (cloud-only): one multica squad task (N=2), `group_size=2`, SCRATCH;
      then a BRANCH run from a closed segment. Verify trajectory harvest + advantage +
      one weight update. Skip with explanation when multi-node hardware unavailable
      (per `backend/areal/CLAUDE.md`).
- [ ] 5.7 Pre-commit: `pre-commit run --all-files`; `uv run pytest` for touched modules;
      document test coverage + hardware limitations in the MR.
