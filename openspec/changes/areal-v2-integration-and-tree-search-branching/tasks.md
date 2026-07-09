# Tasks - areal-v2-integration-and-tree-search-branching

Phased. Each task lists primary files. Check off on completion. Builds on
`multica-v2-segment-dag-training` (AssembledDag / close_segment / get_dag /
SuperNodeAssembler.assemble_from_refs / DataProxyTensorResolver /
EnvDispatchBranchDriver) having landed. **Reuse those components - do not
reimplement.** All phases are F-independent (branching uses
`EnvDispatchBranchDriver` via `create_env_dispatch(mode="branch")`, not
Sub-project F sandbox snapshot/fork).

Test runner: `python3 -m pytest` (NOT `uv run pytest` - broken in this env).
Lint: `uvx ruff check`. No GPU in dev; unit tests CPU-only with fakes.

## Phase 1 - `MultiAgentEnvDispatchWorkflow` orchestrator (reuse existing)

- [ ] 1.1 Create `MultiAgentEnvDispatchWorkflow(RolloutWorkflow)`
      (`customized_areal/tree_search/agents/multi_agent_env_dispatch.py`): thin
      orchestrator. `arun_episode` = `await MulticaEnvDispatchClient.create_env_dispatch`
      (mode="scratch") + `asyncio.to_thread(MulticaDagClient.get_dag)` +
      `asyncio.to_thread(SuperNodeAssembler.assemble_from_refs)` -> `{"assembled_dag",
      "execution_dag"}`. Partial-squad validation + success-path cleanup
      (`DataProxyTensorResolver.clear` + `DataProxySessionRemover.remove`). Export from
      `agents/__init__.py`. Test: N=1 SCRATCH with fakes. AReaL NEVER calls `start_session`.
- [ ] 1.2 N>1 squad + partial-squad drop: 2 rollouts both covered -> returns execution_dag;
      one `agent_run_id` missing from `session_to_agent_run` -> returns `None` (no partial
      `SuperNode`).
- [ ] 1.3 Tensor-ref resolve + cleanup ordering: `resolver.resolve` per segment (via
      `assemble_from_refs`); `resolver.clear` with shard_ids; `session_remover.remove` per
      session; on `DAGError` from `assemble_from_refs`, cleanup is NOT called (success-path
      only, matches `run_segment_dag_training_step`).
- [ ] 1.4 Polling timeout + typed DAG errors: `DagTimeout` -> `None` (reject);
      `DagNotFound`/`DagForbidden` handling decided + asserted.

## Phase 2 - `TreeSearchGroupedRolloutWorkflow` wiring (Approach B: SuperNode-preserving)

- [ ] 2.1 `_result_to_nodes` multica branch (`customized_grouped_workflow.py:929`): result
      carrying `"execution_dag"` -> return its `SuperNode`s (preserve structure; do NOT
      flatten to `Node`s). Single-agent dict path unchanged. Return type widens to
      `list[Node] | list[SuperNode] | None`.
- [ ] 2.2 `_finalize_episode` multica branch (`:1754`): fresh `SuperNode`s -> insert via
      `tree_store.insert_super_batch` (NOT `_wrap_leaf_super`); compute advantages over the
      DAG (`TreeAdvantageComputer` if it handles `SuperNode`s, else `assemble_node_advantages`
      over `topological_order()` - confirm during impl). N=1 multica = leaf `SuperNode`
      (parity). Skip judge/distillation (Change 2).
- [ ] 2.3 Activate `multica_dag_client` hook (`:682`/`:852`) + wire `self.workflow` =
      `MultiAgentEnvDispatchWorkflow` when `multica_dag_enabled` (constructed with the
      dispatch/dag clients + resolver/session_remover/assembler). Non-multica paths
      unchanged.
- [ ] 2.4 `group_size=M` parallel multi-agent rollouts (`:1496`): M parallel `arun_episode` ->
      M `AssembledDag`/`ExecutionDAG` aggregated; cached-episode path handles multi-SuperNode.

## Phase 3 - Tree-search branching (F-independent, via `EnvDispatchBranchDriver`)

- [ ] 3.1 `EdgeType.BRANCH` + provenance (`execution_dag.py`): add `BRANCH="branch"`; add
      optional `branch_from_segment_id`/`branch_from_checkpoint_id` to `Edge`; update
      `to_records`/`from_records` + `_coerce_edge_type`.
- [ ] 3.2 `BRANCH` edge parsing in `AssembledDag` (`supernode_assembler.py`): `EdgeSpec.type`
      `"branch"` -> `EdgeType.BRANCH` + provenance in `assemble_from_refs`.
- [ ] 3.3 `Node.visit_count` (`tree_store.py`) + MCTS backup (`customized_areal/tree_search/dag/backup.py`):
      `branch_backup(parent, *, branch_return, discount, visit_count)` running-mean value
      update. Test: discounted return + multi-branch aggregation.
- [ ] 3.4 Branch execution in `MultiAgentEnvDispatchWorkflow` (via `EnvDispatchBranchDriver`):
      branch mode in `arun_episode` forks via `drive_lane` (`create_env_dispatch(mode="branch")`),
      runs branched squad, harvests branched `AssembledDag` with `branch` edge. SCRATCH path
      unchanged. (Branch *selection* stays in the grouped workflow's `select_branch_candidate`.)
- [ ] 3.5 `max_group_size` bound + circuit breaker (`customized_grouped_workflow.py`): total
      branches per query <= `max_group_size - initial_group_size`; consecutive-failure breaker.
- [ ] 3.6 MCTS backup wiring + branch cleanup (`customized_grouped_workflow.py`): branched
      return -> `branch_backup` to parent checkpoint `Node`; branched sessions removed on cleanup.

## Phase 4 - v2 online training integration (verification)

- [ ] 4.1 Staleness gates `rollout_batch` admission, not `start_session` minting; excess
      `AssembledDag`s hold at Multica's `202`. (Assert/verify the contract.)
- [ ] 4.2 Cross-step staleness (R2): `rollout_batch` awaits all `arun_episode`s (incl. branch
      polling) before `set_version`; no trajectory leaks across steps.
- [ ] 4.3 Session lifecycle: N sessions minted by **Multica** (AReaL never calls
      `start_session`); harvested via `DataProxySessionRemover.remove` (`remove_session=True`);
      `close_segment` per segment. No pre-grant / no capacity ratchet.

## Phase 5 - Tests and end-to-end

- [ ] 5.1 Consolidate Phase 1 unit suite; `python3 -m pytest test_multi_agent_env_dispatch.py -v`.
- [ ] 5.2 Integration: M=2 x N=2 parallel sessions on a fake v2 gateway; router routes by
      `session_key`; no `429` from `SessionStore` (minting unbounded).
- [ ] 5.3 Staleness backpressure: Multica mints faster than trainer admits -> `AssembledDag`s
      hold at `202`; trainer drains in later batches.
- [ ] 5.4 Multi-level branch-tree backup test (extends 3.3/3.6): 2-level branch tree ->
      `branch_backup` aggregates correctly at each checkpoint.
- [ ] 5.5 E2E (cloud-only, hardware-gated): N=2 squad, `group_size=2`, SCRATCH; then a `BRANCH`
      from a closed segment. Verify harvest + advantage + one weight update.
      `pytest.mark.skipif(not GPU, ...)`.
- [ ] 5.6 Pre-commit (`pre-commit run --all-files`) + `python3 -m pytest` for touched modules;
      check off all tasks.md items.

## Test runners / constraints

- AReaL tests run from `backend/areal` with `python3 -m pytest customized_areal/tree_search/tests/`.
- No GPU in dev env; unit/integration tests CPU-only with fakes. E2E is hardware-gated (skip in CI).
- This change integrates the existing v2-segment-dag path + adds branching. Judge / V / GAE
  (Change 2) and the v2 segment-DAG data path itself (Change 1) are NOT modified here.
