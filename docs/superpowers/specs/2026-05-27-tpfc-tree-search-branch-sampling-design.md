# TPFC Tree Search Branch Sampling

## Problem

TPFC tree search currently samples fresh episodes only from the original dataset prompt. Leagent already records entropy metadata on assistant messages for entropy-aware TPFC runs, including `entropy_stats`, `need_branch`, and `branch_sandbox_id`. The tree search workflow does not yet consume that metadata, so high-entropy intermediate decisions cannot be reused as branch points.

## Goal

Add tree-search sampling modes that can start new TPFC episodes either from scratch or from a high-entropy intermediate node. A branch episode must resume from the selected node's conversation prefix and sandbox state, then insert the new episode's nodes back into the tree.

## Sampling Modes

`TreeSearchGroupedRolloutWorkflow` gets a `sample_source` enum-like parameter:

- `scratch`: current behavior. Every needed fresh episode calls the wrapped workflow with the original data.
- `branch`: sample from an existing high-entropy node when one is available. If no branch candidate exists, fall back to scratch and log the reason.
- `mixed`: for each needed fresh episode, branch with probability `branch_probability` when a candidate exists; otherwise sample from scratch. Default `branch_probability` is `0.5`.

The mode is resolved per needed fresh episode, not once per query, so a single batch may contain both scratch and branch episodes in `mixed` mode.

## Tree Node Metadata

Extend `customized_areal.tree_search.mcts_tree_store.Node` with:

- `task_id: str = ""`: Leagent task that produced this episode.
- `entropy_stats: dict[str, Any] | None = None`: copied from the assistant message metadata for this turn.
- `need_branch: bool = False`: copied from message metadata.
- `branch_sandbox_id: str | None = None`: sandbox id saved by Leagent for branchable high-entropy turns.

`interactions_dict_to_nodes()` still creates token/tree nodes from AReaL interactions. After `run_backend()` returns task metadata, the workflow annotates nodes by matching turn order to assistant messages with entropy metadata. The last node in each episode also carries the episode-level `task_id` for traceability.

## TPFC Run Result Contract

Change `customized_areal.tpfc.backend_run.run_backend()` to return the same message and answer data plus task metadata. Return a `BackendRunResult` dataclass with tuple-compatible iteration so existing unpacking can be migrated safely:

- `messages`: parsed messages for reward extraction.
- `raw_messages`: raw DB rows including `message_id`, `role`, `content`, `created_at`, `updated_at`, and `metadata`.
- `task_id`: Leagent task id.
- `final_answer`: extracted answer.
- `log_path`: existing log path.

Its `__iter__` yields `(messages, final_answer, log_path, None)` for temporary compatibility with current callers. New tree-search code reads named attributes only.

Existing callers in `tpfc_agent.py`, `tree_search/core/agent.py`, and TPFC benchmark scripts are updated in the same patch so they read the new contract explicitly.

## Branch Candidate Selection

For a query, candidates are existing tree nodes where:

- `node.query_id == query_id`
- `node.need_branch is True`
- `node.branch_sandbox_id` is non-empty
- `node.task_id` is non-empty

Selection policy is deterministic:

1. Prefer candidates with higher `entropy_stats.max_entropy`.
2. Break ties by insertion order.
3. Do not add visit-count weighting in the first implementation.

## Branch Episode Creation

When a branch candidate is selected:

1. Read raw messages for `candidate.task_id` from Supabase with metadata included.
2. Find the assistant message corresponding to `candidate.turn_idx`.
3. Truncate to the prefix before that assistant turn's response. For example, if step two is high entropy, keep only step one messages.
4. Create a new task for the same account and agent.
5. Insert the truncated messages for the new task, preserving `role`, `content`, `metadata`, and `is_meta` when present.
6. Deep-copy the candidate sandbox state and write a `sandboxes` row binding the new task to the copied sandbox id.
7. Start the agent for the new task with the same model/proxy arguments as scratch sampling.
8. After completion, convert the result to tree nodes and insert them into the same tree store.

The branch task is a normal Leagent task, so the agent runtime resumes from DB messages as context source of truth and uses the task-bound sandbox id during sandbox setup.

## Sandbox Deep Copy

The first implementation keeps sandbox cloning in AReaL's TPFC helper layer, not in the Leagent API. This keeps the training-only branch workflow independent from product API contracts.

The helper should:

- Use the Daytona SDK to snapshot or clone `branch_sandbox_id`.
- Create a new sandbox from that snapshot/clone.
- Insert `sandboxes(sandbox_id, task_id, account_id)` for the new task.
- Never reuse the original `branch_sandbox_id` directly.

If Daytona snapshot/clone is unavailable in the installed SDK or for the current workspace, the branch attempt logs the reason and falls back to scratch for that episode. The fallback preserves training progress without silently sharing mutable sandbox state.

## Cleanup

Scratch runs keep the existing cleanup behavior. Branch runs also clean up their copied sandbox after the run reaches a terminal state. The source branch sandbox is never deleted by a branch run.

If cleanup is skipped because a run may still be active, the copied sandbox id remains discoverable through the new task's `sandboxes` row and existing cleanup scripts can handle it during normal cleanup passes.

## Error Handling

- Missing candidate messages: fall back to scratch.
- Candidate has no `branch_sandbox_id`: exclude it from candidate selection.
- Sandbox clone fails: log and fall back to scratch.
- Truncated-message insertion fails: delete the copied sandbox if one was already created, then fall back to scratch.
- Branch run fails after start: propagate the failure the same way scratch runs do so retry logic can decide whether to retry.

## Testing

Unit tests cover:

- `Node` serialization/checkpoint compatibility with new optional metadata fields.
- Mapping raw assistant message metadata to episode nodes by turn order.
- `sample_source` mode decisions for `scratch`, `branch`, and `mixed`.
- Branch fallback when no candidate or no sandbox clone support exists.
- Branch task construction inserts only messages before the selected high-entropy assistant turn.

Integration-level validation for real Daytona cloning is optional and gated by environment variables because the local test environment may not have Daytona installed or snapshot access.

## Out Of Scope

- Adding a public Leagent branch-task API.
- Changing Leagent entropy computation or thresholds.
- Changing the model/runtime admission path.
- Reusing branch candidates with MCTS UCB or visit-count exploration.
- Sharing the original sandbox instead of cloning it.
