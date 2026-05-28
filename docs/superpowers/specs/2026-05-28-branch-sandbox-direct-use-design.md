# Branch Sandbox Direct Use + Subproc Metadata Propagation

**Date**: 2026-05-28

## Problem

Two issues prevent tree-search branching from working correctly:

1. **Subproc metadata loss**: In `subproc` mode, `TPFCAgent.run()` mutates `data` dict to pass
   `task_id` and `raw_messages` back to the workflow. But `data` is pickled across process
   boundaries, so mutations are lost. The tree search workflow never sees `_backend_run_task_id`
   or `_backend_run_raw_messages`, so `EpisodeRunResult` is never created and
   `annotate_nodes_from_run` is never called. Entropy metadata, `need_branch`, and
   `branch_sandbox_id` never get annotated onto nodes.

2. **Unnecessary sandbox clone**: `build_branch_task` clones `candidate.branch_sandbox_id`
   before binding it to the new task. But the sandbox was already deep-copied by
   `_compute_entropy_metadata` in the agent loop. The clone is redundant and wastes time
   and resources. Additionally, after the branching episode finishes, neither the cloned
   sandbox nor the original `branch_sandbox_id` are cleaned up, causing resource leaks.

## Design

### 1. Structured Return from TPFCAgent

Introduce `TPFCAgentResult` dataclass in `customized_areal/tpfc/tpfc_agent.py`:

```python
@dataclass
class TPFCAgentResult:
    reward: float
    task_id: str = ""
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
```

`TPFCAgent.run()` returns `TPFCAgentResult` instead of mutating `data`:

```python
# Before:
data["_backend_run_task_id"] = run_result.task_id
data["_backend_run_raw_messages"] = run_result.raw_messages
return float(reward)

# After:
return TPFCAgentResult(
    reward=float(reward),
    task_id=run_result.task_id,
    raw_messages=run_result.raw_messages,
)
```

This works in `subproc` mode because `TPFCAgentResult` is a picklable dataclass —
the return value survives the process boundary, unlike `data` dict mutations.

### 2. Backward-Compatible Handling in OpenAIProxyWorkflow

In `OpenAIProxyWorkflow.arun_episode`, after `_run_agent` returns, detect `TPFCAgentResult`
and propagate metadata to `data` in the parent process:

```python
rewards = await self._run_agent(proxy_client.session_api_key, data)

# Handle structured result from agents like TPFCAgent
if isinstance(rewards, TPFCAgentResult):
    data["_backend_run_task_id"] = rewards.task_id
    data["_backend_run_raw_messages"] = rewards.raw_messages
    rewards = rewards.reward  # extract float for reward assignment
```

Then the existing reward-handling logic (`isinstance(rewards, float)` etc.) continues to
work unchanged. The `_with_episode_metadata` function in `tree_search_grouped_workflow.py`
already reads `data["_backend_run_task_id"]` and `data["_backend_run_raw_messages"]`, so
no changes needed there.

For agents that still return `float` or `dict`, behavior is unchanged (backward-compatible).

### 3. Branch Task Preparation — Direct Sandbox Use

Modify `build_branch_task` in `tree_search_grouped_workflow.py` to use
`candidate.branch_sandbox_id` directly instead of cloning:

```python
# Before:
cloned_sandbox_id = await clone_sandbox(candidate.branch_sandbox_id)
if not cloned_sandbox_id:
    return None
branch_task_id = await create_task(...)
await bind_sandbox_to_task(client, sandbox_id=cloned_sandbox_id, ...)
await copy_messages_to_task(client, task_id=branch_task_id, messages=prefix)
# Error cleanup: delete cloned_sandbox_id

# After:
branch_sandbox_id = candidate.branch_sandbox_id
branch_task_id = await create_task(...)
await bind_sandbox_to_task(client, sandbox_id=branch_sandbox_id, ...)
await copy_messages_to_task(client, task_id=branch_task_id, messages=prefix)
return branch_task_id
```

No clone, no error-path sandbox cleanup in `build_branch_task`. The sandbox created by
`_compute_entropy_metadata` in the agent loop is used directly.

### 4. Post-Episode Sandbox Cleanup and Node State Update

After a branching episode finishes in `_run_fresh_episode`, delete the branch sandbox
and update the candidate node state:

```python
if source == SampleSource.BRANCH and candidate is not None:
    branch_task_id = await self._prepare_branch_task(branch_data, candidate)
    if branch_task_id:
        branch_data["task_id"] = branch_task_id
        branch_data["seed_messages_already_inserted"] = True
        result = await self._retry_episode(engine, branch_data, group_idx)
        episode_result = _with_episode_metadata(result, branch_data)
        # Cleanup: delete branch sandbox and mark node as branched
        await self._cleanup_branch(candidate)
        return episode_result
```

`_cleanup_branch` method:

```python
async def _cleanup_branch(self, candidate: Node) -> None:
    """Delete branch sandbox and mark node as branched to prevent re-use."""
    if candidate.branch_sandbox_id:
        try:
            await delete_sandbox(candidate.branch_sandbox_id)
        except Exception:
            logger.warning(
                "Failed to delete branch sandbox_id=%s: ...",
                candidate.branch_sandbox_id,
            )
    candidate.need_branch = False
    candidate.branch_sandbox_id = None
```

This naturally prevents re-branching because `select_branch_candidate` requires both
`need_branch=True` and `branch_sandbox_id` to be set. After cleanup, the node no longer
qualifies as a branch candidate.

## Files Modified

1. `customized_areal/tpfc/tpfc_agent.py` — Add `TPFCAgentResult`, return it from `run()`
2. `areal/experimental/openai/proxy/workflow.py` — Handle `TPFCAgentResult` in `arun_episode`
3. `customized_areal/tree_search/tree_search_grouped_workflow.py` — Modify `build_branch_task`,
   add `_cleanup_branch`, update `_run_fresh_episode`

## Testing

- Unit test for `TPFCAgentResult` pickling (subproc serialization)
- Unit test for `build_branch_task` with direct sandbox use (no clone)
- Unit test for `_cleanup_branch` — sandbox deletion + node state update
- Unit test for `select_branch_candidate` after node is cleaned up (not selected again)
- Integration test for full branch episode flow with subproc mode
