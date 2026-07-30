# Non-training MultiCA diagnosis design

## Goal

Allow a completed non-training environment dispatch to explicitly run the
MultiCA diagnosis agent. The agent must score every eligible turn, persist
those scores in the interaction DAG, and let the AReaL client return the
enriched assembled DAG.

## Scope

This is an opt-in post-completion operation. It does not change training-mode
dispatch semantics, training-session creation, or the existing automatic
diagnosis trigger for a terminal training root task.

## API and lifecycle

MultiCA will expose an authenticated, workspace-scoped diagnosis request for
one environment-dispatch project. The handler will:

1. Verify the caller can access the project and that its root task is terminal.
2. Verify diagnosis and interaction-DAG support are configured; otherwise
   return a clear unavailable response without creating a partial run.
3. Resolve the project's ordered DAG segments and invoke the existing
   on-demand diagnosis runner.
4. Return a run status/report. Repeated requests resume an in-progress run or
   report an already-completed run rather than duplicate rewards.

The existing diagnosis tool server remains the sole writer of `(segment_id,
seq)` step rewards. Its coverage checks ensure every expected turn is scored
before a segment and the run can complete.

## Client behavior

`multica_client.py` will add an explicit debug CLI opt-in (for example,
`--diagnose`). In non-training mode it will:

1. Create the dispatch and wait until the ordinary DAG is terminal.
2. Request on-demand diagnosis for the returned project ID.
3. Poll diagnosis status until completed or failed, with the existing timeout
   and error-redaction conventions.
4. Fetch `/dag` again, validate it as `AssembledDag`, and optionally persist
   the enriched payload through `--dag-out`.

Without `--diagnose`, current non-training behavior is unchanged.

## Global DAG reward normalization

Diagnosis scores belong to training `Node` instances, not to `SuperNode`.
`SuperNode` remains a topology and tensor container. Each training node records
its raw diagnosis value per trajectory in `episode_scores[episode_id]`, where
an episode identifies one trajectory through the DAG. A shared node can
therefore retain distinct observations from different branches.

After the full DAG is available, the client derives each node's mean raw score
from its recorded episode values, then normalizes those means across all scored
nodes in the DAG:

```
node.process_reward = node.mean_episode_score / sum(all_node_mean_scores)
```

The result is a non-negative process-reward distribution whose sum over the
whole DAG is exactly one. A zero-total DAG assigns the uniform distribution
across the scored nodes so the invariant still holds. `outcome_reward` remains
the verifier's terminal reward and is never overwritten; training consumes the
two reward channels together. The implementation must establish an exact
`(segment_id, seq) -> Node` mapping so diagnosis records are applied only to
the matching generated turn.

## Error handling

The server rejects non-terminal, cross-workspace, disabled, and malformed
requests. Client failures identify the operation and redact response bodies.
The client does not claim success merely because the first DAG is terminal:
with `--diagnose` it requires a completed diagnosis report and verifies the
final DAG has a positive `score_max` plus step rewards covering every eligible
LLM-output turn.

## Tests and validation

Server tests will cover authorization, terminal-state gating, configuration
gating, invocation/resumption, and a completed non-training project whose DAG
contains one reward per eligible turn. Client tests will cover the new HTTP
request, polling, final-DAG validation, unchanged default behavior, exact
turn-to-Node mapping, repeated-node episode-score aggregation, global
normalization, and the zero-total fallback.

An integration run needs a configured MultiCA endpoint, API key, workspace,
agent runtime, diagnosis-agent settings, and an external model runtime. The
current workspace has none of those settings, so it cannot safely perform that
live run until they are provided.
