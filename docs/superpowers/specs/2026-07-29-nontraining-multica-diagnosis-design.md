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
request, polling, final-DAG validation, and unchanged default behavior.

An integration run needs a configured MultiCA endpoint, API key, workspace,
agent runtime, diagnosis-agent settings, and an external model runtime. The
current workspace has none of those settings, so it cannot safely perform that
live run until they are provided.
