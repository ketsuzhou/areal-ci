# Brainstorm Summary

- Change: sub-project-f-per-agent-env-and-checkpointing
- Date: 2026-07-07

## Confirmed Technical Approach

### Entropy signal (CONFIRMED — Option A)

Investigation finding: E7 currently injects `logprobs=True` only
(`proxy_rollout_server.py:663`) and stores a **1-D chosen-token logprob tensor**
(`types.py:187,199`). True distribution entropy is NOT computable from stored data —
`top_logprobs=N` is required to get the candidate distribution.

Decision: **Option A — extend E7 to capture the distribution; compute scalar entropy at
the proxy.**

- E7 (`_call_client_create`) additionally requests `top_logprobs=N` (default N=5)
  alongside `logprobs=True`.
- Proxy computes **normalized Shannon entropy** `H(p)/log(N)` per token over the top-k
  distribution, aggregates as **mean** over the tool-call decision span → a single
  scalar entropy score per tool-call decision.
- Persist **only the scalar** entropy score (not the full distribution) — minimizes
  storage impact on E.
- **Span (confirmed default, no objection)**: tool-call tokens only (function name +
  arguments), not the whole assistant turn.
- Graceful degradation: if upstream rejects `top_logprobs` (Anthropic-native without
  gateway), skip entropy-gated checkpoints for that interaction; always-checkpoint
  events still fire. Mirrors E7's existing fallback.
- Touches: E7 proxy path + adds a scalar entropy field on the interaction/checkpoint
  path. E is mid-build — change is timely.

## Key Trade-offs and Risks

- **Entropy storage cost** (resolved): persist scalar only, not distribution.
- **top_logprobs provider support**: Anthropic-native may reject → graceful skip
  (entropy-gated only; always-events unaffected).
- **Threshold calibration** (CANDIDATE): `ENV_CHECKPOINT_ENTROPY_THRESHOLD` — recommend
  a **percentile of the within-rollout entropy distribution** (default 0.75 → checkpoint
  top ~25% highest-entropy tool calls), calibrated from sample rollouts in T1.
  Rationale: absolute entropy depends on N and tokenization; a percentile is
  self-calibrating per rollout. (User confirms at proposal stage.)
- **D-dependency for T7** (RESOLVED — lighter than OpenSpec implied): D's session-open
  hook `maybeOpenTrainingSession` EXISTS and is wired (training.go:148; called at
  env_dispatch.go:673,721 on trained-task creation). F's branch-from-checkpoint extends
  C's `mode=branch` (`CopyProjectSubtree` + `dispatchOne`), so the branched trained task
  gets a new RL session via the existing hook. NOT blocked on D. D's `.comet.yaml`
  (0/16, plan-ready) is STALE — the session lifecycle hooks are in code.
- **E critic loop is COMPLETE** (CORRECTION): `maybeCloseTrainingSessionFromCritic`
  - `parseCriticReward` are implemented AND wired (training.go:354, 524). E7 logprobs
    captured at runtime. F extends E7 with `top_logprobs=N`.
- **Checkpoint concurrency** (CANDIDATE): serialize checkpoint creation per project
  (transaction-scoped) so concurrent squad-member events produce deterministic,
  non-interleaved DB-subtree snapshots.
- **Hook ordering with E** (CONFIRMED RECOMMENDATION): F's `maybeCheckpoint` fires at
  the same trained-terminal chokepoints as E's `RouteTerminalTrainingTask`
  (task.go:956/1380/1571). Checkpoint BEFORE E's routing — captures state at the
  decision point, prior to critic spawn / session close.

## Testing Strategy

- multica Go TDD (scoped to touched packages; pre-existing webpush build + 16 ON
  CONFLICT failures unchanged). Hand-write generated Go (no repo-wide `sqlc generate`).
- AReaL Python: `uv run pytest` on touched tests; `pre-commit run --files`.
- T7 (branch-from-checkpoint) unit-tested with the existing session-open hook (no D
  block); cross-repo E2E gated on a live stack, not on D completion.
- Entropy helper: deterministic/uniform logprob fixtures → known entropy values.

## Spec Patches

- **Delta specs MISSING** (PENDING): this change has no `specs/` dir. The two new
  capabilities (`env-checkpointing`, `per-agent-env-customization`) need delta spec
  files with acceptance scenarios. No `openspec/specs/` main specs exist either (C/D/E
  not archived). To create as part of design phase.
