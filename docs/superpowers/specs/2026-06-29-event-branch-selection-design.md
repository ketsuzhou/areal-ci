# Event Branch-Point Selection — Design

Date: 2026-06-29
Status: Approved (design); implementation pending
Area: `customized_areal/tree_search/agents/`

## 1. Summary

Add a torch-free **branch-point selection policy** that operates over the canonical
`Event` sequence (the completion-ordered linear log of the agent-execution DAG). The
policy ports the existing `select_branch_candidate` criterion — a critic **TD-error
gate** followed by **max-entropy ranking** — onto the `Event`/DAG representation, and
emits **one branch point per `task_id` lane** as `(task_id, seq)` pairs that are
directly consumable by `event_codec.replay_prefix_for`.

This is a **selection policy only**: no new generation, scoring, or rollout. It does
not replace the existing token-level `Node`-based selector; it coexists with it.

## 2. Goals / Non-Goals

### Goals
- A new module `agents/branch_selection.py`, torch-free and I/O-free, consuming an
  `Event` log and returning `list[BranchPoint]`.
- Faithfully reproduce the existing selection criterion (TD-error gate + entropy
  ranking) so behavior is preserved, only the representation changes.
- Emit one branch point per `task_id`, keyed for `replay_prefix_for`.
- Small, independently-testable pure functions (matching the `agents/` package style).

### Non-Goals
- No new branch generation, scoring, or MCTS expansion (selection only).
- No modification to `core/customized_grouped_workflow.py::select_branch_candidate`
  or `core/tree_store.py` (left untouched).
- **Wiring `select_branch_points` into the live workflow is an explicitly-flagged
  follow-up, out of scope for this spec.** This spec delivers the torch-free
  selection layer and its tests only.
- No config schema changes (the function takes raw floats, like `gae.py`).

## 3. Locked Decisions

| # | Decision |
|---|----------|
| Q1 | Branch-point **selection** policy only (no generation/scoring). |
| Q2 | Port the existing criterion **as-is**: critic TD-error gate `|r_t + γ·v(s_{t+1}) − v(s_t)| ≥ td_threshold`, survivors ranked by max entropy. |
| Q3 | **One branch point per `task_id`** (lane). |
| Q4 | **Coexist**: new function in `agents/`; leave `Node`-based `select_branch_candidate` untouched; live-workflow wiring is a separate follow-up. |
| Q5 | Field mapping confirmed (see §6). |
| Structure | **Approach 2**: composable pure functions + orchestrator. |

## 4. Architecture & Placement

New module: `customized_areal/tree_search/agents/branch_selection.py`. Sits beside
`gae.py` and `event_codec.py` in the canonical `agents/` layer. Depends only on
`event_model`, `event_codec`, and `execution_dag`. No torch, no I/O.

Data flow:

```
Event log ──> branch_selection.select_branch_points(events, td_threshold, gamma)
                │  (group by task_id; per lane: gate by |δ_t|, rank by entropy)
                ▼
          list[BranchPoint(task_id, seq, node_id, td_error, entropy)]
                │
                ▼  (one per task_id; consumed downstream — separate follow-up)
   event_codec.replay_prefix_for(events, branch_point=(task_id, seq)) ──> ReplayPrefix
```

New exports added to `agents/__init__.py` (`BranchPoint`, `select_branch_points`, and the
three helpers). The `agents/__init__.py` must stay torch-free, which this module preserves.

## 5. Public API & Data Types

All torch-free, all pure (no input mutation, no I/O, deterministic).

```python
@dataclass(frozen=True)
class BranchPoint:
    task_id: str            # lane identifier
    seq: int                # == Event.branch_seq; (task_id, seq) is replay_prefix_for's key
    node_id: str            # the selected Event's node_id (provenance/debug)
    td_error: float | None  # |δ_t| that passed the gate; None when gate bypassed (no value)
    entropy: float          # Event.metadata["max_entropy"] used for ranking (0.0 if absent)
```

```python
def lane_successor_value(
    event: Event, lane_children: Mapping[str, Event],
) -> tuple[float, float]:
    """Return (r_t, v_next) for the TD term.
    In-lane DAG child exists -> (0.0, child.value or 0.0).
    Terminal (no in-lane child) -> (event.outcome_reward, 0.0)."""

def td_error(event: Event, r_t: float, v_next: float, *, gamma: float) -> float | None:
    """|r_t + gamma*v_next - v(s_t)|, where v(s_t) = event.value.
    Returns None when event.value is None (gate bypassed downstream)."""

def passes_gate(delta: float | None, *, td_threshold: float) -> bool:
    """True if td_threshold <= 0 (gate off), delta is None (bypass/keep),
    or delta >= td_threshold."""

def select_branch_points(
    events: Sequence[Event], *, td_threshold: float = 0.0, gamma: float = 1.0,
) -> list[BranchPoint]:
    """Group eligible events (branch_seq is not None) by task_id; within each
    lane keep gate survivors; return the highest-entropy survivor per lane.
    Deterministic ordering (sorted by task_id)."""
```

API decisions:
- `select_branch_points` is the only entry point most callers need; the three helpers
  are exported for unit testing and reuse (mirrors how `gae.py` exposes both
  `compute_global_gae` and `events_from_nodes`).
- Returns a **list**, one `BranchPoint` per `task_id` that has at least one surviving
  candidate. Lanes with no eligible/surviving events contribute nothing.
- `gamma` defaults to `1.0`, `td_threshold` to `0.0` — matching
  `select_branch_candidate`, so `td_threshold=0.0` reproduces pure entropy selection
  per lane.

## 6. Field Mapping (Event ← old Node)

| Concept | Old (`Node`) | New (`Event`) |
|---------|--------------|---------------|
| Candidate eligibility | `need_branch and task_id and branch_sandbox_id` | `branch_seq is not None`; emit `(event.task_id, event.branch_seq)` |
| TD successor `v(s_{t+1})` | same `episode_id`, `turn_idx + 1` | in-lane **DAG child within the same `task_id` lane** (terminal: `r_t = outcome_reward`, `v_next = 0`) |
| Critic value `v(s_t)` | `tree_store`/`Node.value` | `Event.value` (missing → bypass gate, entropy-only fallback) |
| Entropy ranking key | `entropy_stats["max_entropy"]` | `Event.metadata["max_entropy"]` (codec copies `node.metadata`) |

## 7. Algorithm

`select_branch_points(events, td_threshold, gamma)`:

1. **Build lane children map.** Reconstruct the DAG once via `events_to_dag(events)`.
   For each `task_id` lane, determine each event's in-lane DAG child: follow outgoing
   edges to the child whose `task_id` equals the parent's. If a node has multiple
   in-lane children (unusual for a sequential lane), pick the one with the smallest
   `completion_index` for determinism. Yields `lane_children: dict[node_id -> Event]`.
2. **Filter eligible candidates.** Keep events where `branch_seq is not None`. Group by
   `task_id`.
3. **Per lane, gate each candidate:**
   - `r_t, v_next = lane_successor_value(event, lane_children)`
   - `delta = td_error(event, r_t, v_next, gamma=gamma)` (`None` if `event.value is None`)
   - keep if `passes_gate(delta, td_threshold=td_threshold)`.
4. **Rank survivors.** Within each lane, pick the survivor with the highest
   `Event.metadata.get("max_entropy", 0.0)`. Ties broken by smallest
   `completion_index` (deterministic).
5. **Emit.** One `BranchPoint` per lane with a survivor, list sorted by `task_id`.

This is a faithful port: step 3 mirrors `select_branch_candidate`'s
`delta = abs(r_t + gamma*v_next - v_t)` gate and "missing value bypasses gate"; step 4
mirrors `max(candidates, key=_max_entropy)`. The only semantic shifts are
**lane = `task_id` over DAG edges** (instead of `episode_id` + `turn_idx+1`) and
**multiple outputs (one per lane)** (instead of one global winner) — both per the
locked decisions.

### Value-convention note

In `gae.py`, `Event.value` is documented as `V_{t+1}` (value *after* the turn).
`select_branch_candidate` treats the node's own value as `v(s_t)` and the successor's
as `v(s_{t+1})`. This design **preserves that same relative usage** — `event.value` as
`v(s_t)`, in-lane child's value as `v_next` — so the TD-error has identical shape to
today's. We deliberately keep the existing convention rather than re-deriving from the
GAE indexing; this is a conscious, behavior-preserving choice.

## 8. Error Handling & Edge Cases

- **Empty log** → `[]`.
- **No eligible events** (no `branch_seq` set) → `[]`.
- **Lane with eligible candidates but none survive the gate** → that lane contributes
  nothing (mirrors `select_branch_candidate` returning `None`); other lanes still emit.
- **Missing `Event.value`** (`None`) → `td_error` returns `None` → gate **bypassed**,
  candidate kept (entropy-only fallback), as today.
- **Missing `metadata["max_entropy"]`** → treated as `0.0` for ranking; if all
  survivors in a lane tie at `0.0`, the `completion_index` tiebreak decides.
- **Terminal candidate** (no in-lane child) → `r_t = outcome_reward`, `v_next = 0.0`.
- **Malformed DAG** (asymmetric edges, non-dense `completion_index`, etc.) → surfaced
  by `events_to_dag`, which raises `DAGError`. `branch_selection` does not re-validate;
  it lets `DAGError` propagate (single source of validation = the codec).
- **Negative `gamma`/`td_threshold`** → not validated here (these come from the
  validated `Config`); `branch_selection` stays a pure leaf function with no config
  coupling, matching `gae.py`.

No mutation of input events; no I/O; deterministic output for identical input.

## 9. Testing

Strict TDD (failing tests first). Torch-free; run via `.venv-test/bin/python -m pytest`;
lint with the project ruff. New file `tests/test_branch_selection.py` (alongside
the tests, matching where `test_gae.py` lives).

**Helper-level (unit):**
- `lane_successor_value`: in-lane child → `(0.0, child.value)`; terminal →
  `(outcome_reward, 0.0)`; child with `value=None` → `v_next=0.0`.
- `td_error`: known `|δ|` arithmetic; `value=None` → `None`.
- `passes_gate`: threshold `≤0` → True; `None` delta → True (bypass);
  `delta ≥ threshold` boundary (inclusive); below → False.

**Orchestrator-level (ported from `test_branch_td_gate.py`):**
- `td_threshold=0.0` reproduces entropy-only: highest `max_entropy` per lane wins.
- No value → entropy-only fallback kept.
- Sub-threshold candidate dropped; lower-entropy-but-higher-δ candidate chosen.
- Highest entropy among survivors.
- All-dropped lane → omitted from results.
- Terminal uses `outcome_reward`.
- Missing critic value bypasses gate.

**Lane / multi-output (new):**
- Two `task_id` lanes → exactly two `BranchPoint`s, each the in-lane winner.
- Emitted `(task_id, seq)` round-trips through `replay_prefix_for` without `DAGError`
  (integration guard — proves output is materializer-ready).
- Determinism: shuffled input event order → identical output; `task_id`-sorted result.
- Empty log and no-eligible-events → `[]`.
- Malformed log → `DAGError` propagates from `events_to_dag`.

Coverage target: every helper branch and every orchestrator edge case in §8. No GPU,
no skips needed (fully torch-free).

## 10. Follow-ups (out of scope)

- Wire `select_branch_points` into `core/customized_grouped_workflow.py`: build the
  `Event` log from the live rollout (via `dag_to_events`), call the selector, and feed
  emitted `(task_id, seq)` points to the branch materializer through
  `replay_prefix_for`. This is a separate, explicitly-flagged change touching the live
  workflow.
