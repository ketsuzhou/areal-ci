# Bi-Directional DAG ↔ Linear Event Codec — Design

**Date:** 2026-06-29 **Status:** Approved design — ready for implementation planning
**Package:** `customized_areal/tree_search/dag/` **Related:**
`2026-06-26-multica-dag-rl-design.md` (§3.1 snapshot-at-frontier + transcript replay),
`dag/CRITIC_GAE_INTEGRATION.md` (framework-B reward backup over the global
completion-ordered sequence).

## 1. Overview

The multi-agent execution graph is a DAG of agent runs (nodes = turns, edges =
delegation / mention / completion). Two operations need a **linear** view of that DAG:

1. **DAG → linear events** — for reward backup. Framework-B GAE runs over the global
   completion-ordered turn sequence (a topological linearization of the DAG), not
   per-edge. This direction partially exists today via `gae.events_from_nodes` and
   `critic_observation.build_critic_observations`.
1. **Linear events → DAG** — for environment resume. A persisted linear event log is
   reconstructed into an `ExecutionDAG`, from which a replay prefix (`messages <= seq`
   for a chosen branch point) is derived and handed to the existing
   `BranchMaterializer`.

This design introduces **one canonical `Event` type** as the source of truth for the
linear trajectory, and a **pure, bidirectional codec** around it. The existing forward
helpers are kept as thin adapters over the canonical type with byte-for-byte identical
outputs, so the verified GAE/critic paths do not change behavior.

## 2. Goals and non-goals

### Goals

- A single canonical, JSON-serializable `Event` type that is the persisted
  trajectory-log format.
- Lossless forward (`dag_to_events`) and reverse (`events_to_dag`) conversion — the
  reverse rebuilds an `ExecutionDAG` identical in nodes **and** typed edges.
- A `replay_prefix_for(branch_point)` helper, built on the reconstructed DAG, returning
  plain data shaped to `BranchMaterializer.materialize`'s inputs.
- Existing `events_from_nodes` and `build_critic_observations` produce identical output,
  now expressed in terms of (or validated against) the canonical type.
- Torch-free and I/O-free, consistent with the rest of the `dag/` package.

### Non-goals

- **No edge-routed / tree-style reward backup.** Framework-B GAE over the linear order
  is the credit mechanism; DAG edges are structural only (they fix the causal ordering
  and support resume/frontier queries). The codec stores edges bidirectionally for
  fidelity, not because backup walks them.
- **No changes** to `ExecutionDAG.to_records()` / `from_records()`, `integration.py`
  (`BranchMaterializer`), `environment.py`, or the `AgentRunNode` field set.
- **No resume orchestrator.** The codec is produce-data-only; materialization stays in
  `BranchMaterializer`.
- **No new on-disk migration** of existing serialization onto events.

## 3. Architecture

Two new modules in `customized_areal/tree_search/dag/`, both torch-free and I/O-free:

- **`event_model.py`** — the canonical data definition only:
  - `Event` dataclass (node-granular, frozen).
  - `Event.to_dict()` / `Event.from_dict()` — JSON-round-trippable; the canonical
    persisted trajectory-log format.
  - `message_timeline(events) -> list[dict]` — the derived message-level view used by
    the critic/replay consumers.
- **`event_codec.py`** — the bidirectional conversion:
  - `dag_to_events(dag, *, ordering=None, messages_by_node=None) -> list[Event]`
  - `events_to_dag(events) -> ExecutionDAG`
  - `replay_prefix_for(events, *, branch_point) -> ReplayPrefix`

### Dependency direction (acyclic)

```
execution_dag.py  ◄── event_codec.py ──► event_model.py
                                              ▲
gae.py (adapter) ─────────────────────────────┤
critic_observation.py (adapter) ──────────────┘
```

- `event_model` depends on nothing in the package (pure data).
- `event_codec` depends on `event_model` + `execution_dag`.
- `gae.events_from_nodes` and `critic_observation.build_critic_observations` delegate to
  / are validated against `event_model` (Section 5), keeping their current signatures
  and outputs.
- New names exported from `dag/__init__.py` (kept torch-free): `Event`, `dag_to_events`,
  `events_to_dag`, `replay_prefix_for`, `message_timeline`.

## 4. The `Event` data model

`Event` is node-granular (one per DAG node / completed turn), frozen, torch-free. Fields
group into five concerns.

**Identity** (mirrors `AgentRunNode`):

| Field        | Type          |
| ------------ | ------------- |
| `node_id`    | `str`         |
| `agent_id`   | `str`         |
| `issue_id`   | `str`         |
| `task_id`    | `str`         |
| `session_id` | `str \| None` |

**Ordering** (explicit; the linear axis):

| Field              | Type            | Notes                                                                                                                           |
| ------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `completion_index` | `int`           | 0-based, dense, unique within a log. **Authoritative** ordering.                                                                |
| `completion_time`  | `float \| None` | Optional wall-clock seconds, provenance only. Never used for ordering decisions, so missing/tied times never break round-trips. |

**Structure** (bidirectional; lossless edge recovery):

| Field            | Type                                             |
| ---------------- | ------------------------------------------------ |
| `incoming_edges` | `tuple[(src_node_id: str, type: EdgeType), ...]` |
| `outgoing_edges` | `tuple[(dst_node_id: str, type: EdgeType), ...]` |

Edges are stored in **both** directions. `dag_to_events` derives both from the
reconstructed DAG so they are always consistent at emit time; `events_to_dag` validates
symmetry on decode (Section 6), turning the redundancy into a built-in checksum rather
than a consistency hazard.

**Branch provenance** (for `replay_prefix_for` + resume):

| Field                    | Type          |
| ------------------------ | ------------- |
| `branch_seq`             | `int \| None` |
| `branch_issue_id`        | `str \| None` |
| `branch_env_snapshot_id` | `str \| None` |

**RL signals + payload:**

| Field            | Type               | Notes                                                                              |
| ---------------- | ------------------ | ---------------------------------------------------------------------------------- |
| `value`          | `float \| None`    | V₍t+1₎ (critic next-state value).                                                  |
| `process_reward` | `float`            |                                                                                    |
| `outcome_reward` | `float`            |                                                                                    |
| `messages`       | `tuple[dict, ...]` | The turn's transcript slice, ordered. Drives `message_timeline` + replay prefixes. |
| `metadata`       | `dict`             |                                                                                    |

### Serialization

- `to_dict()` emits a plain JSON-safe dict: `EdgeType` → its `.value` string, tuples →
  lists.
- `from_dict()` is the exact inverse: coerces `type` strings back to `EdgeType`, lists
  back to tuples.
- Round-trip invariant (unit-tested): `Event.from_dict(e.to_dict()) == e`.

### Relationship to `AgentRunNode`

Every `AgentRunNode` field maps onto an `Event` field, so `dag_to_events` →
`events_to_dag` reconstructs nodes verbatim. The fields the `Event` adds beyond
`AgentRunNode` — `completion_index`, `completion_time`, `incoming_edges`,
`outgoing_edges`, `messages` — are the temporal / structural / payload information a
flat node record does not centre on. These added fields live **only in the Event log**;
they round-trip Event↔Event and are **not** written back onto the rebuilt `AgentRunNode`
(which stays a faithful node). The codec sources `messages` from
`messages_by_node[node_id]` if supplied, else from `node.metadata.get("messages", ())`;
`AgentRunNode` is not given new fields.

## 5. Forward path (`dag_to_events` + adapters)

### `dag_to_events(dag, *, ordering=None, messages_by_node=None) -> list[Event]`

1. **Determine completion order.** If `ordering` (a list of `node_id` in completion
   order, e.g. from recorded `completion_time`) is given, use it. Otherwise fall back to
   `dag.topological_order()` — deterministic (Kahn with sorted ties), and valid because
   completion order is always *a* topological order ("prefix = cut").
1. **Validate** the ordering is a permutation of the DAG's node ids and is topologically
   valid (every node appears after all its parents). Otherwise raise `DAGError`.
1. **Build one `Event` per node** in that order: assign dense `completion_index`
   `0..n-1`; copy identity + branch + RL fields off `AgentRunNode`; fill
   `incoming_edges` / `outgoing_edges` from `dag.parents` / `dag.children` (with
   `EdgeType` from the stored edges); attach `messages` from `messages_by_node[node_id]`
   if provided, else `node.metadata.get("messages", ())`; carry `completion_time` if
   known.

Pure function; does not mutate the DAG.

### Adapter 1 — `gae.events_from_nodes(ordered_nodes)`

Unchanged signature; output byte-for-byte identical. Rewritten to build `Event`s for the
ordered nodes via the shared model, then **project** each to a `GlobalEvent` with the
same arithmetic as today (`value = value or 0.0`,
`reward = process_reward + outcome_reward`). A parity test asserts old-vs-new outputs
are identical on a fixture DAG, so `compute_global_gae` and the rest of the GAE path are
untouched.

### Adapter 2 — `critic_observation.build_critic_observations(messages)`

This builder already takes a **message timeline**, not nodes, so it relates to `Event`
through the derived view rather than `dag_to_events`. The function code stays
**unchanged**; `message_timeline(events)` (Section 7) produces exactly the message-list
shape it expects (messages in completion order, turn-outputs tagged with `node_id`). A
parity test proves:

```
build_critic_observations(message_timeline(dag_to_events(dag)))
    == build_critic_observations(original_hand_built_timeline)
```

so the critic path is validated as consistent with the canonical `Event` without
changing its code.

## 6. Reverse path (`events_to_dag` + `replay_prefix_for`)

### `events_to_dag(events) -> ExecutionDAG`

Lossless structural rebuild. Steps (each failure raises `DAGError`):

1. **Order check.** Verify `completion_index` is dense, unique, and non-negative; reject
   gaps / duplicates / negatives.
1. **Edge symmetry check.** Every `A.outgoing (A→B, type)` must have a matching
   `B.incoming (A→B, type)` and vice versa, with matching `EdgeType`. Mismatch →
   `DAGError`. (This is the checksum that justifies storing edges in both directions.)
1. **Add nodes.** Rebuild each `AgentRunNode` from the Event's identity + branch + RL
   fields. `messages` / `completion_index` stay in the Event log, not on the node.
1. **Add edges** from the incoming lists, using `ExecutionDAG.add_edge` (idempotent).
   Symmetry already guarantees the outgoing lists agree, so edges are added once with no
   duplicates.
1. **Topological-order invariant.** Assert the event order (`completion_index`) is a
   valid topological order of the reconstructed edges — for every edge `src→dst`,
   `index(src) < index(dst)`. Violation → `DAGError`. This cheap assertion protects
   reward backup from a corrupted / hand-edited log.

Round-trip guarantee (tested): `dag_to_events` → `events_to_dag` reproduces nodes +
typed edges exactly.

### `replay_prefix_for(events, *, branch_point) -> ReplayPrefix`

Built on the reconstructed DAG. `branch_point = (task_id, seq)`, where `seq` is
`task_message.seq` (matching `2026-06-26-multica-dag-rl-design.md` §3.1).

1. `dag = events_to_dag(events)`.
1. Locate the branch node: the unique Event whose `task_id` matches and whose
   `branch_seq == seq` (the run recorded as allowed to branch at that
   `task_message.seq`). Zero or multiple matches → `DAGError`.
1. Compute its **ancestor set** (`dag.ancestors(node_id)`) plus the branch node itself —
   the causally-prior lane up to and including the branch run.
1. Collect those events in `completion_index` order and flatten their `messages`
   payloads via `message_timeline` restricted to that ancestor sub-DAG, cut at `seq`.

`source_issue_id` in the returned `ReplayPrefix` is the located branch node's `issue_id`
(the issue being forked); `branch_env_snapshot_id` is copied from the same node.

Returns a plain dataclass:

```python
@dataclass(frozen=True)
class ReplayPrefix:
    replay_messages: list[dict]
    task_id: str
    seq: int
    source_issue_id: str
    branch_env_snapshot_id: str | None
```

— exactly the inputs `BranchMaterializer.materialize(...)` already accepts. No I/O, no
materialization. The "DAG reconstructed for resume" is the full lossless graph; the
replay prefix is the ancestor sub-DAG slice cut at the branch point.

### Boundary

`replay_prefix_for` produces data; `BranchMaterializer` (unchanged) consumes it to
snapshot / fork / `agent_start_branch`.

## 7. Derived message-timeline view

`message_timeline(events) -> list[dict]`:

- Walks `events` in `completion_index` order and concatenates each event's `messages`
  payload.
- Tags each turn-output message with its `node_id` (so `build_critic_observations` can
  detect turn outputs); non-output payload messages (seed / user / tool) pass through
  untagged — reproducing exactly the timeline shape the critic builder expects today.
- Pure and deterministic. No `value` / `reward` leakage into the message view (the
  critic sees only whitelisted message fields, consistent with `DEFAULT_CRITIC_FIELDS`).

## 8. Error handling

All errors use the package's existing `DAGError` type. No silent fallback on the reverse
path — corruption is surfaced.

- **Forward:** ordering not a permutation of node ids / not topologically valid.
- **Reverse:** non-dense / duplicate / negative `completion_index`; edge-symmetry
  mismatch; unknown `src` / `dst` in an edge; topological-order invariant violation.
- **`Event.from_dict`:** unknown `EdgeType` string or missing required field (surfaced
  as `DAGError`).
- **`replay_prefix_for`:** branch point not found, `task_id` absent, or multiple nodes
  match `(task_id, branch_seq == seq)`.

## 9. Testing strategy

Torch-free; run via `.venv-test/bin/python -m pytest`, linted with the project ruff. New
types exported from `dag/__init__.py` (kept torch-free).

- **`test_event_model.py`** — `to_dict` / `from_dict` round-trip equality; `EdgeType`
  string coercion; tuple/list normalization; `message_timeline` ordering + `node_id`
  tagging.
- **`test_event_codec.py`** —
  - Forward: `dag_to_events` with explicit `ordering` and with `topological_order()`
    fallback; dense indices; edge fill.
  - Reverse: lossless node+edge round-trip on a multi-lane DAG (the `O0/R0/C0/T0/C1/O1`
    example from `CRITIC_GAE_INTEGRATION.md`).
  - Each `DAGError` path: bad `completion_index`, asymmetric edges, topo-order
    violation, unknown edge endpoint, missing branch point.
  - `replay_prefix_for` ancestor-slice cut at `seq`.
  - Full identity round-trip: `dag → events → dict → events → dag`.
- **Parity tests** (the load-bearing guarantee):
  - `events_from_nodes` old-vs-new output identical.
  - `build_critic_observations(message_timeline(dag_to_events(dag)))` equals feeding the
    original hand-built timeline.

## 10. File summary

| File                                                   | Status         | Contents                                                              |
| ------------------------------------------------------ | -------------- | --------------------------------------------------------------------- |
| `dag/event_model.py`                                   | new            | `Event`, `to_dict`/`from_dict`, `message_timeline`                    |
| `dag/event_codec.py`                                   | new            | `dag_to_events`, `events_to_dag`, `replay_prefix_for`, `ReplayPrefix` |
| `dag/gae.py`                                           | edit           | `events_from_nodes` → thin projection over `Event` (output unchanged) |
| `dag/critic_observation.py`                            | unchanged code | validated against `Event` via `message_timeline` parity test          |
| `dag/__init__.py`                                      | edit           | export new names (stays torch-free)                                   |
| `dag/test_event_model.py`                              | new            | model + serialization + timeline tests                                |
| `dag/test_event_codec.py`                              | new            | forward/reverse/round-trip/error/replay + parity tests                |
| `execution_dag.py`, `integration.py`, `environment.py` | unchanged      | —                                                                     |
