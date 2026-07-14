## ADDED Requirements

### Requirement: Lossless SuperNode serialization
The system SHALL serialize every `SuperNode` field through `to_dict()` and restore it through `from_dict()` with no loss. The serialized envelope MUST include `visit_count` alongside the identity, topology, env, reward, and turn fields. `from_dict()` MUST tolerate a missing `visit_count` key (older checkpoints) by defaulting to 0.

#### Scenario: visit_count survives a round-trip
- **WHEN** a `SuperNode` with `visit_count` set to a non-zero value is serialized via `to_dict()` and restored via `from_dict()`
- **THEN** the restored `SuperNode`'s `visit_count` MUST equal the original value

#### Scenario: Old checkpoints without visit_count deserialize to zero
- **WHEN** `from_dict()` reads a dict that does not contain a `visit_count` key
- **THEN** the restored `SuperNode`'s `visit_count` MUST be 0

### Requirement: Topology-complete v2-assembled SuperNodes
The `SuperNodeAssembler.assemble_from_refs` path SHALL populate each assembled `SuperNode`'s `incoming_edges` and `outgoing_edges` tuples so they are consistent with the `ExecutionDAG.edges` of the containing DAG. A SuperNode serialized via `to_dict()` after v2 assembly MUST retain its full edge topology.

#### Scenario: v2 assembly populates edge tuples
- **WHEN** `assemble_from_refs` builds an `ExecutionDAG` with typed edges between segments
- **THEN** every assembled `SuperNode`'s `incoming_edges` and `outgoing_edges` MUST match the edges in the `ExecutionDAG` (same sources/destinations and edge types)

#### Scenario: v2-assembled topology survives serialization
- **WHEN** a SuperNode produced by `assemble_from_refs` is serialized via `to_dict()` and restored via `from_dict()`
- **THEN** the restored SuperNode's `incoming_edges` and `outgoing_edges` MUST equal the originals

#### Scenario: Leaf segment has empty edge tuples
- **WHEN** a segment with no incoming or outgoing edges is assembled
- **THEN** its `incoming_edges` and `outgoing_edges` MUST both be empty

### Requirement: v2-path fan-in credit is preserved
The v2 assembly path MUST NOT regress fan-in reward credit. When a segment has multiple incoming blocking edges (DELEGATION or COMPLETION), `distribute_reward_over_dag` MUST credit every parent segment.

#### Scenario: Fan-in join credits all parents on the v2 path
- **WHEN** `assemble_from_refs` produces a segment with two incoming DELEGATION edges and `distribute_reward_over_dag` distributes a terminal reward backward
- **THEN** both parent segments MUST receive non-zero credit
