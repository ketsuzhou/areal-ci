# MCTS Tree Backup Design Spec

## Problem

In standard PPO/GAE, advantage estimation is local — each trajectory's advantages are
computed independently using TD-residuals and GAE lambda. When multiple rollouts share
the same prompt but produce different responses, GAE cannot leverage the
cross-trajectory information. This wastes signal: if trajectory A gets reward 1.0 and
trajectory B gets reward 0.0 from the same prompt, GAE treats them independently rather
than recognizing that A's response is better *relative to B's*.

## Solution

Add MCTS tree backup to the RL training loop. Rollouts are inserted turn-by-turn into a
shared compressed trie (keyed by query). MCTS backup propagates rewards from leaves to
root, computing Q-values at each node as the average reward of all trajectories passing
through it. These Q-values replace GAE as the advantage signal.

Key insight: when multiple trajectories share a prefix (same prompt, same early turns),
the shared nodes' Q-values reflect the average outcome of *all* trajectories through
that node. This provides a natural variance reduction — shared prefixes get averaged,
while divergent turns retain their distinct signal.

## Architecture

### Components

| Component               | File                 | Responsibility                                  |
| ----------------------- | -------------------- | ----------------------------------------------- |
| `TreeBackupConfig`      | `config.py`          | Mode enum + config dataclass                    |
| `Turn`                  | `turn_splitter.py`   | Prompt/response token split dataclass           |
| `make_turn_splitter`    | `turn_splitter.py`   | Role-marker-based turn splitting                |
| `TrieNode`              | `trie_node.py`       | Turn-level compressed trie (path indexing only) |
| `MCTSTreeStore`         | `mcts_tree_store.py` | Per-query tree manager + MCTS backup stats      |
| `TreeAdvantageComputer` | `advantage.py`       | Replaces GAE with tree Q-values                 |
| `TreeCheckpointManager` | `checkpoint.py`      | JSON save/load for tree persistence             |
| `TreeBackupPPOTrainer`  | `trainer.py`         | PPOTrainer subclass with method patching        |

### Data Flow

```
Rollout batch (list[dict])
    │
    ├─► TreeBackupPPOTrainer patches PPOActor.compute_advantages
    │       │
    │       ├─► original compute_advantages (GAE path)
    │       │       ├─► batched_call → _compute_advantages
    │       │       │       ├─► [KL rewards, reward scaling, GAE]
    │       │       │       └─► advantages, returns, kl_rewards, tot_rewards
    │       │       └─► split_batch → list[dict] with GAE advantages
    │       │
    │       ├─► MCTSTreeStore.insert_batch(trajectories)
    │       │       ├─► _get_query_id: hash prompt tokens → query key
    │       │       ├─► turn_splitter: split input_ids → list[Turn]
    │       │       ├─► start_sequence → add_turn loop → finish_sequence
    │       │       └─► _backup: walk leaf→root, update visit_count/q_value
    │       │
    │       ├─► TreeAdvantageComputer.compute(trajectories)
    │       │       ├─► get_advantages: Q-values per turn → per-token tensor
    │       │       ├─► get_prompt_mask: zero out prompt token advantages
    │       │       └─→ overwrite trajectories["advantages"], trajectories["returns"]
    │       │
    │       └─► kl_rewards, tot_rewards, loss_mask, logprobs preserved from GAE
    │
    └─► PPO update with tree-based advantages
```

### Turn Splitting

Turn splitting uses **role markers** (e.g., `<|im_start|>assistant`) rather than
delimiters. Each occurrence of the assistant marker in the token sequence marks the
start of a new turn:

- `prompt_tokens` = the marker tokens themselves
- `response_tokens` = everything from marker end to next marker start (or end of
  sequence)

If no marker is provided, the splitter auto-detects from the tokenizer's chat template,
checking for common patterns (Qwen/ChatML, Llama-3, Gemma) with fallback to
`<|im_start|>assistant`.

**Why role markers over delimiters?** Role markers are standard in chat templates and
already present in the tokenizer's vocabulary. Delimiters like `\n\n` are ambiguous —
they appear in natural text and don't reliably separate turns. Role markers provide
unambiguous turn boundaries.

### TrieNode Design

TrieNode stores **only path indexing data** — tokens, children (keyed by first response
token), sequence_ids, and ancestor links. MCTS statistics (visit_count, q_value) are
stored separately in MCTSTreeStore, keyed by `(query_id, id(node))`.

**Why separate MCTS stats?** Two reasons:

1. **Checkpoint decoupling**: The trie structure (which paths exist) changes less
   frequently than the MCTS statistics (which update on every backup). Separating them
   allows independent serialization strategies.
1. **Clean separation of concerns**: TrieNode is a pure data structure for path
   indexing. Adding MCTS stats would couple it to the backup algorithm.

Children are keyed by the **first response token** of the turn. When two trajectories
share a prompt but produce different responses, they diverge at the first response token
— naturally creating branching points in the trie.

### MCTSTreeStore Cursor-Based API

The store provides a cursor-based API for incremental trajectory building:

```
seq_id = store.start_sequence(query_id)     # cursor at root
store.add_turn(query_id, seq_id, turn1)      # cursor advances to child1
store.add_turn(query_id, seq_id, turn2)      # cursor advances to child2
store.finish_sequence(query_id, seq_id, reward)  # backup + clear cursor
```

**Why cursor-based?** Multi-turn rollouts are generated incrementally — the model
produces one turn at a time in a conversation. The cursor API lets the store track
position as turns arrive, rather than requiring the entire trajectory up front. The
convenience `insert_trajectory` method wraps this sequence for batch insertion.

### MCTS Backup

Backup walks from the completed leaf to root, updating every node on the path:

```
q_value = total_value / visit_count
```

where `total_value` accumulates the rewards of all trajectories passing through that
node. This is standard MCTS backup — shared-prefix nodes naturally average the outcomes
of all trajectories through them.

### TreeAdvantageComputer

Replaces GAE advantages with tree Q-values. For each trajectory:

1. Look up Q-values per turn from the tree
1. Expand to per-token tensor using turn boundaries
1. Zero out prompt tokens (only response tokens carry advantage signal)
1. Store as `advantages` and `returns` on the trajectory dict

### TreeBackupPPOTrainer Integration

Rather than modifying `PPOTrainer` directly, `TreeBackupPPOTrainer` subclasses it and
patches `PPOActor.compute_advantages` at runtime.

**Why method patching over subclass?** `compute_advantages` lives on `PPOActor`, not
`PPOTrainer`. Subclassing `PPOTrainer` doesn't give access to override `PPOActor`
methods. Method patching replaces the outer method at the class level, so all `PPOActor`
instances use the patched version.

**Outer vs. inner method:** `PPOActor` has two methods:

- `compute_advantages(self, data: list[dict])` — outer, receives individual trajectory
  dicts (1D tensors), calls `batched_call` to pad/batch them
- `_compute_advantages(self, data: dict)` — inner, receives a single padded batch dict
  with 2D tensors

We patch the **outer** method. This preserves the `list[dict]` format with 1D tensors,
which is the natural format for `insert_batch` and `TreeAdvantageComputer.compute`. The
patched method:

1. Calls the original `compute_advantages` — this runs the full GAE pipeline (KL
   rewards, reward scaling, advantage normalization, etc.) and returns `list[dict]` with
   `advantages`, `returns`, `kl_rewards`, `tot_rewards`, `loss_mask`, `logprobs`
1. Inserts trajectories into the tree via `tree_store.insert_batch(result)` — uses raw
   `traj["rewards"]` (not KL-adjusted)
1. Computes tree-based advantages via `tree_advantage_computer.compute(result)` —
   overwrites `advantages` and `returns` with tree Q-values
1. Returns the result with tree Q-values as advantages

**Why patch the outer method:**

- **No code duplication** — the original method handles all KL/reward/normalization
  logic
- **Correct data format** — `insert_batch` and `compute` receive `list[dict]` with 1D
  tensors as designed
- **Preserved metadata** — `kl_rewards`, `tot_rewards`, `loss_mask`, `logprobs` from the
  original method are kept for logging and other downstream uses

**Reward choice:** Tree uses raw `traj["rewards"]` for Q-value computation. The KL
penalty is a regularization term handled separately in PPO's value estimation and should
not be part of the advantage signal from MCTS perspective.

**Advantage normalization:** Tree Q-values are not additionally normalized. MCTS backup
computes running averages (`total_value / visit_count`), which provides natural
normalization. The original method's `adv_norm` is applied to GAE advantages before they
get overwritten — this is harmless.

### TreeCheckpointManager

Saves/loads the tree structure and MCTS statistics to JSON files. One file per query
tree, plus a `metadata.json` for the seq_id counter. Used in `CROSS_TRAINING` mode to
persist the tree across training runs.

### Configuration

```python
@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF  # OFF | IN_TRAINING | CROSS_TRAINING
    assistant_marker: str = ""  # auto-detected from tokenizer if empty
    checkpoint_dir: str = ""  # required for CROSS_TRAINING mode
```

- **OFF**: Standard GAE, no tree backup
- **IN_TRAINING**: Tree is built and used within a single training run, cleared between
  runs
- **CROSS_TRAINING**: Tree persists across runs via checkpoint, accumulating statistics
  over time

## Constraints

- **Backward compatible**: When mode is OFF, no patching occurs and GAE runs unchanged
- **Outer method patching**: Tree insertion happens in the patched `compute_advantages`,
  which receives `list[dict]` with 1D tensors — the natural format for `insert_batch`
  and `compute`. No format mismatches with `batched_call`
- **Thread safety**: Not yet implemented. Current code assumes single-threaded access
  (patched method runs in main training thread). Future work may add `threading.Lock`
  for async rollout insertion.
