# Fix topk_ids/teacher_logp Data Flow for Multi-Candidate Training

## Problem

When `model_inputs` contains episodes with mixed `topk_ids` (some turns with
`[[-1], [-1], ...]` sentinel, some with `[[1,2,3],[4,5,6],...]`), the data flow
for `topk_ids` and `teacher_logp` is broken in both the standard and tree
training paths. This causes distillation to silently fall back to
single-candidate mode, producing no teacher KL loss.

## Bugs

### Bug 1 — Standard path: topk_ids not split per micro-batch

In `split_padded_tensor_dict_into_mb_list`, `topk_ids` with shape
`[bs, resp_len, max_cand]` has `numel != bs * max_seqlen`, so it went into
`not_to_split`. All micro-batches got the same full-batch tensor instead of
their own subset. **Already fixed** in `areal/utils/data.py:747-755`.

### Bug 2 — Standard path: _prepare_multi_candidate_labels only handles batch_size=1

`_prepare_multi_candidate_labels` squeezes `topk_ids` only when
`shape[0] == 1`. When `mb_bs > 1` (multiple sequences packed into one
micro-batch), the squeeze fails and returns `None`, falling back to
single-candidate.

### Bug 3 — Tree path: _pack_extra_data copies full-batch topk_ids to every tree

In `_pack_extra_data`, non-packable keys like `topk_ids` and `teacher_logp`
are copied as-is from the full batch. Every tree gets `[N, resp_len, max_cand]`
instead of `[num_seqs_in_tree, resp_len, max_cand]`.

### Bug 4 — Tree path: MultiCandidateFSDPEngine tree branch is fundamentally wrong

The custom engine's `_compute_logprobs_and_loss` tree branch calls
`_prepare_multi_candidate_labels` with tree-packed logits. This produces
wrong labels because:
- `ctx.mb_input["topk_ids"]` is full-batch (Bug 3)
- `seq_len` is the packed-tree length, not per-sequence
- The resulting labels don't align with the tree-packed positions

### Bug 5 — Loss function: _compute_teacher_kl_loss assumes batched format

`_compute_teacher_kl_loss` iterates `for b in range(batch_size)` using
`teacher_logprobs.shape[0]` as batch_size. In both the standard path (with
1D packed format) and tree path, `logprobs` and `loss_mask` are 1D concatenated,
not batched. The batch iteration is wrong.

## Design

### Fix 1: Standard path — _prepare_multi_candidate_labels for mb_bs > 1

**File**: `customized_areal/tree_search/engine/fsdp_engine.py`

When `topk_ids.shape[0] > 1`, use `cu_seqlens` from `model_inputs` to locate
each sequence's position range in the 1D packed format.

Algorithm:
1. Get `cu_seqlens` from `model_inputs`
2. For each sequence `i` in `[0, mb_bs)`:
   - `start = cu_seqlens[i]`, `end = cu_seqlens[i+1]`, `seq_len_i = end - start`
   - `prompt_len_i = loss_mask[start:end].bool().argmax()`
   - Create per-sequence labels `[seq_len_i, max_cand]`: fill with
     `rolled_input_ids[start:end]`, then overwrite response positions from
     `topk_ids[i]`
3. Concatenate all per-sequence labels into `[total_len, max_cand]`

### Fix 2: Tree path — _pack_extra_data subsetting

**File**: `areal/models/tree_attn/tree.py`

For 3D+ tensors whose `shape[0]` matches the batch size, subset to
`trie.all_sequence_ids` before copying:

```python
for key in non_packable_keys:
    value = data[key]
    if (torch.is_tensor(value) and value.ndim >= 2
            and value.shape[0] == data["input_ids"].shape[0]):
        extra_data[key] = value[seq_ids]
    else:
        extra_data[key] = value
```

This produces `topk_ids_tree[num_seqs_in_tree, resp_len, max_cand]`.

### Fix 3: Tree path — MultiCandidateFSDPEngine tree branch removal

**File**: `customized_areal/tree_search/engine/fsdp_engine.py`

Remove the custom tree branch from
`MultiCandidateFSDPEngine._compute_logprobs_and_loss`. When
`enable_tree_training=True`, fall through to
`super()._compute_logprobs_and_loss()` which already correctly uses
`gather_packed_tree_logprobs_entropy` to unpack logprobs from the tree.

### Fix 4: Unified 1D path in _compute_teacher_kl_loss

**File**: `customized_areal/tree_search/training/loss.py`

Add a 1D packed-format code path using `cu_seqlens`. When `loss_mask.dim() == 1`
and `cu_seqlens` is available:

```python
cu_seqlens = input_data.get("cu_seqlens")
if loss_mask.dim() == 1 and cu_seqlens is not None:
    for b in range(len(cu_seqlens) - 1):
        start = cu_seqlens[b]
        end = cu_seqlens[b + 1]
        prompt_len = loss_mask[start:end].bool().argmax()
        resp_len = (end - start - prompt_len)
        n_pos = min(resp_len, max_resp)
        student = logprobs[start + prompt_len:start + prompt_len + n_pos, :num_cand]
        teacher = teacher_logprobs[b, :n_pos, :num_cand]
        valid = teacher.abs().sum(dim=-1) > 1e-8
        if valid.any():
            terms.append((student[valid] - teacher[valid]).reshape(-1))
```

This unified path handles both:
- Standard path with mb_bs > 1 (1D packed logprobs + cu_seqlens)
- Tree path (1D per-sequence-concatenated logprobs + cu_seqlens)

The existing batched code path (`loss_mask.dim() == 2`) remains unchanged for
backward compatibility.

### Fix 5: Ensure cu_seqlens is available in mb_input

**File**: `customized_areal/tree_search/training/loss.py`

The loss function needs `cu_seqlens` from `input_data` (= `ctx.mb_input`).

**Standard path**: After `pack_tensor_dict`, `orig_mb` already contains
`cu_seqlens [mb_bs+1]`. This maps each sequence's start/end in the 1D packed
`loss_mask` and `logprobs` tensors.

**Tree path**: The tree packing (`build_packed_tree_batch`) does NOT use
`cu_seqlens` — it uses the trie's `all_sequence_ids` and
`get_sequence_tree_indices`. After `gather_packed_tree_logprobs_entropy`,
logprobs are 1D per-sequence-concatenated, but there's no `cu_seqlens`.

We need to construct `cu_seqlens` from the tree's sequence information. In
`_pack_extra_data` (or in the tree path of `_compute_logprobs_and_loss`), after
getting the per-tree `teacher_logprobs [num_seqs, resp_len, max_cand]`, compute
`cu_seqlens` from `loss_mask` boundaries:

```python
# In the tree path, compute cu_seqlens from trie_node sequence lengths
seq_lens = [trie.get_sequence_tree_indices(sid) for sid in trie.all_sequence_ids]
# Or compute from loss_mask directly using the trie's all_sequence_ids
```

**Alternative simpler approach for tree path**: Since the tree path's
`gather_packed_tree_logprobs_entropy` already returns per-sequence-concatenated
1D logprobs (in `trie.all_sequence_ids` order), and `loss_mask` is also 1D
packed per-sequence, we can compute per-sequence boundaries from `loss_mask`:

For each sequence `b` in `[0, num_seqs_in_tree]`, the prompt_len and resp_len
can be determined from `loss_mask` segments between sequence boundaries. We need
`cu_seqlens` to find these boundaries.

**Practical approach**: Add `cu_seqlens` to the tree micro-batch's
`extra_data` in `_pack_extra_data`. Compute it from `sequence_lens`:

```python
# In _pack_extra_data, after subsetting batch-dim tensors:
if "cu_seqlens" not in extra_data and len(lens) > 0:
    seq_lens_tree = [sequence_lens[sid].item() for sid in seq_ids]
    cu = torch.cumsum(torch.tensor(seq_lens_tree, dtype=torch.int32), dim=0)
    cu = F.pad(cu, (1, 0), value=0)
    extra_data["cu_seqlens"] = cu
```

This ensures both paths have `cu_seqlens` available in `input_data`.

## Files Changed

| File | Change |
|------|--------|
| `areal/utils/data.py` | Already fixed: split 3D+ tensors along dim 0 |
| `customized_areal/tree_search/engine/fsdp_engine.py` | Fix 1, Fix 3 |
| `areal/models/tree_attn/tree.py` | Fix 2, Fix 5 (cu_seqlens) |
| `customized_areal/tree_search/training/loss.py` | Fix 4 |

## Testing

1. Unit test: Create synthetic `topk_ids` with mixed sentinel/valid data,
   verify `_prepare_multi_candidate_labels` produces correct labels for
   both `mb_bs == 1` and `mb_bs > 1`
2. Unit test: Verify `_pack_extra_data` subsets batch-dim tensors correctly
3. Unit test: Verify `_compute_teacher_kl_loss` handles 1D packed format
4. Integration test: Run tree search training with `enable_tree_training=True`
   and verify teacher KL loss is non-zero
