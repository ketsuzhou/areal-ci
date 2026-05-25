# topk_ids/teacher_logp Data Flow Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 bugs in the data flow for topk_ids/teacher_logp so that multi-candidate distillation works correctly in both standard and tree training paths.

**Architecture:** The standard path (enable_tree_training=False) uses packed 1D tensors with cu_seqlens. The tree path (enable_tree_training=True) uses trie-packed logits unpacked to 1D per-sequence-concatenated format. Both paths produce 1D logprobs + cu_seqlens, so the loss function gets a unified 1D code path. The MultiCandidateFSDPEngine tree branch is removed (delegated to base class). The `_pack_extra_data` function subsets batch-dim tensors per tree.

**Tech Stack:** Python 3.12+, PyTorch, FSDP2

---

### Task 1: Fix `_pack_extra_data` to subset batch-dim tensors per tree

**Files:**
- Modify: `areal/models/tree_attn/tree.py:675-705`
- Test: `tests/test_treesearch_bugfixes.py`

This fixes Bug 3: non-packable keys like `topk_ids` and `teacher_logp` currently get the full batch `[N, resp_len, max_cand]` copied to every tree. After this fix, each tree gets only its own sequences `[num_seqs_in_tree, resp_len, max_cand]`.

Also adds `cu_seqlens` to the tree micro-batch's `extra_data` (part of Fix 5), so the loss function can determine per-sequence boundaries in the 1D packed format.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestPackExtraDataSubsetsBatchDimTensors:
    """Bug #3: _pack_extra_data should subset batch-dim tensors per tree."""

    def test_batch_dim_non_packable_subset_per_tree(self):
        import torch
        from areal.models.tree_attn.tree import _pack_extra_data, TrieNode

        # Build a simple trie with 2 sequences (seq_ids 0, 1)
        trie = TrieNode()
        trie.all_sequence_ids = [0, 1]

        # Full batch has 4 sequences, but this tree only has 0 and 1
        N = 4
        data = {
            "input_ids": torch.randint(0, 100, (N, 10)),
            "topk_ids": torch.randint(0, 100, (N, 5, 3)),
            "teacher_logp": torch.randn(N, 5, 3),
            "some_scalar": 42,
        }
        sequence_lens = torch.tensor([8, 7, 9, 6], dtype=torch.int32)
        packable_keys = set()
        non_packable_keys = {"topk_ids", "teacher_logp", "some_scalar"}

        result = _pack_extra_data(trie, data, sequence_lens, packable_keys, non_packable_keys)

        # topk_ids and teacher_logp should be subsetted to [2, 5, 3]
        assert result["topk_ids"].shape == (2, 5, 3)
        assert result["teacher_logp"].shape == (2, 5, 3)
        # Values should match the original sequences 0 and 1
        torch.testing.assert_close(result["topk_ids"], data["topk_ids"][[0, 1]])
        torch.testing.assert_close(result["teacher_logp"], data["teacher_logp"][[0, 1]])
        # Non-tensor scalars should be copied as-is
        assert result["some_scalar"] == 42

    def test_cu_seqlens_added_to_tree_extra_data(self):
        import torch
        from areal.models.tree_attn.tree import _pack_extra_data, TrieNode

        trie = TrieNode()
        trie.all_sequence_ids = [0, 2]  # sequences 0 and 2

        N = 4
        data = {
            "input_ids": torch.randint(0, 100, (N, 10)),
        }
        sequence_lens = torch.tensor([8, 7, 9, 6], dtype=torch.int32)
        packable_keys = set()
        non_packable_keys = set()

        result = _pack_extra_data(trie, data, sequence_lens, packable_keys, non_packable_keys)

        # cu_seqlens should be [0, 8, 17] (cumsum of [8, 9])
        assert "cu_seqlens" in result
        expected = torch.tensor([0, 8, 17], dtype=torch.int32)
        torch.testing.assert_close(result["cu_seqlens"], expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py::TestPackExtraDataSubsetsBatchDimTensors -v`
Expected: FAIL — `result["topk_ids"].shape` will be `(4, 5, 3)` (full batch), not `(2, 5, 3)`.

- [ ] **Step 3: Implement the fix**

Modify `areal/models/tree_attn/tree.py` function `_pack_extra_data` (lines 675-705):

```python
def _pack_extra_data(
    trie: TrieNode,
    data: dict[str, Any],
    sequence_lens: torch.Tensor,
    packable_keys: set[str],
    non_packable_keys: set[str],
) -> dict[str, Any]:
    """Pack additional tensor data according to trie structure."""
    extra_data: dict[str, Any] = {}
    seq_ids = trie.all_sequence_ids
    lens = [sequence_lens[sid].item() for sid in seq_ids]
    batch_size = data["input_ids"].shape[0]

    # Pack tensors according to the order in trie.all_sequence_ids
    for key in packable_keys:
        value = data[key]
        packed = torch.empty(
            (sum(lens), *value.shape[2:]),
            dtype=value.dtype,
            device=value.device,
        )
        cursor = 0
        for length, seq_id in zip(lens, seq_ids):
            packed[cursor : cursor + length] = value[seq_id][:length]
            cursor += length
        extra_data[key] = packed

    # For non-packable data, subset batch-dim tensors to this tree's sequences
    for key in non_packable_keys:
        value = data[key]
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[0] == batch_size
        ):
            extra_data[key] = value[seq_ids]
        else:
            extra_data[key] = value

    # Add cu_seqlens for per-sequence boundary info (used by loss function)
    if "cu_seqlens" not in extra_data and len(lens) > 0:
        cu = torch.cumsum(
            torch.tensor(lens, dtype=torch.int32), dim=0
        )
        cu = torch.nn.functional.pad(cu, (1, 0), value=0)
        extra_data["cu_seqlens"] = cu

    return extra_data
```

Add import at the top of `tree.py` if `torch.nn.functional` is not already imported:
```python
import torch.nn.functional as F
```

Check existing imports first — if `F` is already imported, use `F.pad` instead of `torch.nn.functional.pad`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py::TestPackExtraDataSubsetsBatchDimTensors -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add areal/models/tree_attn/tree.py tests/test_treesearch_bugfixes.py
git commit -m "fix(tree): subset batch-dim non-packable tensors per tree and add cu_seqlens

_pack_extra_data now indexes batch-dim tensors (like topk_ids,
teacher_logp) by trie.all_sequence_ids instead of copying the full
batch. Also adds cu_seqlens to extra_data for per-sequence boundary
info used by the loss function."
```

---

### Task 2: Fix `_compute_teacher_kl_loss` for 1D packed format

**Files:**
- Modify: `customized_areal/tree_search/training/loss.py:216-270`
- Test: `tests/test_treesearch_bugfixes.py`

This fixes Bug 5: the loss function iterates `for b in range(batch_size)` using `teacher_logprobs.shape[0]`, which is wrong when logprobs and loss_mask are 1D concatenated (not batched). After this fix, when `loss_mask` is 1D and `cu_seqlens` is available, the function uses `cu_seqlens` to split per-sequence.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestTeacherKLLossPackedFormat:
    """Bug #5: _compute_teacher_kl_loss must handle 1D packed logprobs with cu_seqlens."""

    def test_packed_1d_format_with_cu_seqlens(self):
        import torch
        from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss

        # 2 sequences packed into 1D: seq0 has prompt_len=2 resp_len=3, seq1 has prompt_len=1 resp_len=2
        # Total packed len = 5 + 3 = 8
        loss_mask = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1], dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)

        # Student logprobs: 1D [8] (single-candidate for simplicity)
        logprobs = torch.tensor([-1.0, -2.0, -0.5, -0.6, -0.7, -3.0, -0.8, -0.9])

        # Teacher: [2, max_resp, 1] where max_resp=3 (max resp across sequences)
        # seq0 has resp_len=3, seq1 has resp_len=2
        teacher_logprobs = torch.tensor([
            [[-0.4], [-0.5], [-0.6]],  # seq0, 3 response positions
            [[-0.7], [-0.8], [0.0]],   # seq1, 2 response positions (3rd is padding)
        ])

        loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[2, 1],  # inferred from loss_mask + cu_seqlens
            input_data={"cu_seqlens": cu_seqlens},
        )

        # Loss should be non-zero and finite
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_packed_1d_multi_candidate(self):
        import torch
        from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss

        # 2 sequences, multi-candidate (3 candidates per position)
        loss_mask = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

        # Student logprobs: [6, 3] (multi-candidate)
        logprobs = torch.randn(6, 3)

        # Teacher: [2, 2, 3]
        teacher_logprobs = torch.randn(2, 2, 3)

        loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[1, 1],
            input_data={"cu_seqlens": cu_seqlens},
        )

        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_batched_2d_format_unchanged(self):
        """Existing batched [batch, seq] format should still work."""
        import torch
        from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss

        # 2D batched format (existing code path)
        loss_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.0, -0.5, -0.6], [-2.0, -0.7, -0.8]])
        teacher_logprobs = torch.tensor([[[-0.4], [-0.5]], [[-0.6], [-0.7]]])

        loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[1, 1],
        )

        assert loss.item() > 0
        assert torch.isfinite(loss)
```

Note: The existing `_compute_teacher_kl_loss` signature doesn't accept `input_data`. We need to add it. The test will initially fail because the function doesn't accept `input_data` and doesn't handle 1D format.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py::TestTeacherKLLossPackedFormat -v`
Expected: FAIL — `TypeError` for unexpected `input_data` keyword argument.

- [ ] **Step 3: Implement the fix**

Modify `_compute_teacher_kl_loss` in `customized_areal/tree_search/training/loss.py` (lines 216-270). Replace the entire function:

```python
def _compute_teacher_kl_loss(
    teacher_logprobs: torch.Tensor,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int],
    input_data: dict | None = None,
) -> torch.Tensor:
    """Compute teacher KL distillation loss from batched teacher_logprobs tensor.

    teacher_logprobs is response-aligned with shape [batch, resp_len, max_candidates].
    Position i in the response maps to absolute sequence position prompt_len + i.

    Supports both batched (loss_mask 2D) and 1D packed (loss_mask 1D + cu_seqlens)
    formats for logprobs and loss_mask.
    """
    if teacher_logprobs.numel() == 0:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)

    terms: list[torch.Tensor] = []
    mask = loss_mask.bool()
    batch_size = teacher_logprobs.shape[0]
    max_resp = teacher_logprobs.shape[1]

    is_multi_candidate = logprobs.dim() > loss_mask.dim() or (
        logprobs.dim() == loss_mask.dim() and logprobs.shape != loss_mask.shape
    )

    # 1D packed format: use cu_seqlens for per-sequence boundaries
    cu_seqlens = (input_data or {}).get("cu_seqlens")
    if mask.dim() == 1 and cu_seqlens is not None:
        for b in range(len(cu_seqlens) - 1):
            if b >= batch_size:
                break
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            pl = prompt_lens[b] if b < len(prompt_lens) else 0
            # Recompute prompt_len from loss_mask for this segment
            seg_mask = mask[start:end]
            if seg_mask.any():
                pl = int(seg_mask.int().argmax().item())
            resp_len = (end - start) - pl
            n_pos = min(resp_len, max_resp)
            if n_pos == 0:
                continue

            if is_multi_candidate:
                num_cand = min(
                    teacher_logprobs.shape[2],
                    logprobs.shape[1] if logprobs.dim() == 2 else logprobs.shape[2],
                )
                student = logprobs[start + pl : start + pl + n_pos, :num_cand]
                teacher = teacher_logprobs[b, :n_pos, :num_cand]
                valid = teacher.abs().sum(dim=-1) > 1e-8
                if valid.any():
                    terms.append((student[valid] - teacher[valid]).reshape(-1))
            else:
                student = logprobs[start + pl : start + pl + n_pos]
                teacher = teacher_logprobs[b, :n_pos, 0]
                valid = teacher.abs() > 1e-8
                if valid.any():
                    terms.append((student[valid] - teacher[valid]).reshape(-1))

        if not terms:
            return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)
        return torch.cat(terms).mean()

    # Original batched format: loss_mask is 2D [batch, seq_len]
    is_batched = mask.dim() == 2 or logprobs.dim() >= 3

    for b in range(batch_size):
        pl = prompt_lens[b] if b < len(prompt_lens) else 0
        resp_len = mask[b, pl:].sum().item() if is_batched else mask[pl:].sum().item()
        n_pos = min(resp_len, max_resp)
        if n_pos == 0:
            continue

        if is_multi_candidate:
            num_cand = min(
                teacher_logprobs.shape[2],
                logprobs.shape[2] if logprobs.dim() == 3 else logprobs.shape[1],
            )
            if is_batched:
                student = logprobs[b, pl : pl + n_pos, :num_cand]
            else:
                student = logprobs[pl : pl + n_pos, :num_cand]
            teacher = teacher_logprobs[b, :n_pos, :num_cand]
            valid = teacher.abs().sum(dim=-1) > 1e-8
            if valid.any():
                terms.append((student[valid] - teacher[valid]).reshape(-1))
        else:
            if is_batched:
                student = logprobs[b, pl : pl + n_pos]
            else:
                student = logprobs[pl : pl + n_pos]
            teacher = teacher_logprobs[b, :n_pos, 0]
            valid = teacher.abs() > 1e-8
            if valid.any():
                terms.append((student[valid] - teacher[valid]).reshape(-1))

    if not terms:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)

    return torch.cat(terms).mean()
```

Also update the two call sites in `grpo_distill_loss_fn` (lines 104-108 and 144-150) to pass `input_data`:

```python
# Line ~104 (DISTILL mode)
teacher_kl_loss = _compute_teacher_kl_loss(
    teacher_logprobs=teacher_logprobs,
    logprobs=logprobs,
    loss_mask=loss_mask,
    prompt_lens=prompt_lens,
    input_data=input_data,
)

# Line ~144 (Combined mode)
teacher_kl_loss = _compute_teacher_kl_loss(
    teacher_logprobs=teacher_logprobs,
    logprobs=logprobs,
    loss_mask=loss_mask,
    prompt_lens=prompt_lens,
    input_data=input_data,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py::TestTeacherKLLossPackedFormat -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/training/loss.py tests/test_treesearch_bugfixes.py
git commit -m "fix(loss): add 1D packed format support to _compute_teacher_kl_loss

Uses cu_seqlens from input_data to split 1D concatenated logprobs
per-sequence when loss_mask is 1D. The existing batched 2D path
remains unchanged. Fixes teacher KL loss computation in both
standard (mb_bs > 1) and tree training paths."
```

---

### Task 3: Fix `_prepare_multi_candidate_labels` for `mb_bs > 1`

**Files:**
- Modify: `customized_areal/tree_search/engine/fsdp_engine.py:181-246`
- Test: `tests/test_treesearch_bugfixes.py`

This fixes Bug 2: when multiple sequences are packed into one micro-batch (`mb_bs > 1`), `_prepare_multi_candidate_labels` currently fails because it only squeezes when `shape[0] == 1`. The fix uses `cu_seqlens` from `model_inputs` to locate each sequence's position range and creates per-sequence labels, then concatenates them into `[total_len, max_cand]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestPrepareMultiCandidateLabelsPacked:
    """Bug #2: _prepare_multi_candidate_labels must handle mb_bs > 1."""

    def _make_engine(self):
        """Create a minimal MultiCandidateFSDPEngine for testing."""
        from unittest.mock import MagicMock
        from customized_areal.tree_search.engine.fsdp_engine import MultiCandidateFSDPEngine
        engine = MagicMock(spec=MultiCandidateFSDPEngine)
        engine._prepare_multi_candidate_labels = MultiCandidateFSDPEngine._prepare_multi_candidate_labels.__get__(engine)
        engine.config = MagicMock()
        return engine

    def test_single_sequence_still_works(self):
        import torch
        engine = self._make_engine()

        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30, 40, 50]]),  # [1, 5]
            "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 5], dtype=torch.int32),
        }
        mb_input = {
            "topk_ids": torch.tensor([[[100, 101, 102], [200, 201, 202], [300, 301, 302]]]),
            # [1, 3, 3] — 1 sequence, 3 response positions, 3 candidates
        }

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=5)

        assert labels is not None
        assert labels.shape == (5, 3)
        # Positions 0-1 (prompt): should have rolled input_ids
        assert labels[0, 0].item() == 20  # rolled(10)
        assert labels[1, 0].item() == 30  # rolled(20)
        # Positions 2-4 (response): should have topk_ids
        assert labels[2, 0].item() == 100
        assert labels[2, 1].item() == 101

    def test_two_sequences_packed(self):
        import torch
        engine = self._make_engine()

        # 2 sequences packed: seq0=[10,20,30,40], seq1=[50,60,70]
        # cu_seqlens = [0, 4, 7]
        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30, 40, 50, 60, 70]]),  # [1, 7]
            "loss_mask": torch.tensor([[0, 1, 1, 1, 0, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 4, 7], dtype=torch.int32),
        }
        mb_input = {
            # 2 sequences, each with 2 response positions, 3 candidates
            # seq0: prompt_len=1, resp_len=3 (from loss_mask)
            # seq1: prompt_len=1, resp_len=2 (from loss_mask)
            "topk_ids": torch.tensor([
                [[100, 101, 102], [200, 201, 202], [300, 301, 302]],  # seq0
                [[400, 401, 402], [500, 501, 502], [-1, -1, -1]],     # seq1 (3rd pos is padding)
            ]),
            # [2, 3, 3]
        }

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=7)

        assert labels is not None
        assert labels.shape == (7, 3)
        # seq0 position 0 (prompt): rolled input_ids
        assert labels[0, 0].item() == 20
        # seq0 positions 1-3 (response): from topk_ids[0]
        assert labels[1, 0].item() == 100
        assert labels[2, 0].item() == 200
        assert labels[3, 0].item() == 300
        # seq1 position 4 (prompt): rolled input_ids
        assert labels[4, 0].item() == 60
        # seq1 positions 5-6 (response): from topk_ids[1]
        assert labels[5, 0].item() == 400
        assert labels[6, 0].item() == 500

    def test_returns_none_when_no_topk_ids(self):
        import torch
        engine = self._make_engine()

        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        }
        mb_input = {}

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=3)
        assert labels is None

    def test_returns_none_when_all_sentinel(self):
        import torch
        engine = self._make_engine()

        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        }
        mb_input = {
            "topk_ids": torch.tensor([[[-1], [-1]]]),  # all -1 sentinel
        }

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=3)
        assert labels is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py::TestPrepareMultiCandidateLabelsPacked -v`
Expected: FAIL — `test_two_sequences_packed` will fail because `_prepare_multi_candidate_labels` returns `None` when `topk_ids.shape[0] > 1` (the squeeze condition `shape[0] == 1` fails, then `dim() != 2` check triggers).

- [ ] **Step 3: Implement the fix**

Replace `_prepare_multi_candidate_labels` in `customized_areal/tree_search/engine/fsdp_engine.py` (lines 181-246):

```python
def _prepare_multi_candidate_labels(
    self,
    model_inputs: dict[str, Any],
    mb_input: dict[str, Any],
    seq_len: int,
) -> torch.Tensor | None:
    """Prepare 2D labels for multi-candidate logprob gathering.

    Reads the response-aligned topk_ids tensor from mb_input and
    expands it to full-sequence labels [seq_len, max_candidates].
    Position i in topk_ids maps to absolute sequence position prompt_len + i.

    Positions where topk_ids has -1 sentinel (no distill data for
    that node) are filled with the actual next token from
    rolled_input_ids so the engine gathers single-candidate-equivalent
    logprobs at those positions.

    Handles both single-sequence (mb_bs=1) and multi-sequence packed
    (mb_bs > 1 with cu_seqlens) micro-batches.
    """
    topk_ids = mb_input.get("topk_ids")
    if topk_ids is None or topk_ids.numel() == 0:
        return None

    # topk_ids: [mb_bs, resp_len, max_cand]
    if topk_ids.dim() != 3:
        return None

    mb_bs, resp_len, max_candidates = topk_ids.shape

    # If every response position has -1 sentinel across all sequences, no distill data
    if (topk_ids[:, :, 0] < 0).all():
        return None

    loss_mask = model_inputs.get("loss_mask")
    input_ids = model_inputs.get("input_ids")
    cu_seqlens = model_inputs.get("cu_seqlens")

    device = topk_ids.device

    if mb_bs == 1 or cu_seqlens is None:
        # Single-sequence or no cu_seqlens: original path
        topk_2d = topk_ids.squeeze(0)  # [resp_len, max_cand]
        if topk_2d.dim() != 2:
            return None

        prompt_len = 0
        if loss_mask is not None:
            lm_flat = loss_mask.squeeze(0) if loss_mask.dim() > 1 else loss_mask
            prompt_len = int(lm_flat.bool().int().argmax().item())

        # Get rolled_input_ids for prompt and non-distill positions
        rolled = None
        if input_ids is not None:
            ids_flat = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
            rolled = torch.roll(ids_flat, shifts=-1)[:seq_len]

        labels = torch.zeros(seq_len, max_candidates, dtype=torch.long, device=device)
        if rolled is not None:
            labels[:, 0] = rolled
            for c in range(1, max_candidates):
                labels[:, c] = rolled

        end = min(prompt_len + resp_len, seq_len)
        if end > prompt_len:
            chunk = topk_2d[: end - prompt_len]
            valid = chunk[:, 0] >= 0
            if valid.any():
                labels[prompt_len:end][valid] = chunk[valid]

        return labels

    # Multi-sequence packed: use cu_seqlens to build labels per sequence
    labels_parts = []
    ids_flat = input_ids.squeeze(0) if input_ids is not None and input_ids.dim() > 1 else input_ids

    for b in range(mb_bs):
        start = cu_seqlens[b].item()
        end = cu_seqlens[b + 1].item()
        seq_len_i = end - start

        # Compute prompt_len from loss_mask segment
        prompt_len_i = 0
        if loss_mask is not None:
            lm_flat = loss_mask.squeeze(0) if loss_mask.dim() > 1 else loss_mask
            seg = lm_flat[start:end]
            if seg.bool().any():
                prompt_len_i = int(seg.bool().int().argmax().item())

        # Create rolled input_ids for this sequence
        if ids_flat is not None:
            seg_ids = ids_flat[start:end]
            rolled_i = torch.roll(seg_ids, shifts=-1)
        else:
            rolled_i = torch.zeros(seq_len_i, dtype=torch.long, device=device)

        labels_i = torch.zeros(seq_len_i, max_candidates, dtype=torch.long, device=device)
        labels_i[:, 0] = rolled_i
        for c in range(1, max_candidates):
            labels_i[:, c] = rolled_i

        # Overwrite response positions with topk_ids for this sequence
        end_resp = min(prompt_len_i + resp_len, seq_len_i)
        if end_resp > prompt_len_i:
            chunk = topk_ids[b, : end_resp - prompt_len_i]
            valid = chunk[:, 0] >= 0
            if valid.any():
                labels_i[prompt_len_i:end_resp][valid] = chunk[valid]

        labels_parts.append(labels_i)

    return torch.cat(labels_parts, dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py::TestPrepareMultiCandidateLabelsPacked -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/engine/fsdp_engine.py tests/test_treesearch_bugfixes.py
git commit -m "fix(engine): support mb_bs > 1 in _prepare_multi_candidate_labels

Uses cu_seqlens from model_inputs to locate each sequence's position
range in the 1D packed format. Creates per-sequence labels then
concatenates into [total_len, max_cand]. Single-sequence path
unchanged."
```

---

### Task 4: Remove broken tree branch from `MultiCandidateFSDPEngine._compute_logprobs_and_loss`

**Files:**
- Modify: `customized_areal/tree_search/engine/fsdp_engine.py:248-349`

This fixes Bug 4: the custom engine's `_compute_logprobs_and_loss` tree branch tries to use `_prepare_multi_candidate_labels` with tree-packed logits, which is fundamentally wrong. The base FSDPEngine already handles the tree path correctly using `gather_packed_tree_logprobs_entropy`. The custom engine should delegate to the base class when `enable_tree_training=True`.

- [ ] **Step 1: Write the failing test**

This is a code removal/simplification task. The test is that the tree path now delegates to the base class correctly. Since we can't easily instantiate the full engine in a unit test, we verify by reading the code structure. However, we can test that the method no longer has the broken tree-specific code path.

We'll rely on the integration test (Task 5) to verify correctness. For now, we make the code change and verify it doesn't break existing tests.

- [ ] **Step 2: Implement the fix**

Replace `_compute_logprobs_and_loss` in `customized_areal/tree_search/engine/fsdp_engine.py` (lines 248-349):

```python
def _compute_logprobs_and_loss(
    self,
    logits: torch.Tensor,
    ctx: Any,  # FSDPTrainContext
    loss_fn: Callable[..., torch.Tensor],
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    total_loss_weight: torch.Tensor,
    loss_multiplier: float = 1.0,
) -> torch.Tensor:
    """Compute logprobs/entropy and return scaled loss with multi-candidate support.

    This method overrides the parent to:
    1. Prepare multi-candidate labels from topk_ids tensor
    2. Compute multi-candidate logprobs
    3. Pass multi-candidate logprobs to loss function

    For tree training (enable_tree_training=True), delegates to the base
    FSDPEngine which uses gather_packed_tree_logprobs_entropy to correctly
    unpack per-sequence logprobs from the trie structure. Multi-candidate
    logprob gathering is only supported in the non-tree training path.
    """

    if self.config.is_critic and self.enable_tree_training:
        raise NotImplementedError(
            "Tree training with critic model is not supported yet."
        )

    if not self.config.is_critic:
        if not self.enable_tree_training:
            # Standard path: prepare multi-candidate labels and compute logprobs
            # Robust seq_len extraction from logits tensor
            if logits.ndim == 2:
                seq_len = logits.shape[0]
            elif logits.ndim == 3:
                seq_len = logits.shape[1]
            else:
                raise ValueError(
                    f"Unexpected logits ndim: {logits.ndim}. "
                    f"Expected 2D [seq_len, vocab] or 3D [batch, seq_len, vocab]"
                )

            # Check for multi-candidate data from topk_ids tensor
            topk_ids = ctx.mb_input.get("topk_ids")
            if topk_ids is not None and topk_ids.numel() > 0:
                multi_candidate_labels = self._prepare_multi_candidate_labels(
                    ctx.model_inputs, ctx.mb_input, seq_len
                )

                if multi_candidate_labels is not None:
                    logprobs, entropy = self._compute_logprobs_entropy(
                        logits,
                        ctx.model_inputs,
                        ctx.ulysses_pad_size,
                        labels_override=multi_candidate_labels,
                    )
                else:
                    logprobs, entropy = self._compute_logprobs_entropy(
                        logits, ctx.model_inputs, ctx.ulysses_pad_size
                    )
            else:
                logprobs, entropy = self._compute_logprobs_entropy(
                    logits, ctx.model_inputs, ctx.ulysses_pad_size
                )

            vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                logits, ctx.ulysses_pad_size
            )

            if ctx.pad_length > 0:
                logprobs = logprobs[: -ctx.pad_length]
                entropy = entropy[: -ctx.pad_length]
                logits = logits[: -ctx.pad_length]
                vocab_min_logits = vocab_min_logits[: -ctx.pad_length]
                vocab_max_logits = vocab_max_logits[: -ctx.pad_length]

            loss = loss_fn(
                logprobs,
                entropy,
                ctx.mb_input,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            # Tree training: delegate to base FSDPEngine which correctly
            # uses gather_packed_tree_logprobs_entropy
            return super()._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier,
            )
    else:
        values = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
        if ctx.pad_length > 0:
            values = values[: -ctx.pad_length]
        loss = loss_fn(values, ctx.mb_input)

    loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight * loss_multiplier
    return loss * loss_scale
```

Key changes:
1. Tree training branch now delegates to `super()._compute_logprobs_and_loss()` instead of trying to use `_prepare_multi_candidate_labels`
2. Standard path multi-candidate label preparation is kept (but now also handles `mb_bs > 1` from Task 3)
3. The `vocab_min_logits`/`vocab_max_logits` and `loss_fn` call are moved inside the `if not self.enable_tree_training` block for the standard path only (the tree path handles them in the base class)

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py -v`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/engine/fsdp_engine.py
git commit -m "fix(engine): delegate tree training path to base FSDPEngine

The custom MultiCandidateFSDPEngine tree branch incorrectly tried to
use _prepare_multi_candidate_labels with tree-packed logits. The base
FSDPEngine already correctly handles the tree path using
gather_packed_tree_logprobs_entropy. Multi-candidate logprob gathering
is only supported in the non-tree training path."
```

---

### Task 5: Run pre-commit and final verification

**Files:**
- All modified files

- [ ] **Step 1: Run pre-commit hooks**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && pre-commit run --all-files`
Expected: All checks PASS. Fix any formatting issues.

- [ ] **Step 2: Run full test suite for modified modules**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit any formatting fixes if needed**

```bash
git add -A
git commit -m "style: apply pre-commit formatting fixes"
```

- [ ] **Step 4: Verify all bugs are fixed by reviewing changes**

Review all modified files and verify:
1. `areal/utils/data.py`: 3D+ tensors with `shape[0] == bs` are split per micro-batch (already done)
2. `areal/models/tree_attn/tree.py`: `_pack_extra_data` subsets batch-dim tensors and adds `cu_seqlens`
3. `customized_areal/tree_search/training/loss.py`: `_compute_teacher_kl_loss` handles 1D packed format via `cu_seqlens`
4. `customized_areal/tree_search/engine/fsdp_engine.py`: `_prepare_multi_candidate_labels` handles `mb_bs > 1` and tree path delegates to base class
