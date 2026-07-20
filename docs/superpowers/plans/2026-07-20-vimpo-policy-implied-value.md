---
change: add-vimpo-critic-mode
design-doc: docs/superpowers/specs/2026-07-20-vimpo-policy-implied-value-design.md
base-ref: c95b9c210a8fd77b7c4b6a2eb1fe1632bd0db8e8
---

# VIMPO Policy-Implied Value Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `VIMPO` advantage mode that trains one FSDP actor with the paper's policy-implied terminal value objective while scoring a frozen initial-policy reference through SGLang with configurable actor-selected top-k candidate KL.

**Architecture:** The rollout workflow emits episode identity and centered terminal targets but no learned-critic values. A dedicated FSDP PPO actor snapshots sampled-token and actor-selected candidate probabilities, obtains matching full-softmax reference scores from a frozen SGLang service, computes detached reverse-lambda advantages, and performs one combined PPO/value backward pass over episode-atomic microbatches.

**Tech Stack:** Python 3.12, PyTorch/FSDP2, `torch.distributed`, AReaL PPO and tree-search APIs, SGLang native HTTP API, `httpx`, pytest.

## Global Constraints

- The fixed reference is the actor's initial checkpoint `pi_ref = pi_0`; it is never updated or checkpointed by this trainer.
- Only the FSDP actor backend is supported in version 1; Megatron, Archon, and learned/generative critic paths fail validation.
- `vimpo_top_k` is configurable and uses actor-selected candidates with full-vocabulary softmax normalization; candidates are never renormalized.
- Candidate KL is exact only when effective `k == vocab_size`; all smaller values are reported as truncated candidate KL with retained mass.
- Reference tensors, candidate KL, old/proximal log-probabilities, terminal targets, and advantages are detached.
- Multi-turn episodes are indivisible at outer PPO-minibatch and inner FSDP-microbatch boundaries.
- PPO token means and terminal episode means use separate distributed denominators in one backward/optimizer step.
- VIMPO does not require `actor.kl_ctl > 0`, does not create an FSDP reference, and does not attach `critic_train_data`.
- Do not modify `areal/api/cli_args.py`, add dependencies, or change launcher/scheduler allocation logic.
- Preserve all existing TREE, GAE, HYBRID_GAE, VERSIONED_BACKUP, distillation, clip-cov, Muon, and generative-critic behavior.

---

## File Structure

- Modify `customized_areal/tree_search/config.py`: enum, VIMPO defaults, and mode-specific validation.
- Modify `customized_areal/tree_search/core/advantage.py`: candidate KL, episode reverse-lambda scan, distributed masked whitening, and `VIMPOAdvantageComputer`.
- Modify `customized_areal/tree_search/core/tree_store.py`: stable VIMPO fields on `Node` and sequence-shaped tensorization.
- Modify `customized_areal/tree_search/core/customized_grouped_workflow.py`: query-local centered reward metadata and VIMPO dispatch that bypasses critic paths.
- Modify `customized_areal/tree_search/engine/fsdp_engine.py`: no-grad actor candidate statistics and the VIMPO two-denominator training primitive.
- Modify `customized_areal/tree_search/engine/__init__.py`: lazy export for the VIMPO actor.
- Create `customized_areal/tree_search/training/vimpo_reference.py`: frozen SGLang identity and candidate-score adapter.
- Create `customized_areal/tree_search/training/vimpo_batching.py`: episode-atomic padded batch allocator.
- Create `customized_areal/tree_search/training/losses/vimpo.py`: pure PPO and terminal-value numerators plus metrics.
- Modify `customized_areal/tree_search/training/losses/__init__.py`: VIMPO loss exports.
- Modify `customized_areal/tree_search/training/actor.py`: `VIMPOFSDPPPOActor` orchestration and lifecycle.
- Modify `customized_areal/tree_search/training/trainer.py`: VIMPO engine selection and actor-config propagation.
- Modify `customized_areal/tree_search/__init__.py`: lazy public exports.
- Modify `customized_areal/tree_search/README.md`: configuration and deployment contract.
- Create focused tests under `customized_areal/tree_search/tests/`, plus one hardware-gated integration test.

### Task 1: Configuration Contract and Pure VIMPO Mathematics

**Files:**
- Modify: `customized_areal/tree_search/config.py`
- Modify: `customized_areal/tree_search/core/advantage.py`
- Modify: `customized_areal/tree_search/__init__.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_config.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_advantage.py`

**Interfaces:**
- Consumes: existing `Config`, `AdvantageMode`, and explicit actor data-parallel process groups.
- Produces: `AdvantageMode.VIMPO`; the exact config fields in the design; `candidate_forward_kl`, `masked_episode_reverse_lambda`, `masked_distributed_whiten`, and `VIMPOAdvantageComputer.compute`.

- [x] **Step 1: Write failing configuration tests**

```python
# customized_areal/tree_search/tests/test_vimpo_config.py
import pytest

from customized_areal.tree_search.config import AdvantageMode, Config, LossMode


def test_vimpo_defaults_and_string_round_trip() -> None:
    cfg = Config(advantage_mode="vimpo", vimpo_ref_base_url="http://ref:30000")
    assert cfg.advantage_mode is AdvantageMode.VIMPO
    assert cfg.loss_mode is LossMode.GRPO
    assert cfg.vimpo_beta == pytest.approx(5e-4)
    assert cfg.vimpo_actor_coeff == pytest.approx(5e-3)
    assert cfg.vimpo_value_loss_weight == pytest.approx(1.0)
    assert cfg.vimpo_gamma == pytest.approx(1.0)
    assert cfg.vimpo_lambda == pytest.approx(1.0)
    assert cfg.vimpo_top_k == 128
    assert cfg.vimpo_whiten_advantages is True
    assert cfg.vimpo_detach_kl is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vimpo_beta": 0.0}, "vimpo_beta must be finite and > 0"),
        ({"vimpo_actor_coeff": -1.0}, "vimpo_actor_coeff must be finite and >= 0"),
        ({"vimpo_value_loss_weight": -1.0}, "vimpo_value_loss_weight must be finite and >= 0"),
        ({"vimpo_actor_coeff": 0.0, "vimpo_value_loss_weight": 0.0}, "at least one VIMPO loss coefficient"),
        ({"vimpo_gamma": 0.99}, "vimpo_gamma must equal 1.0"),
        ({"vimpo_lambda": 1.1}, "vimpo_lambda must be in [0, 1]"),
        ({"vimpo_top_k": 0}, "vimpo_top_k must be > 0"),
        ({"vimpo_detach_kl": False}, "vimpo_detach_kl=False is not supported"),
        ({"vimpo_ref_base_url": ""}, "vimpo_ref_base_url is required"),
        ({"enable_generative_critic": True}, "incompatible with enable_generative_critic"),
        ({"use_clip_cov": True}, "incompatible with use_clip_cov"),
        ({"loss_mode": LossMode.DISTILL}, "requires loss_mode='grpo'"),
    ],
)
def test_vimpo_rejects_invalid_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Config(
            advantage_mode=AdvantageMode.VIMPO,
            vimpo_ref_base_url="http://ref:30000",
            **kwargs,
        )


def test_non_vimpo_mode_does_not_require_reference_url() -> None:
    assert Config().advantage_mode is AdvantageMode.TREE
```

- [x] **Step 2: Run configuration tests and verify the new enum/fields are absent**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_config.py -q`

Expected: FAIL during collection or construction because `AdvantageMode.VIMPO` and the `vimpo_*` fields do not exist.

- [x] **Step 3: Add the enum, fields, and mode-gated validation**

```python
# customized_areal/tree_search/config.py
import math

class AdvantageMode(str, Enum):
    GAE = "gae"
    TREE = "tree"
    HYBRID_GAE = "hybrid_gae"
    VERSIONED_BACKUP = "versioned_backup"
    VIMPO = "vimpo"

# Add after loss_mode in Config.
vimpo_beta: float = 5e-4
vimpo_actor_coeff: float = 5e-3
vimpo_value_loss_weight: float = 1.0
vimpo_gamma: float = 1.0
vimpo_lambda: float = 1.0
vimpo_top_k: int = 128
vimpo_whiten_advantages: bool = True
vimpo_detach_kl: bool = True
vimpo_ref_base_url: str = ""
vimpo_ref_timeout: float = 300.0
vimpo_ref_max_concurrency: int = 8
vimpo_ref_max_retries: int = 3

# Add after self.advantage_mode = AdvantageMode(self.advantage_mode).
if self.advantage_mode is AdvantageMode.VIMPO:
    if not math.isfinite(self.vimpo_beta) or self.vimpo_beta <= 0:
        raise ValueError("vimpo_beta must be finite and > 0")
    if not math.isfinite(self.vimpo_actor_coeff) or self.vimpo_actor_coeff < 0:
        raise ValueError("vimpo_actor_coeff must be finite and >= 0")
    if not math.isfinite(self.vimpo_value_loss_weight) or self.vimpo_value_loss_weight < 0:
        raise ValueError("vimpo_value_loss_weight must be finite and >= 0")
    if self.vimpo_actor_coeff == 0 and self.vimpo_value_loss_weight == 0:
        raise ValueError("at least one VIMPO loss coefficient must be positive")
    if self.vimpo_gamma != 1.0:
        raise ValueError("vimpo_gamma must equal 1.0 in VIMPO version 1")
    if not math.isfinite(self.vimpo_lambda) or not 0 <= self.vimpo_lambda <= 1:
        raise ValueError("vimpo_lambda must be in [0, 1]")
    if self.vimpo_top_k <= 0:
        raise ValueError("vimpo_top_k must be > 0")
    if not self.vimpo_detach_kl:
        raise ValueError("vimpo_detach_kl=False is not supported in VIMPO version 1")
    if not self.vimpo_ref_base_url.startswith(("http://", "https://")):
        raise ValueError("vimpo_ref_base_url is required and must be HTTP(S)")
    if not math.isfinite(self.vimpo_ref_timeout) or self.vimpo_ref_timeout <= 0:
        raise ValueError("vimpo_ref_timeout must be finite and > 0")
    if self.vimpo_ref_max_concurrency <= 0:
        raise ValueError("vimpo_ref_max_concurrency must be > 0")
    if self.vimpo_ref_max_retries < 0:
        raise ValueError("vimpo_ref_max_retries must be >= 0")
    if self.loss_mode is not LossMode.GRPO:
        raise ValueError("VIMPO requires loss_mode='grpo'")
    if self.enable_generative_critic:
        raise ValueError("VIMPO is incompatible with enable_generative_critic")
    if self.use_clip_cov:
        raise ValueError("VIMPO is incompatible with use_clip_cov")
```

- [x] **Step 4: Write failing pure-math and gradient-isolation tests**

```python
# customized_areal/tree_search/tests/test_vimpo_advantage.py
import torch

from customized_areal.tree_search.core.advantage import (
    VIMPOAdvantageComputer,
    candidate_forward_kl,
    masked_episode_reverse_lambda,
    masked_distributed_whiten,
)


def test_candidate_kl_is_unrenormalized_and_exact_at_full_vocab() -> None:
    policy = torch.log(torch.tensor([[[0.50, 0.30, 0.20]]]))
    reference = torch.log(torch.tensor([[[0.25, 0.25, 0.50]]]))
    mask = torch.tensor([[True]])
    kl, mass = candidate_forward_kl(policy, reference, mask)
    expected = (policy.exp() * (policy - reference)).sum(-1)
    torch.testing.assert_close(kl, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(mass, torch.ones_like(mass), rtol=0, atol=1e-6)


def test_candidate_kl_keeps_missing_probability_mass_outside_topk() -> None:
    policy = torch.log(torch.tensor([[[0.50, 0.30]]]))
    reference = torch.log(torch.tensor([[[0.25, 0.25]]]))
    kl, mass = candidate_forward_kl(policy, reference, torch.tensor([[True]]))
    torch.testing.assert_close(mass, torch.tensor([[0.80]]), rtol=0, atol=1e-6)
    torch.testing.assert_close(
        kl,
        torch.tensor([[0.50 * torch.log(torch.tensor(2.0)) + 0.30 * torch.log(torch.tensor(1.2))]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_reverse_lambda_crosses_turns_and_resets_between_episodes() -> None:
    td = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, False], [True, False, False]])
    episode = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    turn = torch.tensor([[1, 1, 1], [2, 2, 2], [1, 1, 1]])
    actual = masked_episode_reverse_lambda(td, mask, episode, turn, gamma=1.0, lam=0.5)
    expected = torch.tensor([[2.75, 3.50, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_whitening_one_token_returns_zero_and_detaches() -> None:
    values = torch.tensor([[4.0, 0.0]], requires_grad=True)
    actual = masked_distributed_whiten(values, torch.tensor([[True, False]]), group=None)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    assert actual.requires_grad is False


def test_vimpo_computer_detaches_kl_reference_and_advantage() -> None:
    policy = torch.log(torch.tensor([[[0.6, 0.3]]], requires_grad=True))
    reference = torch.log(torch.tensor([[[0.4, 0.2]]], requires_grad=True))
    batch = {
        "vimpo_predict_mask": torch.tensor([[True]]),
        "vimpo_sample_logp": torch.tensor([[-0.2]], requires_grad=True),
        "vimpo_ref_sample_logp": torch.tensor([[-0.4]], requires_grad=True),
        "vimpo_candidate_logp": policy,
        "vimpo_ref_candidate_logp": reference,
        "vimpo_episode_index": torch.tensor([[0]]),
        "vimpo_turn_index": torch.tensor([[1]]),
    }
    out = VIMPOAdvantageComputer(beta=0.5, gamma=1.0, lam=1.0, whiten=False).compute(batch)
    assert out["vimpo_candidate_kl"].requires_grad is False
    assert out["advantages"].requires_grad is False
```

- [x] **Step 5: Run the math tests and verify imports fail**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_advantage.py -q`

Expected: FAIL during collection because the VIMPO helpers do not exist.

- [x] **Step 6: Implement the pure tensor helpers and computer**

```python
# customized_areal/tree_search/core/advantage.py
def candidate_forward_kl(
    policy_candidate_logp: torch.Tensor,
    reference_candidate_logp: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy_candidate_logp.shape != reference_candidate_logp.shape:
        raise ValueError("policy and reference candidate log-probability shapes differ")
    if policy_candidate_logp.ndim != 3 or mask.shape != policy_candidate_logp.shape[:2]:
        raise ValueError("candidate log-probabilities must be [B, S, K] with mask [B, S]")
    valid = mask.bool()
    if not torch.isfinite(policy_candidate_logp[valid]).all() or not torch.isfinite(reference_candidate_logp[valid]).all():
        raise ValueError("valid candidate log-probabilities must be finite")
    policy_logp = policy_candidate_logp.float()
    reference_logp = reference_candidate_logp.float()
    probability = policy_logp.exp()
    kl = (probability * (policy_logp - reference_logp)).sum(-1)
    mass = probability.sum(-1)
    return torch.where(valid, kl, 0.0).detach(), torch.where(valid, mass, 0.0).detach()


def masked_episode_reverse_lambda(
    td: torch.Tensor,
    mask: torch.Tensor,
    episode_index: torch.Tensor,
    turn_index: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    result = torch.zeros_like(td, dtype=torch.float32)
    valid = mask.bool()
    for episode in torch.unique(episode_index[valid], sorted=True):
        rows = torch.unique(torch.nonzero((episode_index == episode) & valid, as_tuple=False)[:, 0], sorted=True)
        ordered_rows = sorted(rows.tolist(), key=lambda row: int(turn_index[row][valid[row]][0]))
        positions = [(row, pos) for row in ordered_rows for pos in torch.nonzero(valid[row], as_tuple=False).flatten().tolist()]
        carry = td.new_zeros((), dtype=torch.float32)
        for row, pos in reversed(positions):
            carry = td[row, pos].float() + gamma * lam * carry
            result[row, pos] = carry
    return result


def masked_distributed_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    group: torch.distributed.ProcessGroup | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    valid = mask.bool()
    count = valid.sum(dtype=torch.float64)
    total = values.double().masked_fill(~valid, 0).sum()
    total_sq = values.double().square().masked_fill(~valid, 0).sum()
    stats = torch.stack((count, total, total_sq))
    if torch.distributed.is_initialized() and group is not None:
        torch.distributed.all_reduce(stats, group=group)
    if stats[0] == 0:
        raise ValueError("cannot whiten zero valid VIMPO tokens")
    mean = stats[1] / stats[0]
    variance = (stats[2] / stats[0] - mean.square()).clamp_min(0)
    normalized = (values.float() - mean.float()) / torch.sqrt(variance.float() + eps)
    if stats[0] == 1 or variance == 0:
        normalized = torch.zeros_like(normalized)
    return normalized.masked_fill(~valid, 0).detach()


class VIMPOAdvantageComputer:
    def __init__(self, *, beta: float, gamma: float, lam: float, whiten: bool, dp_group=None) -> None:
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.whiten = whiten
        self.dp_group = dp_group

    def compute(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["vimpo_predict_mask"].bool()
        kl, mass = candidate_forward_kl(batch["vimpo_candidate_logp"], batch["vimpo_ref_candidate_logp"], mask)
        td = self.beta * (batch["vimpo_sample_logp"].float() - batch["vimpo_ref_sample_logp"].float() - kl)
        advantages = masked_episode_reverse_lambda(td.detach(), mask, batch["vimpo_episode_index"], batch["vimpo_turn_index"], gamma=self.gamma, lam=self.lam)
        if self.whiten:
            advantages = masked_distributed_whiten(advantages, mask, self.dp_group)
        batch.update(vimpo_candidate_kl=kl, vimpo_retained_mass=mass, advantages=advantages.detach())
        return batch
```

Also reject duplicate valid candidate IDs in `VIMPOAdvantageComputer.compute`, export the class lazily from `customized_areal/tree_search/__init__.py`, and keep all reductions in float32 except whitening sufficient statistics.

- [x] **Step 7: Run tests and commit**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_config.py customized_areal/tree_search/tests/test_vimpo_advantage.py -q`

Expected: PASS.

```bash
git add customized_areal/tree_search/config.py customized_areal/tree_search/core/advantage.py customized_areal/tree_search/__init__.py customized_areal/tree_search/tests/test_vimpo_config.py customized_areal/tree_search/tests/test_vimpo_advantage.py
git commit -m "feat(tree-search): add VIMPO config and advantage math"
```

### Task 2: Workflow Episode Metadata and Centered Targets

**Files:**
- Modify: `customized_areal/tree_search/core/tree_store.py`
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_workflow.py`

**Interfaces:**
- Consumes: `Node.query_id`, `Node.episode_id`, `Node.turn_idx`, `Node.outcome_reward`, and `AdvantageMode.VIMPO`.
- Produces: `annotate_vimpo_episode_metadata(nodes: list[Node]) -> None` and tensor keys `vimpo_query_index`, `vimpo_episode_index`, `vimpo_turn_index`, `vimpo_centered_reward`, `vimpo_predict_mask`.

- [x] **Step 1: Write failing metadata and dispatch tests**

```python
# customized_areal/tree_search/tests/test_vimpo_workflow.py
import torch
import pytest

from customized_areal.tree_search.core.customized_grouped_workflow import annotate_vimpo_episode_metadata
from customized_areal.tree_search.core.tree_store import Node, _node_to_tensor_dict


def _node(query: str, episode: str, turn: int, reward: float) -> Node:
    return Node(input_ids=[10, 11, 12], logprobs=[0.0, -0.2, -0.3], loss_mask=[0, 1, 1], versions=[0, 0, 0], outcome_reward=reward, node_id=f"{episode}-{turn}", query_id=query, episode_id=episode, turn_idx=turn)


def test_centered_rewards_count_distinct_episodes_not_turns() -> None:
    nodes = [_node("q", "a", 1, 1.0), _node("q", "a", 2, 1.0), _node("q", "b", 1, 0.0)]
    annotate_vimpo_episode_metadata(nodes)
    assert [n.vimpo_centered_reward for n in nodes] == [0.5, 0.5, -0.5]
    assert nodes[0].vimpo_episode_index == nodes[1].vimpo_episode_index
    assert nodes[2].vimpo_episode_index != nodes[0].vimpo_episode_index


def test_single_episode_query_has_zero_target() -> None:
    nodes = [_node("q", "a", 1, 0.7)]
    annotate_vimpo_episode_metadata(nodes)
    assert nodes[0].vimpo_centered_reward == pytest.approx(0.0)


def test_tensorizer_uses_next_token_coordinates() -> None:
    node = _node("q", "a", 1, 1.0)
    annotate_vimpo_episode_metadata([node])
    data = _node_to_tensor_dict(node, "q", node.node_id, advantage_mode="vimpo")
    assert data["vimpo_predict_mask"].dtype is torch.bool
    assert data["vimpo_predict_mask"].tolist() == [[True, True, False]]
    assert data["vimpo_episode_index"].shape == data["input_ids"].shape


def test_metadata_rejects_duplicate_turn_and_inconsistent_reward() -> None:
    with pytest.raises(ValueError, match="duplicate turn_idx"):
        annotate_vimpo_episode_metadata([_node("q", "a", 1, 1), _node("q", "a", 1, 1)])
    with pytest.raises(ValueError, match="inconsistent outcome_reward"):
        annotate_vimpo_episode_metadata([_node("q", "a", 1, 1), _node("q", "a", 2, 0)])
```

- [x] **Step 2: Run tests and verify the metadata helper is missing**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_workflow.py -q`

Expected: FAIL during collection because `annotate_vimpo_episode_metadata` does not exist.

- [x] **Step 3: Add stable Node fields and query-local annotation**

```python
# Add to Node in customized_areal/tree_search/core/tree_store.py
vimpo_query_index: int = -1
vimpo_episode_index: int = -1
vimpo_centered_reward: float = 0.0

# customized_areal/tree_search/core/customized_grouped_workflow.py
def annotate_vimpo_episode_metadata(nodes: list[Node]) -> None:
    grouped: dict[str, dict[str, list[Node]]] = {}
    for node in nodes:
        if not node.query_id:
            raise ValueError("VIMPO requires a non-empty query_id")
        episode_id = node.episode_id or node.node_id
        if not episode_id:
            raise ValueError("VIMPO requires a non-empty episode_id or node_id")
        grouped.setdefault(node.query_id, {}).setdefault(episode_id, []).append(node)
    episode_counter = 0
    for query_index, query_id in enumerate(sorted(grouped)):
        episodes = grouped[query_id]
        rewards: dict[str, float] = {}
        for episode_id, episode_nodes in episodes.items():
            turns = [node.turn_idx for node in episode_nodes]
            if len(turns) != len(set(turns)):
                raise ValueError(f"duplicate turn_idx in episode {episode_id!r}")
            values = {float(node.outcome_reward) for node in episode_nodes}
            if len(values) != 1:
                raise ValueError(f"inconsistent outcome_reward in episode {episode_id!r}")
            rewards[episode_id] = values.pop()
        mean_reward = sum(rewards.values()) / len(rewards)
        for episode_id in sorted(episodes):
            for node in episodes[episode_id]:
                node.vimpo_query_index = query_index
                node.vimpo_episode_index = episode_counter
                node.vimpo_centered_reward = rewards[episode_id] - mean_reward
            episode_counter += 1
```

Extend `_node_to_tensor_dict(node, query_id, node_id, max_tokens=0, loss_mode=None, advantage_mode=None)` and `_nodes_to_batched_tensor_dict` so VIMPO emits sequence-shaped metadata. Build the canonical mask exactly as:

```python
predict_mask = torch.roll(traj["loss_mask"].bool(), shifts=-1, dims=-1)
predict_mask[:, -1] = False
traj["vimpo_predict_mask"] = predict_mask
for key, value, dtype in (
    ("vimpo_query_index", node.vimpo_query_index, torch.int64),
    ("vimpo_episode_index", node.vimpo_episode_index, torch.int64),
    ("vimpo_turn_index", node.turn_idx, torch.int64),
    ("vimpo_centered_reward", node.vimpo_centered_reward, torch.float32),
):
    traj[key] = torch.full((1, seq_len), value, dtype=dtype)
```

In `_finalize_episode`, call `annotate_vimpo_episode_metadata(all_nodes)` only for VIMPO, skip every Node advantage computer, skip `_annotate_critic_values`, pass `advantage_mode` to tensorization, and never attach `critic_train_data`.

- [x] **Step 4: Run focused and regression workflow tests**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_workflow.py customized_areal/tree_search/tests/test_tree_store_loo.py customized_areal/tree_search/tests/test_gae_advantage.py customized_areal/tree_search/tests/test_critic_smoke.py -q`

Expected: PASS; existing modes retain their original tensor keys and critic dispatch.

- [x] **Step 5: Commit**

```bash
git add customized_areal/tree_search/core/tree_store.py customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/tests/test_vimpo_workflow.py
git commit -m "feat(tree-search): preserve VIMPO episode targets"
```

### Task 3: Episode-Atomic PPO and FSDP Microbatch Allocation

**Files:**
- Create: `customized_areal/tree_search/training/vimpo_batching.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_batching.py`

**Interfaces:**
- Consumes: padded tensor dictionaries, `MicroBatchSpec`, and repeated `vimpo_episode_index`.
- Produces: `split_episode_atomic_batches(data, mb_spec, episode_key="vimpo_episode_index") -> MicroBatchList` with complete episodes in every `mb`.

- [ ] **Step 1: Write failing allocator tests**

```python
# customized_areal/tree_search/tests/test_vimpo_batching.py
import pytest
import torch
from areal.api.cli_args import MicroBatchSpec
from customized_areal.tree_search.training.vimpo_batching import split_episode_atomic_batches


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(24).view(4, 6),
        "attention_mask": torch.tensor([[1,1,1,1,0,0], [1,1,1,0,0,0], [1,1,1,1,1,0], [1,1,0,0,0,0]], dtype=torch.bool),
        "vimpo_episode_index": torch.tensor([[0]*6, [0]*6, [1]*6, [2]*6]),
        "vimpo_predict_mask": torch.tensor([[1,1,0,0,0,0], [1,0,0,0,0,0], [1,1,1,0,0,0], [1,0,0,0,0,0]], dtype=torch.bool),
    }


def test_allocator_never_splits_episode_rows() -> None:
    result = split_episode_atomic_batches(_batch(), MicroBatchSpec(n_mbs=2))
    memberships = [{int(v) for v in mb["vimpo_episode_index"][:, 0]} for mb in result.mbs]
    assert sum(0 in members for members in memberships) == 1
    containing = next(mb for mb in result.mbs if 0 in {int(v) for v in mb["vimpo_episode_index"][:, 0]})
    assert (containing["vimpo_episode_index"][:, 0] == 0).sum() == 2


def test_allocator_rejects_episode_larger_than_token_limit() -> None:
    with pytest.raises(ValueError, match="episode 0.*exceeds max_tokens_per_mb=6"):
        split_episode_atomic_batches(_batch(), MicroBatchSpec(n_mbs=2, max_tokens_per_mb=6))
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_batching.py -q`

Expected: FAIL during collection because `vimpo_batching.py` does not exist.

- [ ] **Step 3: Implement deterministic episode grouping and balancing**

Implement the allocator with this exact public contract and ordering:

```python
def split_episode_atomic_batches(
    data: dict[str, torch.Tensor],
    mb_spec: MicroBatchSpec,
    *,
    episode_key: str = "vimpo_episode_index",
) -> MicroBatchList:
    attention_mask = data["attention_mask"].bool()
    episode_rows: dict[int, list[int]] = {}
    for row in range(attention_mask.shape[0]):
        values = torch.unique(data[episode_key][row])
        if values.numel() != 1:
            raise ValueError(f"row {row} belongs to more than one VIMPO episode")
        episode_rows.setdefault(int(values[0]), []).append(row)
    costs = {episode: int(attention_mask[rows].sum()) for episode, rows in episode_rows.items()}
    limit = mb_spec.max_tokens_per_mb
    if limit is not None:
        for episode, cost in costs.items():
            if cost > limit:
                raise ValueError(f"episode {episode} token cost {cost} exceeds max_tokens_per_mb={limit}")
    groups: list[list[int]] = [[] for _ in range(mb_spec.n_mbs)]
    loads = [0] * mb_spec.n_mbs
    for episode in sorted(episode_rows, key=lambda value: (-costs[value], value)):
        eligible = [index for index, load in enumerate(loads) if limit is None or load + costs[episode] <= limit]
        if not eligible:
            raise ValueError(f"episode {episode} cannot fit any VIMPO microbatch")
        target = min(eligible, key=lambda index: (loads[index], index))
        groups[target].extend(episode_rows[episode])
        loads[target] += costs[episode]
    groups = [rows for rows in groups if rows]
    forward_indices = [row for rows in groups for row in rows]
    backward_indices = [0] * len(forward_indices)
    for new, old in enumerate(forward_indices):
        backward_indices[old] = new
    mbs = [{key: value[rows] if torch.is_tensor(value) and value.shape[:1] == attention_mask.shape[:1] else value for key, value in data.items()} for rows in groups]
    return MicroBatchList(data=data, mb_spec=mb_spec, mbs=mbs, group_lens=loads[:len(groups)], forward_indices=forward_indices, backward_indices=backward_indices)
```

Before returning, reject empty input and enforce contiguous one-based turn rows per episode. Preserve non-row tensor/list metadata exactly as the generic splitter does.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_batching.py -q`

Expected: PASS.

```bash
git add customized_areal/tree_search/training/vimpo_batching.py customized_areal/tree_search/tests/test_vimpo_batching.py
git commit -m "feat(tree-search): add episode-atomic VIMPO batching"
```

### Task 4: Frozen SGLang Reference Candidate Scorer

**Files:**
- Create: `customized_areal/tree_search/training/vimpo_reference.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_reference.py`

**Interfaces:**
- Consumes: actor initial checkpoint/tokenizer identity and per-position actor candidate IDs.
- Produces: `ReferenceIdentity`, `ReferenceScoreRequest`, `ReferenceScore`, `VIMPOReferenceScorer`, and `SGLangVIMPOReferenceScorer` with synchronous `validate_identity`, `score`, and `close` methods suitable for PPO worker RPC.

- [ ] **Step 1: Write failing mocked HTTP contract tests**

```python
# customized_areal/tree_search/tests/test_vimpo_reference.py
import httpx
import pytest

from customized_areal.tree_search.training.vimpo_reference import ReferenceIdentity, ReferenceScoreRequest, SGLangVIMPOReferenceScorer


def test_actor_selected_candidates_and_sample_dedup_keep_order() -> None:
    seen: list[dict] = []
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/init", "revision": "main", "vocab_size": 8, "tokenizer_vocab_size": 8, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 2, "temperature": 1.0, "quantized": False})
        seen.append(payload)
        token_ids = payload["token_ids_logprob"]
        return httpx.Response(200, json={"meta_info": {"token_ids_logprob": [[-float(token), token, str(token)] for token in token_ids]}})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ref")
    scorer = SGLangVIMPOReferenceScorer("http://ref", timeout=2, max_concurrency=2, max_retries=0, client=client)
    scorer.validate_identity(ReferenceIdentity("/models/init", "main", 8, 8, 1, 2, 2))
    result = scorer.score([ReferenceScoreRequest((0, 1), [1, 3], 5, [7, 5, 4])])
    assert seen[0]["token_ids_logprob"] == [7, 5, 4]
    assert result[0].candidate_logp == [-7.0, -5.0, -4.0]
    assert result[0].sampled_logp == -5.0


def test_out_of_order_completion_is_restored_by_key() -> None:
    requests = [ReferenceScoreRequest((0, 3), [1,2,3], 4, [5]), ReferenceScoreRequest((0, 1), [1], 2, [3])]
    scorer = _scripted_scorer({(0, 1): (-2.0, [-3.0]), (0, 3): (-4.0, [-5.0])})
    assert [score.key for score in scorer.score(requests)] == [(0, 3), (0, 1)]


@pytest.mark.parametrize("field", ["model_path", "revision", "vocab_size", "tokenizer_vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"])
def test_identity_mismatch_fails_before_scoring(field: str) -> None:
    scorer = _identity_scorer(**{field: "wrong" if field in {"model_path", "revision"} else 99})
    with pytest.raises(ValueError, match=field):
        scorer.validate_identity(ReferenceIdentity("/models/init", "main", 8, 8, 1, 2, 2))
```

In this file, `_scripted_scorer` and `_identity_scorer` are complete local fixtures backed by `httpx.MockTransport`; they must also cover HTTP 429/500 retry, HTTP 400 no-retry, timeout exhaustion, missing token ID, non-finite score, candidate chunking, endpoint identity change after reconnect, bounded worker count, and idempotent close.

- [ ] **Step 2: Run tests and verify the scorer module is missing**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_reference.py -q`

Expected: FAIL during collection because `vimpo_reference.py` does not exist.

- [ ] **Step 3: Implement immutable identity and request/response types**

```python
@dataclass(frozen=True)
class ReferenceIdentity:
    model_path: str
    revision: str
    vocab_size: int
    tokenizer_vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None

@dataclass(frozen=True)
class ReferenceScoreRequest:
    key: tuple[int, int]
    prefix_ids: list[int]
    sampled_token_id: int
    candidate_token_ids: list[int]

@dataclass(frozen=True)
class ReferenceScore:
    key: tuple[int, int]
    sampled_logp: float
    candidate_logp: list[float]

class VIMPOReferenceScorer(Protocol):
    def validate_identity(self, expected: ReferenceIdentity) -> None:
        raise NotImplementedError
    def score(self, requests: list[ReferenceScoreRequest]) -> list[ReferenceScore]:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 4: Implement the generic SGLang adapter**

Use a bounded `concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency)` around one shared `httpx.Client`. Sort work by `(len(prefix_ids), key)`, restore result order by the caller's keys, and send:

```python
payload = {
    "input_ids": request.prefix_ids,
    "sampling_params": {"max_new_tokens": 1, "temperature": 1.0},
    "return_logprob": True,
    "token_ids_logprob": deduplicated_token_ids,
    "stream": False,
}
```

Parse `meta_info["token_ids_logprob"]` into a token-ID map; require every requested ID exactly once and finite. Chunk `token_ids_logprob` when the service reports a limit, joining chunks by ID. Retry only `httpx.TransportError`, HTTP 429, and HTTP 5xx up to `max_retries`; run identity validation again after a transport reconnect. Compare every `ReferenceIdentity` field, `temperature == 1.0`, and `quantized is False` from `/get_model_info`. Never call a weight-update endpoint.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_reference.py -q`

Expected: PASS with no real network access.

```bash
git add customized_areal/tree_search/training/vimpo_reference.py customized_areal/tree_search/tests/test_vimpo_reference.py
git commit -m "feat(tree-search): add frozen SGLang VIMPO scorer"
```

### Task 5: FSDP Actor Candidate Statistics

**Files:**
- Modify: `customized_areal/tree_search/engine/fsdp_engine.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_fsdp_stats.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_fsdp_distributed.py`

**Interfaces:**
- Consumes: actor logits, next-token labels/mask, explicit TP/SP groups, and effective `top_k`.
- Produces: `VIMPOCandidateStats` and `compute_vimpo_candidate_stats(data, top_k)` without exporting full-vocabulary logits.

- [ ] **Step 1: Write failing single-rank numerical tests**

```python
# customized_areal/tree_search/tests/test_vimpo_fsdp_stats.py
import torch
from customized_areal.tree_search.engine.fsdp_engine import vimpo_candidate_stats_from_logits


def test_stats_use_full_vocab_normalizer_and_actor_topk() -> None:
    logits = torch.tensor([[[0.0, 3.0, 2.0, 1.0], [2.0, 0.0, 1.0, 3.0]]])
    labels = torch.tensor([[1, 3]])
    mask = torch.tensor([[True, True]])
    stats = vimpo_candidate_stats_from_logits(logits, labels, mask, top_k=2)
    expected = logits.log_softmax(-1)
    assert stats.candidate_ids.tolist() == [[[1, 2], [3, 0]]]
    torch.testing.assert_close(stats.candidate_logp, expected.gather(-1, stats.candidate_ids), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(stats.sampled_logp, expected.gather(-1, labels.unsqueeze(-1)).squeeze(-1), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(stats.retained_mass, stats.candidate_logp.exp().sum(-1), rtol=1e-6, atol=1e-6)


def test_topk_is_capped_at_vocab_and_masked_rows_are_sentinel() -> None:
    stats = vimpo_candidate_stats_from_logits(torch.zeros(1, 2, 3), torch.tensor([[0, 1]]), torch.tensor([[True, False]]), top_k=8)
    assert stats.candidate_ids.shape == (1, 2, 3)
    assert stats.candidate_ids[0, 1].tolist() == [-1, -1, -1]
    assert stats.predict_mask.tolist() == [[True, False]]
```

- [ ] **Step 2: Run tests and verify the stats API is absent**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_fsdp_stats.py -q`

Expected: FAIL during collection because `vimpo_candidate_stats_from_logits` does not exist.

- [ ] **Step 3: Implement the return type and single-rank helper**

```python
@dataclass(frozen=True)
class VIMPOCandidateStats:
    sampled_logp: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_logp: torch.Tensor
    retained_mass: torch.Tensor
    predict_mask: torch.Tensor

def vimpo_candidate_stats_from_logits(logits, labels, predict_mask, *, top_k):
    effective_k = min(top_k, logits.shape[-1])
    logp = logits.float().log_softmax(-1)
    candidate_logp, candidate_ids = torch.topk(logp, effective_k, dim=-1)
    sampled_logp = logp.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    candidate_ids = candidate_ids.masked_fill(~predict_mask.unsqueeze(-1), -1)
    candidate_logp = candidate_logp.masked_fill(~predict_mask.unsqueeze(-1), 0)
    sampled_logp = sampled_logp.masked_fill(~predict_mask, 0)
    retained_mass = candidate_logp.exp().sum(-1).masked_fill(~predict_mask, 0)
    return VIMPOCandidateStats(sampled_logp, candidate_ids, candidate_logp, retained_mass, predict_mask.bool())
```

- [ ] **Step 4: Add fake-process-group tests for sharded normalization and global top-k**

In `test_vimpo_fsdp_distributed.py`, spawn two CPU `gloo` ranks with `torch.multiprocessing.spawn`. Give rank 0 vocabulary IDs `[0, 2)` and rank 1 `[2, 4)`, compare global candidate IDs/log-probabilities and sampled-token log-probability against concatenated brute force, and monkeypatch `torch.distributed.all_reduce/all_gather` wrappers to assert the configured `tp_group` is passed. Mark the true SP/packed-tree GPU cases:

```python
@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="VIMPO TP/SP FSDP test requires at least 2 CUDA devices")
def test_vimpo_stats_tp_sp_and_packed_tree_alignment() -> None:
    torch.multiprocessing.spawn(
        _tp_sp_packed_tree_worker,
        args=(2,),
        nprocs=2,
        join=True,
    )
```

Define `_tp_sp_packed_tree_worker(rank: int, world_size: int)` in the same test module using the existing distributed-test setup/teardown pattern. It initializes a temporary-file `gloo` control group and NCCL model groups, constructs identical packed and non-packed token batches, invokes `compute_vimpo_candidate_stats` on both, and uses `torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)` on sampled log-probabilities, candidate log-probabilities, retained mass, and scattered candidate IDs. Do not mock FSDP or DTensor internals.

- [ ] **Step 5: Implement TP/SP and packed-tree collection**

Add `MultiCandidateFSDPEngine.compute_vimpo_candidate_stats(self, data, *, top_k)` as a no-grad/eval forward. For each vocab shard, calculate global max with `all_reduce(MAX, group=tp_group)`, global shifted exponential sum with `all_reduce(SUM, group=tp_group)`, add the shard's global vocab offset to local top-k IDs, `all_gather` at most `K` candidates per rank, then select global top-k with token ID as deterministic tie-breaker. Gather sampled logits only from the owning shard. Reuse `_sp_all_gather` and `gather_packed_tree_vocab_stats`/trie mappings before returning `[B,S,...]`. Force temperature `1.0`; restore the previous train/eval mode after the snapshot.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_fsdp_stats.py customized_areal/tree_search/tests/test_vimpo_fsdp_distributed.py -q`

Expected: CPU tests PASS; hardware test SKIP with the explicit CUDA reason when fewer than two GPUs are present.

```bash
git add customized_areal/tree_search/engine/fsdp_engine.py customized_areal/tree_search/tests/test_vimpo_fsdp_stats.py customized_areal/tree_search/tests/test_vimpo_fsdp_distributed.py
git commit -m "feat(tree-search): collect VIMPO actor candidate stats"
```

### Task 6: Combined Episode-Level VIMPO Loss and Two-Denominator Backward

**Files:**
- Create: `customized_areal/tree_search/training/losses/vimpo.py`
- Modify: `customized_areal/tree_search/training/losses/__init__.py`
- Modify: `customized_areal/tree_search/engine/fsdp_engine.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_loss.py`

**Interfaces:**
- Consumes: differentiable sampled-token log-probabilities and the enriched VIMPO batch.
- Produces: `VIMPOLossTerms`, `vimpo_loss_terms`, `vimpo_loss_fn`, and `MultiCandidateFSDPEngine.train_vimpo_batch` performing one zero-grad/backward/step.

- [ ] **Step 1: Write failing loss and gradient tests**

```python
# customized_areal/tree_search/tests/test_vimpo_loss.py
import torch
from customized_areal.tree_search.training.losses.vimpo import vimpo_loss_terms


def test_terminal_loss_averages_complete_episodes_not_tokens() -> None:
    current = torch.tensor([[-0.2, -0.4, 0.0], [-0.3, 0.0, 0.0]], requires_grad=True)
    data = {
        "vimpo_predict_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "vimpo_episode_index": torch.tensor([[0,0,0], [1,1,1]]),
        "vimpo_centered_reward": torch.tensor([[0.5,0.5,0.5], [-0.5,-0.5,-0.5]]),
        "vimpo_ref_sample_logp": torch.tensor([[-0.5,-0.5,0.0], [-0.5,0.0,0.0]], requires_grad=True),
        "vimpo_candidate_kl": torch.tensor([[0.1,0.2,0.0], [0.3,0.0,0.0]], requires_grad=True),
        "logprobs": torch.tensor([[-0.25,-0.45,0.0], [-0.35,0.0,0.0]], requires_grad=True),
        "prox_logp": torch.tensor([[-0.25,-0.45,0.0], [-0.35,0.0,0.0]], requires_grad=True),
        "advantages": torch.tensor([[1.0,2.0,0.0], [-1.0,0.0,0.0]], requires_grad=True),
    }
    terms = vimpo_loss_terms(current, data, beta=0.5, eps_clip=0.2, eps_clip_higher=None)
    assert terms.episode_count == 2
    assert terms.valid_token_count == 3
    total = terms.value_sum / terms.episode_count + 0.005 * terms.ppo_sum / terms.valid_token_count
    total.backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert data["vimpo_ref_sample_logp"].grad is None
    assert data["vimpo_candidate_kl"].grad is None
    assert data["advantages"].grad is None
    assert data["prox_logp"].grad is None


def test_loss_rejects_partial_episode_before_backward() -> None:
    data = _complete_two_turn_batch()
    data["vimpo_expected_turn_count"] = torch.full_like(data["vimpo_turn_index"], 2)
    partial = {key: value[:1] for key, value in data.items()}
    with pytest.raises(ValueError, match="episode 0 is incomplete"):
        vimpo_loss_terms(torch.zeros_like(partial["vimpo_ref_sample_logp"], requires_grad=True), partial, beta=5e-4, eps_clip=0.2, eps_clip_higher=None)
```

- [ ] **Step 2: Run tests and verify the loss module is absent**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_loss.py -q`

Expected: FAIL during collection because `training/losses/vimpo.py` does not exist.

- [ ] **Step 3: Implement numerator-returning pure loss math**

```python
@dataclass(frozen=True)
class VIMPOLossTerms:
    ppo_sum: torch.Tensor
    value_sum: torch.Tensor
    valid_token_count: torch.Tensor
    episode_count: torch.Tensor
    terminal_prediction: torch.Tensor
    terminal_target: torch.Tensor
    terminal_residual: torch.Tensor

def vimpo_loss_terms(logprobs, data, *, beta, eps_clip, eps_clip_higher):
    mask = data["vimpo_predict_mask"].bool()
    old = data.get("prox_logp", data["logprobs"]).detach().float()
    advantage = data["advantages"].detach().float()
    ratio = torch.exp(logprobs.float() - old)
    lower = 1.0 - eps_clip
    upper = 1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher)
    ppo_token = -torch.minimum(ratio * advantage, ratio.clamp(lower, upper) * advantage)
    ppo_sum = ppo_token.masked_select(mask).sum()
    predictions, targets = [], []
    for episode in torch.unique(data["vimpo_episode_index"][mask], sorted=True):
        episode_mask = mask & (data["vimpo_episode_index"] == episode)
        prediction = beta * (logprobs.float() - data["vimpo_ref_sample_logp"].detach().float() - data["vimpo_candidate_kl"].detach().float())
        predictions.append(prediction.masked_select(episode_mask).sum())
        repeated_target = data["vimpo_centered_reward"].masked_select(episode_mask).float()
        if not torch.allclose(repeated_target, repeated_target[0].expand_as(repeated_target)):
            raise ValueError(f"episode {int(episode)} has inconsistent centered rewards")
        targets.append(repeated_target[0].detach())
    prediction_tensor = torch.stack(predictions)
    target_tensor = torch.stack(targets)
    residual = prediction_tensor - target_tensor
    return VIMPOLossTerms(ppo_sum, 0.5 * residual.square().sum(), mask.sum(), torch.tensor(len(predictions), device=logprobs.device), prediction_tensor, target_tensor, residual)
```

Use the repository's `ppo_actor_loss_fn` clipping semantics when integrating, but retain the numerator/count contract. Validate complete turn counts, constant metadata per row, finite inputs, and nonzero token/episode counts before returning.

- [ ] **Step 4: Add one-backward/two-denominator engine tests**

Create a fake subclass whose `forward_backward_batch`, `optimizer_zero_grad`, and `optimizer_step` record calls. Assert `train_vimpo_batch` calls each exactly once, all-reduces `[valid_token_count, episode_count]` over `dp_group`, scales each microbatch as:

```python
loss = self.parallel_helper.dp_size * (
    actor_coeff * terms.ppo_sum / global_valid_tokens
    + value_loss_weight * terms.value_sum / global_episodes
)
```

and validates all batches/reference tensors before `optimizer_zero_grad()`.

- [ ] **Step 5: Implement `train_vimpo_batch`**

Normalize input, call the episode-atomic splitter, pack/pad with the same helpers used by `_prepare_mb_list`, compute and all-reduce the two float64 denominators once, then use `forward_backward_batch` with a process callback that gathers only sampled-action log-probabilities and calls `vimpo_loss_terms`. Accumulate detached metric numerators; call `optimizer_step()` once. Do not call generic `train_batch`, whose single `loss_weight_fn` cannot represent both means.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_loss.py customized_areal/tree_search/tests/test_vimpo_batching.py -q`

Expected: PASS.

```bash
git add customized_areal/tree_search/training/losses/vimpo.py customized_areal/tree_search/training/losses/__init__.py customized_areal/tree_search/engine/fsdp_engine.py customized_areal/tree_search/tests/test_vimpo_loss.py
git commit -m "feat(tree-search): add combined VIMPO actor loss"
```

### Task 7: Dedicated VIMPO FSDP Actor and Trainer Wiring

**Files:**
- Modify: `customized_areal/tree_search/training/actor.py`
- Modify: `customized_areal/tree_search/training/trainer.py`
- Modify: `customized_areal/tree_search/engine/__init__.py`
- Modify: `customized_areal/tree_search/__init__.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_actor.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_trainer.py`

**Interfaces:**
- Consumes: Tasks 1–6 contracts and dynamic actor config attributes copied by `CustomizedPPOTrainer`.
- Produces: `VIMPOFSDPPPOActor.compute_advantages`, `.ppo_update`, `.destroy`, controller construction, and VIMPO-only trainer selection.

- [ ] **Step 1: Write failing actor orchestration tests with fake scorer/engine**

```python
# customized_areal/tree_search/tests/test_vimpo_actor.py
def test_compute_advantages_orders_snapshot_reference_then_advantage() -> None:
    events: list[str] = []
    actor = _fake_vimpo_actor(events)
    enriched = actor.compute_advantages([_vimpo_rollout_batch()])
    assert events == ["validate_identity", "actor_snapshot", "reference_score", "advantage"]
    assert "vimpo_candidate_kl" in enriched[0]
    assert enriched[0]["advantages"].requires_grad is False


def test_reference_failure_happens_before_optimizer_mutation() -> None:
    actor, optimizer = _fake_vimpo_actor_with_failing_reference()
    with pytest.raises(RuntimeError, match="reference unavailable"):
        actor.compute_advantages([_vimpo_rollout_batch()])
    assert optimizer.zero_grad_calls == 0
    assert optimizer.step_calls == 0


def test_ppo_update_uses_one_combined_train_call() -> None:
    actor, engine = _fake_vimpo_actor_with_engine()
    actor.ppo_update([_complete_enriched_batch()])
    assert engine.train_vimpo_batch_calls == 1
    assert engine.train_batch_calls == 0
```

- [ ] **Step 2: Write failing trainer selection tests**

```python
# customized_areal/tree_search/tests/test_vimpo_trainer.py
def test_vimpo_selects_dedicated_fsdp_actor_and_copies_config(monkeypatch) -> None:
    trainer = CustomizedPPOTrainer.__new__(CustomizedPPOTrainer)
    trainer.tree_search_config = Config(advantage_mode="vimpo", vimpo_ref_base_url="http://ref")
    trainer.scheduler = object()
    actor = trainer._create_train_engine(_actor_config(backend="fsdp:d1"), _allocation("fsdp"))
    assert actor.__class__.__name__ == "VIMPOFSDPPPOActor"
    assert actor.config.vimpo_top_k == 128
    assert actor.config.vimpo_ref_base_url == "http://ref"


def test_vimpo_rejects_non_fsdp_before_actor_creation() -> None:
    trainer = _uninitialized_trainer(Config(advantage_mode="vimpo", vimpo_ref_base_url="http://ref"))
    with pytest.raises(ValueError, match="VIMPO requires FSDP actor backend"):
        trainer._create_train_engine(_actor_config(backend="megatron:d1"), _allocation("megatron"))
```

- [ ] **Step 3: Run tests and verify the dedicated actor is absent**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_actor.py customized_areal/tree_search/tests/test_vimpo_trainer.py -q`

Expected: FAIL because `VIMPOFSDPPPOActor` is not defined or selected.

- [ ] **Step 4: Implement the dedicated actor without a global monkey patch**

```python
class VIMPOFSDPPPOActor(MultiCandidateFSDPEngine):
    def __init__(self, config: PPOActorConfig, scorer: VIMPOReferenceScorer | None = None):
        super().__init__(config)
        self.actor = PPOActor(config, self)
        self.reference_scorer = scorer or SGLangVIMPOReferenceScorer(
            config.vimpo_ref_base_url,
            timeout=config.vimpo_ref_timeout,
            max_concurrency=config.vimpo_ref_max_concurrency,
            max_retries=config.vimpo_ref_max_retries,
        )
        self.vimpo_advantage = VIMPOAdvantageComputer(beta=config.vimpo_beta, gamma=config.vimpo_gamma, lam=config.vimpo_lambda, whiten=config.vimpo_whiten_advantages, dp_group=self.dp_group)

    @torch.no_grad()
    def compute_advantages(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return batched_call(self._compute_vimpo_advantages, data, pass_meta=True)

    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._vimpo_update, data, unpack=False)

    def destroy(self) -> None:
        self.reference_scorer.close()
        super().destroy()
```

`_compute_vimpo_advantages` validates identity, calls `compute_vimpo_candidate_stats`, builds one `ReferenceScoreRequest` per valid `(row, prediction_position)` using `input_ids[row, :position+1]`, scores only on the model-parallel head, aligns results by key, broadcasts tensors over `mp_group`, and invokes `VIMPOAdvantageComputer`. Record snapshot version before scoring. `_vimpo_update` validates every required key, logs token/episode denominators, outer-splits with `split_episode_atomic_batches`, and calls `train_vimpo_batch` once per complete PPO minibatch.

- [ ] **Step 5: Wire trainer selection and lifecycle**

Import `AdvantageMode` in `training/trainer.py`. At the start of `_create_train_engine`, before distillation/Muon/clip-cov branches, validate `alloc.backend == "fsdp"`, copy every `vimpo_*` setting plus expected actor/tokenizer identity to `actor_config`, select `VIMPOFSDPPPOActor` (or a Muon wrapper that only patches optimizer construction), create its process group, and return it. Do not install the distillation or combined-critic monkey patches. Add lazy exports in both `engine/__init__.py` and the package root.

Because base `PPOTrainer` creates `self.ref` only when `config.actor.kl_ctl > 0 and config.ref is not None`, document and test that a VIMPO run uses `actor.kl_ctl = 0` and `config.ref = None`. Assert `trainer.ref is None` and `trainer.critic is None` in a constructor-level fake test.

- [ ] **Step 6: Run actor/trainer tests and existing patch regressions**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_actor.py customized_areal/tree_search/tests/test_vimpo_trainer.py customized_areal/tree_search/tests/test_critic_update.py customized_areal/tree_search/tests/test_trainer_integration_critic_gae.py -q`

Expected: PASS; VIMPO uses no generic reference/critic engine and existing critic patches still install/restore normally.

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/training/actor.py customized_areal/tree_search/training/trainer.py customized_areal/tree_search/engine/__init__.py customized_areal/tree_search/__init__.py customized_areal/tree_search/tests/test_vimpo_actor.py customized_areal/tree_search/tests/test_vimpo_trainer.py
git commit -m "feat(tree-search): integrate VIMPO FSDP actor training"
```

### Task 8: Metrics, CPU End-to-End Smoke Test, and Documentation

**Files:**
- Modify: `customized_areal/tree_search/training/actor.py`
- Modify: `customized_areal/tree_search/engine/fsdp_engine.py`
- Modify: `customized_areal/tree_search/README.md`
- Create: `customized_areal/tree_search/tests/test_vimpo_smoke.py`
- Create: `customized_areal/tree_search/tests/test_vimpo_fsdp_sglang_integration.py`

**Interfaces:**
- Consumes: complete VIMPO training pipeline.
- Produces: required `stats_tracker` metrics, an offline CPU smoke test, and an explicit hardware/service integration gate.

- [ ] **Step 1: Write the failing CPU smoke test**

```python
# customized_areal/tree_search/tests/test_vimpo_smoke.py
def test_vimpo_cpu_pipeline_changes_actor_only() -> None:
    config = Config(advantage_mode="vimpo", vimpo_ref_base_url="http://fake", vimpo_top_k=3)
    nodes = [_node("q", "a", 1, 1.0), _node("q", "a", 2, 1.0), _node("q", "b", 1, 0.0)]
    annotate_vimpo_episode_metadata(nodes)
    batch = _tensorize(nodes)
    actor = _tiny_cpu_vimpo_actor(config, frozen_reference=_deterministic_reference())
    actor_before = [parameter.detach().clone() for parameter in actor.model.parameters()]
    reference_before = actor.reference_scorer.identity_and_version()
    enriched = actor.compute_advantages([batch])
    actor.ppo_update(enriched)
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, actor.model.parameters()))
    assert actor.reference_scorer.identity_and_version() == reference_before
    assert all(torch.isfinite(torch.tensor(value)) for value in actor.last_vimpo_metrics.values())
```

- [ ] **Step 2: Add and assert observability**

Use `stats_tracker.denominator` for valid tokens and complete episodes, `stats_tracker.stat` for candidate KL, retained-mass mean/min/quantiles, raw/normalized advantages, terminal prediction/target/residual/RMSE, and `stats_tracker.scalar` for component/combined losses, reference latency/retries, effective top-k, exact-KL flag, and snapshot policy version. Exact names:

```text
vimpo/candidate_kl
vimpo/retained_mass
vimpo/raw_advantage
vimpo/normalized_advantage
vimpo/terminal_prediction
vimpo/terminal_target
vimpo/terminal_residual
vimpo/terminal_rmse
vimpo/value_loss
vimpo/ppo_actor_loss
vimpo/combined_loss
vimpo/reference_latency_ms
vimpo/reference_retries
vimpo/effective_top_k
vimpo/exact_kl
vimpo/snapshot_policy_version
```

Do not call `.item()` or `.tolist()` on hot-path GPU tensors; feed tensors directly to the stats tracker. `vimpo/exact_kl` is one only when effective `K == vocab_size`.

- [ ] **Step 3: Add the hardware-gated integration test**

```python
# customized_areal/tree_search/tests/test_vimpo_fsdp_sglang_integration.py
import os
import pytest
import torch

pytestmark = [pytest.mark.slow]

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="VIMPO FSDP+SGLang integration requires at least 2 CUDA devices")
def test_one_vimpo_step_keeps_initial_sglang_reference_frozen() -> None:
    base_url = os.environ.get("VIMPO_TEST_SGLANG_URL")
    if not base_url:
        pytest.skip("VIMPO_TEST_SGLANG_URL is not set to a frozen initial-model SGLang service")
    result = _run_one_distributed_vimpo_step(base_url)
    assert result.actor_parameters_changed
    assert result.reference_identity_before == result.reference_identity_after
    assert result.optimizer_steps == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values())
```

The helper uses the repository's torchrun test harness, the same tokenizer/checkpoint for actor initialization and SGLang, and reads identity twice; it never launches or updates the reference service.

- [ ] **Step 4: Document configuration and deployment**

Add a VIMPO section to `customized_areal/tree_search/README.md` with this runnable configuration fragment:

```yaml
tree_search:
  advantage_mode: vimpo
  loss_mode: grpo
  vimpo_beta: 0.0005
  vimpo_actor_coeff: 0.005
  vimpo_value_loss_weight: 1.0
  vimpo_gamma: 1.0
  vimpo_lambda: 1.0
  vimpo_top_k: 128
  vimpo_whiten_advantages: true
  vimpo_detach_kl: true
  vimpo_ref_base_url: http://vimpo-reference:30000
  vimpo_ref_timeout: 300.0
  vimpo_ref_max_concurrency: 8
  vimpo_ref_max_retries: 3
```

State that the URL must serve the unmodified initial actor checkpoint with identical tokenizer/special IDs, temperature-one full-normalized token-ID log-probabilities, no quantization in paper-faithful mode, and no weight updates. Explain that `top_k < vocab_size` is truncated candidate KL, retained mass diagnoses approximation quality, `top_k == vocab_size` is exact but expensive, the actor backend must be FSDP, `actor.kl_ctl` may be zero, and no `ref` FSDP allocation should be configured.

- [ ] **Step 5: Run smoke, integration gate, and focused suite**

Run: `uv run pytest customized_areal/tree_search/tests/test_vimpo_*.py -q`

Expected: all CPU/mocked tests PASS; GPU/SGLang tests either PASS or SKIP with their explicit environment reason.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/training/actor.py customized_areal/tree_search/engine/fsdp_engine.py customized_areal/tree_search/README.md customized_areal/tree_search/tests/test_vimpo_smoke.py customized_areal/tree_search/tests/test_vimpo_fsdp_sglang_integration.py
git commit -m "test(tree-search): cover VIMPO end-to-end training"
```

### Task 9: Full Regression, Graph Refresh, and Comet Evidence

**Files:**
- Modify: `openspec/changes/add-vimpo-critic-mode/tasks.md`
- Generated/modified: `graphify-out/**`

**Interfaces:**
- Consumes: the complete implementation and OpenSpec task checklist.
- Produces: verified repository state, refreshed graph, and checked task evidence without broadening scope.

- [ ] **Step 1: Run the complete targeted tree-search suite**

Run: `uv run pytest customized_areal/tree_search/tests -q`

Expected: PASS, with only pre-existing or explicit hardware/service skips.

- [ ] **Step 2: Run static checks on all touched Python files**

Run: `uv run ruff check customized_areal/tree_search`

Expected: `All checks passed!`

Run: `uv run ruff format --check customized_areal/tree_search`

Expected: no files require reformatting.

- [ ] **Step 3: Run repository pre-commit hooks**

Run: `source .venv/bin/activate && pre-commit run --all-files`

Expected: every hook passes. If hooks format files, review the diff, rerun the targeted tests, and rerun pre-commit until it exits zero; do not skip hooks.

- [ ] **Step 4: Refresh and inspect graph relationships**

Run: `graphify update .`

Expected: graph update exits zero.

Run: `graphify path VIMPOFSDPPPOActor vimpo_loss_fn`

Expected: the graph shows the actor update path reaching the combined VIMPO loss.

- [ ] **Step 5: Validate OpenSpec and mark completed tasks**

Run: `openspec validate add-vimpo-critic-mode --type change --strict --json --no-interactive`

Expected: JSON reports one passed change and zero failures.

After matching each checkbox to test evidence, change the corresponding entries in `openspec/changes/add-vimpo-critic-mode/tasks.md` from `- [ ]` to `- [x]`. Keep item 7.1 checked only after confirming the native ragged SGLang endpoint and non-FSDP backends remain follow-up scope.

- [ ] **Step 6: Review the final diff and commit verification artifacts**

Run: `git status --short && git diff --check && git diff --stat c95b9c210a8fd77b7c4b6a2eb1fe1632bd0db8e8`

Expected: no whitespace errors; only VIMPO/OpenSpec/docs/graph changes plus known pre-existing user changes appear.

```bash
git add openspec/changes/add-vimpo-critic-mode/tasks.md graphify-out
git commit -m "docs(tree-search): record VIMPO verification evidence"
```

Do not squash or create a PR unless the user separately requests it.
