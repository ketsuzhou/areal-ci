"""Unit tests for TreeSearchGroupedRolloutWorkflow.

Exercises each pipeline stage in isolation so you can profile which step is slow:

1. interactions_dict_to_nodes  — rollout result → Node conversion
2. _result_to_nodes            — per-episode wrapper
3. _setup_distill_provider     — teacher client/provider init
4. _prepare_distill_for_episode — diagnose + teacher logprobs (THE BOTTLENECK)
5. _prepare_distill_for_node_groups — parallel episode distillation
6. tree_store insert / cache lookup
7. tree_advantage_computer     — GRPO normalization
8. _nodes_to_batched_tensor_dict — final tensor conversion
9. Full _arun_episode_impl     — end-to-end with mocked I/O

Usage:
    uv run pytest tests/test_tree_search/test_tree_search_grouped_workflow.py -v
    uv run pytest tests/test_tree_search/test_tree_search_grouped_workflow.py -v -k "test_prepare_distill"
    uv run pytest tests/test_tree_search/test_tree_search_grouped_workflow.py -v -k "test_full_arun" -s

To profile:
    uv run pytest tests/test_tree_search/test_tree_search_grouped_workflow.py -v --durations=0
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from customized_areal.tree_search.config import AdvantageMode, CacheMode, LossMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    *,
    node_id: str = "",
    episode_id: str = "ep1",
    query_id: str = "q1",
    turn_idx: int = 1,
    seq_len: int = 50,
    response_len: int = 10,
    outcome_reward: float = 1.0,
    guidance: dict[int, str] | None = None,
    topk_ids: list[list[int]] | None = None,
    topk_logp: list[list[float]] | None = None,
) -> Node:
    """Create a synthetic Node for testing."""
    input_ids = list(range(seq_len))
    loss_mask = [0] * (seq_len - response_len) + [1] * response_len
    logprobs = [0.0] * (seq_len - response_len) + [-2.0] * response_len
    versions = [-1] * (seq_len - response_len) + [0] * response_len
    return Node(
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=logprobs,
        versions=versions,
        outcome_reward=outcome_reward,
        node_id=node_id or uuid.uuid4().hex[:8],
        episode_id=episode_id,
        query_id=query_id,
        turn_idx=turn_idx,
        guidance=guidance,
        topk_ids=topk_ids,
        topk_logp=topk_logp,
    )


def _make_model_response(
    seq_len: int = 50,
    response_len: int = 10,
) -> "ModelResponse":
    """Create a ModelResponse with correct constructor (no input_len/output_len)."""
    from areal.api import ModelResponse

    resp = ModelResponse(
        input_tokens=list(range(seq_len)),
        output_tokens=list(range(seq_len, seq_len + response_len)),
        output_logprobs=[-1.0 - 0.1 * j for j in range(response_len)],
        output_versions=[0] * response_len,
    )
    # output_top_logprobs is accessed dynamically by interactions_dict_to_nodes
    # but is not a dataclass field — must set it as attribute
    resp.output_top_logprobs = None
    return resp


def _make_interaction_dict(
    n_turns: int = 2, seq_len: int = 50, response_len: int = 10
) -> dict[str, Any]:
    """Create a synthetic interactions dict (InteractionWithTokenLogpReward items)."""
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    interactions = {}
    for i in range(n_turns):
        resp = _make_model_response(seq_len=seq_len, response_len=response_len)
        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=float(i),
            chat_template_type="individual",
        )
        # Set interaction_id via the private setter
        interaction._interaction_id = f"int_{i}"
        interactions[f"int_{i}"] = interaction
    return interactions


def _make_workflow(
    *,
    group_size: int = 2,
    loss_mode: LossMode = LossMode.DISTILL,
    advantage_mode: AdvantageMode = AdvantageMode.GAE,
    cache_mode: CacheMode = CacheMode.CROSS_TRAINING,
    checkpoint_dir: str = "/tmp/test_tree_search_ckpt",
    teacher_provider: str = "external",
) -> "TreeSearchGroupedRolloutWorkflow":
    """Create a workflow instance with a mock inner workflow."""
    from customized_areal.tree_search.tree_search_grouped_workflow import (
        TreeSearchGroupedRolloutWorkflow,
    )

    # TRAIN_ID is required when loading a CROSS_TRAINING checkpoint
    if "TRAIN_ID" not in os.environ:
        os.environ["TRAIN_ID"] = "test_train_id"

    inner = AsyncMock()
    return TreeSearchGroupedRolloutWorkflow(
        workflow=inner,
        group_size=group_size,
        checkpoint_dir=checkpoint_dir,
        advantage_mode=advantage_mode,
        loss_mode=loss_mode,
        cache_mode=cache_mode,
        tokenizer_path="fake/tokenizer",
        teacher_provider=teacher_provider,
        teacher_base_url="http://localhost:9999",
        teacher_backend="openai",
        teacher_top_k=5,
    )


# ---------------------------------------------------------------------------
# 1. interactions_dict_to_nodes
# ---------------------------------------------------------------------------


class TestInteractionsDictToNodes:
    def test_basic_conversion(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            interactions_dict_to_nodes,
        )

        interactions = _make_interaction_dict(n_turns=2)
        nodes = interactions_dict_to_nodes(interactions)
        assert len(nodes) == 2
        for node in nodes:
            assert isinstance(node, Node)
            assert len(node.input_ids) == 60  # seq_len + response_len
            assert sum(node.loss_mask) == 10  # response_len

    def test_empty_dict(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            interactions_dict_to_nodes,
        )

        nodes = interactions_dict_to_nodes({})
        assert nodes == []

    def test_concat_mode_with_parent(self):
        """Test concat mode where parent logprobs are carried forward."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        from customized_areal.tree_search.tree_search_grouped_workflow import (
            interactions_dict_to_nodes,
        )

        # First interaction (root)
        resp1 = _make_model_response(seq_len=20, response_len=10)
        int1 = InteractionWithTokenLogpReward(
            model_response=resp1,
            reward=0.5,
            chat_template_type="concat",
        )
        int1._interaction_id = "root"

        # Second interaction with parent reference
        resp2 = _make_model_response(seq_len=35, response_len=10)
        int2 = InteractionWithTokenLogpReward(
            model_response=resp2,
            reward=1.0,
            chat_template_type="concat",
            parent=int1,
        )
        int2._interaction_id = "child"

        nodes = interactions_dict_to_nodes({"root": int1, "child": int2})
        assert len(nodes) == 2
        assert nodes[0].outcome_reward == 0.5
        # Child in concat mode: parent logprobs are carried forward
        assert nodes[1].outcome_reward == 1.0

    def test_skips_non_interaction_type(self):
        """Non-InteractionWithTokenLogpReward values are skipped."""
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            interactions_dict_to_nodes,
        )

        interactions = {"a": "not_an_interaction", "b": 42}
        nodes = interactions_dict_to_nodes(interactions)
        assert nodes == []

    def test_with_top_logprobs(self):
        """output_top_logprobs on ModelResponse is converted to topk_ids/topk_logp."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        from customized_areal.tree_search.tree_search_grouped_workflow import (
            interactions_dict_to_nodes,
        )

        resp = _make_model_response(seq_len=10, response_len=3)
        # Set output_top_logprobs dynamically (not a dataclass field)
        # The code unpacks as (token_id, logprob) per line 115 of the workflow
        resp.output_top_logprobs = [
            [(100, -0.1), (200, -0.5)],
            [(101, -0.2), (201, -0.6)],
            [(102, -0.3), (202, -0.7)],
        ]
        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
            chat_template_type="individual",
        )
        interaction._interaction_id = "int_topk"

        nodes = interactions_dict_to_nodes({"int_topk": interaction})
        assert len(nodes) == 1
        assert nodes[0].topk_ids is not None
        assert nodes[0].topk_logp is not None
        assert len(nodes[0].topk_ids) == 3
        assert nodes[0].topk_ids[0] == [100, 200]


# ---------------------------------------------------------------------------
# 2. _result_to_nodes
# ---------------------------------------------------------------------------


class TestResultToNodes:
    def test_dict_result(self):
        wf = _make_workflow(loss_mode=LossMode.GRPO)
        interactions = _make_interaction_dict(n_turns=2)
        nodes = wf._result_to_nodes(interactions, "q1", 0)
        assert len(nodes) == 2
        assert all(n.query_id == "q1" for n in nodes)
        assert all(n.episode_id.startswith("q1_0_") for n in nodes)

    def test_list_result(self):
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        wf = _make_workflow(loss_mode=LossMode.GRPO)
        resp = _make_model_response(seq_len=30, response_len=10)
        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=0.8,
            chat_template_type="individual",
        )
        interaction._interaction_id = "int_0"

        nodes = wf._result_to_nodes([interaction], "q2", 1)
        assert len(nodes) == 1
        assert nodes[0].query_id == "q2"

    def test_unsupported_result_type(self):
        wf = _make_workflow(loss_mode=LossMode.GRPO)
        assert wf._result_to_nodes("bad_type", "q1", 0) is None
        assert wf._result_to_nodes(42, "q1", 0) is None


# ---------------------------------------------------------------------------
# 3. _setup_distill_provider
# ---------------------------------------------------------------------------


class TestSetupDistillProvider:
    @pytest.mark.asyncio
    async def test_external_provider_init(self):
        wf = _make_workflow()
        mock_engine = MagicMock()
        mock_engine._proxy_gateway_addr = ""
        mock_engine.addresses = []
        mock_engine.config = MagicMock()
        mock_engine.config.admin_api_key = ""

        with patch.dict("os.environ", {
            "WORKSPACE_OPENAI_API_KEY": "test-key",
            "WORKSPACE_OPENAI_API_BASE": "http://test:8080/v1",
        }):
            provider, client = await wf._setup_distill_provider(mock_engine)

        assert provider is not None
        assert client is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_engine_provider_sglang(self):
        wf = _make_workflow(teacher_provider="engine")
        mock_engine = MagicMock()
        mock_engine._proxy_gateway_addr = ""
        mock_engine.addresses = ["10.0.0.1:8000"]
        mock_engine.config = MagicMock()
        mock_engine.config.admin_api_key = "admin123"
        mock_engine.backend = MagicMock()
        type(mock_engine.backend).__name__ = "SGLangBackend"

        with patch.dict("os.environ", {
            "WORKSPACE_OPENAI_API_KEY": "test-key",
            "WORKSPACE_OPENAI_API_BASE": "http://test:8080/v1",
        }):
            provider, client = await wf._setup_distill_provider(mock_engine)

        assert client.config.teacher_backend == "sglang"
        assert client.config.teacher_base_url == "http://10.0.0.1:8000"
        await client.close()


# ---------------------------------------------------------------------------
# 4. _prepare_distill_for_episode  (the bottleneck)
# ---------------------------------------------------------------------------


class TestPrepareDistillForEpisode:
    @pytest.mark.asyncio
    async def test_skips_when_guidance_cached(self):
        """If the last node's guidance is already set, diagnosis call is skipped."""
        wf = _make_workflow()
        nodes = [
            _make_node(
                node_id="n1",
                turn_idx=1,
                seq_len=30,
                response_len=5,
            ),
            _make_node(
                node_id="n2",
                turn_idx=2,
                seq_len=30,
                response_len=5,
                # Set guidance on the LAST node — that's what _prepare checks
                guidance={2: "Improve turn 2"},
            ),
        ]

        mock_provider = AsyncMock()
        mock_provider.diagnose_episode = AsyncMock()
        mock_provider.get_logprobs_for_prompt = AsyncMock(
            return_value=[[-1.0]] * 5
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with (
            patch(
                "customized_areal.tree_search.tree_search_grouped_workflow._input_ids_to_messages",
                return_value=[{"role": "user", "content": "hello"}],
            ),
            patch(
                "customized_areal.tree_search.core.selected_turn_distill.selected_turn_to_position_rewards",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result_nodes, rewards = await wf._prepare_distill_for_episode(
                nodes=nodes,
                data={"answer": "42"},
                engine=MagicMock(),
                provider=mock_provider,
                tokenizer=mock_tokenizer,
            )

        # Diagnosis should NOT be called because guidance is cached on last node
        mock_provider.diagnose_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_diagnose_when_no_guidance(self):
        """If the last node's guidance is None/falsy, diagnosis is called."""
        wf = _make_workflow()
        nodes = [_make_node(node_id="n1", turn_idx=1, seq_len=30, response_len=5)]
        nodes[0].guidance = None  # explicit

        mock_provider = AsyncMock()
        mock_provider.diagnose_episode = AsyncMock(return_value=(
            "```xml\n"
            "<diagnosis><turns><turn><turn_idx>1</turn_idx>"
            "<should_improve>true</should_improve>"
            "<guidance>Fix this</guidance></turn></turns></diagnosis>\n"
            "```"
        ))
        mock_provider.get_logprobs_for_prompt = AsyncMock(
            return_value=[[-1.5]] * 5
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with (
            patch(
                "customized_areal.tree_search.tree_search_grouped_workflow._input_ids_to_messages",
                return_value=[{"role": "user", "content": "hello"}],
            ),
            patch(
                "customized_areal.tree_search.core.selected_turn_distill.selected_turn_to_position_rewards",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result_nodes, rewards = await wf._prepare_distill_for_episode(
                nodes=nodes,
                data={"answer": "42"},
                engine=MagicMock(),
                provider=mock_provider,
                tokenizer=mock_tokenizer,
            )

        mock_provider.diagnose_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_distill_mode_empty_on_no_selected_turns(self):
        """In DISTILL mode, no selected turns → empty result."""
        wf = _make_workflow(loss_mode=LossMode.DISTILL)
        nodes = [_make_node(node_id="n1", turn_idx=1, seq_len=30, response_len=5)]

        mock_provider = AsyncMock()
        mock_provider.diagnose_episode = AsyncMock(return_value=(
            "```xml\n<diagnosis><turns /></diagnosis>\n```"
        ))
        mock_tokenizer = MagicMock()

        with patch(
            "customized_areal.tree_search.tree_search_grouped_workflow._input_ids_to_messages",
            return_value=[{"role": "user", "content": "hello"}],
        ):
            result_nodes, rewards = await wf._prepare_distill_for_episode(
                nodes=nodes,
                data={"answer": "42"},
                engine=MagicMock(),
                provider=mock_provider,
                tokenizer=mock_tokenizer,
            )

        assert result_nodes == []
        assert rewards == {}

    @pytest.mark.asyncio
    async def test_guidance_cached_on_last_node_after_diagnose(self):
        """After first diagnosis, guidance is cached on last node for subsequent calls."""
        wf = _make_workflow()
        nodes = [_make_node(node_id="n1", turn_idx=1, seq_len=30, response_len=5)]
        nodes[0].guidance = None

        mock_provider = AsyncMock()
        mock_provider.diagnose_episode = AsyncMock(return_value=(
            "```xml\n"
            "<diagnosis><turns><turn><turn_idx>1</turn_idx>"
            "<should_improve>true</should_improve>"
            "<guidance>Fix this</guidance></turn></turns></diagnosis>\n"
            "```"
        ))
        mock_provider.get_logprobs_for_prompt = AsyncMock(
            return_value=[[-1.5]] * 5
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with (
            patch(
                "customized_areal.tree_search.tree_search_grouped_workflow._input_ids_to_messages",
                return_value=[{"role": "user", "content": "hello"}],
            ),
            patch(
                "customized_areal.tree_search.core.selected_turn_distill.selected_turn_to_position_rewards",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result_nodes, rewards = await wf._prepare_distill_for_episode(
                nodes=nodes,
                data={"answer": "42"},
                engine=MagicMock(),
                provider=mock_provider,
                tokenizer=mock_tokenizer,
            )

        # After diagnose, guidance should be cached on last node
        assert nodes[0].guidance is not None
        assert 1 in nodes[0].guidance


# ---------------------------------------------------------------------------
# 5. _prepare_distill_for_node_groups (parallel episode distillation)
# ---------------------------------------------------------------------------


class TestPrepareDistillForNodeGroups:
    @pytest.mark.asyncio
    async def test_parallel_episodes(self):
        wf = _make_workflow()
        ep1 = [_make_node(node_id="a", episode_id="ep1", turn_idx=1, guidance={1: "fix"})]
        ep2 = [_make_node(node_id="b", episode_id="ep2", turn_idx=1, guidance={1: "fix"})]

        mock_provider = AsyncMock()
        mock_tokenizer = MagicMock()

        with patch.object(
            wf, "_prepare_distill_for_episode", new_callable=AsyncMock
        ) as mock_ep:
            mock_ep.side_effect = [
                (ep1, {"a": []}),
                (ep2, {"b": []}),
            ]
            all_nodes, rewards = await wf._prepare_distill_for_node_groups(
                [ep1, ep2], {"answer": "42"}, MagicMock(), mock_provider, mock_tokenizer
            )

        assert len(all_nodes) == 2
        assert mock_ep.call_count == 2

    @pytest.mark.asyncio
    async def test_exception_in_one_episode_doesnt_kill_others(self):
        wf = _make_workflow(loss_mode=LossMode.GRPO)
        ep1 = [_make_node(node_id="a", episode_id="ep1", turn_idx=1)]
        ep2 = [_make_node(node_id="b", episode_id="ep2", turn_idx=1)]

        mock_provider = AsyncMock()
        mock_tokenizer = MagicMock()

        with patch.object(
            wf, "_prepare_distill_for_episode", new_callable=AsyncMock
        ) as mock_ep:
            mock_ep.side_effect = [
                RuntimeError("boom"),
                (ep2, {"b": []}),
            ]
            all_nodes, rewards = await wf._prepare_distill_for_node_groups(
                [ep1, ep2], {"answer": "42"}, MagicMock(), mock_provider, mock_tokenizer
            )

        assert len(all_nodes) >= 1


# ---------------------------------------------------------------------------
# 6. tree_store insert / cache lookup
# ---------------------------------------------------------------------------


class TestTreeStoreOperations:
    def test_insert_and_count(self):
        store = MCTSTreeStore()
        n1 = _make_node(node_id="n1", query_id="q1", episode_id="ep1")
        n2 = _make_node(node_id="n2", query_id="q1", episode_id="ep1")
        store.insert_batch([n1, n2])

        assert store.get_untrained_episode_count("q1") == 1
        assert store.get_untrained_count("q1") == 2

    def test_mark_trained(self):
        with patch.dict("os.environ", {"TRAIN_ID": "run1"}, clear=False):
            store = MCTSTreeStore()
            n1 = _make_node(node_id="n1", query_id="q1", episode_id="ep1")
            store.insert_batch([n1])
            assert store.get_untrained_episode_count("q1") == 1

            store.set_trained("n1", True)
            assert store.get_untrained_episode_count("q1") == 0

    def test_cross_episode_counting(self):
        with patch.dict("os.environ", {"TRAIN_ID": "run1"}, clear=False):
            store = MCTSTreeStore()
            store.current_train_id = "run1"
            for ep in ("ep1", "ep2"):
                for t in (1, 2):
                    n = _make_node(
                        node_id=f"n_{ep}_{t}",
                        query_id="q1",
                        episode_id=ep,
                        turn_idx=t,
                    )
                    store.insert_batch([n])

            assert store.get_untrained_episode_count("q1") == 2

            for t in (1, 2):
                store.set_trained(f"n_ep1_{t}", True)
            assert store.get_untrained_episode_count("q1") == 1

    def test_load_untrained_episodes(self):
        with patch.dict("os.environ", {"TRAIN_ID": "run1"}, clear=False):
            store = MCTSTreeStore()
            store.current_train_id = "run1"
            for ep in ("ep1", "ep2", "ep3"):
                for t in (1, 2):
                    n = _make_node(
                        node_id=f"n_{ep}_{t}",
                        query_id="q1",
                        episode_id=ep,
                        turn_idx=t,
                    )
                    store.insert_batch([n])

            nodes = store.load_untrained_episodes("q1", 2)
            episode_ids = {n.episode_id for n in nodes}
            assert len(episode_ids) == 2
            assert len(nodes) == 4  # 2 episodes × 2 turns each


# ---------------------------------------------------------------------------
# 7. tree_advantage_computer
# ---------------------------------------------------------------------------


class TestTreeAdvantageComputer:
    def test_grpo_normalization(self):
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)

        # Need ≥2 episodes with DIFFERENT rewards for non-zero normalization
        nodes = []
        for i, reward in enumerate([0.2, 0.8]):
            n = _make_node(
                node_id=f"n_{i}",
                query_id="q1",
                episode_id=f"ep_{i}",
                outcome_reward=reward,
                seq_len=30,
                response_len=5,
            )
            nodes.append(n)
        store.insert_batch(nodes)

        computer.compute(nodes)

        # Advantages should be normalized within query group
        for node in nodes:
            assert node.advantages is not None
            assert node.returns is not None
            # Response positions should have non-zero advantages
            # (2 different rewards → non-zero after normalization)
            adv = node.advantages
            assert adv.dim() == 1
            response_adv = adv[torch.tensor(node.loss_mask) == 1]
            assert response_adv.abs().sum().item() > 0

    def test_single_episode_zero_advantage(self):
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)

        # Single episode → advantages are 0 (no normalization possible)
        n = _make_node(
            node_id="n0",
            query_id="q1",
            episode_id="ep0",
            outcome_reward=1.0,
            seq_len=30,
            response_len=5,
        )
        store.insert_batch([n])
        computer.compute([n])
        assert n.advantages is not None
        assert n.advantages.abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# 8. _nodes_to_batched_tensor_dict
# ---------------------------------------------------------------------------


class TestNodesToBatchedTensorDict:
    def test_basic_conversion(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _nodes_to_batched_tensor_dict,
        )

        nodes = [
            _make_node(node_id="n1", seq_len=30, response_len=5),
            _make_node(node_id="n2", seq_len=40, response_len=8),
        ]
        result = _nodes_to_batched_tensor_dict(nodes)
        assert result is not None
        assert "input_ids" in result
        assert "loss_mask" in result
        assert "logprobs" in result
        assert result["input_ids"].shape[0] == 2  # batch dim

    def test_empty_nodes(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _nodes_to_batched_tensor_dict,
        )

        assert _nodes_to_batched_tensor_dict([]) is None

    def test_max_tokens_truncation(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _nodes_to_batched_tensor_dict,
        )

        nodes = [_make_node(node_id="n1", seq_len=100, response_len=10)]
        result = _nodes_to_batched_tensor_dict(nodes, max_tokens=50)
        assert result is not None
        assert result["input_ids"].shape[1] == 50

    def test_distill_mode_adds_teacher_logp(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _nodes_to_batched_tensor_dict,
        )

        nodes = [_make_node(node_id="n1", seq_len=30, response_len=5)]
        result = _nodes_to_batched_tensor_dict(nodes, loss_mode="distill")
        assert "teacher_logp" in result

    def test_grpo_mode_no_teacher_logp(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _nodes_to_batched_tensor_dict,
        )

        nodes = [_make_node(node_id="n1", seq_len=30, response_len=5)]
        result = _nodes_to_batched_tensor_dict(nodes, loss_mode="grpo")
        assert "teacher_logp" not in result

    def test_with_topk_ids(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _nodes_to_batched_tensor_dict,
        )

        n = _make_node(
            node_id="n1",
            seq_len=30,
            response_len=5,
            topk_ids=[[1, 2, 3]] * 5,
            topk_logp=[[-0.1, -0.5, -1.0]] * 5,
        )
        result = _nodes_to_batched_tensor_dict([n], loss_mode="distill")
        assert "topk_ids" in result
        assert result["topk_ids"].shape[-1] == 3  # top-k dimension


# ---------------------------------------------------------------------------
# 9. Full _arun_episode_impl  (end-to-end with mocked I/O)
# ---------------------------------------------------------------------------


class TestFullArunEpisode:
    @pytest.mark.asyncio
    async def test_grpo_mode_no_distill(self):
        """GRPO mode skips distillation entirely — fastest path."""
        wf = _make_workflow(loss_mode=LossMode.GRPO, cache_mode=CacheMode.OFF)

        interactions = _make_interaction_dict(n_turns=2)
        wf.workflow.arun_episode = AsyncMock(return_value=interactions)

        result = await wf._arun_episode_impl(
            engine=MagicMock(),
            data={"query_id": "q1", "answer": "42"},
        )

        assert result is not None
        assert "input_ids" in result
        # group_size=2, so 2 episodes × 2 turns = 4 nodes in result
        assert result["input_ids"].shape[0] == 4

    @pytest.mark.asyncio
    async def test_cross_training_cache_reuse(self):
        """CROSS_TRAINING mode loads cached nodes and generates fewer fresh episodes."""
        wf = _make_workflow(
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.CROSS_TRAINING,
            group_size=2,
        )

        # Pre-seed the tree store with 1 untrained episode
        cached_node = _make_node(
            node_id="cached_n1",
            query_id="q1",
            episode_id="cached_ep1",
            turn_idx=1,
        )
        wf.tree_store.insert_batch([cached_node])

        # Only 1 fresh episode needed (group_size=2, cached=1)
        interactions = _make_interaction_dict(n_turns=1)
        wf.workflow.arun_episode = AsyncMock(return_value=interactions)

        result = await wf._arun_episode_impl(
            engine=MagicMock(),
            data={"query_id": "q1", "answer": "42"},
        )

        assert result is not None
        # Should only call inner workflow once (not twice)
        assert wf.workflow.arun_episode.call_count == 1

    @pytest.mark.asyncio
    async def test_distill_mode_with_mocked_teacher(self):
        """DISTILL mode with mocked teacher calls — verify provider setup and teardown."""
        wf = _make_workflow(loss_mode=LossMode.BOTH, cache_mode=CacheMode.OFF)

        interactions = _make_interaction_dict(n_turns=2)
        wf.workflow.arun_episode = AsyncMock(return_value=interactions)

        mock_provider = AsyncMock()
        mock_provider.diagnose_episode = AsyncMock(return_value=(
            "```xml\n<diagnosis><turns /></diagnosis>\n```"
        ))

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with (
            patch.object(
                wf, "_setup_distill_provider",
                new_callable=AsyncMock,
                return_value=(mock_provider, mock_client),
            ),
            patch.object(
                wf, "_get_tokenizer",
                new_callable=AsyncMock,
                return_value=mock_tokenizer,
            ),
            patch(
                "customized_areal.tree_search.tree_search_grouped_workflow._input_ids_to_messages",
                return_value=[{"role": "user", "content": "hello"}],
            ),
        ):
            result = await wf._arun_episode_impl(
                engine=MagicMock(),
                data={"query_id": "q1", "answer": "42"},
            )

        # BOTH mode returns nodes even when no turns selected for distill
        assert result is not None
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_query_id_no_cache(self):
        """No query_id means no cache lookup, always generate fresh."""
        wf = _make_workflow(loss_mode=LossMode.GRPO)

        interactions = _make_interaction_dict(n_turns=1)
        wf.workflow.arun_episode = AsyncMock(return_value=interactions)

        result = await wf._arun_episode_impl(
            engine=MagicMock(),
            data={"answer": "42"},  # no query_id
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_failed_episode_returns_none(self):
        """If all episodes fail, arun_episode returns None."""
        wf = _make_workflow(loss_mode=LossMode.GRPO)

        wf.workflow.arun_episode = AsyncMock(return_value=RuntimeError("boom"))

        result = await wf.arun_episode(
            engine=MagicMock(),
            data={"query_id": "q1"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_all_cached_no_generation(self):
        """When cache has enough episodes, no fresh generation needed."""
        wf = _make_workflow(
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.CROSS_TRAINING,
            group_size=2,
        )

        # Pre-seed with 2 untrained episodes (enough for group_size=2)
        for i in range(2):
            n = _make_node(
                node_id=f"cached_n{i}",
                query_id="q1",
                episode_id=f"cached_ep{i}",
                turn_idx=1,
            )
            wf.tree_store.insert_batch([n])

        wf.workflow.arun_episode = AsyncMock()

        result = await wf._arun_episode_impl(
            engine=MagicMock(),
            data={"query_id": "q1", "answer": "42"},
        )

        assert result is not None
        # No fresh episodes needed — inner workflow not called
        wf.workflow.arun_episode.assert_not_called()


# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------


class TestProfiling:
    """Add timing to identify bottlenecks. Run with -s to see output."""

    @pytest.mark.asyncio
    async def test_profile_stages(self):
        """Time each pipeline stage with synthetic data."""
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _nodes_to_batched_tensor_dict,
            _group_nodes_by_episode,
        )
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        N_EPISODES = 4
        TURNS_PER_EP = 3
        SEQ_LEN = 200
        RESP_LEN = 30

        # Stage 1: Node creation
        t0 = time.perf_counter()
        all_nodes = []
        for ep in range(N_EPISODES):
            for t in range(TURNS_PER_EP):
                n = _make_node(
                    node_id=f"n_ep{ep}_t{t}",
                    query_id="q1",
                    episode_id=f"ep{ep}",
                    turn_idx=t + 1,
                    seq_len=SEQ_LEN,
                    response_len=RESP_LEN,
                    outcome_reward=float(ep),
                )
                all_nodes.append(n)
        t1 = time.perf_counter()
        print(f"\n[PROFILE] Node creation ({N_EPISODES * TURNS_PER_EP} nodes): {t1 - t0:.4f}s")

        # Stage 2: Tree store insert
        store = MCTSTreeStore()
        t0 = time.perf_counter()
        store.insert_batch(all_nodes)
        t1 = time.perf_counter()
        print(f"[PROFILE] Tree store insert: {t1 - t0:.4f}s")

        # Stage 3: Cache lookup
        t0 = time.perf_counter()
        count = store.get_untrained_episode_count("q1")
        cached = store.load_untrained_episodes("q1", count)
        t1 = time.perf_counter()
        print(f"[PROFILE] Cache lookup ({count} episodes, {len(cached)} nodes): {t1 - t0:.4f}s")

        # Stage 4: Advantage computation
        computer = TreeAdvantageComputer(store)
        t0 = time.perf_counter()
        computer.compute(all_nodes)
        t1 = time.perf_counter()
        print(f"[PROFILE] Advantage computation: {t1 - t0:.4f}s")

        # Stage 5: Group nodes by episode
        t0 = time.perf_counter()
        groups = _group_nodes_by_episode(all_nodes)
        t1 = time.perf_counter()
        print(f"[PROFILE] Group nodes by episode ({len(groups)} groups): {t1 - t0:.4f}s")

        # Stage 6: Tensor dict conversion
        t0 = time.perf_counter()
        result = _nodes_to_batched_tensor_dict(all_nodes, loss_mode="distill")
        t1 = time.perf_counter()
        print(f"[PROFILE] Tensor dict conversion: {t1 - t0:.4f}s")
        assert result is not None

        # Stage 7: Mark trained
        t0 = time.perf_counter()
        for node in all_nodes:
            if node.node_id:
                store.set_trained(node.node_id, True)
        t1 = time.perf_counter()
        print(f"[PROFILE] Mark trained: {t1 - t0:.4f}s")

        print(
            "[PROFILE] Note: diagnose + teacher logprobs are network-bound and NOT "
            "included here — mock them and time separately if needed."
        )


# ---------------------------------------------------------------------------
# _input_ids_to_messages
# ---------------------------------------------------------------------------


class TestInputIdsToMessages:
    def test_basic_parsing(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _input_ids_to_messages,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi<|im_end|>"
        )
        mock_tokenizer.decode.return_value = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi<|im_end|>"
        )

        messages = _input_ids_to_messages(list(range(20)), mock_tokenizer)
        assert len(messages) >= 1

    def test_fallback_on_decode_failure(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _input_ids_to_messages,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("no template")
        mock_tokenizer.decode.return_value = "Just some text"

        messages = _input_ids_to_messages(list(range(10)), mock_tokenizer)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# _filter_distill_episode_failure
# ---------------------------------------------------------------------------


class TestFilterDistillEpisodeFailure:
    def test_distill_mode_returns_empty(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _filter_distill_episode_failure,
        )

        nodes = [_make_node(node_id="n1")]
        result = _filter_distill_episode_failure(nodes, LossMode.DISTILL)
        assert result == []

    def test_non_distill_mode_returns_nodes(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _filter_distill_episode_failure,
        )

        nodes = [_make_node(node_id="n1")]
        result = _filter_distill_episode_failure(nodes, LossMode.GRPO)
        assert result is nodes


# ---------------------------------------------------------------------------
# _group_nodes_by_episode
# ---------------------------------------------------------------------------


class TestGroupNodesByEpisode:
    def test_basic_grouping(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _group_nodes_by_episode,
        )

        n1 = _make_node(node_id="a", episode_id="ep1", turn_idx=1)
        n2 = _make_node(node_id="b", episode_id="ep1", turn_idx=2)
        n3 = _make_node(node_id="c", episode_id="ep2", turn_idx=1)

        groups = _group_nodes_by_episode([n1, n2, n3])
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_missing_episode_id(self):
        from customized_areal.tree_search.tree_search_grouped_workflow import (
            _group_nodes_by_episode,
        )

        n1 = _make_node(node_id="a", episode_id="", turn_idx=1)
        groups = _group_nodes_by_episode([n1])
        assert len(groups) == 1
