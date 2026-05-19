import pytest

from customized_areal.tree_search.config import TreeBackupConfig
from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    EpisodeDiagnosis,
    PositionRewardInfo,
)
from customized_areal.tree_search.mcts_tree_store import Node


class FakeTokenizer:
    def encode(self, text):
        if "Improve this selected assistant turn using this guidance:" in text:
            return [900, 901]
        return [ord(ch) for ch in text]


class FakeProvider:
    def __init__(self, teacher_logprobs):
        self.teacher_logprobs = teacher_logprobs
        self.calls = []

    async def get_logprobs_for_prompt(
        self, prompt_ids, generation_ids, candidate_token_ids
    ):
        self.calls.append(
            {
                "prompt_ids": prompt_ids,
                "generation_ids": generation_ids,
                "candidate_token_ids": candidate_token_ids,
            }
        )
        return self.teacher_logprobs


class FakeTopKEngine:
    def __init__(self, topk_ids, topk_logp):
        self.topk_ids = topk_ids
        self.topk_logp = topk_logp
        self.calls = []

    async def get_topk_logprobs(self, input_ids, loss_mask, top_k):
        self.calls.append(
            {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "top_k": top_k,
            }
        )
        return self.topk_ids, self.topk_logp


class AwaitableTopK:
    def __init__(self, result):
        self.result = result

    def __await__(self):
        async def _return_result():
            return self.result

        return _return_result().__await__()


class AwaitableReturningTopKEngine:
    def __init__(self, topk_ids, topk_logp):
        self.topk_ids = topk_ids
        self.topk_logp = topk_logp

    def get_topk_logprobs(self, input_ids, loss_mask, top_k):
        return AwaitableTopK((self.topk_ids, self.topk_logp))


def test_position_reward_info_carries_teacher_logprobs():
    info = PositionRewardInfo(
        position=0,
        candidate_token_ids=[11, 12],
        logprobs=[-0.7, -1.3],
        teacher_logprobs=[-0.4, -2.0],
        rewards=[-0.3, 0.7],
        sample_index=3,
    )

    assert info.teacher_logprobs == [-0.4, -2.0]
    assert info.sample_index == 3


def test_diagnosis_turn_requires_guidance_for_selected_turns():
    selected = DiagnosisTurn(turn_idx=2, should_improve=True, guidance="Use the tool.")
    skipped = DiagnosisTurn(turn_idx=3, should_improve=False, guidance="")
    blank_guidance = DiagnosisTurn(turn_idx=4, should_improve=True, guidance="   ")

    assert selected.is_selected is True
    assert skipped.is_selected is False
    assert blank_guidance.is_selected is False


def test_episode_diagnosis_returns_only_selected_turn_guidance():
    diagnosis = EpisodeDiagnosis(
        turns=(
            DiagnosisTurn(turn_idx=0, should_improve=False, guidance="Ignore this."),
            DiagnosisTurn(turn_idx=1, should_improve=True, guidance="Use the tool."),
            DiagnosisTurn(turn_idx=2, should_improve=True, guidance=""),
            DiagnosisTurn(turn_idx=3, should_improve=True, guidance="   "),
        )
    )

    assert diagnosis.selected_turns == {1: "Use the tool."}


def test_package_exports_selected_turn_diagnosis_types():
    from customized_areal.tree_search import (
        DiagnosisTurn as ExportedDiagnosisTurn,
        EpisodeDiagnosis as ExportedEpisodeDiagnosis,
    )

    assert ExportedDiagnosisTurn is DiagnosisTurn
    assert ExportedEpisodeDiagnosis is EpisodeDiagnosis


def test_tree_backup_config_has_distill_defaults():
    config = TreeBackupConfig()

    assert config.topk_distill is False
    assert config.teacher_provider == "external"
    assert config.teacher_base_url == "http://localhost:8001"
    assert config.teacher_model_name == ""
    assert config.teacher_top_k == 10
    assert config.teacher_max_retries == 3
    assert config.teacher_timeout == 60.0
    assert config.teacher_missing_logprob == -23.0
    assert config.diagnose_model_name == ""
    assert config.diagnose_max_tokens == 1024
    assert config.diagnose_temperature == 0.0
    assert config.strict_distill_json is True


def test_parse_episode_diagnosis_keeps_only_selected_turns():
    from customized_areal.tree_search.core.selected_turn_distill import (
        parse_episode_diagnosis,
    )

    diagnosis = parse_episode_diagnosis(
        """
        {
          "turns": [
            {"turn_idx": 1, "should_improve": true, "guidance": "Use exact units."},
            {"turn_idx": 2, "should_improve": false, "guidance": "Ignore this."},
            {"turn_idx": 3, "should_improve": true, "guidance": "   "}
          ]
        }
        """
    )

    assert diagnosis.selected_turns == {1: "Use exact units."}


def test_response_token_span_returns_first_contiguous_one_span():
    from customized_areal.tree_search.core.selected_turn_distill import (
        response_token_span,
    )

    assert response_token_span([0, 0, 1, 1, 0]) == (2, 4)
    assert response_token_span([0, 1, 1, 0, 0, 1, 1]) == (1, 3)


def test_build_teacher_prompt_ids_excludes_generation_from_prefix():
    from customized_areal.tree_search.core.selected_turn_distill import (
        build_teacher_prompt_ids,
    )

    node = Node(
        input_ids=[10, 11, 20, 21],
        loss_mask=[0, 0, 1, 1],
        logprobs=[0.0, 0.0, -0.3, -0.4],
        versions=[-1, -1, 0, 0],
    )

    prompt_ids, generation_ids = build_teacher_prompt_ids(
        node, "Be more direct.", FakeTokenizer()
    )

    assert generation_ids == [20, 21]
    assert prompt_ids == [10, 11, 900, 901]


def test_build_teacher_prompt_ids_uses_first_response_span():
    from customized_areal.tree_search.core.selected_turn_distill import (
        build_teacher_prompt_ids,
    )

    node = Node(
        input_ids=[10, 20, 21, 11, 12, 30, 31],
        loss_mask=[0, 1, 1, 0, 0, 1, 1],
        logprobs=[0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4],
        versions=[-1, 0, 0, -1, -1, 0, 0],
    )

    prompt_ids, generation_ids = build_teacher_prompt_ids(
        node, "Fix the first turn.", FakeTokenizer()
    )

    assert generation_ids == [20, 21]
    assert prompt_ids == [10, 900, 901]


@pytest.mark.asyncio
async def test_selected_turn_to_position_rewards_single_candidate_path():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    provider = FakeProvider([[-1.0], [-2.5]])
    node = Node(
        input_ids=[10, 11, 20, 21],
        loss_mask=[0, 0, 1, 1],
        logprobs=[0.0, 0.0, -0.3, -0.4],
        versions=[-1, -1, 0, 0],
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Be more direct.",
        tokenizer=FakeTokenizer(),
        provider=provider,
        sample_index=7,
        topk_distill=False,
        engine=None,
        teacher_top_k=10,
    )

    assert provider.calls == [
        {
            "prompt_ids": [10, 11, 900, 901],
            "generation_ids": [20, 21],
            "candidate_token_ids": [[20], [21]],
        }
    ]
    assert rewards == [
        PositionRewardInfo(
            position=0,
            candidates=["20"],
            candidate_token_ids=[20],
            logprobs=[-0.3],
            teacher_logprobs=[-1.0],
            rewards=[0.7],
            chosen_index=0,
            sample_index=7,
        ),
        PositionRewardInfo(
            position=1,
            candidates=["21"],
            candidate_token_ids=[21],
            logprobs=[-0.4],
            teacher_logprobs=[-2.5],
            rewards=[2.1],
            chosen_index=0,
            sample_index=7,
        ),
    ]


@pytest.mark.asyncio
async def test_selected_turn_to_position_rewards_topk_moves_generated_token_first():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    provider = FakeProvider([[-1.0, -2.0, -3.0]])
    node = Node(
        input_ids=[10, 11, 20],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.3],
        versions=[-1, -1, 0],
        topk_ids=[[30, 20, 40]],
        topk_logp=[[-0.1, -0.3, -0.8]],
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Be more direct.",
        tokenizer=FakeTokenizer(),
        provider=provider,
        sample_index=0,
        topk_distill=True,
        engine=None,
        teacher_top_k=10,
    )

    assert provider.calls[0]["candidate_token_ids"] == [[20, 30, 40]]
    assert rewards[0].candidate_token_ids == [20, 30, 40]
    assert rewards[0].logprobs == [-0.3, -0.1, -0.8]
    assert rewards[0].chosen_index == 0


@pytest.mark.asyncio
async def test_selected_turn_to_position_rewards_topk_uses_first_response_rows():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    provider = FakeProvider([[-1.0, -2.0], [-1.5, -2.5]])
    node = Node(
        input_ids=[10, 20, 21, 11, 12, 30],
        loss_mask=[0, 1, 1, 0, 0, 1],
        logprobs=[0.0, -0.1, -0.2, 0.0, 0.0, -0.3],
        versions=[-1, 0, 0, -1, -1, 0],
        topk_ids=[
            [20, 50],
            [21, 51],
            [30, 60],
        ],
        topk_logp=[
            [-0.1, -1.1],
            [-0.2, -1.2],
            [-0.3, -1.3],
        ],
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Fix the first turn.",
        tokenizer=FakeTokenizer(),
        provider=provider,
        sample_index=0,
        topk_distill=True,
        engine=None,
        teacher_top_k=10,
    )

    assert provider.calls[0]["generation_ids"] == [20, 21]
    assert provider.calls[0]["candidate_token_ids"] == [[20, 50], [21, 51]]
    assert rewards[0].candidate_token_ids == [20, 50]
    assert rewards[0].logprobs == [-0.1, -1.1]
    assert rewards[1].candidate_token_ids == [21, 51]
    assert rewards[1].logprobs == [-0.2, -1.2]


@pytest.mark.asyncio
async def test_selected_turn_to_position_rewards_topk_accepts_selected_response_rows():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    provider = FakeProvider([[-1.0, -2.0], [-1.5, -2.5]])
    node = Node(
        input_ids=[10, 20, 21, 11, 12, 30],
        loss_mask=[0, 1, 1, 0, 0, 1],
        logprobs=[0.0, -0.1, -0.2, 0.0, 0.0, -0.3],
        versions=[-1, 0, 0, -1, -1, 0],
        topk_ids=[[20, 50], [21, 51]],
        topk_logp=[[-0.1, -1.1], [-0.2, -1.2]],
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Fix the first turn.",
        tokenizer=FakeTokenizer(),
        provider=provider,
        sample_index=0,
        topk_distill=True,
        engine=None,
        teacher_top_k=10,
    )

    assert provider.calls[0]["generation_ids"] == [20, 21]
    assert provider.calls[0]["candidate_token_ids"] == [[20, 50], [21, 51]]
    assert rewards[0].candidate_token_ids == [20, 50]
    assert rewards[0].logprobs == [-0.1, -1.1]
    assert rewards[1].candidate_token_ids == [21, 51]
    assert rewards[1].logprobs == [-0.2, -1.2]


@pytest.mark.asyncio
async def test_selected_turn_topk_requires_engine_for_missing_cache():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    node = Node(
        input_ids=[10, 11, 20],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.3],
        versions=[-1, -1, 0],
    )

    with pytest.raises(NotImplementedError, match="get_topk_logprobs"):
        await selected_turn_to_position_rewards(
            node=node,
            guidance="Be more direct.",
            tokenizer=FakeTokenizer(),
            provider=FakeProvider([[-1.0]]),
            sample_index=0,
            topk_distill=True,
            engine=None,
            teacher_top_k=2,
        )


@pytest.mark.asyncio
async def test_selected_turn_topk_recomputes_missing_cache_from_full_sequence_rows():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    provider = FakeProvider([[-1.0, -2.0], [-1.5, -2.5]])
    engine = FakeTopKEngine(
        topk_ids=[
            [10, 50],
            [11, 51],
            [20, 60],
            [21, 61],
        ],
        topk_logp=[
            [0.0, -5.0],
            [0.0, -5.1],
            [-0.3, -1.3],
            [-0.4, -1.4],
        ],
    )
    node = Node(
        input_ids=[10, 11, 20, 21],
        loss_mask=[0, 0, 1, 1],
        logprobs=[0.0, 0.0, -0.3, -0.4],
        versions=[-1, -1, 0, 0],
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Be more direct.",
        tokenizer=FakeTokenizer(),
        provider=provider,
        sample_index=0,
        topk_distill=True,
        engine=engine,
        teacher_top_k=2,
    )

    assert engine.calls == [
        {
            "input_ids": [10, 11, 20, 21],
            "loss_mask": [0, 0, 1, 1],
            "top_k": 2,
        }
    ]
    assert node.topk_ids == [[20, 60], [21, 61]]
    assert node.topk_logp == [[-0.3, -1.3], [-0.4, -1.4]]
    assert provider.calls[0]["candidate_token_ids"] == [[20, 60], [21, 61]]
    assert rewards[0].logprobs == [-0.3, -1.3]
    assert rewards[1].logprobs == [-0.4, -1.4]


@pytest.mark.asyncio
async def test_selected_turn_topk_recomputes_missing_cache_from_all_response_rows():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    provider = FakeProvider([[-1.0, -2.0], [-1.5, -2.5]])
    engine = FakeTopKEngine(
        topk_ids=[
            [20, 50],
            [21, 51],
            [30, 60],
        ],
        topk_logp=[
            [-0.1, -1.1],
            [-0.2, -1.2],
            [-0.3, -1.3],
        ],
    )
    node = Node(
        input_ids=[10, 20, 21, 11, 12, 30],
        loss_mask=[0, 1, 1, 0, 0, 1],
        logprobs=[0.0, -0.1, -0.2, 0.0, 0.0, -0.3],
        versions=[-1, 0, 0, -1, -1, 0],
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Fix the first turn.",
        tokenizer=FakeTokenizer(),
        provider=provider,
        sample_index=0,
        topk_distill=True,
        engine=engine,
        teacher_top_k=2,
    )

    assert node.topk_ids == [[20, 50], [21, 51]]
    assert node.topk_logp == [[-0.1, -1.1], [-0.2, -1.2]]
    assert provider.calls[0]["generation_ids"] == [20, 21]
    assert provider.calls[0]["candidate_token_ids"] == [[20, 50], [21, 51]]
    assert rewards[0].candidate_token_ids == [20, 50]
    assert rewards[0].logprobs == [-0.1, -1.1]
    assert rewards[1].candidate_token_ids == [21, 51]
    assert rewards[1].logprobs == [-0.2, -1.2]


@pytest.mark.asyncio
async def test_selected_turn_topk_accepts_callable_returning_awaitable():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    provider = FakeProvider([[-1.0, -2.0]])
    engine = AwaitableReturningTopKEngine(
        topk_ids=[[20, 60]],
        topk_logp=[[-0.3, -1.3]],
    )
    node = Node(
        input_ids=[10, 11, 20],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.3],
        versions=[-1, -1, 0],
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Be direct.",
        tokenizer=FakeTokenizer(),
        provider=provider,
        sample_index=0,
        topk_distill=True,
        engine=engine,
        teacher_top_k=2,
    )

    assert rewards[0].candidate_token_ids == [20, 60]
    assert node.topk_ids == [[20, 60]]


@pytest.mark.asyncio
async def test_selected_turn_topk_rejects_mismatched_id_logprob_layouts():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    engine = FakeTopKEngine(
        topk_ids=[
            [10, 50],
            [20, 60],
        ],
        topk_logp=[[-0.3, -1.3]],
    )
    node = Node(
        input_ids=[10, 20],
        loss_mask=[0, 1],
        logprobs=[0.0, -0.3],
        versions=[-1, 0],
    )

    with pytest.raises(ValueError, match="same layout"):
        await selected_turn_to_position_rewards(
            node=node,
            guidance="Be direct.",
            tokenizer=FakeTokenizer(),
            provider=FakeProvider([[-1.0, -2.0]]),
            sample_index=0,
            topk_distill=True,
            engine=engine,
            teacher_top_k=2,
        )


@pytest.mark.asyncio
async def test_selected_turn_topk_rejects_invalid_same_length_row_layouts():
    from customized_areal.tree_search.core.selected_turn_distill import (
        selected_turn_to_position_rewards,
    )

    engine = FakeTopKEngine(
        topk_ids=[
            [99, -1],
            [20, 50],
            [21, 51],
            [30, 60],
        ],
        topk_logp=[
            [-9.9, -9.1],
            [-0.1, -1.1],
            [-0.2, -1.2],
            [-0.3, -1.3],
        ],
    )
    node = Node(
        input_ids=[10, 20, 21, 11, 12, 30],
        loss_mask=[0, 1, 1, 0, 0, 1],
        logprobs=[0.0, -0.1, -0.2, 0.0, 0.0, -0.3],
        versions=[-1, 0, 0, -1, -1, 0],
    )

    with pytest.raises(ValueError, match="all-response aligned"):
        await selected_turn_to_position_rewards(
            node=node,
            guidance="Fix the first turn.",
            tokenizer=FakeTokenizer(),
            provider=FakeProvider([[-1.0, -2.0]]),
            sample_index=0,
            topk_distill=True,
            engine=engine,
            teacher_top_k=2,
        )
