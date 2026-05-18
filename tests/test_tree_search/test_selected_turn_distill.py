from customized_areal.tree_search.config import TreeBackupConfig
from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    PositionRewardInfo,
)


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


def test_tree_backup_config_has_distill_defaults():
    config = TreeBackupConfig()

    assert config.topk_distill is False
    assert config.teacher_provider == "external"
    assert config.teacher_top_k == 10
    assert config.diagnose_temperature == 0.0
    assert config.strict_distill_json is True
