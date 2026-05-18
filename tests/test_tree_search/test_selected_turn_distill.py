from customized_areal.tree_search.config import TreeBackupConfig
from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    EpisodeDiagnosis,
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
