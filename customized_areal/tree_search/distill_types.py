"""Distillation types: PositionRewardInfo and InteractionWithTokenLevelReward."""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.experimental.openai.types import InteractionWithTokenLogpReward


@dataclass
class PositionRewardInfo:
    """Reward information for a single generation position.

    Stores candidate tokens, their log probabilities, and computed rewards
    at a specific position. Used for KL-based rewards.
    """

    position: int
    candidates: list[str] = field(default_factory=list)
    candidate_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] | None = None
    teacher_logprobs: list[float] | None = None
    rewards: list[float] = field(default_factory=list)
    chosen_index: int = 0
    sample_index: int = 0


@dataclass(frozen=True)
class DiagnosisTurn:
    turn_idx: int
    should_improve: bool
    guidance: str = ""

    @property
    def is_selected(self) -> bool:
        return self.should_improve and bool(self.guidance.strip())


@dataclass(frozen=True)
class EpisodeDiagnosis:
    turns: tuple[DiagnosisTurn, ...]

    @property
    def selected_turns(self) -> dict[int, str]:
        return {turn.turn_idx: turn.guidance for turn in self.turns if turn.is_selected}


@dataclass
class InteractionWithTokenLevelReward(InteractionWithTokenLogpReward):
    """Extended interaction class that supports token-level rewards.

    The ``rewards`` key in the tensor dict is always the trajectory-level
    scalar reward (used by tree backup and GAE). Per-token position-level
    rewards are stored in the separate ``token_rewards`` key (used for
    distillation).
    """

    token_rewards: list[float] | None = None
    token_reward_mask: list[int] | None = None

    def __post_init__(self):
        """Validate token-level reward dimensions."""
        if self.model_response is not None and self.token_rewards is not None:
            expected_len = len(self.model_response.output_tokens)
            if len(self.token_rewards) != expected_len:
                raise ValueError(
                    f"token_rewards length ({len(self.token_rewards)}) must match "
                    f"output_tokens length ({expected_len})"
                )
