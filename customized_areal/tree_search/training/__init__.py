"""Lazy exports for tree-search training components."""

__all__ = [
    "MultiCandidateFSDPPPOActor",
    "patch_ppo_actor_class_to_use_distill_loss",
    "unpatch_ppo_actor_distill_loss",
    "grpo_distill_loss_fn",
    "gather_logprobs_entropy_multi_candidates",
    "CustomizedPPOTrainer",
]


def __getattr__(name: str):
    if name in {
        "MultiCandidateFSDPPPOActor",
        "patch_ppo_actor_class_to_use_distill_loss",
        "unpatch_ppo_actor_distill_loss",
    }:
        from .actor import (
            MultiCandidateFSDPPPOActor,
            patch_ppo_actor_class_to_use_distill_loss,
            unpatch_ppo_actor_distill_loss,
        )

        exports = {
            "MultiCandidateFSDPPPOActor": MultiCandidateFSDPPPOActor,
            "patch_ppo_actor_class_to_use_distill_loss": (
                patch_ppo_actor_class_to_use_distill_loss
            ),
            "unpatch_ppo_actor_distill_loss": unpatch_ppo_actor_distill_loss,
        }
        return exports[name]
    if name == "gather_logprobs_entropy_multi_candidates":
        from .logprobs import gather_logprobs_entropy_multi_candidates

        return gather_logprobs_entropy_multi_candidates
    if name == "grpo_distill_loss_fn":
        from .loss import grpo_distill_loss_fn

        return grpo_distill_loss_fn
    if name == "CustomizedPPOTrainer":
        from .trainer import CustomizedPPOTrainer

        return CustomizedPPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
