"""Lazy exports for tree-search FSDP engine components."""

from .fsdp_engine import MultiCandidateFSDPEngine

__all__ = ["MultiCandidateFSDPEngine", "MultiCandidateFSDPPPOActor"]


def __getattr__(name: str):
    if name == "MultiCandidateFSDPPPOActor":
        from ..training.actor import MultiCandidateFSDPPPOActor

        return MultiCandidateFSDPPPOActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
