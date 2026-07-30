"""Lazy exports for tree-search FSDP engine components."""

from .fsdp_engine import MultiCandidateFSDPEngine

__all__ = [
    "MultiCandidateFSDPEngine",
    "MultiCandidateFSDPPPOActor",
    "VIMPOFSDPPPOActor",
    "MuonVIMPOFSDPPPOActor",
]


def __getattr__(name: str):
    if name == "MultiCandidateFSDPPPOActor":
        from ..training.actor import MultiCandidateFSDPPPOActor

        return MultiCandidateFSDPPPOActor
    if name == "VIMPOFSDPPPOActor":
        from ..training.actor import VIMPOFSDPPPOActor

        return VIMPOFSDPPPOActor
    if name == "MuonVIMPOFSDPPPOActor":
        from ..training.actor import MuonVIMPOFSDPPPOActor

        return MuonVIMPOFSDPPPOActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
