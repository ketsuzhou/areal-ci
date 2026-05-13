"""Customized AReaL components for TPFC Agent."""

# Delay imports to avoid circular dependency issues when importing just configs
# Use direct imports from submodules instead:
#   from customized_areal.tpfc.tpfc_config import TPFCConfig
#   from customized_areal.tpfc.tpfc_agent import TPFCAgent

__all__ = ["TPFCConfig", "TPFCAgent"]


# Lazy imports
def __getattr__(name):
    if name == "TPFCConfig":
        from customized_areal.tpfc.tpfc_config import TPFCConfig

        return TPFCConfig
    elif name == "TPFCAgent":
        from customized_areal.tpfc.tpfc_agent import TPFCAgent

        return TPFCAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
