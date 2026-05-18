"""Builtin agent configurations."""

from __future__ import annotations

TPFC_CONFIG = {
    "id": "tpfc",
    "version": "v1",
    "name": "TPFC",
    "description": "TPFC is a specialized agent designed to learn from experience, store task solutions, strategies, skills as well as retrieve them when needed to improve future task completions.",
    "is_default": False,
    "metadata": {"visible": False},
    "config": {
        "system_prompt": "",
        "model": "openrouter/qwen/qwen3-vl-8b-thinking",
        "max_iterations": 50,
        "max_tokens": 16384,
        "tools": {
            "builtin": [
                {"type": "tool", "name": t, "enabled": True}
                for t in [
                    "sb_shell_tool",
                    "searching_tool",
                    "document_reading_tool",
                    "sb_files_tool",
                    "audio_analysis_tool",
                    "video_analysis_tool",
                    "sb_vision_tool",
                ]
            ],
            "mcp": [],
        },
        "triggers": [],
        "context_manager_type": "tpfc",
    },
    "restrictions": {},
}
