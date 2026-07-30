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
        "model": "anthropic/claude-sonnet-4-6-20260217",
        "max_tokens": 16384,
        "tools": {
            "builtin": [
                {"type": "tool", "name": t, "enabled": True}
                for t in [
                    "sb_vision_tool",
                    "sb_shell_tool",
                    "sb_upload_file_tool",
                    "searching_tool",
                    "sb_document_parser",
                    "sb_files_tool",
                    "audio_analysis_tool",
                ]
            ],
            "mcp": [],
        },
        "triggers": [],
        "context_manager_type": "tpfc",
    },
    "restrictions": {},
}
