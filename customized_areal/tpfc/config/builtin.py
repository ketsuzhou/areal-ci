"""Builtin agent configurations."""

from __future__ import annotations

_TOOL_METHODS = [
    "shell",
    # sb_vision_tool methods
    "load_image",
    "clear_images_from_context",
    # document_reading_tool methods
    "read_as_markdown",
    "read_pdf",
    "read_document",
    "search_in_document",
    # sb_files_tool methods
    "create_file",
    "upsert_file",
    "str_replace",
    "full_file_rewrite",
    "delete_file",
    "edit_file",
    # audio_analysis_tool methods
    "audio_transcription",
    "audio_question_answering",
    # video_analysis_tool methods
    "analyze_video",
    # searching_tool methods
    "google_search",
    "scrape_website",
    "wiki_get_page_content",
    "search_wiki_revision",
]

TPFC_CONFIG = {
    "id": "tpfc",
    "role": "main",
    "name": "TPFC",
    "description": "TPFC is a specialized agent designed to learn from experience, store task solutions, strategies, skills as well as retrieve them when needed to improve future task completions.",
    "is_default": False,
    "metadata": {"visible": False},
    "run_config": {"max_iterations": 50},
    "config": {
        "system_prompt": "",
        "max_tokens": 16384,
        "tools": {
            "builtin": [
                {"type": "tool", "name": t, "enabled": True} for t in _TOOL_METHODS
            ],
            "mcp": [],
        },
        "triggers": [],
        "context_manager_type": "tpfc",
        "entropy_aware": True,
    },
    "restrictions": {},
}
