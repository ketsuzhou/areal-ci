"""TPFC reward function using LLM-as-judge.

Refactored from gaia_final_reward.py to match AReaL reward function interface.
"""

import json
import os
import ssl
import urllib.error
import urllib.request

from areal.utils import logging

logger = logging.getLogger("TPFCReward")


def _flatten_content(content):
    """Convert OpenAI-style content (str, dict, or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", part.get("content", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    if isinstance(content, dict):
        return content.get("text", content.get("content", str(content)))
    return str(content)


def _extract_answer_tag(response_text: str) -> str | None:
    """Extract content between <answer> tags."""
    response_text = _flatten_content(response_text)
    start_tag = "<answer>"
    end_tag = "</answer>"
    start_positions = [
        i for i in range(len(response_text)) if response_text.startswith(start_tag, i)
    ]
    end_positions = [
        i for i in range(len(response_text)) if response_text.startswith(end_tag, i)
    ]

    for start_pos in reversed(start_positions):
        for end_pos in end_positions:
            if end_pos > start_pos:
                content_start = start_pos + len(start_tag)
                return response_text[content_start:end_pos].strip()
        break
    return None


def _build_judge_prompt(model_answer: str, ground_truth: str, user_query: str) -> str:
    """Build the LLM judge prompt."""
    return f"""You are a strict correctness evaluator. Assign a score of 1.0 only if the model answer is fully correct and appropriate; otherwise 0.0.

    User Question: "{user_query}"
    Model Answer: "{model_answer}"
    Ground Truth: "{ground_truth}"

    Scoring:
    - 1.0: Answer is factually correct, complete, relevant, and obeys format constraints (e.g., "no abbreviations" -> "USA" is invalid; "numeric only" -> no text). Minor differences in units, rounding, articles, punctuation, or whitespace are acceptable.
    - 0.0: Anything else - including missing info, wrong fact, irrelevant content, or violation of explicit/implicit format.

    Output ONLY in the format:
    <score>1.0</score> or <score>0.0</score>
    (no explanations, no extra text)
    """.strip()


def _call_llm_judge(prompt: str) -> str:
    """Call LLM judge API with environment-based configuration."""
    # Get configuration from environment variables
    model_name = os.getenv("TPFC_JUDGE_MODEL", "qwen2.5-72b-instruct")
    base_url = os.getenv(
        "TPFC_JUDGE_BASE_URL",
        "http://10.254.10.192:8443/service-large-64-1772072563849/llm/v1",
    )
    api_key = os.getenv(
        "TPFC_JUDGE_API_KEY",
        "RlkgHzgBa2zbPcQrw96rdn68dr7kXk8Mv84Nhs5Trrk6gfq8Cqw5CcDQ787p6d6bPrAWDrw8b97gq8N7jWWxhasnVP76FD76r8tJ2688mdPnSP7N6V7gl1VLKD9LasJq",
    )
    timeout = float(os.getenv("TPFC_JUDGE_TIMEOUT", "120.0"))

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 8192,
        "top_p": 0.95,
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )

    # Use unverified SSL context for internal services
    ssl_context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=ssl_context
        ) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM judge HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM judge request failed: {exc}") from exc

    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"].strip()


def _parse_judge_score(judge_response_text: str) -> float:
    """Parse the score from judge response."""
    stripped = judge_response_text.strip()
    start_pos = stripped.rfind("<score>")
    end_pos = stripped.rfind("</score>")
    if start_pos < 0 or end_pos < 0 or start_pos >= end_pos:
        return 0.0

    content = stripped[start_pos + 7 : end_pos].strip()
    try:
        score = float(content)
        return 1.0 if score >= 0.9 else 0.0
    except (TypeError, ValueError):
        return 0.0


def tpfc_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """TPFC reward function using LLM-as-judge.

    This reward function extracts the answer from the model's completion using
    <answer> tags, then uses an LLM judge to evaluate correctness against the
    ground truth.

    Args:
        prompt: The original prompt (messages or string)
        completions: Model's generated completion
        prompt_ids: Token IDs of prompt
        completion_ids: Token IDs of completion
        answer: Ground truth answer
        **kwargs: Additional arguments

    Returns:
        Reward score (0.0 or 1.0)
    """
    try:
        # Extract model answer from completion
        completion_str = (
            str(completions) if not isinstance(completions, str) else completions
        )
        model_answer = _extract_answer_tag(completion_str)

        if model_answer is None:
            logger.debug("No <answer> tag found in completion")
            return 0.0

        # Extract ground truth
        ground_truth = str(answer) if answer is not None else ""

        # Extract user query from prompt
        user_query = ""
        if isinstance(prompt, list) and len(prompt) > 0:
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break

        # Build and send judge prompt
        judge_prompt = _build_judge_prompt(model_answer, ground_truth, user_query)
        judge_response = _call_llm_judge(judge_prompt)

        # Parse score
        score = _parse_judge_score(judge_response)

        logger.debug(
            f"TPFC reward: score={score}, model_answer={model_answer[:100]}..."
        )

        return score

    except Exception as e:
        logger.warning(f"Exception in tpfc_reward_fn: {e}", exc_info=True)
        return 0.0
