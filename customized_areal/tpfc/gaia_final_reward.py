"""
Minimal final reward logic extracted from `verl_new/trainer/main_ppo.py`.

Only standard-library imports are used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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


def extract_answer(response_text):
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


def build_evaluate_final_answer_prompt(model_answer, ground_truth, user_query):
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


def call_openai_compatible_model(
    prompt,
    model_name,
    base_url,
    api_key,
    system_prompt="You are a helpful assistant.",
    timeout=120.0,
    verify_ssl=False,
    temperature=0.7,
    max_tokens=8192,
    top_p=0.95,
):
    import httpx

    http_client = httpx.Client(verify=False)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        extra_body={"enable_thinking": True},
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


def parse_judge_score(judge_response_text):
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


def evaluate_final_answer(
    model_answer,
    ground_truth,
    user_query,
    model_name,
    base_url,
    api_key,
    system_prompt="You are a helpful assistant.",
    timeout=120.0,
    verify_ssl=False,
):
    prompt = build_evaluate_final_answer_prompt(
        model_answer=model_answer,
        ground_truth=ground_truth,
        user_query=user_query,
    )
    judge_response_text = call_openai_compatible_model(
        prompt=prompt,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    answer_reward = parse_judge_score(judge_response_text)
    return {
        "judge_prompt": prompt,
        "judge_response_text": judge_response_text,
        "answer_reward": answer_reward,
    }


def compute_reward(
    ground_truth,
    user_query,
    answer,
    model_name=None,
    base_url=None,
    api_key=None,
    system_prompt="You are a helpful assistant.",
    timeout=120.0,
    verify_ssl=False,
):
    if answer is None or ground_truth is None:
        raise

    # Resolve judge config: explicit args > env vars > defaults
    _model_name = model_name or os.environ.get(
        "TPFC_JUDGE_MODEL", "qwen/qwen3.5-397b-a17b"
    )
    _base_url = base_url or os.environ.get(
        "WORKSPACE_OPENAI_API_BASE", "https://openrouter.ai/api/v1"
    )
    _api_key = api_key or os.environ.get("WORKSPACE_OPENAI_API_KEY", "")

    try:
        judge_result = evaluate_final_answer(
            model_answer=answer,
            ground_truth=ground_truth,
            user_query=user_query,
            model_name=_model_name,
            base_url=_base_url,
            api_key=_api_key,
            system_prompt=system_prompt,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
    except Exception:
        # Fallback to OpenRouter if WORKSPACE endpoint fails
        fallback_url = "https://openrouter.ai/api/v1"
        fallback_key = os.environ.get("OPENROUTER_API_KEY", "")
        if _base_url != fallback_url or _api_key != fallback_key:
            judge_result = evaluate_final_answer(
                model_answer=answer,
                ground_truth=ground_truth,
                user_query=user_query,
                model_name=_model_name,
                base_url=fallback_url,
                api_key=fallback_key,
                system_prompt=system_prompt,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
        else:
            raise

    return {
        "raw_answer": answer,
        "judge_prompt": judge_result["judge_prompt"],
        "judge_response_text": judge_result["judge_response_text"],
        "answer_reward": judge_result["answer_reward"],
    }


if __name__ == "__main__":
    judge_base_url = os.environ.get("WORKSPACE_OPENAI_API_BASE")
    judge_api_key = os.environ.get("WORKSPACE_OPENAI_API_KEY")

    result = compute_reward(
        response_text="The capital of France is Paris. <answer>Paris</answer>",
        ground_truth="Paris",
        user_query="What is the capital of France?",
        model_name="deepseek-v3",
        base_url=judge_base_url,
        api_key=judge_api_key,
    )
    print(json.dumps(result, indent=2))


__all__ = [
    "extract_answer",
    "build_evaluate_final_answer_prompt",
    "call_openai_compatible_model",
    "parse_judge_score",
    "evaluate_final_answer",
    "compute_reward",
]
