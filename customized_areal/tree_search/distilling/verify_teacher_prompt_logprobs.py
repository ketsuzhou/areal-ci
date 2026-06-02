#!/usr/bin/env python3
"""Verify the teacher API can return prompt logprobs via echo mode.

Tests the same request pattern used by TeacherClient._get_logprobs_openai():
  - Sends token IDs as the prompt (input_ids + output_ids)
  - Uses echo=True + max_tokens=1 + logprobs=N
  - Extracts top-k logprobs at the output positions (prompt logprobs)

Usage:
    uv run customized_areal/tree_search/distilling/verify_teacher_prompt_logprobs.py

    # With custom args:
    uv run customized_areal/tree_search/distilling/verify_teacher_prompt_logprobs.py \
        --base-url http://10.254.244.168:8443/.../llm \
        --model Qwen3.5_397B_A17B_FP8 \
        --api-key KEY \
        --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx
from transformers import AutoTokenizer


DEFAULT_BASE_URL = "http://10.254.244.168:8443/service-large-64-1775803465274/llm/v1"
DEFAULT_MODEL = "Qwen3.5_397B_A17B_FP8"
DEFAULT_TOKENIZER = "/dfs/share-groups/letrain/ckpt/Qwen3.5-9B"
DEFAULT_TOP_K = 10
DEFAULT_API_KEY = "aH6867Z7ppZWqN7NQGsaPhSlMrk68AGQnr18Z66KsSL0864mkjaBP7Tr868pKvnCJcWa0VvHZDHk64MrKG5Z5gWrZGx88mrmAv7pk88p786gFB6XXr6MrtSckKs9QJsr"


async def check_health(client: httpx.AsyncClient, base_url: str, api_key: str) -> bool:
    """Check /v1/models endpoint."""
    resp = await client.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if resp.status_code != 200:
        print(f"[FAIL] /v1/models returned {resp.status_code}: {resp.text[:200]}")
        return False
    data = resp.json()
    models = [m["id"] for m in data.get("data", [])]
    print(f"[OK]   /v1/models: {models}")
    return True


async def test_prompt_logprobs(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    tokenizer: AutoTokenizer,
    top_k: int,
) -> bool:
    """Test echo=True prompt logprobs — the core pattern used by TeacherClient."""
    input_text = "The capital of France is"
    output_text = " Paris"

    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    all_ids = input_ids + output_ids

    print(f"\n--- Prompt Logprob Test ---")
    print(f"  Input text:  {input_text!r}")
    print(f"  Output text: {output_text!r}")
    print(f"  input_ids:   {input_ids}")
    print(f"  output_ids:  {output_ids}")
    print(f"  all_ids:     {all_ids} (len={len(all_ids)})")
    print(f"  prompt_len:  {len(input_ids)}, output_len: {len(output_ids)}")

    payload = {
        "prompt": all_ids,
        "model": model,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": top_k,
        "echo": True,
    }

    t0 = time.time()
    try:
        resp = await client.post(
            f"{base_url}/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(120.0),
        )
    except httpx.HTTPError as e:
        print(f"[FAIL] Request error: {e}")
        return False
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"[FAIL] /v1/completions returned {resp.status_code}: {resp.text[:300]}")
        return False

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        print(f"[FAIL] No choices in response")
        return False

    logprobs_data = choices[0].get("logprobs", {})
    top_logprobs_list = logprobs_data.get("top_logprobs", [])

    total_positions = len(top_logprobs_list)
    output_top_logprobs = top_logprobs_list[len(input_ids) : len(input_ids) + len(output_ids)]

    print(f"\n  Response in {elapsed:.2f}s")
    print(f"  Total logprob positions returned: {total_positions}")
    print(f"  Output logprob positions: {len(output_top_logprobs)} (expected {len(output_ids)})")

    if len(output_top_logprobs) < len(output_ids):
        print(f"[FAIL] Fewer output logprob positions than expected")
        return False

    print(f"\n  Prompt logprobs at output positions:")
    all_ok = True
    for i, pos_entry in enumerate(output_top_logprobs):
        if pos_entry is None:
            print(f"    pos {i}: None (missing)")
            all_ok = False
            continue

        token_entries = list(pos_entry.items()) if isinstance(pos_entry, dict) else pos_entry
        print(f"    pos {i} (output token {output_ids[i]} = {tokenizer.decode([output_ids[i]])!r}):")
        for entry in token_entries[:3]:
            if isinstance(entry, dict):
                tid = entry.get("token_id") or entry.get("id")
                lp = entry.get("logprob", "?")
                txt = tokenizer.decode([tid]) if isinstance(tid, int) else "?"
                print(f"      token_id={tid} ({txt!r}) logprob={lp}")
            elif isinstance(entry, tuple) and len(entry) >= 2:
                print(f"      {entry}")

    prompt_top_logprobs = top_logprobs_list[1 : len(input_ids)]
    prompt_with_logprobs = sum(1 for p in prompt_top_logprobs if p is not None)
    print(f"\n  Prompt positions with logprobs: {prompt_with_logprobs}/{len(prompt_top_logprobs)}")

    print(f"\n  Prompt token logprobs:")
    for i, pos_entry in enumerate(prompt_top_logprobs):
        token_id = input_ids[i + 1]  # offset by 1 since we sliced from pos 1
        token_text = tokenizer.decode([token_id])
        if pos_entry is None:
            print(f"    pos {i+1}: token_id={token_id} ({token_text!r}) logprob=None (missing)")
            continue
        # Look for the actual token's logprob in the top-k entries
        token_logp = None
        if isinstance(pos_entry, dict):
            for key, entry in pos_entry.items():
                if isinstance(entry, dict):
                    tid = entry.get("token_id") or entry.get("id")
                    if tid == token_id:
                        token_logp = entry.get("logprob")
                        break
                elif isinstance(key, int) and key == token_id:
                    token_logp = entry if isinstance(entry, (int, float)) else entry.get("logprob") if isinstance(entry, dict) else None
                    break
        if token_logp is not None:
            print(f"    pos {i+1}: token_id={token_id} ({token_text!r}) logp={token_logp:.6f}")
        else:
            # Fallback: print top-3 entries so we can inspect
            token_entries = list(pos_entry.items()) if isinstance(pos_entry, dict) else pos_entry
            print(f"    pos {i+1}: token_id={token_id} ({token_text!r}) logp=NOT_IN_TOP_K  top3:")
            for entry in (token_entries[:3] if isinstance(token_entries, list) else []):
                if isinstance(entry, tuple) and len(entry) >= 2:
                    print(f"      {entry}")

    if all_ok and len(output_top_logprobs) == len(output_ids):
        print(f"\n[PASS] Prompt logprobs working correctly")
        return True
    else:
        print(f"\n[FAIL] Prompt logprobs incomplete or missing")
        return False


async def test_raw_text_logprobs(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    top_k: int,
) -> bool:
    """Quick sanity check: text prompt with echo=True."""
    payload = {
        "prompt": "Hello",
        "model": model,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": top_k,
        "echo": True,
    }

    print(f"\n--- Raw Text Echo Test ---")
    try:
        resp = await client.post(
            f"{base_url}/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(60.0),
        )
    except httpx.HTTPError as e:
        print(f"[FAIL] Request error: {e}")
        return False

    if resp.status_code != 200:
        print(f"[FAIL] Returned {resp.status_code}: {resp.text[:200]}")
        return False

    data = resp.json()
    logprobs_data = data["choices"][0].get("logprobs", {})
    top_logprobs = logprobs_data.get("top_logprobs", [])
    print(f"  Positions with logprobs: {len(top_logprobs)}")
    for i, pos in enumerate(top_logprobs[:3]):
        if pos is None:
            print(f"    pos {i}: None")
            continue
        items = list(pos.items())[:2] if isinstance(pos, dict) else list(pos)[:2]
        print(f"    pos {i}: {items}")

    has_prompt_lps = any(p is not None for p in top_logprobs[:-1])
    if has_prompt_lps:
        print(f"[PASS] Echo logprobs include prompt positions")
        return True
    else:
        print(f"[FAIL] No prompt position logprobs in echo response")
        return False


async def main() -> int:
    parser = argparse.ArgumentParser(description="Verify teacher API prompt logprobs")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Teacher API key (reads WORKSPACE_OPENAI_API_KEY env if not set)")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    api_key = args.api_key or __import__("os").environ.get("WORKSPACE_OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] No API key provided. Use --api-key or set TEACHER_API_KEY.")
        return 1

    print(f"Teacher API:  {args.base_url}")
    print(f"Model:        {args.model}")
    print(f"Tokenizer:    {args.tokenizer}")
    print(f"Top-k:        {args.top_k}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    async with httpx.AsyncClient() as client:
        results = []

        ok = await check_health(client, args.base_url, api_key)
        if not ok:
            print("\nTeacher API is unreachable. Aborting.")
            return 1
        results.append(("Health check", ok))

        ok = await test_raw_text_logprobs(client, args.base_url, api_key, args.model, args.top_k)
        results.append(("Raw text echo logprobs", ok))

        ok = await test_prompt_logprobs(
            client, args.base_url, api_key, args.model, tokenizer, args.top_k
        )
        results.append(("Token-ID prompt logprobs", ok))

    print(f"\n{'='*50}")
    print("Summary:")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    all_pass = all(ok for _, ok in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
