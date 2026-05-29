"""Quick test: can the WORKSPACE_OPENAI API obtain logprobs?"""

import asyncio
import json
import os

import httpx

API_KEY = os.environ["WORKSPACE_OPENAI_API_KEY"]
# The full base URL already includes /llm/v1
API_BASE = "http://10.254.244.168:8443/service-large-64-1775803465274/llm/v1"


async def main():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        # Test 1: /v1/completions with echo + logprobs
        print("=== Test 1: /v1/completions (echo + logprobs) ===")
        payload = {
            "model": "Qwen3.5_397B_A17B_FP8",
            "prompt": "what your name?",
            "max_tokens": 10,
            "temperature": 0.0,
            "logprobs": 5,
            "echo": True,
        }
        try:
            resp = await client.post(
                f"{API_BASE}/completions",
                content=json.dumps(payload),
                headers=headers,
            )
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    lp = choices[0].get("logprobs", {})
                    tokens = lp.get("tokens", [])
                    top_logprobs = lp.get("top_logprobs", [])
                    print(f"Tokens ({len(tokens)}): {tokens[:20]}")
                    for i, tlp in enumerate(top_logprobs[:5]):
                        print(
                            f"  pos {i} token={tokens[i] if i < len(tokens) else '?'}: {tlp}"
                        )
                else:
                    print(f"Response: {data}")
            else:
                print(f"Error: {resp.text[:500]}")
        except Exception as e:
            print(f"Exception: {e}")

        # Test 2: /v1/chat/completions with logprobs
        print("\n=== Test 2: /v1/chat/completions (logprobs) ===")
        chat_payload = {
            "model": "Qwen3.5_397B_A17B_FP8",
            "messages": [{"role": "user", "content": "what your name?"}],
            "max_tokens": 20,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
        }
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                content=json.dumps(chat_payload),
                headers=headers,
            )
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    lp = choices[0].get("logprobs", {})
                    print(f"Reply: {content[:200]}")
                    if lp:
                        tokens = lp.get("tokens", [])
                        top_lps = lp.get("top_logprobs", [])
                        print(f"Logprob tokens ({len(tokens)}): {tokens[:10]}")
                        for i, tlp in enumerate(top_lps[:5]):
                            print(f"  pos {i}: {tlp}")
                    else:
                        print("No logprobs in chat response")
                else:
                    print(f"Response: {data}")
            else:
                print(f"Error: {resp.text[:500]}")
        except Exception as e:
            print(f"Exception: {e}")


if __name__ == "__main__":
    asyncio.run(main())
