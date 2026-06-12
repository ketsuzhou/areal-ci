import json
import math

import requests

API_KEY = "pAx3ob61ByPD8NDYcDRGDJ2k7LjObP8M"

url = "https://api.deepinfra.com/v1/openai/completions"
prompt_text = "what is your name"
payload = {
    "model": "deepseek-ai/DeepSeek-V3",
    "max_tokens": 1,
    "temperature": 0.6,
    "logprobs": 5,
    "echo": True,
    "prompt": prompt_text,
}
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
data = response.json()
lp = data["choices"][0]["logprobs"]

prompt_len = len(prompt_text)

print("=== Log probs for prompt tokens ===")
print(f"{'Token':<20} {'LogP':>10} {'Prob':>12}")
print("-" * 45)
total_log_p = 0.0
for token, log_p, offset in zip(lp["tokens"], lp["token_logprobs"], lp["text_offset"]):
    if offset < prompt_len:
        prob = math.exp(log_p)
        print(f"{repr(token):<20} {log_p:>10.6f} {prob:>12.8f}")
        total_log_p += log_p
print("-" * 45)
print(f"{'TOTAL':<20} {total_log_p:>10.6f} {math.exp(total_log_p):>12.8e}")

print()
print("=== Top-K log probs at each prompt token position ===")
for i, (token, log_p, top_lp, offset) in enumerate(
    zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"], lp["text_offset"])
):
    if offset >= prompt_len:
        break
    print(f"\nPosition {i}: {repr(token)}  (logP={log_p:.6f})")
    if not top_lp:
        print("  (no top-K data)")
        continue
    sorted_top = sorted(top_lp.items(), key=lambda x: -x[1])
    for rank, (t, p) in enumerate(sorted_top):
        print(
            f"  top-{rank + 1}: {repr(t):<20} logP={p:>10.6f}  prob={math.exp(p):.8f}"
        )
