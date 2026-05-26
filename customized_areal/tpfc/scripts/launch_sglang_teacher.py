#!/usr/bin/env python3
"""Launch a standalone SGLang server for external teacher distillation.

Usage:
    # Start server only (for training):
    uv run customized_areal/tpfc/scripts/launch_sglang_teacher.py \
        --model-path /dfs/share-groups/letrain/ckpt/Qwen3.5-9B \
        --port 32735 \
        --gpu-id 0 \
        --serve-only

    # Start server + run self-test:
    uv run customized_areal/tpfc/scripts/launch_sglang_teacher.py \
        --model-path /dfs/share-groups/letrain/ckpt/Qwen3.5-9B \
        --port 32735 \
        --gpu-id 0

This script:
1. Builds the SGLang launch command via SGLangConfig.build_cmd()
2. Starts the server as a subprocess
3. Waits for /v1/models to be ready
4. (Optional) Sends a test /generate request matching the teacher distill payload format
5. Validates the response contains top-k logprobs

Configured as the external teacher for config_tpfc_Qwen3-5L-9B-opd.yaml:
  tree_search:
    teacher_provider: external
    teacher_base_url: http://localhost:32735
    teacher_backend: sglang
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import signal

import requests

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from areal.api.cli_args import SGLangConfig

DEFAULT_MODEL_PATH = "/dfs/share-groups/letrain/ckpt/Qwen3.5-9B"
DEFAULT_PORT = 32735


def build_sglang_cmd(model_path: str, port: int, gpu_id: int) -> list[str]:
    config = SGLangConfig(
        model_path=model_path,
        random_seed=1,
        skip_tokenizer_init=True,
        context_length=16384,
        mem_fraction_static=0.50,
        max_running_requests=1,
        enable_multimodal=True,
        disable_radix_cache=True,
        attention_backend="fa3",
        dtype="bfloat16",
    )
    cmd = SGLangConfig.build_cmd(
        config,
        tp_size=1,
        base_gpu_id=gpu_id,
        host="0.0.0.0",
        port=port,
        dist_init_addr="localhost:29500",
        n_nodes=1,
        node_rank=0,
    )
    return cmd


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                print(f"  Server ready after {time.time() - start:.1f}s")
                time.sleep(3)
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def test_teacher_generate(base_url: str, top_k: int = 10) -> bool:
    """Send a /generate request matching the teacher distill payload format."""
    from transformers import AutoTokenizer

    model_path = os.environ.get("TEST_MODEL_PATH", "")
    if not model_path:
        print("  WARNING: TEST_MODEL_PATH not set, using dummy token IDs")
        prompt_text = "The answer is"
        prompt_ids = [1, 2, 3, 4, 5]
        output_ids = [6, 7, 8]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_text = "The answer is"
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        output_ids = prompt_ids[-2:]

    all_ids = prompt_ids + output_ids
    prompt_len = len(prompt_ids)

    payload = {
        "input_ids": all_ids,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_len,
        "top_logprobs_num": top_k,
        "stream": False,
    }

    print(f"  Sending /generate with {len(all_ids)} tokens, logprob_start_len={prompt_len}, top_k={top_k}")
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)

    if resp.status_code != 200:
        print(f"  FAILED: HTTP {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")
        return False

    data = resp.json()
    meta = data.get("meta_info", {})
    input_top = meta.get("input_top_logprobs")
    output_top = meta.get("output_top_logprobs")

    n_input = len(input_top) if input_top else 0
    n_output = len(output_top) if output_top else 0
    print(f"  Response OK — input_top_logprobs positions: {n_input}, output_top_logprobs positions: {n_output}")

    if input_top and len(input_top) > 0 and input_top[0] is not None:
        sample = input_top[0]
        print(f"  Sample logprobs at position 0: {len(sample)} entries")
        if sample:
            print(f"    First entry: logprob={sample[0][0]:.4f}, token_id={sample[0][1]}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Launch SGLang server for external teacher distillation")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to model weights")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for SGLang server")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k logprobs for teacher distill test")
    parser.add_argument("--timeout", type=int, default=300, help="Server startup timeout in seconds")
    parser.add_argument("--serve-only", action="store_true", help="Only start the server, skip self-test")
    args = parser.parse_args()

    os.environ["TEST_MODEL_PATH"] = args.model_path

    print(f"=== SGLang Teacher Server (External Provider) ===")
    print(f"Model: {args.model_path}")
    print(f"Port:  {args.port}")
    print(f"GPU:   {args.gpu_id}")
    print(f"Mode:  {'serve-only' if args.serve_only else 'serve + self-test'}")

    # Build and print the launch command
    cmd = build_sglang_cmd(args.model_path, args.port, args.gpu_id)
    print(f"\nLaunch command:\n  {' '.join(cmd)}")

    # Launch server
    print(f"\nStarting SGLang server on GPU {args.gpu_id}...")
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{args.port}"

    def _shutdown():
        print(f"\nShutting down SGLang server (PID={proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server stopped.")

    try:
        # Wait for server
        print(f"Waiting for server (timeout={args.timeout}s)...")
        if not wait_for_server(base_url, timeout=args.timeout):
            print("FAILED: Server did not become ready in time")
            _shutdown()
            sys.exit(1)

        # Health check
        print(f"\n--- Health check ---")
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        print(f"  /v1/models: {resp.status_code}")

        if args.serve_only:
            print(f"\nServer is running at {base_url}")
            print(f"Config: teacher_base_url=http://localhost:{args.port}, teacher_backend=sglang")
            print(f"Press Ctrl+C to stop.")
            proc.wait()
        else:
            # Run self-test
            print(f"\n--- Teacher distill /generate test ---")
            ok = test_teacher_generate(base_url, top_k=args.top_k)

            if ok:
                print("\n=== ALL TESTS PASSED ===")
            else:
                print("\n=== TESTS FAILED ===")
                _shutdown()
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if proc.poll() is None:
            _shutdown()


if __name__ == "__main__":
    main()
