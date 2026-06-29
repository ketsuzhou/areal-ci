"""Run a single GAIA task by task_id through the TPFC benchmark harness.

Mirrors benchmark_run_base.main() but overrides the whitelist to a single task
and sets max_concurrent=1. Used to verify a targeted fix end-to-end.
"""

import asyncio
import os

import dotenv
from omegaconf import OmegaConf

from customized_areal.tpfc.scripts.benchmark_run_base import entrypoint

dotenv.load_dotenv()

TASK_ID = os.environ.get("GAIA_TASK_ID", "a1e91b78-d3d8-4675-bb8d-62741b4b68a6")

cfg = OmegaConf.create(
    {
        "benchmark": {
            "name": "gaia-validation",
            "data": {
                "data_dir": "customized_areal/dataset/gaia-benchmark/gaia/2023/validation",
                "metadata_file": "metadata.jsonl",
                "whitelist": [TASK_ID],
            },
            "execution": {"max_concurrent": 1, "max_tasks": 166, "pass_at_k": 1},
        },
        "llm": {
            "provider": "openai",
            "model_name": "openrouter/qwen/qwen3.5-9b",
            "reasoning_effort": "low",
            "stream": False,
        },
        "env": {"openai_api_key": "", "openrouter_api_key": ""},
        "level": 1,
        "user_id": "62ec5137-d121-4c8c-b175-ee165bdf38e4",
        "agent_id": os.environ.get("main_agent_id", ""),
        "backend_mode": True,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
    }
)

cfg.tags = [
    f"{cfg.benchmark.name}",
    f"{cfg.llm.model_name}",
    "base_0617",
    f"level_{cfg.level}",
]
cfg.output_dir = f"logs/{'-'.join(cfg.tags[:-1])}/{cfg.tags[-1]}"

if __name__ == "__main__":
    asyncio.run(entrypoint(cfg))
