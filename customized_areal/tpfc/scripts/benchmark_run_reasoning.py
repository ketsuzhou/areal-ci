import asyncio
import os

import dotenv
from omegaconf import OmegaConf

from customized_areal.tpfc.scripts.benchmark_run_base import entrypoint


def main():
    dotenv.load_dotenv()

    # Create configuration using OmegaConf
    cfg = OmegaConf.create(
        {
            "benchmark": {
                "name": "gaia-validation",
                "data": {
                    "data_dir": "customized_areal/dataset/gaia-benchmark/gaia/2023/validation",
                    "metadata_file": "metadata.jsonl",
                    "whitelist": [],
                },
                "execution": {"max_concurrent": 10, "max_tasks": 166, "pass_at_k": 1},
            },
            "llm": {
                "provider": "openai",
                # "model_name": "openrouter/gpt-5",
                # "model_name": "openai-compatible/gpt-5",
                # "model_name": "openrouter/qwen/qwen3.5-9b",
                "model_name": "openrouter/qwen/qwen3-vl-8b-thinking",
                # "model_name": "openrouter/qwen/qwen3-32b",
                "enable_thinking": False,
                "reasoning_effort": "low",
                "stream": False,
            },
            "env": {"openai_api_key": ""},
            "level": 1,
            "user_id": "62ec5137-d121-4c8c-b175-ee165bdf38e4",
            "agent_id": os.environ.get("main_agent_id", ""),
            "backend_mode": True,
            "base_url": os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        }
    )

    # Compute derived values
    cfg.tags = [
        f"{cfg.benchmark.name}",
        f"{cfg.llm.model_name}",
        "reasoning_add_audio_video_change_doc_tool_0424",
        # "compression_1w",
        f"level_{cfg.level}",
    ]
    cfg.output_dir = f"logs/{'-'.join(cfg.tags[:-1])}/{cfg.tags[-1]}"

    asyncio.run(entrypoint(cfg))


if __name__ == "__main__":
    main()
