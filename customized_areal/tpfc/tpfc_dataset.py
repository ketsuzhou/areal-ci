"""TPFC Dataset loader for AReaL RL training.

This module provides dataset loading functions for the TPFC (Task Planning
with Function Calling) generated training data stored in parquet format.

The dataset format matches openai/gsm8k RL format:
- messages: List of dicts with 'role' and 'content' keys
- answer: Ground truth answer string
- files_path (optional): List of image file paths for multimodal models
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def convert_image_to_bytes(
    image_input: dict[str, Any],
    fixed_width: int = 448,
    fixed_height: int = 448,
) -> bytes:
    """Convert image input from parquet to JPEG bytes.

    Args:
        image_input: Dict with 'image' key containing file path (e.g., {'image': 'file://...'})
        fixed_width: Target width for resizing.
        fixed_height: Target height for resizing.

    Returns:
        JPEG image bytes.
    """
    # Handle dict format from parquet (e.g., {'image': 'file://...'})
    if isinstance(image_input, dict) and "image" in image_input:
        image_path = image_input["image"]
        if image_path.startswith("file://"):
            image_path = image_path[7:]  # Remove file:// prefix
        image = Image.open(image_path)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize to fixed dimensions
    if image.size != (fixed_width, fixed_height):
        image = image.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save to JPEG bytes
    import io

    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def get_tpfc_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    **kwargs,
):
    """Load TPFC RL dataset from parquet file.

    The parquet file has columns:
    - data_source: Source identifier
    - prompt: Array of message dicts with 'role' and 'content' keys
    - ability: Task ability type
    - images: Array of image dicts with 'image' key (file paths)
    - reward_model: Dict with 'ground_truth' and 'style'
    - extra_info: Dict with additional metadata

    Returns dataset with same format as gsm8k RL:
    - messages: List of dicts with 'role' and 'content'
    - answer: Ground truth string
    - files_path (optional): List of image file paths for multimodal

    Args:
        path: Path to the parquet file.
        split: Dataset split (not used for parquet, but kept for API consistency).
        tokenizer: Optional tokenizer for length filtering.
        max_length: Optional maximum sequence length for filtering.

    Returns:
        HuggingFace Dataset with 'messages', 'answer', and optionally 'files_path'.
    """
    from datasets import Dataset

    # Load parquet file
    parquet_path = Path(path)
    if parquet_path.is_dir():
        parquet_files = list(parquet_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in directory: {path}")
        parquet_path = parquet_files[0]

    # Read parquet using pandas
    df = pd.read_parquet(parquet_path)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    def process(sample):
        """Process a single sample to match gsm8k RL format."""
        # Extract prompt messages (numpy array of dicts -> list of dicts)
        prompt_array = sample["prompt"]
        if isinstance(prompt_array, np.ndarray):
            messages = [
                {"role": msg.get("role"), "content": msg.get("content", "")}
                for msg in prompt_array
            ]
        elif isinstance(prompt_array, list):
            messages = prompt_array
        else:
            messages = []

        # Get ground truth from reward_model
        reward_model = sample.get("reward_model", {})
        if isinstance(reward_model, dict):
            answer = reward_model.get("ground_truth", "")
        else:
            answer = ""

        # Process images if present
        images_array = sample.get("images", [])
        files_path = []
        if isinstance(images_array, (list, np.ndarray)) and len(images_array) > 0:
            for img_data in images_array:
                if isinstance(img_data, dict) and "image" in img_data:
                    image_path = img_data["image"]
                    if image_path.startswith("file://"):
                        image_path = image_path[7:]
                    files_path.append(image_path)

        # Extract query_id from extra_info (UUID generated at data prep time)
        extra_info = sample.get("extra_info", {})
        query_id = (
            extra_info.get("query_id", "") if isinstance(extra_info, dict) else ""
        )

        # Extract query text from the last user message after "<User Query>: ",
        # stripping any leading "<context>...</context>" prefix
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                marker = "<User Query>: "
                idx = content.rfind(marker)
                if idx != -1:
                    query = content[idx + len(marker) :]
                    query = re.sub(
                        r"^\s*<context>.*?</context>\s*", "", query, flags=re.DOTALL
                    )
                    query = re.sub(
                        r"^\s*<context>.*?<context>\s*", "", query, flags=re.DOTALL
                    )
                break

        result = {
            "messages": messages,
            "answer": answer,
            "files_path": files_path,
            "query_id": query_id,
            "query": query,
        }

        return result

    dataset = dataset.map(process, remove_columns=dataset.column_names)

    # Filter by length if requested
    if max_length is not None and tokenizer is not None:

        def filter_length(sample):
            try:
                # Concatenate all message content for length check
                text = "\n".join(msg.get("content", "") for msg in sample["messages"])
                tokens = tokenizer.encode(text)
                return len(tokens) <= max_length
            except Exception:
                # If filtering fails, keep the sample
                return True

        dataset = dataset.filter(filter_length)

    return dataset
