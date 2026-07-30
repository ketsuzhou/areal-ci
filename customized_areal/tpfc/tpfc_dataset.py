"""TPFC Dataset loader for AReaL RL training.

This module provides dataset loading functions for the TPFC (Task Planning
with Function Calling) generated training data stored in parquet format.

The dataset format matches openai/gsm8k RL format:
- messages: List of dicts with 'role' and 'content' keys
- answer: Ground truth answer string
- files_path (optional): List of image file paths for multimodal models
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return []


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        nested = content.get("content", content.get("text", ""))
        return _content_to_text(nested)
    if isinstance(content, list):
        return "".join(_content_to_text(part) for part in content)
    return str(content)


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
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"Invalid image path: {image_path!r}")
        if image_path.startswith("file://"):
            image_path = image_path[7:]  # Remove file:// prefix
        with Image.open(image_path) as opened:
            image = opened.copy()
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
        parquet_files = sorted(parquet_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in directory: {path}")
        parquet_path = parquet_files[0]
    if not parquet_path.exists():
        raise FileNotFoundError(f"TPFC dataset parquet not found: {parquet_path}")

    # Read parquet using pandas
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"TPFC dataset parquet has no rows: {parquet_path}")

    def process(sample):
        """Process a single sample to match gsm8k RL format."""
        # Extract prompt messages (numpy array of dicts -> list of dicts)
        prompt_array = _to_list(sample.get("prompt"))
        messages = []
        for msg in prompt_array:
            msg_dict = _to_dict(msg)
            if not msg_dict:
                continue
            role = msg_dict.get("role")
            if not isinstance(role, str) or not role:
                continue
            messages.append(
                {"role": role, "content": _content_to_text(msg_dict.get("content"))}
            )

        # Get ground truth from reward_model
        reward_model = _to_dict(sample.get("reward_model"))
        answer = _content_to_text(reward_model.get("ground_truth", ""))

        # Process images if present
        images_array = _to_list(sample.get("images"))
        files_path = []
        for img_data in images_array:
            img_dict = _to_dict(img_data)
            image_path = img_dict.get("image")
            if not isinstance(image_path, str) or not image_path:
                continue
            if image_path.startswith("file://"):
                image_path = image_path[7:]
            files_path.append(image_path)

        # Extract query_id from extra_info (UUID generated at data prep time)
        extra_info = _to_dict(sample.get("extra_info"))
        query_id = _content_to_text(extra_info.get("query_id", ""))

        # Extract query text from the last user message after "<User Query>: ",
        # stripping any leading "<context>...</context>" prefix
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = _content_to_text(msg.get("content"))
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
                else:
                    query = re.sub(
                        r"^\s*<context>.*?</context>\s*", "", content, flags=re.DOTALL
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

    dataset = Dataset.from_list(
        [process(sample) for sample in df.to_dict(orient="records")]
    )

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
