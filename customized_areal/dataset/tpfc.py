"""TPFC (Task Planning with Function Calling) dataset loader."""

from datasets import load_dataset


def get_tpfc_rl_dataset(
    path: str,
    split: str | None,
    tokenizer,
    max_length: int | None = None,
    **kwargs,
):
    """Load TPFC dataset for RL training.

    The dataset is expected to be a parquet file with the following columns:
    - prompt: Array of message objects with 'content' and 'role'
    - reward_model: Dict with 'ground_truth' and 'style'
    - images: Optional array of image objects with 'image' key containing file path
    - extra_info: Additional metadata

    Args:
        path: Path to the parquet file
        split: Dataset split (train/test) - not used for parquet, kept for API compatibility
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length to include
        **kwargs: Additional arguments

    Returns:
        Dataset with 'messages', 'ground_truth', and optional 'files_path' columns
    """
    # Load the parquet file
    dataset = load_dataset("parquet", data_files=path, split="train")

    def process(sample):
        """Process a single sample into the format expected by RL workflows."""
        # Extract messages from prompt column (already in OpenAI format)
        messages = list(sample["prompt"])

        # Extract ground truth from reward_model
        reward_model = sample.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "")
        style = reward_model.get("style", "none")

        # Build the processed sample
        # Note: Use 'answer' as column name since AReaL passes it as 'answer' to reward functions
        result = {
            "messages": messages,
            "answer": ground_truth,  # AReaL expects 'answer' column for ground truth
            "style": style,
            "files_path": [],  # Always include for consistent schema
            "extra_info": {},  # Always include for consistent schema
        }

        # Populate files_path if images present
        images = sample.get("images")
        if images is not None and len(images) > 0:
            result["files_path"] = [
                img["image"][7:]
                if isinstance(img, dict) and img.get("image", "").startswith("file://")
                else img.get("image", "")
                if isinstance(img, dict)
                else img
                for img in images
            ]

        # Populate extra_info if present
        extra_info = sample.get("extra_info")
        if extra_info is not None:
            result["extra_info"] = extra_info

        return result

    dataset = dataset.map(process)

    # Remove original columns
    columns_to_remove = [
        col
        for col in dataset.column_names
        if col not in ["messages", "answer", "style", "files_path", "extra_info"]
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    # Filter by max_length if provided
    if max_length is not None:

        def filter_length(sample):
            # Estimate token count from messages
            total_chars = sum(len(m.get("content", "")) for m in sample["messages"])
            # Rough estimate: 1 token ~ 4 characters
            estimated_tokens = total_chars // 4
            return estimated_tokens <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
