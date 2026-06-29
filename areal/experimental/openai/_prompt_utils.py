# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from areal.utils.hf_utils import apply_chat_template

__all__ = [
    "apply_chat_template",
    "_DATA_URI_RE",
    "_ensure_message_dict_list",
    "_extract_images_from_messages",
]


def _ensure_message_dict_list(
    name: str,
    value: list[Any],
) -> list[dict[str, Any]]:
    """Validate that ``value`` is a list of dictionaries or BaseModel objects.

    Args:
        name: Name of the argument being validated (for error messages).
        value: The list provided by the caller.

    Returns:
        A list containing only dictionaries. BaseModel objects are
        converted into their dictionary representation with
        `model_dump(exclude_none=True)`; dictionaries are preserved.

    Raises:
        TypeError: If ``value`` is not a list or an element cannot be converted to a dict.
    """

    if not isinstance(value, list):
        raise TypeError(
            f"{name} must be provided as a list, got {type(value).__name__}"
        )

    def _normalize(item: Any):
        # we should convert BaseModel first, because BaseModel is also Iterable
        if isinstance(item, BaseModel):
            return item.model_dump(exclude_none=True)
        elif isinstance(item, Mapping):
            return {k: _normalize(v) for k, v in item.items() if v is not None}
        elif (
            isinstance(item, Iterable)
            and not isinstance(item, str)
            and not isinstance(item, bytes)
            and not isinstance(item, bytearray)
        ):
            return [_normalize(sub_item) for sub_item in item]
        else:
            return item

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict) or isinstance(item, BaseModel):
            normalized.append(_normalize(item))
        else:
            raise TypeError(
                f"{name}[{index}] must be a dict or a BaseModel; got {type(item).__name__}"
            )
    return normalized


# Regex for data URI: data:image/<subtype>;base64,<data>
_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)


def _extract_images_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract image data from OpenAI-format messages.

    Scans message ``content`` lists for ``image_url`` content parts,
    extracts base64 data (or raw URLs), and converts messages to a
    HuggingFace-compatible format for ``apply_chat_template``.

    Args:
        messages: Normalized list of message dicts (OpenAI format).

    Returns:
        A 3-tuple of:

        - **image_data** – list of base64 image strings (no data-URI prefix)
          or raw URL strings for each image found.
        - **messages_for_tokenizer** – deep copy of *messages* where every
          ``{"type": "image_url", ...}`` part is replaced by
          ``{"type": "image"}`` so that HuggingFace VLM tokenizers insert
          the correct image-placeholder tokens.
        - **vision_messages_for_vllm** – deep copy of *messages* where
          ``image_url`` parts retain the ``image_url`` key but the ``url``
          value is replaced with a placeholder (the actual base64 data URI
          is injected later by the vLLM backend from *image_data*).
    """
    image_data: list[str] = []
    messages_for_tokenizer: list[dict[str, Any]] = []
    vision_messages_for_vllm: list[dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            messages_for_tokenizer.append(deepcopy(msg))
            vision_messages_for_vllm.append(deepcopy(msg))
            continue

        tok_parts: list[dict[str, Any]] = []
        vllm_parts: list[dict[str, Any]] = []

        for part in content:
            if not isinstance(part, dict):
                tok_parts.append(part)
                vllm_parts.append(deepcopy(part))
                continue

            if part.get("type") == "image_url":
                image_url_obj = part.get("image_url", {})
                url = (
                    image_url_obj.get("url", "")
                    if isinstance(image_url_obj, dict)
                    else ""
                )

                if not url:
                    raise ValueError(
                        "image_url content part has an empty or missing URL. "
                        "Provide a valid data URI or HTTP(S) URL in "
                        "image_url.url."
                    )

                # Extract base64 payload from data URIs; keep raw URLs as-is.
                m = _DATA_URI_RE.match(url)
                if m:
                    image_data.append(m.group(1))
                else:
                    image_data.append(url)

                tok_parts.append({"type": "image"})

                # vLLM backend injects actual data URI from req.image_data.
                vllm_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "placeholder"},
                    }
                )
            else:
                tok_parts.append(deepcopy(part))
                vllm_parts.append(deepcopy(part))

        tok_msg = {**msg, "content": tok_parts}
        vllm_msg = {**msg, "content": vllm_parts}
        messages_for_tokenizer.append(tok_msg)
        vision_messages_for_vllm.append(vllm_msg)

    return image_data, messages_for_tokenizer, vision_messages_for_vllm
