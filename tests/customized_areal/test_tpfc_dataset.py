import json

import pandas as pd
import pytest

from customized_areal.tpfc.tpfc_dataset import get_tpfc_rl_dataset


class WhitespaceTokenizer:
    def encode(self, text):
        return text.split()


def test_get_tpfc_rl_dataset_normalizes_malformed_rows(tmp_path, monkeypatch):
    parquet_path = tmp_path / "tpfc.parquet"
    parquet_path.touch()
    rows = [
        {
            "prompt": [
                {"role": "system", "content": None},
                {
                    "role": "user",
                    "content": (
                        "<context>hidden</context>\n"
                        "Prefix <User Query>: What is the answer?"
                    ),
                },
                None,
                {"role": "", "content": "ignored"},
            ],
            "reward_model": json.dumps({"ground_truth": 42}),
            "images": [
                {"image": "file:///tmp/example.png"},
                {"image": None},
                "not-json",
            ],
            "extra_info": json.dumps({"query_id": "query-1"}),
        },
        {
            "prompt": [
                {
                    "role": "user",
                    "content": {"content": "<context>x</context>\nPlain task"},
                }
            ],
            "reward_model": {},
            "images": None,
            "extra_info": {},
        },
    ]
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: pd.DataFrame(rows),
    )

    dataset = get_tpfc_rl_dataset(str(parquet_path), split="train")

    first = dataset[0]
    assert first["messages"] == [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": (
                "<context>hidden</context>\nPrefix <User Query>: What is the answer?"
            ),
        },
    ]
    assert first["answer"] == "42"
    assert first["files_path"] == ["/tmp/example.png"]
    assert first["query_id"] == "query-1"
    assert first["query"] == "What is the answer?"

    second = dataset[1]
    assert second["query"] == "Plain task"
    assert second["answer"] == ""
    assert second["files_path"] == []


def test_get_tpfc_rl_dataset_filters_by_length(tmp_path, monkeypatch):
    parquet_path = tmp_path / "tpfc.parquet"
    parquet_path.touch()
    rows = [
        {
            "prompt": [{"role": "user", "content": "short"}],
            "reward_model": {},
            "images": [],
            "extra_info": {},
        },
        {
            "prompt": [{"role": "user", "content": "too many words"}],
            "reward_model": {},
            "images": [],
            "extra_info": {},
        },
    ]
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: pd.DataFrame(rows),
    )

    dataset = get_tpfc_rl_dataset(
        str(parquet_path),
        split="train",
        tokenizer=WhitespaceTokenizer(),
        max_length=1,
    )

    assert len(dataset) == 1
    assert dataset[0]["query"] == "short"


def test_get_tpfc_rl_dataset_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="TPFC dataset parquet not found"):
        get_tpfc_rl_dataset(str(tmp_path / "missing.parquet"), split="train")


def test_get_tpfc_rl_dataset_raises_for_empty_parquet(tmp_path, monkeypatch):
    parquet_path = tmp_path / "empty.parquet"
    parquet_path.touch()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="TPFC dataset parquet has no rows"):
        get_tpfc_rl_dataset(str(parquet_path), split="train")
