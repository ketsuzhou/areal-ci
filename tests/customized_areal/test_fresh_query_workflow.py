from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from customized_areal.tree_search.config import AdvantageMode, CacheMode, LossMode
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
    _apply_fresh_query_row,
)


class _FakeNotFilter:
    def __init__(self, query):
        self.query = query

    def contains(self, column, value):
        self.query.not_contains_calls.append((column, value))
        return self.query


class _FakeSelectQuery:
    def __init__(self, rows):
        self.rows = rows

    def select(self, columns):
        self.columns = columns
        return self

    def limit(self, limit):
        self.limit_value = limit
        return self

    async def execute(self):
        return SimpleNamespace(data=self.rows)


class _FakeUpdateQuery:
    def __init__(self, client, payload, result_rows):
        self.client = client
        self.payload = payload
        self.result_rows = result_rows
        self.query_id = None
        self.not_contains_calls = []
        self.not_ = _FakeNotFilter(self)

    def eq(self, column, value):
        assert column == "query_id"
        self.query_id = value
        return self

    async def execute(self):
        self.client.updates.append(
            {
                "query_id": self.query_id,
                "payload": self.payload,
                "not_contains_calls": list(self.not_contains_calls),
            }
        )
        return SimpleNamespace(data=self.result_rows)


class _FakeTable:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def select(self, columns):
        self.client.selected_tables.append((self.name, columns))
        return _FakeSelectQuery(self.client.rows)

    def update(self, payload):
        result_rows = self.client.update_results.pop(0)
        return _FakeUpdateQuery(self.client, payload, result_rows)


class _FakeClient:
    def __init__(self, rows, update_results):
        self.rows = rows
        self.update_results = list(update_results)
        self.selected_tables = []
        self.updates = []

    def table(self, name):
        return _FakeTable(self, name)


class _FakeDBConnection:
    client = None

    async def get_client(self):
        return self.client


def _workflow(tmp_path):
    return TreeSearchGroupedRolloutWorkflow(
        MagicMock(),
        group_size=1,
        checkpoint_dir=str(tmp_path),
        advantage_mode=AdvantageMode.TREE,
        loss_mode=LossMode.GRPO,
        cache_mode=CacheMode.OFF,
        use_fresh_query=True,
        fresh_query_table="query_bank",
    )


def test_apply_fresh_query_row_normalizes_database_fields():
    data = {"existing": "kept"}
    row = {
        "query_id": "q1",
        "query": "What happened?",
        "gold_answer": "42",
        "evaluation_rubric": ["correct", "concise"],
        "used4train": ["old-train"],
    }

    result = _apply_fresh_query_row(data, row)

    assert result["existing"] == "kept"
    assert result["query_id"] == "q1"
    assert result["query"] == "What happened?"
    assert result["answer"] == "42"
    assert result["evaluation_rubric"] == ["correct", "concise"]
    assert result["used4train"] == ["old-train"]


@pytest.mark.asyncio
async def test_load_fresh_query_claims_first_eligible_row(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[
            {"query_id": "used", "used4train": ["train-a"]},
            {
                "query_id": "q1",
                "query": "Question",
                "gold_answer": "Answer",
                "evaluation_rubric": ["rubric"],
                "used4train": ["other"],
            },
        ],
        update_results=[[{"query_id": "q1"}]],
    )
    _FakeDBConnection.client = client
    monkeypatch.setattr(
        "customized_areal.db_service.DBConnection",
        _FakeDBConnection,
    )

    result = await _workflow(tmp_path)._load_fresh_query_data({})

    assert result is not None
    assert result["query_id"] == "q1"
    assert result["answer"] == "Answer"
    assert result["used4train"] == ["other", "train-a"]
    assert client.selected_tables == [
        (
            "query_bank",
            "query_id,query,gold_answer,evaluation_rubric,used4train",
        )
    ]
    assert client.updates == [
        {
            "query_id": "q1",
            "payload": {"used4train": ["other", "train-a"]},
            "not_contains_calls": [("used4train", ["train-a"])],
        }
    ]


@pytest.mark.asyncio
async def test_load_fresh_query_retries_when_claim_loses_race(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[
            {"query_id": "q1", "used4train": []},
            {"query_id": "q2", "query": "Q2", "gold_answer": "A2", "used4train": []},
        ],
        update_results=[[], [{"query_id": "q2"}]],
    )
    _FakeDBConnection.client = client
    monkeypatch.setattr(
        "customized_areal.db_service.DBConnection",
        _FakeDBConnection,
    )

    result = await _workflow(tmp_path)._load_fresh_query_data({})

    assert result is not None
    assert result["query_id"] == "q2"
    assert [update["query_id"] for update in client.updates] == ["q1", "q2"]


@pytest.mark.asyncio
async def test_load_fresh_query_returns_none_when_no_eligible_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[{"query_id": "q1", "used4train": ["train-a"]}],
        update_results=[],
    )
    _FakeDBConnection.client = client
    monkeypatch.setattr(
        "customized_areal.db_service.DBConnection",
        _FakeDBConnection,
    )

    result = await _workflow(tmp_path)._load_fresh_query_data({})

    assert result is None
    assert client.updates == []
