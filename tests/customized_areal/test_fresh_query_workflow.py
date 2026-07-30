from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    Config,
    LossMode,
)
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
from customized_areal.tree_search.core.fresh_query import _apply_fresh_query_row


class _FakeNotFilter:
    def __init__(self, query):
        self.query = query

    def contains(self, column, value):
        self.query.not_contains_calls.append((column, value))
        return self.query


class _FakeSelectQuery:
    def __init__(self, rows):
        self.rows = rows
        self.not_contains_calls = []
        self.not_ = _FakeNotFilter(self)

    def select(self, columns):
        self.columns = columns
        return self

    def limit(self, limit):
        self.limit_value = limit
        return self

    def eq(self, column, value):
        assert column == "query_id"
        self.rows = [r for r in self.rows if r.get("query_id") == value]
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
        config=Config(
            checkpoint_dir=str(tmp_path),
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            mode=CacheMode.OFF,
            use_fresh_query=True,
            fresh_query_table="query_bank",
        ),
    )


def _patch_db(monkeypatch, client):
    _FakeDBConnection.client = client
    monkeypatch.setattr(
        "customized_areal.db_service.DBConnection",
        _FakeDBConnection,
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
async def test_select_fresh_query_returns_first_eligible_row_without_claiming(
    tmp_path, monkeypatch
):
    """Selection is read-only: used4train is only updated after a usable batch."""
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
        update_results=[],
    )
    _patch_db(monkeypatch, client)

    wf = _workflow(tmp_path)
    result = await wf._select_fresh_query_data({})

    assert result is not None
    data, (query_id, train_id) = result
    assert data["query_id"] == "q1"
    assert data["answer"] == "Answer"
    # Claim is deferred: the row's used4train is passed through unchanged and
    # no update is issued at selection time.
    assert data["used4train"] == ["other"]
    assert (query_id, train_id) == ("q1", "train-a")
    assert client.updates == []
    assert "q1" in wf._inflight_fresh_queries


@pytest.mark.asyncio
async def test_select_fresh_query_skips_inflight_rows(tmp_path, monkeypatch):
    """A query already selected by a concurrent arun_episode is skipped."""
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[
            {"query_id": "q1", "query": "Q1", "gold_answer": "A1", "used4train": []},
            {"query_id": "q2", "query": "Q2", "gold_answer": "A2", "used4train": []},
        ],
        update_results=[],
    )
    _patch_db(monkeypatch, client)

    wf = _workflow(tmp_path)
    wf._inflight_fresh_queries.add("q1")
    result = await wf._select_fresh_query_data({})

    assert result is not None
    data, (query_id, _) = result
    assert query_id == "q2"
    assert data["answer"] == "A2"


@pytest.mark.asyncio
async def test_select_fresh_query_returns_none_when_no_eligible_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[{"query_id": "q1", "used4train": ["train-a"]}],
        update_results=[],
    )
    _patch_db(monkeypatch, client)

    result = await _workflow(tmp_path)._select_fresh_query_data({})

    assert result is None
    assert client.updates == []


@pytest.mark.asyncio
async def test_claim_fresh_query_appends_train_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[{"query_id": "q1", "used4train": ["other"]}],
        update_results=[[{"query_id": "q1"}]],
    )
    _patch_db(monkeypatch, client)

    await _workflow(tmp_path)._claim_fresh_query("q1", "train-a")

    assert client.updates == [
        {
            "query_id": "q1",
            "payload": {"used4train": ["other", "train-a"]},
            "not_contains_calls": [("used4train", ["train-a"])],
        }
    ]


@pytest.mark.asyncio
async def test_claim_fresh_query_is_noop_when_already_claimed(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[{"query_id": "q1", "used4train": ["train-a"]}],
        update_results=[],
    )
    _patch_db(monkeypatch, client)

    await _workflow(tmp_path)._claim_fresh_query("q1", "train-a")

    assert client.updates == []


def _make_nodes_factory(rewards):
    """Return a _result_to_nodes replacement yielding one 2-turn episode per
    call, cycling through ``rewards`` per episode."""
    from customized_areal.tree_search.core.tree_store import Node

    call_count = 0

    def make_nodes(result, query_id, group_idx):
        nonlocal call_count
        reward = rewards[call_count % len(rewards)]
        call_count += 1
        nodes = []
        for i in range(2):
            node = Node(
                input_ids=[1, 2, 3],
                loss_mask=[0, 1, 1],
                logprobs=[0.0, -0.5, -0.3],
                versions=[-1, 0, 0],
                outcome_reward=reward,
            )
            node.query_id = query_id
            node.episode_id = f"ep_{group_idx}"
            node.node_id = f"node_{group_idx}_{i}"
            node.turn_idx = i + 1
            nodes.append(node)
        return nodes

    return make_nodes


@pytest.mark.asyncio
async def test_arun_episode_claims_query_after_successful_batch(tmp_path, monkeypatch):
    """End to end: a usable batch triggers the deferred claim."""
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[{"query_id": "q1", "query": "Q", "gold_answer": "A", "used4train": []}],
        update_results=[[{"query_id": "q1"}]],
    )
    _patch_db(monkeypatch, client)

    base = MagicMock()
    base.arun_episode = AsyncMock(return_value={})
    wf = TreeSearchGroupedRolloutWorkflow(
        base,
        group_size=2,
        config=Config(
            checkpoint_dir=str(tmp_path),
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            mode=CacheMode.OFF,
            use_fresh_query=True,
            fresh_query_table="query_bank",
        ),
    )
    wf._result_to_nodes = _make_nodes_factory([0.0, 1.0])

    result = await wf.arun_episode(MagicMock(), {})

    assert result is not None
    assert client.updates == [
        {
            "query_id": "q1",
            "payload": {"used4train": ["train-a"]},
            "not_contains_calls": [("used4train", ["train-a"])],
        }
    ]
    assert "q1" not in wf._inflight_fresh_queries


@pytest.mark.asyncio
async def test_arun_episode_does_not_claim_discarded_query(tmp_path, monkeypatch):
    """A zero-variance-discarded rollout leaves the query eligible (no claim)."""
    monkeypatch.setenv("TRAIN_ID", "train-a")
    client = _FakeClient(
        rows=[{"query_id": "q1", "query": "Q", "gold_answer": "A", "used4train": []}],
        update_results=[],
    )
    _patch_db(monkeypatch, client)

    base = MagicMock()
    base.arun_episode = AsyncMock(return_value={})
    wf = TreeSearchGroupedRolloutWorkflow(
        base,
        group_size=2,
        config=Config(
            checkpoint_dir=str(tmp_path),
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            mode=CacheMode.OFF,
            use_fresh_query=True,
            fresh_query_table="query_bank",
        ),
    )
    wf._result_to_nodes = _make_nodes_factory([1.0])

    result = await wf.arun_episode(MagicMock(), {})

    assert result is None
    assert client.updates == []
    assert "q1" not in wf._inflight_fresh_queries
