"""Tests for the max_group_size branch budget + circuit breaker (Task 3.5).

The dynamic loop caps successful branch episodes at
``max_group_size - initial_group_size`` (``_branch_budget``) and stops on
``_MAX_CONSECUTIVE_FAILED_ADDITIONS`` consecutive failures. Failed episodes do
not count against the budget (they count against the circuit breaker), so both
bounds remain operative.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    Config,
    LossMode,
)
from customized_areal.tree_search.core.customized_grouped_workflow import (
    _MAX_CONSECUTIVE_FAILED_ADDITIONS,
    TreeSearchGroupedRolloutWorkflow,
)


class _NoneWorkflow:
    """Base workflow whose arun_episode always returns None (failed episode)."""

    def __init__(self) -> None:
        self.calls = 0

    async def arun_episode(self, engine, data):
        self.calls += 1
        return None


def _make(
    tmp_path,
    *,
    initial_group_size: int,
    max_group_size: int,
    workflow,
) -> TreeSearchGroupedRolloutWorkflow:
    return TreeSearchGroupedRolloutWorkflow(
        workflow=workflow,
        group_size=initial_group_size,
        config=Config(
            checkpoint_dir=str(tmp_path),
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            mode=CacheMode.OFF,
            dynamic_group_size=True,
            initial_group_size=initial_group_size,
            max_group_size=max_group_size,
        ),
    )


def test_branch_budget_property(tmp_path):
    wf = _make(
        tmp_path,
        initial_group_size=2,
        max_group_size=4,
        workflow=_NoneWorkflow(),
    )
    assert wf._branch_budget == 2  # 4 - 2

    wf0 = _make(
        tmp_path,
        initial_group_size=2,
        max_group_size=2,
        workflow=_NoneWorkflow(),
    )
    assert wf0._branch_budget == 0  # no room for branches


@pytest.mark.asyncio
async def test_budget_zero_prevents_branch_episodes(tmp_path):
    # max_group_size == initial_group_size -> branch_budget == 0 -> the dynamic
    # loop never enters (no branch episodes). Only the initial round runs.
    fake = _NoneWorkflow()
    wf = _make(tmp_path, initial_group_size=2, max_group_size=2, workflow=fake)
    await wf._arun_episode_dynamic(engine=None, data={}, query_id="q1")
    assert fake.calls == 2  # initial_group_size only; no branches attempted


@pytest.mark.asyncio
async def test_consecutive_failures_trip_circuit_breaker(tmp_path):
    # All episodes fail (None). branch_samples stays 0 so the budget does not
    # bind; the consecutive-failure circuit breaker stops the loop after
    # _MAX_CONSECUTIVE_FAILED_ADDITIONS failures.
    fake = _NoneWorkflow()
    wf = _make(tmp_path, initial_group_size=2, max_group_size=4, workflow=fake)
    await wf._arun_episode_dynamic(engine=None, data={}, query_id="q1")
    # initial (2) + max_failed_additions (= 3) consecutive failures.
    assert fake.calls == 2 + _MAX_CONSECUTIVE_FAILED_ADDITIONS
