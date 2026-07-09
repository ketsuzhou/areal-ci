"""Tests for the multica_dag_client hook wiring (Task 2.3).

When ``multica_dag_enabled=True`` the grouped workflow constructs a
``MultiAgentEnvDispatchWorkflow`` as its base ``self.workflow`` from the multica
component hooks, and ``self._multica_dag_client`` is non-None. The non-multica
path keeps the caller-supplied ``workflow`` unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

from customized_areal.tree_search.agents.multi_agent_env_dispatch import (
    MultiAgentEnvDispatchWorkflow,
)
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
)
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)


def _make(tmp_path, **multica_kwargs):
    return TreeSearchGroupedRolloutWorkflow(
        workflow=SimpleNamespace(),  # ignored on the multica path
        group_size=1,
        checkpoint_dir=str(tmp_path),
        advantage_mode=AdvantageMode.TREE,
        loss_mode=LossMode.GRPO,
        cache_mode=CacheMode.OFF,
        **multica_kwargs,
    )


def test_multica_enabled_constructs_multi_agent_workflow(tmp_path):
    dispatch = SimpleNamespace()
    dag_client = SimpleNamespace()
    assembler = SimpleNamespace()
    resolver = SimpleNamespace()
    session_remover = SimpleNamespace()

    wf = _make(
        tmp_path,
        multica_dag_enabled=True,
        multica_dag_client=dag_client,
        multica_dispatch_client=dispatch,
        multica_assembler=assembler,
        multica_resolver=resolver,
        multica_session_remover=session_remover,
    )

    # self.workflow is the constructed MultiAgentEnvDispatchWorkflow...
    assert isinstance(wf.workflow, MultiAgentEnvDispatchWorkflow)
    # ...wired with the supplied components (dag_client is the get_dag poller;
    # dispatch_client is the create_env_dispatch client - distinct concerns).
    assert wf.workflow._dag_client is dag_client
    assert wf.workflow._dispatch is dispatch
    assert wf.workflow._assembler is assembler
    assert wf.workflow._resolver is resolver
    assert wf.workflow._session_remover is session_remover
    # The hook is active.
    assert wf._multica_dag_enabled is True
    assert wf._multica_dag_client is dag_client


def test_multica_disabled_keeps_passed_workflow(tmp_path):
    base_workflow = SimpleNamespace()

    wf = TreeSearchGroupedRolloutWorkflow(
        workflow=base_workflow,  # multica_dag_enabled defaults to False
        group_size=1,
        checkpoint_dir=str(tmp_path),
        advantage_mode=AdvantageMode.TREE,
        loss_mode=LossMode.GRPO,
        cache_mode=CacheMode.OFF,
    )

    assert wf._multica_dag_enabled is False
    # The caller-supplied workflow is retained (no override).
    assert wf.workflow is base_workflow
