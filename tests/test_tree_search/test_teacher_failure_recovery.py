import asyncio
import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT / "customized_areal/tree_search/core/customized_grouped_workflow.py"
)


class LossMode(str, Enum):
    GRPO = "grpo"
    DISTILL = "distill"
    BOTH = "both"


class TeacherServiceError(RuntimeError):
    pass


@dataclass
class Node:
    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    versions: list[int]
    node_id: str = ""
    parent_node_id: str | None = None
    episode_id: str = ""
    turn_idx: int = 0
    query_id: str = ""
    train_id: str = ""
    discarded: bool = False
    task_id: str = ""
    entropy_stats: dict | None = None
    need_branch: bool = False
    branch_sandbox_id: str | None = None
    outcome_reward: float = 0.0
    advantages: object | None = None
    returns: object | None = None
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None
    guidance: dict[int, str] | None = None


@dataclass
class Reward:
    teacher_logprobs: list[float]
    candidate_token_ids: list[int]
    sample_index: int = 0


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False):
        return "<|im_start|>user\nX<|im_end|>\n"

    def decode(self, input_ids, skip_special_tokens=False):
        return (
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\nAnswer<|im_end|>\n"
        )


class DiagnoseFailureProvider:
    async def diagnose_episode(self, conversation, gold_answer, temperature=None):
        raise TeacherServiceError("teacher diagnose backend unavailable")


class StubSelectedTurnModule:
    def __init__(self, planned_results):
        self.planned_results = list(planned_results)
        self.calls = []

    def parse_episode_diagnosis(self, raw_text):
        raise AssertionError("parse_episode_diagnosis should not be called in this test")

    async def selected_turn_to_position_rewards(
        self,
        *,
        node,
        guidance,
        tokenizer,
        provider,
        sample_index,
        topk_distill,
        engine,
        teacher_top_k,
    ):
        self.calls.append(
            {
                "node_id": node.node_id,
                "guidance": guidance,
                "sample_index": sample_index,
            }
        )
        result = self.planned_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _install_workflow_stubs(selected_turn_module: StubSelectedTurnModule) -> None:
    customized_areal_pkg = types.ModuleType("customized_areal")
    customized_areal_pkg.__path__ = []
    sys.modules["customized_areal"] = customized_areal_pkg

    tree_search_pkg = types.ModuleType("customized_areal.tree_search")
    tree_search_pkg.__path__ = []
    sys.modules["customized_areal.tree_search"] = tree_search_pkg

    core_pkg = types.ModuleType("customized_areal.tree_search.core")
    core_pkg.__path__ = []
    sys.modules["customized_areal.tree_search.core"] = core_pkg

    distilling_pkg = types.ModuleType("customized_areal.tree_search.distilling")
    distilling_pkg.__path__ = []
    sys.modules["customized_areal.tree_search.distilling"] = distilling_pkg

    db_service = types.ModuleType("customized_areal.db_service")
    for name in (
        "bind_sandbox_to_task",
        "copy_messages_to_task",
        "create_task",
        "delete_sandbox",
        "truncate_messages_before_turn",
    ):
        setattr(db_service, name, lambda *args, **kwargs: None)
    sys.modules["customized_areal.db_service"] = db_service

    backend_run = types.ModuleType("customized_areal.tpfc.backend_run")
    backend_run._get_raw_messages_with_client = lambda *args, **kwargs: []
    sys.modules["customized_areal.tpfc.backend_run"] = backend_run

    config_module = types.ModuleType("customized_areal.tree_search.config")

    class AdvantageMode(str, Enum):
        TREE = "tree"

    class CacheMode(str, Enum):
        CROSS_TRAINING = "cross_training"

    class SampleSource(str, Enum):
        SCRATCH = "scratch"
        BRANCH = "branch"
        MIXED = "mixed"

    config_module.AdvantageMode = AdvantageMode
    config_module.CacheMode = CacheMode
    config_module.LossMode = LossMode
    config_module.SampleSource = SampleSource
    sys.modules["customized_areal.tree_search.config"] = config_module

    tree_store_module = types.ModuleType("customized_areal.tree_search.core.tree_store")
    tree_store_module.Node = Node
    sys.modules["customized_areal.tree_search.core.tree_store"] = tree_store_module

    uncertainty_module = types.ModuleType(
        "customized_areal.tree_search.core.uncertainty"
    )
    uncertainty_module.should_discard_query = lambda *args, **kwargs: False
    sys.modules["customized_areal.tree_search.core.uncertainty"] = uncertainty_module

    selected_turn_module_obj = types.ModuleType(
        "customized_areal.tree_search.distilling.selected_turn_distill"
    )
    selected_turn_module_obj.parse_episode_diagnosis = (
        selected_turn_module.parse_episode_diagnosis
    )
    selected_turn_module_obj.selected_turn_to_position_rewards = (
        selected_turn_module.selected_turn_to_position_rewards
    )
    sys.modules[
        "customized_areal.tree_search.distilling.selected_turn_distill"
    ] = selected_turn_module_obj

    teacher_client_module = types.ModuleType(
        "customized_areal.tree_search.distilling.teacher_client"
    )
    teacher_client_module.TeacherServiceError = TeacherServiceError
    sys.modules[
        "customized_areal.tree_search.distilling.teacher_client"
    ] = teacher_client_module

    areal_pkg = types.ModuleType("areal")
    areal_pkg.__path__ = []
    sys.modules["areal"] = areal_pkg

    areal_api = types.ModuleType("areal.api")

    class RolloutWorkflow:
        pass

    areal_api.RolloutWorkflow = RolloutWorkflow
    sys.modules["areal.api"] = areal_api

    areal_utils = types.ModuleType("areal.utils")
    logging_module = types.ModuleType("areal.utils.logging")
    logging_module.getLogger = logging.getLogger
    areal_utils.logging = logging_module
    sys.modules["areal.utils"] = areal_utils
    sys.modules["areal.utils.logging"] = logging_module


def _load_workflow_module(selected_turn_module: StubSelectedTurnModule):
    module_name = "tests._teacher_failure_recovery_workflow"
    sys.modules.pop(module_name, None)
    _install_workflow_stubs(selected_turn_module)
    spec = importlib.util.spec_from_file_location(module_name, WORKFLOW_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_workflow(module, loss_mode: LossMode):
    workflow = object.__new__(module.TreeSearchGroupedRolloutWorkflow)
    workflow.loss_mode = loss_mode
    workflow.topk_distill = False
    workflow.teacher_top_k = 4
    return workflow


def _make_node(node_id: str, episode_id: str, turn_idx: int) -> Node:
    return Node(
        input_ids=[10, 11, 12],
        loss_mask=[0, 1, 1],
        logprobs=[0.0, 0.0, 0.0],
        versions=[0, 0, 0],
        node_id=node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
    )


def test_prepare_distill_for_episode_falls_back_after_diagnose_failure():
    selected_turn = StubSelectedTurnModule(
        planned_results=[[Reward(teacher_logprobs=[-0.2], candidate_token_ids=[11])]]
    )
    module = _load_workflow_module(selected_turn)
    workflow = _make_workflow(module, LossMode.DISTILL)
    provider = DiagnoseFailureProvider()
    node = _make_node("node-1", "episode-1", 1)

    prepared_nodes, rewards_by_node_id = asyncio.run(
        workflow._prepare_distill_for_episode(
            nodes=[node],
            data={"answer": "42"},
            engine=None,
            provider=provider,
            tokenizer=DummyTokenizer(),
        )
    )

    assert prepared_nodes == [node]
    assert list(rewards_by_node_id) == ["node-1"]
    assert rewards_by_node_id["node-1"][0].teacher_logprobs == [-0.2]
    assert selected_turn.calls == [
        {
            "node_id": "node-1",
            "guidance": "",
            "sample_index": 0,
        }
    ]


def test_prepare_distill_for_episode_skips_only_failed_nodes():
    selected_turn = StubSelectedTurnModule(
        planned_results=[
            [Reward(teacher_logprobs=[-0.1], candidate_token_ids=[11])],
            TeacherServiceError("teacher logprob backend unavailable"),
        ]
    )
    module = _load_workflow_module(selected_turn)
    workflow = _make_workflow(module, LossMode.DISTILL)
    provider = DiagnoseFailureProvider()
    failed_node = _make_node("node-fail", "episode-2", 1)
    kept_node = _make_node("node-keep", "episode-2", 2)

    prepared_nodes, rewards_by_node_id = asyncio.run(
        workflow._prepare_distill_for_episode(
            nodes=[failed_node, kept_node],
            data={"answer": "42"},
            engine=None,
            provider=provider,
            tokenizer=DummyTokenizer(),
        )
    )

    assert prepared_nodes == [failed_node, kept_node]
    assert list(rewards_by_node_id) == ["node-keep"]
    assert failed_node.teacher_logp is None
    assert kept_node.teacher_logp == [[-0.1]]


def test_prepare_distill_for_episode_drops_distill_episode_if_all_nodes_fail():
    selected_turn = StubSelectedTurnModule(
        planned_results=[TeacherServiceError("teacher logprob backend unavailable")]
    )
    module = _load_workflow_module(selected_turn)
    workflow = _make_workflow(module, LossMode.DISTILL)
    provider = DiagnoseFailureProvider()
    node = _make_node("node-drop", "episode-3", 1)

    prepared_nodes, rewards_by_node_id = asyncio.run(
        workflow._prepare_distill_for_episode(
            nodes=[node],
            data={"answer": "42"},
            engine=None,
            provider=provider,
            tokenizer=DummyTokenizer(),
        )
    )

    assert prepared_nodes == []
    assert rewards_by_node_id == {}
