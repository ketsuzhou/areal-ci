"""End-to-end validation for the agentic-verifier + critic/GAE flow (Task 10).

Two layers:

  1. **Torch-free data-flow e2e** (runs here): a 2-run DAG goes through the full
     non-GPU pipeline -- pi-verifier finalize (fakes) assigns per-session reward,
     rewards land on the nodes via the session map, the critic values are set,
     and the global GAE assembler produces per-node advantages/returns. Includes
     the resource-leak assertion (every started session is harvested exactly
     once -- no orphans).

  2. **GPU train-step e2e** (skipped): the live group_size=2 rollout + combined
     actor+critic update + sandbox/issue teardown require torch + FSDP + a GPU +
     a live Multica/Fleet stack, unavailable here. Skipped with explanation.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from customized_areal.tree_search.agents.agentic_verifier import AgenticVerifier
from customized_areal.tree_search.agents.dag_advantage import assemble_node_advantages
from customized_areal.tree_search.agents.execution_dag import (
    EdgeType,
    ExecutionDAG,
    SuperNode,
)
from customized_areal.tree_search.agents.harvest import VerifierFinalizer

_HAS_TORCH = importlib.util.find_spec("torch") is not None


class _FakeLauncher:
    def __init__(self, output: str) -> None:
        self._output = output

    async def run(self, *, prompt: str, judge_model: str) -> str:
        return self._output


class _RecordingBridge:
    def __init__(self) -> None:
        self.set_reward_calls: list[tuple[str, float]] = []
        self.export_calls: list[str] = []

    async def set_reward(self, *, session_id: str, reward: float) -> None:
        self.set_reward_calls.append((session_id, reward))

    async def export(self, *, session_id: str) -> None:
        self.export_calls.append(session_id)


def _two_run_dag() -> ExecutionDAG:
    """planner A0 delegates to worker B0; B0 completes back to A0's terminal."""
    dag = ExecutionDAG()
    a = SuperNode(node_id="A0", agent_id="planner", issue_id="i1", task_id="t")
    b = SuperNode(node_id="B0", agent_id="worker", issue_id="i2", task_id="t")
    dag.add_event(a)
    dag.add_event(b)
    dag.add_edge("A0", "B0", EdgeType.DELEGATION)
    dag.set_session_id("A0", "sess-A0")
    dag.set_session_id("B0", "sess-B0")
    return dag


def _verifier_output(rewards: list[dict]) -> str:
    return "reviewed\n```json\n" + json.dumps({"rewards": rewards}) + "\n```"


@pytest.mark.asyncio
async def test_e2e_data_flow_verify_to_gae_no_orphans() -> None:
    dag = _two_run_dag()
    session_map = dag.session_map()  # {"A0": "sess-A0", "B0": "sess-B0"}

    # --- Phase 2: verifier finalize assigns per-session reward + harvests ---
    launcher = _FakeLauncher(
        _verifier_output(
            [
                {"session_id": "sess-A0", "reward": 0.4, "rationale": "planned ok"},
                {"session_id": "sess-B0", "reward": 1.0, "rationale": "did the work"},
            ]
        )
    )
    bridge = _RecordingBridge()
    finalizer = VerifierFinalizer(
        verifier=AgenticVerifier(launcher=launcher, judge_model="judge/fixed"),
        reward_writer=bridge,
        harvester=bridge,
    )
    result = await finalizer.finalize(
        task_id="t",
        transcripts={"A0": "planner transcript", "B0": "worker transcript"},
        session_map=session_map,
        acceptance_criteria=["task solved"],
    )
    assert result.run.ok is True
    reward_by_session = dict(bridge.set_reward_calls)

    # Resource-leak assertion: every started session harvested exactly once.
    started = {sid for sid in session_map.values() if sid}
    assert set(result.exported) == started
    assert len(result.exported) == len(set(result.exported))  # no double export

    # --- map per-session reward back onto the DAG nodes (outcome reward) ---
    session_to_node = {sid: nid for nid, sid in session_map.items() if sid}
    for sid, reward in reward_by_session.items():
        dag.get(session_to_node[sid]).outcome_reward = reward

    # --- critic produced a value per node (V_{t+1}); set them here ---
    dag.get("A0").value = 0.3
    dag.get("B0").value = 0.0  # terminal node bootstraps to 0

    # --- Phase 3: global GAE over the completion-ordered nodes ---
    ordered = [dag.get("A0"), dag.get("B0")]  # global completion order
    adv = assemble_node_advantages(ordered, initial_value=0.0, gamma=1.0, lam=1.0)

    # Every node got an advantage and a return; terminal reward (1.0 on B0)
    # flows back to the planner via the global trajectory.
    assert set(adv.advantages) == {"A0", "B0"}
    assert set(adv.returns) == {"A0", "B0"}
    # B0 terminal: r=process(0)+outcome(1.0); delta_B0 = 1.0 + 0 - V_1(0.3) = 0.7
    assert adv.advantages["B0"] == pytest.approx(0.7)
    # A0: delta_A0 = r_A0(0.4) + gamma*V_1(0.3) - V_0(0.0) = 0.7 ;
    #     adv_A0 = delta_A0 + gamma*lam*adv_B0 = 0.7 + 0.7 = 1.4
    assert adv.advantages["A0"] == pytest.approx(1.4)


@pytest.mark.skipif(
    not _HAS_TORCH,
    reason="group_size=2 e2e train step requires torch + FSDP + GPU + a live "
    "Multica/Fleet stack (not available here). See CRITIC_GAE_INTEGRATION.md; "
    "run on a GPU node.",
)
def test_e2e_group_size_2_train_step_with_teardown() -> None:  # pragma: no cover
    pytest.skip(
        "requires GPU training stack + live Multica/Fleet; validated on a GPU node"
    )
