"""Verifier finalization helper for tree-search RL sessions.

Runs the verifier over a completed run and writes the resulting reward to the
RL session (replacing the constant ``set_reward(1.0)``).

Branching is driven entirely through the env-dispatch primitive
(``EnvDispatchBranchDriver``); the former cloud-branch transport
(``BranchMaterializer`` / ``MulticaIssueForker`` / ``agent_start_branch``) has
been removed.
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.agents.verifier import VerifierResult


async def finalize_with_verifier(
    *,
    verifier: Any,
    writer: Any,
    session_id: str,
    run: dict[str, Any],
) -> VerifierResult:
    """Run the verifier and write the reward to the RL session.

    Replaces the constant ``set_reward(1.0)``. Returns the
    :class:`VerifierResult` so the caller can inspect it.
    """
    result = await verifier.verify(run)
    await writer.finalize(session_id=session_id, verifier_result=result)
    return result
