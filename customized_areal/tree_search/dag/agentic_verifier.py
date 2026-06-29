"""AgenticVerifier -- pi-agent verifier driver (Phase 2, Task 3).

Replaces the in-process Python verifier with a *pi agent on a fixed judge
model* that reviews a finished multi-agent task and assigns a reward per RL
``session_id`` (design decisions 1-3). The agent reads the Multica transcripts
(read-only) and acceptance criteria, reasons about each agent's contribution,
and reports per-session rewards; this driver builds the prompt, launches the
agent, parses the structured verdict, validates it against the task's
``session_map``, and returns an auditable :class:`VerifierRun`.

On any launch/parse error the driver falls back to a safe *neutral* reward for
every session rather than crashing finalization.

Torch-free: uses stdlib :mod:`logging` so the ``dag`` package stays importable
without the training stack (consistent with ``execution_dag`` / ``environment``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger("AgenticVerifier")

# Reward range the verifier may assign per session.
_REWARD_MIN = 0.0
_REWARD_MAX = 1.0

# Matches a ```json ... ``` fenced block (the verifier's structured verdict).
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class VerifierReward:
    """One agent's reward as assigned by the verifier agent."""

    session_id: str
    reward: float
    rationale: str = ""


@dataclass
class VerifierRun:
    """Auditable record of one verifier-agent pass over a finished task.

    ``ok`` is ``False`` when the agent launch or output parse failed and the
    neutral fallback was applied; ``error`` then carries the reason.
    """

    task_id: str
    rewards: list[VerifierReward]
    judge_model: str
    ok: bool = True
    error: str | None = None
    raw_output: str = ""

    def reward_map(self) -> dict[str, float]:
        """``{session_id: reward}`` -- the per-session credit to write."""
        return {r.session_id: r.reward for r in self.rewards}


class PiVerifierLauncher(Protocol):
    """Seam that runs the pi verifier agent and returns its final output text.

    A real implementation launches the pi CLI (fixed judge model, the
    ``verifier-rl`` extension loaded) in the verifier sandbox and returns the
    agent's final message. Tests inject a fake that returns canned output.
    """

    async def run(self, *, prompt: str, judge_model: str) -> str: ...


def parse_verifier_output(text: str) -> list[tuple[str, float, str]]:
    """Extract ``[(session_id, reward, rationale)]`` from the agent output.

    The verifier agent appends a ```json {"rewards": [...]}``` block. Raises
    :class:`ValueError` when no valid block is present (the driver treats this
    as a parse error and applies the neutral fallback).
    """
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no ```json``` rewards block in verifier output")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"verifier output JSON invalid: {exc}") from exc
    rewards = payload.get("rewards")
    if not isinstance(rewards, list):
        raise ValueError("verifier output missing 'rewards' list")
    out: list[tuple[str, float, str]] = []
    for item in rewards:
        if not isinstance(item, dict):
            continue
        session_id = item.get("session_id")
        reward = item.get("reward")
        if not isinstance(session_id, str) or not isinstance(reward, (int, float)):
            continue
        rationale = item.get("rationale", "")
        out.append((session_id, float(reward), str(rationale)))
    return out


_PROMPT_TEMPLATE = """You are the verifier for a finished multi-agent task. \
Several agents collaborated; judge how much each agent contributed to \
satisfying the acceptance criteria, and assign each a reward in [0.0, 1.0].

Task id: {task_id}

Acceptance criteria:
{criteria}

Per-agent transcripts (one block per agent run):
{transcripts}

Agent run -> RL session id (assign a reward to each session id):
{session_lines}

For each session id, call /rl/set_reward <session_id> <reward> with a reward in \
[0.0, 1.0]. Then output ONLY a JSON block in the exact format:

```json
{{"rewards": [{{"session_id": "<id>", "reward": <0.0-1.0>, "rationale": "<why>"}}]}}
```
"""


def build_verifier_prompt(
    *,
    task_id: str,
    transcripts: dict[str, str],
    session_map: dict[str, str | None],
    acceptance_criteria: list[str],
) -> str:
    """Render the verifier prompt (transcripts + criteria + session map)."""
    criteria = "\n".join(f"- {c}" for c in acceptance_criteria) or "- (none provided)"
    transcript_blocks = (
        "\n".join(f"[agent {node_id}]\n{text}" for node_id, text in transcripts.items())
        or "(no transcripts)"
    )
    session_lines = (
        "\n".join(
            f"- {node_id} -> {sid}"
            for node_id, sid in session_map.items()
            if sid is not None
        )
        or "(no sessions)"
    )
    return _PROMPT_TEMPLATE.format(
        task_id=task_id,
        criteria=criteria,
        transcripts=transcript_blocks,
        session_lines=session_lines,
    )


class AgenticVerifier:
    """Drives the pi verifier agent and returns a :class:`VerifierRun`.

    Parameters
    ----------
    launcher:
        Runs the pi agent and returns its final output text.
    judge_model:
        Fixed judge model id. NOT trained, NOT routed through ``areal/...``.
    neutral_reward:
        Reward assigned to every session when the agent fails (safe fallback).
    """

    def __init__(
        self,
        *,
        launcher: PiVerifierLauncher,
        judge_model: str,
        neutral_reward: float = 0.0,
    ) -> None:
        self._launcher = launcher
        self._judge_model = judge_model
        self._neutral = float(neutral_reward)

    async def verify_task(
        self,
        *,
        task_id: str,
        transcripts: dict[str, str],
        session_map: dict[str, str | None],
        acceptance_criteria: list[str],
    ) -> VerifierRun:
        valid_sessions = {sid for sid in session_map.values() if sid is not None}
        prompt = build_verifier_prompt(
            task_id=task_id,
            transcripts=transcripts,
            session_map=session_map,
            acceptance_criteria=acceptance_criteria,
        )
        try:
            output = await self._launcher.run(
                prompt=prompt, judge_model=self._judge_model
            )
            parsed = parse_verifier_output(output)
        except Exception as exc:
            logger.error(
                "verifier agent failed (task_id=%s); applying neutral reward %.3f: %s",
                task_id,
                self._neutral,
                exc,
            )
            return self._neutral_run(task_id, valid_sessions, error=str(exc))

        rewards: list[VerifierReward] = []
        seen: set[str] = set()
        for session_id, reward, rationale in parsed:
            if session_id not in valid_sessions:
                logger.warning(
                    "verifier assigned reward to unknown session %s (task_id=%s); "
                    "dropping",
                    session_id,
                    task_id,
                )
                continue
            if session_id in seen:
                continue
            seen.add(session_id)
            clamped = min(_REWARD_MAX, max(_REWARD_MIN, reward))
            rewards.append(
                VerifierReward(
                    session_id=session_id, reward=clamped, rationale=rationale
                )
            )
        logger.info(
            "verifier assigned %d/%d session rewards (task_id=%s)",
            len(rewards),
            len(valid_sessions),
            task_id,
        )
        return VerifierRun(
            task_id=task_id,
            rewards=rewards,
            judge_model=self._judge_model,
            ok=True,
            raw_output=output,
        )

    def _neutral_run(
        self, task_id: str, sessions: set[str], *, error: str
    ) -> VerifierRun:
        rewards = [
            VerifierReward(
                session_id=sid, reward=self._neutral, rationale="neutral fallback"
            )
            for sid in sorted(sessions)
        ]
        return VerifierRun(
            task_id=task_id,
            rewards=rewards,
            judge_model=self._judge_model,
            ok=False,
            error=error,
        )


__all__ = [
    "AgenticVerifier",
    "PiVerifierLauncher",
    "VerifierReward",
    "VerifierRun",
    "build_verifier_prompt",
    "parse_verifier_output",
]
