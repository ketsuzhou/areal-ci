# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenRouter remote rollout proxy.

Spec: docs/superpowers/specs/2026-06-29-openrouter-remote-rollout-proxy-design.md
"""

from __future__ import annotations

import logging

import pytest

from areal.api import cli_args as cli_args_module
from areal.api.cli_args import PPOActorConfig

# ---------------------------------------------------------------------------
# Tests: PPOActorConfig enable_remote_rollout warning (spec tests 18-20)
# ---------------------------------------------------------------------------


# Areal's getLogger() replaces Logger.root/manager on import, which orphans
# caplog's root-attached handler and breaks `logging.getLogger("CLIArgs")`
# lookups. Attach a handler directly to the exact logger object the
# cli_args module holds, so capture is robust to that root replacement.
class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def cliargs_capture():
    handler = _ListHandler()
    logger = cli_args_module.logger
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


class TestEnableRemoteRolloutWarning:
    def test_warns_when_recompute_disabled(self, cliargs_capture):
        """enable_remote_rollout=True with no recompute path → warning."""
        PPOActorConfig(
            enable_remote_rollout=True,
            recompute_logprob=False,
            use_decoupled_loss=False,
        )
        assert any(
            "enable_remote_rollout" in rec.getMessage()
            and "recompute" in rec.getMessage()
            for rec in cliargs_capture.records
        ), (
            f"expected recompute warning, got: {[r.getMessage() for r in cliargs_capture.records]}"
        )

    def test_no_warning_when_recompute_logprob_true(self, cliargs_capture):
        """enable_remote_rollout=True + recompute_logprob=True → no warning."""
        PPOActorConfig(
            enable_remote_rollout=True,
            recompute_logprob=True,
            use_decoupled_loss=False,
        )
        assert not any(
            "enable_remote_rollout" in rec.getMessage()
            for rec in cliargs_capture.records
        ), f"unexpected warning: {[r.getMessage() for r in cliargs_capture.records]}"

    def test_no_warning_when_decoupled_loss_true(self, cliargs_capture):
        """enable_remote_rollout=True + use_decoupled_loss=True → no warning."""
        PPOActorConfig(
            enable_remote_rollout=True,
            recompute_logprob=False,
            use_decoupled_loss=True,
        )
        assert not any(
            "enable_remote_rollout" in rec.getMessage()
            for rec in cliargs_capture.records
        ), f"unexpected warning: {[r.getMessage() for r in cliargs_capture.records]}"
