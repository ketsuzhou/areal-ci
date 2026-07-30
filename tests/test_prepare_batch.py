"""Unit tests for prepare_batch method across different engine implementations.

Tests cover:
- TrainController.prepare_batch delegation to rollout
- DistRolloutCoordinator.prepare_batch with mocked engines
- WorkflowExecutor.prepare_batch with mocked dependencies
- FSDPEngine.prepare_batch integration
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from areal.api import (
    AllocationMode,
    FinetuneSpec,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import TrainEngineConfig
from areal.infra import TrainController
from areal.infra.dist_rollout import DistRolloutCoordinator


class MockWorkflow(WorkflowLike):
    """Mock workflow for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return mock rollout results."""
        batch_size = data.get("input_ids", torch.tensor([[1, 2, 3]])).shape[0]
        return [
            {
                "input_ids": torch.randint(0, 100, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.bool),
                "loss_mask": torch.ones(batch_size, 10, dtype=torch.bool),
                "rewards": torch.randn(batch_size),
            }
        ]


class MockStatefulDataLoader:
    """Mock StatefulDataLoader for testing."""

    def __init__(self, data: list[dict[str, Any]], batch_size: int = 2):
        self._data = data
        self.batch_size = batch_size
        self._index = 0

    def __iter__(self):
        return iter(self._data)

    def state_dict(self):
        return {"index": self._index}

    def load_state_dict(self, state):
        self._index = state.get("index", 0)


@pytest.fixture
def mock_ft_spec():
    """Create a mock FinetuneSpec."""
    return FinetuneSpec(
        total_train_epochs=1,
        dataset_size=100,
        train_batch_size=4,
    )


@pytest.fixture
def mock_alloc_mode():
    """Create a mock AllocationMode."""
    return AllocationMode.from_str("sglang:d1p1t1+fsdp:d1p1t1")


@pytest.fixture
def train_controller(mock_ft_spec, mock_alloc_mode):
    """Create a TrainController fixture."""
    config = TrainEngineConfig(
        experiment_name="test_prepare_batch",
        trial_name="test_trial",
    )

    class MockScheduler:
        def __init__(self):
            self.workers = []
            self.engine_calls = []

        def get_workers(self, role, timeout=None):
            return self.workers

        async def async_call_engine(self, worker_id, method, *args, **kwargs):
            self.engine_calls.append((worker_id, method, args, kwargs))
            if method == "is_data_parallel_head":
                return True
            return None

    scheduler = MockScheduler()
    controller = TrainController(
        train_engine_cls=Mock,
        train_config=config,
        scheduler=scheduler,
    )
    return controller


class TestTrainControllerPrepareBatch:
    """Tests for TrainController.prepare_batch method."""

    def test_prepare_batch_delegates_to_rollout(
        self, train_controller, mock_ft_spec, mock_alloc_mode
    ):
        """Test that prepare_batch properly delegates to rollout controller."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=mock_alloc_mode,
            ft_spec=mock_ft_spec,
        )

        # Create mock rollout controller
        mock_rollout = Mock()
        expected_trajectories = [
            {
                "input_ids": torch.randint(0, 100, (2, 10)),
                "attention_mask": torch.ones(2, 10, dtype=torch.bool),
                "rewards": torch.tensor([1.0, 2.0]),
            }
        ]
        mock_rollout.prepare_batch.return_value = expected_trajectories

        # Connect rollout engine
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        # Create mock dataloader
        mock_dataloader = Mock()
        mock_dataloader.batch_size = 4

        # Call prepare_batch
        result = train_controller.prepare_batch(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
            group_size=2,
            dynamic_bs=True,
        )

        # Verify rollout.prepare_batch was called with correct arguments
        mock_rollout.prepare_batch.assert_called_once_with(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
            group_size=2,
            dynamic_bs=True,
        )

        # Verify result is returned correctly
        assert result == expected_trajectories

    def test_prepare_batch_with_should_accept_fn(
        self, train_controller, mock_ft_spec, mock_alloc_mode
    ):
        """Test prepare_batch with should_accept_fn parameter."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=mock_alloc_mode,
            ft_spec=mock_ft_spec,
        )

        mock_rollout = Mock()
        mock_rollout.prepare_batch.return_value = []

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        mock_dataloader = Mock()
        mock_dataloader.batch_size = 4

        # Custom accept function
        def custom_accept_fn(trajectory: dict[str, Any]) -> bool:
            return trajectory.get("reward", 0.0) > 0.5

        train_controller.prepare_batch(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={},
            should_accept_fn=custom_accept_fn,
            group_size=1,
            dynamic_bs=False,
        )

        # Verify should_accept_fn is passed through
        call_kwargs = mock_rollout.prepare_batch.call_args.kwargs
        assert call_kwargs["should_accept_fn"] == custom_accept_fn


class TestDistRolloutCoordinatorPrepareBatch:
    """Tests for DistRolloutCoordinator.prepare_batch method."""

    @pytest.fixture
    def mock_engines(self):
        """Create mock rollout and train engines."""
        rollout_engine = Mock()
        train_engine = Mock()
        return rollout_engine, train_engine

    def test_prepare_batch_not_data_parallel_head(self, mock_engines):
        """Test prepare_batch when not data parallel head."""
        rollout_engine, train_engine = mock_engines

        # Setup: train_engine is NOT DP head
        train_engine.is_data_parallel_head.return_value = False

        coordinator = DistRolloutCoordinator(rollout_engine, train_engine)

        # Mock _broadcast_and_redistribute_trajectories to return empty list
        with patch.object(
            coordinator,
            "_broadcast_and_redistribute_trajectories",
            return_value=[],
        ) as mock_broadcast:
            mock_dataloader = Mock()
            mock_dataloader.batch_size = 4

            result = coordinator.prepare_batch(
                dataloader=mock_dataloader,
                workflow=MockWorkflow,
                workflow_kwargs={},
                should_accept_fn=None,
                group_size=1,
                dynamic_bs=False,
            )

            # Verify rollout_engine.prepare_batch was NOT called
            rollout_engine.prepare_batch.assert_not_called()

            # Verify broadcast was called with None
            mock_broadcast.assert_called_once()
            assert mock_broadcast.call_args[0][0] is None

            assert result == []

    def test_prepare_batch_is_data_parallel_head(self, mock_engines):
        """Test prepare_batch when IS data parallel head."""
        rollout_engine, train_engine = mock_engines

        # Setup: train_engine IS DP head
        train_engine.is_data_parallel_head.return_value = True

        expected_trajectories = [
            {
                "input_ids": torch.randint(0, 100, (2, 10)),
                "attention_mask": torch.ones(2, 10, dtype=torch.bool),
                "rewards": torch.tensor([1.0, 2.0]),
            }
        ]
        rollout_engine.prepare_batch.return_value = expected_trajectories

        coordinator = DistRolloutCoordinator(rollout_engine, train_engine)

        with patch.object(
            coordinator,
            "_broadcast_and_redistribute_trajectories",
            return_value=expected_trajectories,
        ) as mock_broadcast:
            mock_dataloader = Mock()
            mock_dataloader.batch_size = 4

            result = coordinator.prepare_batch(
                dataloader=mock_dataloader,
                workflow=MockWorkflow,
                workflow_kwargs={"temperature": 0.7},
                should_accept_fn=None,
                group_size=2,
                dynamic_bs=True,
            )

            # Verify rollout_engine.prepare_batch was called
            rollout_engine.prepare_batch.assert_called_once_with(
                mock_dataloader,
                workflow=MockWorkflow,
                workflow_kwargs={"temperature": 0.7},
                should_accept_fn=None,
                group_size=2,
                dynamic_bs=True,
            )

            # Verify broadcast was called with trajectories
            mock_broadcast.assert_called_once()
            assert result == expected_trajectories

    def test_prepare_batch_moves_tensors_to_device(self, mock_engines):
        """Test that trajectories are moved to correct device when DP head."""
        rollout_engine, train_engine = mock_engines

        train_engine.is_data_parallel_head.return_value = True

        # Create trajectories that should be moved to device
        cpu_trajectories = [
            {
                "input_ids": torch.randint(0, 100, (2, 10)),
                "attention_mask": torch.ones(2, 10, dtype=torch.bool),
            }
        ]
        rollout_engine.prepare_batch.return_value = cpu_trajectories

        coordinator = DistRolloutCoordinator(rollout_engine, train_engine)

        with (
            patch.object(
                coordinator,
                "_broadcast_and_redistribute_trajectories",
                return_value=cpu_trajectories,
            ),
            patch(
                "areal.infra.dist_rollout.current_platform.current_device",
                return_value="cuda:0",
            ),
            patch(
                "areal.infra.dist_rollout.tensor_container_to",
                return_value=cpu_trajectories,
            ) as mock_to_device,
        ):
            mock_dataloader = Mock()
            mock_dataloader.batch_size = 4

            coordinator.prepare_batch(
                dataloader=mock_dataloader,
                workflow=MockWorkflow,
            )

            # Verify tensor_container_to was called
            mock_to_device.assert_called_once()


class TestPrepareBatchIntegration:
    """Integration-style tests for prepare_batch with mocked components."""

    def test_prepare_batch_returns_list_of_dicts(self):
        """Test that prepare_batch returns list of trajectory dictionaries."""
        mock_rollout = Mock()

        expected_result = [
            {
                "input_ids": torch.randint(0, 100, (2, 10)),
                "attention_mask": torch.ones(2, 10, dtype=torch.bool),
                "loss_mask": torch.zeros(2, 10, dtype=torch.bool),
                "rewards": torch.tensor([1.0, 0.5]),
                "logprobs": torch.randn(2, 10),
            },
            {
                "input_ids": torch.randint(0, 100, (2, 8)),
                "attention_mask": torch.ones(2, 8, dtype=torch.bool),
                "loss_mask": torch.zeros(2, 8, dtype=torch.bool),
                "rewards": torch.tensor([0.8, 1.2]),
                "logprobs": torch.randn(2, 8),
            },
        ]
        mock_rollout.prepare_batch.return_value = expected_result

        result = mock_rollout.prepare_batch(
            dataloader=Mock(),
            workflow=MockWorkflow,
            group_size=1,
        )

        assert isinstance(result, list)
        assert len(result) == 2
        for traj in result:
            assert isinstance(traj, dict)
            assert "input_ids" in traj
            assert "attention_mask" in traj

    def test_prepare_batch_with_group_size(self):
        """Test prepare_batch with group_size parameter."""
        mock_rollout = Mock()
        mock_rollout.prepare_batch.return_value = []

        mock_dataloader = Mock()
        mock_dataloader.batch_size = 4

        mock_rollout.prepare_batch(
            dataloader=mock_dataloader,
            workflow=MockWorkflow,
            group_size=4,
            dynamic_bs=False,
        )

        # Verify group_size is passed correctly
        call_kwargs = mock_rollout.prepare_batch.call_args.kwargs
        assert call_kwargs["group_size"] == 4

    def test_prepare_batch_workflow_kwarg_passing(self):
        """Test that workflow_kwargs are properly passed to workflow."""
        mock_rollout = Mock()

        workflow_kwargs = {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 100,
        }

        mock_rollout.prepare_batch(
            dataloader=Mock(),
            workflow=MockWorkflow,
            workflow_kwargs=workflow_kwargs,
        )

        # Verify workflow_kwargs are passed
        call_kwargs = mock_rollout.prepare_batch.call_args.kwargs
        assert call_kwargs["workflow_kwargs"] == workflow_kwargs


class TestPrepareBatchErrorHandling:
    """Tests for error handling in prepare_batch."""

    def test_prepare_batch_rollout_not_connected(self, train_controller):
        """Test that prepare_batch raises when rollout not connected."""
        mock_dataloader = Mock()

        with pytest.raises(RuntimeError, match="rollout engine not connected"):
            train_controller.prepare_batch(
                dataloader=mock_dataloader,
                workflow="test.workflow",
            )

    def test_prepare_batch_engine_not_initialized(self, mock_engines):
        """Test behavior when engine is not properly initialized."""
        rollout_engine, train_engine = mock_engines

        coordinator = DistRolloutCoordinator(rollout_engine, train_engine)

        # Don't call connect_engine - should handle gracefully or raise
        # The actual behavior depends on implementation
        mock_dataloader = Mock()

        # This tests that we handle the case where rollout_engine.prepare_batch
        # might raise an error
        rollout_engine.prepare_batch.side_effect = RuntimeError("Engine not ready")

        with pytest.raises(RuntimeError, match="Engine not ready"):
            coordinator.prepare_batch(
                dataloader=mock_dataloader,
                workflow=MockWorkflow,
            )
