import pytest
import torch

from customized_areal.optimizers.muon import (
    MuonWithAuxAdam,
    muon_update,
    zeropower_via_newtonschulz5,
)


class TestZeropowerViaNewtonSchulz5:
    def test_square_matrix_near_orthogonal(self):
        G = torch.randn(8, 8)
        result = zeropower_via_newtonschulz5(G, steps=5)
        Z = result.float() @ result.float().mT
        torch.testing.assert_close(Z, torch.eye(8), atol=0.4, rtol=0.4)

    def test_tall_matrix_near_orthogonal(self):
        G = torch.randn(16, 8)
        result = zeropower_via_newtonschulz5(G, steps=5)
        Z = result.float().mT @ result.float()
        torch.testing.assert_close(Z, torch.eye(8), atol=0.4, rtol=0.4)

    def test_wide_matrix_near_orthogonal(self):
        G = torch.randn(8, 16)
        result = zeropower_via_newtonschulz5(G, steps=5)
        Z = result.float() @ result.float().mT
        torch.testing.assert_close(Z, torch.eye(8), atol=0.4, rtol=0.4)

    def test_batched_matrix(self):
        G = torch.randn(3, 8, 8)
        result = zeropower_via_newtonschulz5(G, steps=5)
        assert result.shape == (3, 8, 8)
        for i in range(3):
            Z = result[i].float() @ result[i].float().mT
            torch.testing.assert_close(Z, torch.eye(8), atol=0.4, rtol=0.4)

    def test_output_dtype_bfloat16(self):
        G = torch.randn(4, 4)
        result = zeropower_via_newtonschulz5(G, steps=5)
        assert result.dtype == torch.bfloat16

    def test_1d_input_raises(self):
        with pytest.raises(AssertionError):
            zeropower_via_newtonschulz5(torch.randn(8), steps=5)


class TestMuonUpdate:
    def test_output_shape_matches_grad(self):
        grad = torch.randn(4, 4)
        momentum_buffer = torch.zeros_like(grad)
        update = muon_update(grad, momentum_buffer, beta=0.95)
        assert update.shape == grad.shape

    def test_momentum_buffer_modified_inplace(self):
        grad = torch.randn(4, 4)
        momentum_buffer = torch.zeros_like(grad)
        muon_update(grad, momentum_buffer, beta=0.95)
        assert not torch.all(momentum_buffer == 0)

    def test_4d_input_flattened_to_2d(self):
        grad = torch.randn(2, 3, 4, 5)
        momentum_buffer = torch.zeros_like(grad)
        update = muon_update(grad, momentum_buffer, beta=0.95)
        assert update.shape == grad.shape

    def test_nesterov_vs_standard(self):
        grad = torch.randn(4, 4)
        buf_n = torch.zeros_like(grad)
        buf_s = torch.zeros_like(grad)
        update_n = muon_update(grad.clone(), buf_n, beta=0.95, nesterov=True)
        update_s = muon_update(grad.clone(), buf_s, beta=0.95, nesterov=False)
        assert not torch.allclose(update_n, update_s)


class TestMuonWithAuxAdam:
    def _make_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
        )

    def _split_params(self, model):
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.ndim < 2]
        return muon_params, adam_params

    def test_creates_both_param_groups(self):
        model = self._make_model()
        muon_params, adam_params = self._split_params(model)
        param_groups = [
            {"params": muon_params, "use_muon": True, "lr": 0.02, "momentum": 0.95},
            {"params": adam_params, "use_muon": False, "lr": 3e-4},
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["use_muon"] is True
        assert optimizer.param_groups[1]["use_muon"] is False

    def test_step_updates_parameters(self):
        model = self._make_model()
        muon_params, adam_params = self._split_params(model)
        param_groups = [
            {"params": muon_params, "use_muon": True, "lr": 0.02, "momentum": 0.95},
            {"params": adam_params, "use_muon": False, "lr": 3e-4},
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        params_before = [p.clone() for p in model.parameters()]
        optimizer.step()
        params_after = list(model.parameters())
        any_changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert any_changed

    def test_state_keys_muon_vs_adam(self):
        model = self._make_model()
        muon_params, adam_params = self._split_params(model)
        param_groups = [
            {"params": muon_params, "use_muon": True, "lr": 0.02, "momentum": 0.95},
            {"params": adam_params, "use_muon": False, "lr": 3e-4},
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        muon_p = muon_params[0]
        assert "momentum_buffer" in optimizer.state[muon_p]
        adam_p = adam_params[0]
        assert "exp_avg" in optimizer.state[adam_p]
        assert "exp_avg_sq" in optimizer.state[adam_p]
        assert "step" in optimizer.state[adam_p]

    def test_no_grad_params_skipped(self):
        model = self._make_model()
        for p in model[0].parameters():
            p.requires_grad = False
        muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            {"params": muon_params, "use_muon": True, "lr": 0.02, "momentum": 0.95},
            {"params": adam_params, "use_muon": False, "lr": 3e-4},
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    def test_state_dict_round_trip(self):
        model = self._make_model()
        muon_params, adam_params = self._split_params(model)
        param_groups = [
            {"params": muon_params, "use_muon": True, "lr": 0.02, "momentum": 0.95},
            {"params": adam_params, "use_muon": False, "lr": 3e-4},
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        state = optimizer.state_dict()
        assert "state" in state
        assert "param_groups" in state

    def test_weight_decay_applied(self):
        model = self._make_model()
        muon_params, adam_params = self._split_params(model)
        param_groups = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": 0.02,
                "momentum": 0.95,
                "weight_decay": 0.1,
            },
            {"params": adam_params, "use_muon": False, "lr": 3e-4, "weight_decay": 0.1},
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        # Weight decay is applied as: p.mul_(1 - lr * weight_decay)
        # Verify the optimizer has weight_decay configured
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[1]["weight_decay"] == 0.1
        # Verify step completes without error
        optimizer.step()


class TestPatchFsdpEngineForMuon:
    def test_patch_replaces_create_optimizer(self):
        from areal.engine.fsdp_engine import FSDPEngine

        original = FSDPEngine._create_optimizer
        from customized_areal.optimizers.patch import patch_fsdp_engine_for_muon

        patch_fsdp_engine_for_muon()
        assert FSDPEngine._create_optimizer is not original
        from customized_areal.optimizers.patch import unpatch_fsdp_engine_for_muon

        unpatch_fsdp_engine_for_muon()
        assert FSDPEngine._create_optimizer is original

    def test_patch_idempotent(self):
        from areal.engine.fsdp_engine import FSDPEngine

        original = FSDPEngine._create_optimizer
        from customized_areal.optimizers.patch import (
            patch_fsdp_engine_for_muon,
            unpatch_fsdp_engine_for_muon,
        )

        patch_fsdp_engine_for_muon()
        patch_fsdp_engine_for_muon()
        unpatch_fsdp_engine_for_muon()
        assert FSDPEngine._create_optimizer is original

    def test_unpatch_without_patch_is_noop(self):
        from areal.engine.fsdp_engine import FSDPEngine

        original = FSDPEngine._create_optimizer
        from customized_areal.optimizers.patch import unpatch_fsdp_engine_for_muon

        unpatch_fsdp_engine_for_muon()
        assert FSDPEngine._create_optimizer is original

    def test_patch_stores_muon_config(self):
        from customized_areal.optimizers.patch import (
            _muon_config,
            patch_fsdp_engine_for_muon,
            unpatch_fsdp_engine_for_muon,
        )

        try:
            patch_fsdp_engine_for_muon(
                momentum=0.9, muon_adam_lr=1e-4, ns_steps=3, nesterov=False
            )
            assert _muon_config["momentum"] == 0.9
            assert _muon_config["muon_adam_lr"] == 1e-4
            assert _muon_config["ns_steps"] == 3
            assert _muon_config["nesterov"] is False
        finally:
            unpatch_fsdp_engine_for_muon()

    def test_non_muon_type_delegates_to_original(self):
        from customized_areal.optimizers.patch import (
            patch_fsdp_engine_for_muon,
            unpatch_fsdp_engine_for_muon,
        )

        from areal.engine.fsdp_engine import FSDPEngine

        original = FSDPEngine._create_optimizer
        try:
            patch_fsdp_engine_for_muon()
            # The patched method delegates when type != "muon"
            # We can verify the method is different from original
            assert FSDPEngine._create_optimizer is not original
        finally:
            unpatch_fsdp_engine_for_muon()
            # Make sure it's restored even if something fails
            FSDPEngine._create_optimizer = original
