from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute approximate zero-power (orthogonal) matrix via Newton-Schulz iteration.

    Uses a 5th-order cubic polynomial iteration that is numerically stable in bfloat16.
    Coefficients from: https://github.com/KellerJordan/Muon

    Args:
        G: Input matrix of shape (..., m, n) with m >= n recommended.
        steps: Number of Newton-Schulz iterations.

    Returns:
        Approximate zero-power matrix of same shape as G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute Muon update: momentum + Newton-Schulz orthogonalization.

    Args:
        grad: Current gradient.
        momentum_buffer: EMA of past gradients (modified in-place).
        beta: Momentum coefficient.
        ns_steps: Number of Newton-Schulz iterations.
        nesterov: Whether to use Nesterov-style momentum lookahead.

    Returns:
        Orthogonalized update of same shape as grad.
    """
    original_shape = grad.shape
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update.view(original_shape)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices + Adam for all other parameters.

    Adapted from SingleDeviceMuonWithAuxAdam in https://github.com/KellerJordan/Muon.
    FSDP2-compatible: no dist.all_gather (FSDP handles distributed state).

    Expects two param groups:
      - use_muon=True: 2D weight matrices optimized with Muon (momentum + Newton-Schulz)
      - use_muon=False: non-2D params (biases, embeddings, norms) optimized with Adam

    Args:
        param_groups: List of dicts, each with a "use_muon" key and optimizer-specific
            hyperparameters (lr, momentum, weight_decay for Muon; lr, betas, eps,
            weight_decay for Adam).
        ns_steps: Number of Newton-Schulz iterations for Muon updates.
        nesterov: Whether to use Nesterov momentum lookahead for Muon updates.
    """

    def __init__(
        self,
        param_groups: list[dict],
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, dict())
        self.ns_steps = ns_steps
        self.nesterov = nesterov

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=self.ns_steps,
                        nesterov=self.nesterov,
                    )
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    grad = p.grad
                    beta1, beta2 = group["betas"]

                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] / bias_correction1
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
