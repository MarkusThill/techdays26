from __future__ import annotations

from collections.abc import Iterable

import torch


class TCL(torch.optim.Optimizer):
    """Temporal-Coherence Learning optimizer (Bagheri et al., 2015).

    Per-weight learning rate is scaled by the sign-coherence of past
    gradients — well-suited to the sparse LUT updates of n-tuple networks.

    For each weight ``w_i``::

        N_i ← N_i + g_i              # signed sum of grads
        A_i ← A_i + |g_i|            # absolute sum of grads
        η_i = lr · |N_i| / (A_i + ε) # ∈ [0, lr]
        w_i ← w_i − η_i · g_i

    Coherent gradients (consistent sign) → effective rate ≈ lr.
    Oscillating gradients              → effective rate ≈ 0.

    Args:
        params: Iterable of parameters to optimize.
        lr:     Base learning rate. For n-tuple Connect-4 networks try
                values in the range 0.1 – 1.0 (much larger than for Adam,
                because the |N|/A factor is typically << 1).
        beta:   Decay factor for N and A.
                ``1.0`` (default) = classic unbounded sums.
                ``< 1.0``         = exponential moving average — useful for
                                    non-stationary targets.
        eps:    Numerical-stability term added to the denominator.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"Invalid beta: {beta} (must be in (0, 1])")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["N"] = torch.zeros_like(p)
                    state["A"] = torch.zeros_like(p)
                N, A = state["N"], state["A"]

                if beta == 1.0:
                    N.add_(g)
                    A.add_(g.abs())
                else:
                    N.mul_(beta).add_(g)
                    A.mul_(beta).add_(g.abs())

                eta = lr * N.abs() / (A + eps)
                p.add_(-eta * g)

        return loss
