from __future__ import annotations

import numpy as np
import torch


def plot_adam_effective_lr(
    optimizer: torch.optim.Optimizer, title_suffix: str = ""
) -> None:
    """Plot histograms of Adam's per-parameter effective learning rates.

    For Adam the actual per-weight update is::

        W -= lr * m_hat / (sqrt(v_hat) + eps)

    so the effective learning rate per weight is ``lr / (sqrt(v_hat) + eps)``,
    where ``v_hat = v / (1 - beta2^t)`` is the bias-corrected second moment.

    Three subplots are shown:

    1. Histogram of effective LR values.
    2. Histogram of log10(effective LR).
    3. Histogram of the ratio effective LR / scheduled LR.

    Args:
        optimizer:    A ``torch.optim.Adam`` instance that has been stepped at
                      least once (so optimizer state is populated).
        title_suffix: Optional string appended to the figure title.
    """
    import matplotlib.pyplot as plt

    for group in optimizer.param_groups:
        lr = group["lr"]
        _beta1, beta2 = group["betas"]
        eps = group["eps"]

        for p in group["params"]:
            state = optimizer.state[p]
            if not state:
                print("No optimizer state yet (run at least 1 step first)")
                return

            step = state["step"]
            v = state["exp_avg_sq"]

            bias_correction2 = 1.0 - beta2**step
            v_hat = v / bias_correction2
            eff_lr = lr / (torch.sqrt(v_hat) + eps)

            mask = v > 0
            eff_lr_active = eff_lr[mask].detach().cpu().numpy()

            if eff_lr_active.size == 0:
                print("No active weights found.")
                return

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            # 1) Linear histogram
            ax = axes[0]
            ax.hist(eff_lr_active, bins=100, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Effective LR")
            ax.set_ylabel("Count")
            ax.set_title(f"Effective LR distribution (step {step})")
            ax.axvline(lr, color="r", linestyle="--", label=f"scheduled lr={lr:.2e}")
            ax.legend()

            # 2) Log-scale histogram
            ax = axes[1]
            ax.hist(
                np.log10(eff_lr_active),
                bins=100,
                edgecolor="black",
                alpha=0.7,
                color="orange",
            )
            ax.set_xlabel("log10(Effective LR)")
            ax.set_ylabel("Count")
            ax.set_title("log10(Effective LR) distribution")
            ax.axvline(
                np.log10(lr),
                color="r",
                linestyle="--",
                label=f"log10(scheduled lr)={np.log10(lr):.2f}",
            )
            ax.legend()

            # 3) Ratio effective / scheduled
            ax = axes[2]
            ax.hist(
                eff_lr_active / lr,
                bins=100,
                edgecolor="black",
                alpha=0.7,
                color="green",
            )
            ax.set_xlabel("Effective LR / Scheduled LR")
            ax.set_ylabel("Count")
            ax.set_title("Ratio effective / scheduled LR")
            ax.axvline(1.0, color="r", linestyle="--", label="ratio = 1")
            ax.legend()

            fig.suptitle(
                f"Adam effective LR analysis{title_suffix}\n"
                f"step={step}, scheduled_lr={lr:.2e}, "
                f"active weights={mask.sum().item()}/{v.numel()}, "
                f"eff_lr: median={np.median(eff_lr_active):.2e}, "
                f"min={eff_lr_active.min():.2e}, max={eff_lr_active.max():.2e}",
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()

            v_hat_active = v_hat[mask].detach().cpu().numpy()
            print(
                f"v_hat stats for {mask.sum().item()} active weights:  "
                f"min={v_hat_active.min():.2e}  max={v_hat_active.max():.2e}  "
                f"median={np.median(v_hat_active):.2e}  mean={v_hat_active.mean():.2e}"
            )
