"""Shared losses and optimizers for monitor distillation."""

from __future__ import annotations

from typing import Any

import torch

MUON_LR_ADJUSTMENTS = ("match_rms_adamw", "original", "spectral_unclamped")


def configure_mean_loss_accumulation(trainer: Any) -> None:
    """Tell Trainer our custom loss returns means, not token-count-normalized sums.

    Model forward signatures can advertise loss kwargs even when a custom
    compute_loss ignores num_items_in_batch. Without this override, completion
    labels make Trainer skip its gradient-accumulation division.
    """
    trainer.model_accepts_loss_kwargs = False


def muon_update_scale(
    matrix: torch.Tensor,
    adjustment: str = "match_rms_adamw",
) -> float:
    """Return the per-matrix multiplier for a Muon polar update.

    ``match_rms_adamw`` follows the Moonshot rule: a full-rank polar factor for
    an ``[A, B]`` matrix has RMS ``1 / sqrt(max(A, B))``, so multiplying by
    ``0.2 * sqrt(max(A, B))`` targets the empirically observed AdamW update RMS.
    """
    if matrix.ndim != 2:
        raise ValueError("Muon learning-rate adjustment requires a matrix")
    rows, columns = matrix.shape
    if adjustment == "match_rms_adamw":
        return 0.2 * max(rows, columns) ** 0.5
    if adjustment == "original":
        return max(1.0, rows / columns) ** 0.5
    if adjustment == "spectral_unclamped":
        return (rows / columns) ** 0.5
    raise ValueError(
        f"unknown Muon learning-rate adjustment {adjustment!r}; "
        f"expected one of {MUON_LR_ADJUSTMENTS}"
    )


def zeropower_via_newtonschulz5(
    matrix: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    """Approximate ``UV^T`` for ``USV^T`` using quintic Newton-Schulz."""
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a matrix")
    a, b, c = (3.4445, -4.7750, 2.0315)
    value = matrix.bfloat16()
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        value = value.T
    value = value / (value.norm() + 1e-7)
    for _ in range(steps):
        covariance = value @ value.T
        value = a * value + (b * covariance + c * covariance @ covariance) @ value
    return value.T if transposed else value


class MuonAdamW(torch.optim.Optimizer):
    """Use Muon for 2D trainable matrices and AdamW for other parameters."""

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str = "match_rms_adamw",
    ) -> None:
        if muon_adjust_lr_fn not in MUON_LR_ADJUSTMENTS:
            raise ValueError(
                f"muon_adjust_lr_fn must be one of {MUON_LR_ADJUSTMENTS}"
            )
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "muon_momentum": muon_momentum,
            "muon_nesterov": muon_nesterov,
            "muon_ns_steps": muon_ns_steps,
            "muon_adjust_lr_fn": muon_adjust_lr_fn,
            "weight_decay": 0.0,
            "algorithm": "adamw",
        }
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        # PyTorch and Transformers schedulers mutate the standard ``lr`` key.
        # Reading it here keeps Muon warmup and decay synchronized with AdamW.
        lr = group["lr"]
        momentum = group["muon_momentum"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.ndim != 2:
                raise RuntimeError("Muon received a non-2D parameter")
            if group["weight_decay"]:
                parameter.mul_(1.0 - lr * group["weight_decay"])
            state = self.state[parameter]
            buffer = state.setdefault(
                "momentum_buffer",
                torch.zeros_like(parameter, dtype=torch.float32),
            )
            gradient = parameter.grad.float()
            buffer.mul_(momentum).add_(gradient)
            update = (
                gradient.add(buffer, alpha=momentum)
                if group["muon_nesterov"]
                else buffer
            )
            update = zeropower_via_newtonschulz5(
                update,
                steps=group["muon_ns_steps"],
            )
            scale = muon_update_scale(parameter, group["muon_adjust_lr_fn"])
            parameter.add_(update.to(parameter.dtype), alpha=-lr * scale)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if group["weight_decay"]:
                parameter.mul_(1.0 - lr * group["weight_decay"])
            gradient = parameter.grad.float()
            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    dtype=torch.float32,
                )
            state["step"] += 1
            average = state["exp_avg"]
            square_average = state["exp_avg_sq"]
            average.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            square_average.mul_(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            correction1 = 1.0 - beta1 ** state["step"]
            correction2 = 1.0 - beta2 ** state["step"]
            denominator = (
                square_average.sqrt().div_(correction2**0.5).add_(group["eps"])
            )
            parameter.addcdiv_(
                average.to(parameter.dtype),
                denominator.to(parameter.dtype),
                value=-(lr / correction1),
            )


def muon_adamw_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Partition trainable parameters by optimizer and decay behavior."""
    muon_parameters = []
    adamw_decay_parameters = []
    adamw_no_decay_parameters = []
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2:
            muon_parameters.append(parameter)
        elif any(token in name for token in no_decay):
            adamw_no_decay_parameters.append(parameter)
        else:
            adamw_decay_parameters.append(parameter)

    groups: list[dict[str, Any]] = []
    if muon_parameters:
        groups.append(
            {
                "params": muon_parameters,
                "weight_decay": weight_decay,
                "algorithm": "muon",
            }
        )
    if adamw_decay_parameters:
        groups.append(
            {
                "params": adamw_decay_parameters,
                "weight_decay": weight_decay,
                "algorithm": "adamw",
            }
        )
    if adamw_no_decay_parameters:
        groups.append(
            {
                "params": adamw_no_decay_parameters,
                "weight_decay": 0.0,
                "algorithm": "adamw",
            }
        )
    return groups
