"""Small, framework-local adversarial attack references for educational testing."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pgd_linf(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    targeted: bool = False,
) -> torch.Tensor:
    """Generate a bounded L-infinity PGD example against a PyTorch classifier.

    The function assumes `inputs` are already in the model's expected input
    space. `epsilon` and `step_size` therefore use those same units.
    """
    if epsilon < 0 or step_size <= 0 or steps < 1:
        raise ValueError("epsilon>=0, step_size>0, and steps>=1 are required")
    if clamp_min >= clamp_max:
        raise ValueError("clamp_min must be smaller than clamp_max")

    model.eval()
    original = inputs.detach()
    adv = original.clone()

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, targets)
        grad = torch.autograd.grad(loss, adv)[0]
        direction = -grad.sign() if targeted else grad.sign()
        adv = adv.detach() + step_size * direction
        delta = torch.clamp(adv - original, min=-epsilon, max=epsilon)
        adv = torch.clamp(original + delta, min=clamp_min, max=clamp_max)

    return adv.detach()
