"""Losses: focal loss for future-event occupancy, SiLog for metric depth."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def voxel_focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Class-balanced focal loss on binarized event occupancy.

    Event voxels are overwhelmingly empty (~95-99%); alpha is derived from the
    batch's positive fraction so polarity/scene density needs no hand tuning.
    """
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = ce * (1 - p_t) ** gamma

    positive_fraction = target.mean().clamp(1e-6, 1 - 1e-6)
    alpha = 1 - positive_fraction
    loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    return loss.mean()


def silog_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, lam: float = 0.5
) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al.). pred/target: (B, H, W), meters."""
    d = torch.log(pred[valid]) - torch.log(target[valid])
    return torch.sqrt((d**2).mean() - lam * d.mean() ** 2 + 1e-8)


@torch.no_grad()
def depth_metric_sums(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Per-pixel sums for abs_rel / RMSE / delta<1.25 plus the valid count,
    so callers can aggregate exactly across batches."""
    p = pred[valid]
    t = target[valid]
    thresh = torch.maximum(p / t, t / p)
    return {
        "abs_rel": float((torch.abs(p - t) / t).sum()),
        "sq_err": float(((p - t) ** 2).sum()),
        "d1": float((thresh < 1.25).float().sum()),
        "n": float(valid.sum()),
    }
