#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import torch


def select_score_threshold_mask(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Select every finite score strictly above an absolute ADC threshold."""
    scores = scores.detach().flatten()
    return torch.logical_and(torch.isfinite(scores), scores > float(threshold))


def select_low_score_death_mask(
    scores: torch.Tensor,
    opacity: torch.Tensor,
    min_opacity: float,
    donor_pool_opacity: float,
) -> torch.Tensor:
    """Replace the vanilla opacity-prune count with lowest-score donors.

    The number of selected donors equals the number that vanilla would prune at
    ``min_opacity``. Candidate identities are chosen by ascending score from a
    slightly broader low-opacity pool, matching the paper's death controller.
    """
    scores = torch.nan_to_num(
        scores.detach().flatten(),
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("-inf"),
    )
    opacity = opacity.detach().flatten().to(device=scores.device)
    if scores.numel() != opacity.numel():
        raise ValueError("scores and opacity must have the same number of entries")

    standard_dead_count = int((opacity < float(min_opacity)).sum().item())
    selected = torch.zeros_like(opacity, dtype=torch.bool)
    if standard_dead_count == 0:
        return selected

    pool_threshold = max(float(donor_pool_opacity), float(min_opacity))
    candidate_indices = torch.nonzero(opacity <= pool_threshold, as_tuple=False).flatten()
    if candidate_indices.numel() < standard_dead_count:
        raise ValueError("donor pool cannot preserve the vanilla opacity-prune count")

    lowest_local = torch.topk(
        scores[candidate_indices],
        k=standard_dead_count,
        largest=False,
        sorted=False,
    ).indices
    selected[candidate_indices[lowest_local]] = True
    return selected


def select_topk_mask(scores: torch.Tensor, eligible_mask: torch.Tensor, quota: int) -> torch.Tensor:
    """Select exactly ``quota`` highest-scoring eligible entries."""
    scores = scores.flatten()
    eligible_mask = eligible_mask.flatten().to(device=scores.device, dtype=torch.bool)
    if scores.numel() != eligible_mask.numel():
        raise ValueError("scores and eligible_mask must have the same number of entries")

    selected = torch.zeros_like(eligible_mask)
    if quota <= 0:
        return selected

    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
    if quota >= eligible_indices.numel():
        return eligible_mask.clone()

    eligible_scores = scores[eligible_indices]
    top_local = torch.topk(eligible_scores, k=int(quota), largest=True, sorted=False).indices
    selected[eligible_indices[top_local]] = True
    return selected


def restrict_to_topk_shortlist(
    scores: torch.Tensor,
    eligible_mask: torch.Tensor,
    quota: int,
    shortlist_factor: int,
) -> torch.Tensor:
    """Keep at most ``shortlist_factor * quota`` top eligible proposals.

    A nonpositive factor disables shortlisting. The shortlist is never smaller
    than the requested quota, so a downstream selector can still fill it.
    """
    eligible_mask = eligible_mask.flatten().to(device=scores.device, dtype=torch.bool)
    factor = int(shortlist_factor)
    if factor <= 0:
        return eligible_mask.clone()

    eligible_count = int(eligible_mask.sum().item())
    quota = min(max(0, int(quota)), eligible_count)
    shortlist_size = min(eligible_count, max(quota, factor * quota))
    return select_topk_mask(scores, eligible_mask, shortlist_size)


def rank_normalize_masked(scores: torch.Tensor, eligible_mask: torch.Tensor) -> torch.Tensor:
    """Return ordinal ranks in [0, 1] for eligible entries, zero elsewhere."""
    scores = scores.flatten()
    eligible_mask = eligible_mask.flatten().to(device=scores.device, dtype=torch.bool)
    if scores.numel() != eligible_mask.numel():
        raise ValueError("scores and eligible_mask must have the same number of entries")

    normalized = torch.zeros_like(scores, dtype=torch.float32)
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
    count = int(eligible_indices.numel())
    if count == 0:
        return normalized
    if count == 1:
        normalized[eligible_indices] = 1.0
        return normalized

    order = torch.argsort(scores[eligible_indices], descending=False)
    ordinal = torch.linspace(0.0, 1.0, count, device=scores.device)
    normalized[eligible_indices[order]] = ordinal
    return normalized
