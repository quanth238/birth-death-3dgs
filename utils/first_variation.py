#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import json
import random
import time

import torch

from utils.general_utils import build_rotation
from utils.loss_utils import ssim


def load_birth_quota_schedule(path):
    """Load a unique, nonnegative iteration-to-birth-quota control schedule."""
    schedule = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            iteration = int(row["iteration"])
            quota = int(row["birth_quota"])
            if iteration in schedule:
                raise ValueError(f"duplicate iteration {iteration} in birth quota log")
            if quota < 0:
                raise ValueError(f"negative birth quota on line {line_number}")
            schedule[iteration] = quota
    if not schedule:
        raise ValueError("birth quota log contains no events")
    return schedule


def select_score_views(cameras, num_views: int, seed: int):
    """Select score views without changing the training RNG state."""
    cameras = list(cameras)
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if not cameras:
        raise ValueError("at least one training camera is required")
    rng = random.Random(int(seed))
    indices = list(range(len(cameras)))
    rng.shuffle(indices)
    return [cameras[index] for index in indices[: min(num_views, len(indices))]]


def select_positive_weighted_mask(scores, eligible_mask, quota: int, seed: int):
    """Sample unique parents proportional to positive score using an isolated RNG."""
    scores = scores.detach().flatten()
    eligible_mask = eligible_mask.flatten().to(device=scores.device, dtype=torch.bool)
    if scores.numel() != eligible_mask.numel():
        raise ValueError("scores and eligible_mask must have the same number of entries")

    selected = torch.zeros_like(eligible_mask)
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
    quota = min(max(0, int(quota)), int(eligible_indices.numel()))
    if quota == 0:
        return selected, {
            "sampling_seed": int(seed),
            "positive_weight_count": 0,
            "positive_weight_fallback": False,
        }

    eligible_weights = scores[eligible_indices].clamp_min(0.0)
    positive_local = torch.nonzero(eligible_weights > 0, as_tuple=False).flatten()
    generator = torch.Generator(device=scores.device)
    generator.manual_seed(int(seed))

    positive_weight_fallback = int(positive_local.numel()) < quota
    if not positive_weight_fallback:
        sampled_local = torch.multinomial(
            eligible_weights,
            quota,
            replacement=False,
            generator=generator,
        )
        selected[eligible_indices[sampled_local]] = True
    else:
        selected[eligible_indices[positive_local]] = True
        remaining = quota - int(positive_local.numel())
        zero_local = torch.nonzero(eligible_weights <= 0, as_tuple=False).flatten()
        fallback_order = torch.randperm(
            zero_local.numel(),
            generator=generator,
            device=scores.device,
        )[:remaining]
        selected[eligible_indices[zero_local[fallback_order]]] = True

    return selected, {
        "sampling_seed": int(seed),
        "positive_weight_count": int(positive_local.numel()),
        "positive_weight_fallback": positive_weight_fallback,
    }


def select_threshold_with_growth_limit(
    scores,
    eligible_mask,
    threshold: float,
    max_growth_fraction: float,
    seed: int,
    max_births=None,
):
    """Select q-threshold parents under event and cumulative birth limits.

    A zero growth fraction disables the limit and selects every eligible score
    strictly above ``threshold``. Otherwise at most
    ``floor(max_growth_fraction * population)`` unique candidates are sampled
    without replacement, proportional to their positive excess over the
    threshold. ``max_births`` optionally supplies an independent remaining
    population budget for the current event. For the intended q_i > 0
    experiment, this is exactly score-proportional sampling by q_i.
    """
    scores = scores.detach().flatten()
    eligible_mask = eligible_mask.flatten().to(device=scores.device, dtype=torch.bool)
    if scores.numel() != eligible_mask.numel():
        raise ValueError("scores and eligible_mask must have the same number of entries")

    max_growth_fraction = float(max_growth_fraction)
    if not 0.0 <= max_growth_fraction <= 1.0:
        raise ValueError("max_growth_fraction must be in [0, 1]")

    candidate_mask = torch.logical_and(
        eligible_mask,
        torch.logical_and(torch.isfinite(scores), scores > float(threshold)),
    )
    candidate_count = int(candidate_mask.sum().item())
    event_birth_limit = candidate_count
    growth_limit_enabled = max_growth_fraction > 0.0
    if growth_limit_enabled:
        event_birth_limit = int(max_growth_fraction * scores.numel())

    event_limited_count = min(candidate_count, event_birth_limit)
    population_budget_enabled = max_births is not None
    population_budget_birth_limit = candidate_count
    if population_budget_enabled:
        population_budget_birth_limit = max(0, int(max_births))
    birth_limit = min(event_limited_count, population_budget_birth_limit)

    growth_limit_applied = candidate_count > event_birth_limit
    population_budget_limit_applied = event_limited_count > population_budget_birth_limit
    selection_limit_applied = candidate_count > birth_limit
    if selection_limit_applied:
        selection_mask, sampling_stats = select_positive_weighted_mask(
            scores - float(threshold),
            candidate_mask,
            birth_limit,
            seed,
        )
        if population_budget_limit_applied and growth_limit_applied:
            birth_selection = "q_proportional_without_replacement_under_growth_and_population_limits"
        elif population_budget_limit_applied:
            birth_selection = "q_proportional_without_replacement_under_population_limit"
        else:
            birth_selection = "q_proportional_without_replacement_under_growth_limit"
    else:
        selection_mask = candidate_mask
        sampling_stats = {
            "sampling_seed": int(seed),
            "positive_weight_count": candidate_count,
            "positive_weight_fallback": False,
        }
        birth_selection = "all_q_above_absolute_threshold"

    sampling_stats.update({
        "birth_selection": birth_selection,
        "first_variation_birth_threshold": float(threshold),
        "threshold_candidate_count": candidate_count,
        "max_growth_fraction": max_growth_fraction,
        "max_growth_birth_limit": event_birth_limit,
        "growth_limit_enabled": growth_limit_enabled,
        "growth_limit_applied": growth_limit_applied,
        "population_budget_enabled": population_budget_enabled,
        "population_budget_birth_limit": population_budget_birth_limit,
        "population_budget_limit_applied": population_budget_limit_applied,
        "effective_birth_limit": birth_limit,
    })
    return selection_mask, sampling_stats


def taming_population_target(
    initial_count: int,
    final_budget: int,
    iteration: int,
    densify_from_iter: int,
    budget_end_iter: int,
):
    """Return the Taming-3DGS parabolic cumulative population target.

    The target follows ``N0 + (B-N0) * (2p-p^2)`` for progress ``p`` between
    the start and end of budgeted densification. It is monotonic, front-loaded,
    and reaches the exact requested budget at ``budget_end_iter``.
    """
    initial_count = int(initial_count)
    final_budget = int(final_budget)
    iteration = int(iteration)
    densify_from_iter = int(densify_from_iter)
    budget_end_iter = int(budget_end_iter)
    if initial_count < 0:
        raise ValueError("initial_count must be nonnegative")
    if final_budget < initial_count:
        raise ValueError("final_budget must be at least initial_count")
    if budget_end_iter <= densify_from_iter:
        raise ValueError("budget_end_iter must be after densify_from_iter")

    progress = (iteration - densify_from_iter) / float(
        budget_end_iter - densify_from_iter
    )
    progress = min(1.0, max(0.0, progress))
    growth = final_budget - initial_count
    target = initial_count + int(growth * (2.0 * progress - progress * progress))
    return min(final_budget, max(initial_count, target))


def select_positive_weighted_with_ranked_fallback(
    scores,
    eligible_mask,
    fallback_scores,
    quota: int,
    seed: int,
):
    """Sample positive utilities, then fill an unmatched quota by native ranking.

    The positive-utility subset is sampled without replacement in proportion to
    its score. If it is smaller than the requested quota, every positive item is
    retained and the remaining slots are filled by descending fallback score.
    This keeps the native controller as the conservative fallback instead of
    injecting random zero/negative-utility proposals.
    """
    scores = scores.detach().flatten()
    fallback_scores = fallback_scores.detach().flatten().to(device=scores.device)
    eligible_mask = eligible_mask.flatten().to(device=scores.device, dtype=torch.bool)
    if scores.numel() != eligible_mask.numel() or scores.numel() != fallback_scores.numel():
        raise ValueError(
            "scores, fallback_scores, and eligible_mask must have the same number of entries"
        )

    selected = torch.zeros_like(eligible_mask)
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
    quota = min(max(0, int(quota)), int(eligible_indices.numel()))
    eligible_weights = scores[eligible_indices].clamp_min(0.0)
    positive_local = torch.nonzero(eligible_weights > 0, as_tuple=False).flatten()
    positive_count = int(positive_local.numel())
    fallback_count = max(0, quota - positive_count)

    if quota == 0:
        return selected, {
            "sampling_seed": int(seed),
            "positive_weight_count": positive_count,
            "positive_weight_fallback": False,
            "ranked_fallback_count": 0,
            "ranked_fallback": "vanilla_position_gradient_top_r",
        }

    if positive_count >= quota:
        generator = torch.Generator(device=scores.device)
        generator.manual_seed(int(seed))
        sampled_local = torch.multinomial(
            eligible_weights,
            quota,
            replacement=False,
            generator=generator,
        )
        selected[eligible_indices[sampled_local]] = True
        fallback_count = 0
    else:
        selected[eligible_indices[positive_local]] = True
        remaining_eligible = torch.logical_and(eligible_mask, ~selected)
        remaining_indices = torch.nonzero(remaining_eligible, as_tuple=False).flatten()
        if fallback_count > 0:
            fallback_order = torch.topk(
                fallback_scores[remaining_indices],
                k=fallback_count,
                largest=True,
                sorted=False,
            ).indices
            selected[remaining_indices[fallback_order]] = True

    return selected, {
        "sampling_seed": int(seed),
        "positive_weight_count": positive_count,
        "positive_weight_fallback": fallback_count > 0,
        "ranked_fallback_count": int(fallback_count),
        "ranked_fallback": "vanilla_position_gradient_top_r",
    }


def first_variation_image_objective(pred, gt, lambda_dssim: float, fused_ssim_fn=None):
    l1_term = (pred - gt).abs().mean()
    if fused_ssim_fn is None:
        ssim_value = ssim(pred, gt)
    else:
        ssim_value = fused_ssim_fn(pred.unsqueeze(0), gt.unsqueeze(0))
    return (1.0 - lambda_dssim) * l1_term + lambda_dssim * (1.0 - ssim_value)


def compute_first_variation_score(
    cameras,
    gaussians,
    pipe,
    background,
    lambda_dssim: float,
    *,
    fused_ssim_fn=None,
    separate_sh=False,
    use_trained_exp=False,
):
    """Compute q=-dF/dm at m=1 while all scene parameters remain detached."""
    from gaussian_renderer import render_mass_gated

    if not cameras:
        raise ValueError("at least one score view is required")
    device = gaussians.get_xyz.device
    mass_gate = torch.ones_like(gaussians.get_opacity.detach(), device=device, requires_grad=True)
    num_views = max(1, len(cameras))
    objective_sum = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    started = time.perf_counter()
    with torch.enable_grad():
        for view in cameras:
            package = render_mass_gated(
                view,
                gaussians,
                pipe,
                background,
                mass_gate,
                separate_sh=separate_sh,
                use_trained_exp=use_trained_exp,
            )
            prediction = package["render"]
            if view.alpha_mask is not None:
                prediction = prediction * view.alpha_mask.to(device)
            ground_truth = view.original_image.to(device)
            objective = first_variation_image_objective(
                prediction,
                ground_truth,
                lambda_dssim,
                fused_ssim_fn=fused_ssim_fn,
            )
            objective_sum += float(objective.detach().item())
            (objective / num_views).backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    if mass_gate.grad is None:
        raise RuntimeError("mass-gated score pass produced no mass gradient")
    raw_score = -mass_gate.grad.detach().flatten()
    elapsed = time.perf_counter() - started
    return raw_score, {
        "score_compute_sec": elapsed,
        "score_objective": objective_sum / num_views,
        "score_views": len(cameras),
        "score_view_names": [view.image_name for view in cameras],
        "raw_score_mean": float(raw_score.mean().item()),
        "raw_score_std": float(raw_score.std(unbiased=False).item()),
        "raw_score_min": float(raw_score.min().item()),
        "raw_score_max": float(raw_score.max().item()),
        "raw_positive_fraction": float((raw_score > 0).float().mean().item()),
    }


def build_standard_vanilla_probes(
    gaussians,
    candidate_mask,
    scene_extent: float,
    seed: int,
    split_children: int = 2,
):
    """Generate the exact native clone/split children without mutating the model."""
    if split_children <= 0:
        raise ValueError("split_children must be positive")
    device = gaussians.get_xyz.device
    candidate_mask = candidate_mask.flatten().to(device=device, dtype=torch.bool)
    point_count = gaussians.get_xyz.shape[0]
    if candidate_mask.numel() != point_count:
        raise ValueError("candidate_mask must match the current Gaussian population")

    scale_max = gaussians.get_scaling.detach().max(dim=1).values
    split_mask = torch.logical_and(
        candidate_mask,
        scale_max > gaussians.percent_dense * float(scene_extent),
    )
    clone_mask = torch.logical_and(candidate_mask, ~split_mask)
    clone_indices = torch.nonzero(clone_mask, as_tuple=False).flatten()
    split_indices = torch.nonzero(split_mask, as_tuple=False).flatten()

    split_parent_indices = split_indices.repeat(split_children)
    split_local_samples = torch.empty(
        (0, 3), device=device, dtype=gaussians.get_xyz.dtype
    )
    split_xyz = torch.empty_like(split_local_samples)
    if split_indices.numel() > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        split_stds = gaussians.get_scaling.detach()[split_indices].repeat(
            split_children, 1
        )
        split_local_samples = torch.randn(
            split_stds.shape,
            generator=generator,
            device=device,
            dtype=split_stds.dtype,
        ) * split_stds
        split_rotations = build_rotation(
            gaussians._rotation.detach()[split_indices]
        ).repeat(split_children, 1, 1)
        split_xyz = torch.bmm(
            split_rotations, split_local_samples.unsqueeze(-1)
        ).squeeze(-1) + gaussians.get_xyz.detach()[split_indices].repeat(
            split_children, 1
        )

    split_sample_table = torch.full(
        (point_count, split_children, 3),
        float("nan"),
        device=device,
        dtype=gaussians.get_xyz.dtype,
    )
    if split_indices.numel() > 0:
        split_sample_table[split_indices] = split_local_samples.view(
            split_children, split_indices.numel(), 3
        ).permute(1, 0, 2)

    parent_indices = torch.cat((clone_indices, split_parent_indices), dim=0)
    xyz = torch.cat((gaussians.get_xyz.detach()[clone_indices], split_xyz), dim=0)
    opacity = gaussians.get_opacity.detach()[parent_indices]
    rotation = gaussians.get_rotation.detach()[parent_indices]
    features_dc = gaussians.get_features_dc.detach()[parent_indices]
    features_rest = gaussians.get_features_rest.detach()[parent_indices]
    clone_scaling = gaussians.get_scaling.detach()[clone_indices]
    split_scaling = gaussians.get_scaling.detach()[split_parent_indices] / (
        0.8 * split_children
    )
    scaling = torch.cat((clone_scaling, split_scaling), dim=0)

    return {
        "xyz": xyz,
        "opacity": opacity,
        "scaling": scaling,
        "rotation": rotation,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "parent_indices": parent_indices,
        "candidate_mask": candidate_mask,
        "clone_mask": clone_mask,
        "split_mask": split_mask,
        "clone_count": int(clone_indices.numel()),
        "split_count": int(split_indices.numel()),
        "split_children": int(split_children),
        "split_indices": split_indices,
        "split_local_samples": split_local_samples,
        "split_sample_table": split_sample_table,
        "proposal_seed": int(seed),
    }


def aggregate_vanilla_proposal_utilities(
    base_scores,
    child_scores,
    child_parent_indices,
    candidate_mask,
    split_mask,
):
    """Map support scores to clone/split finite-move first-order utilities."""
    base_scores = base_scores.flatten()
    child_scores = child_scores.flatten().to(base_scores)
    child_parent_indices = child_parent_indices.flatten().to(
        device=base_scores.device, dtype=torch.long
    )
    candidate_mask = candidate_mask.flatten().to(
        device=base_scores.device, dtype=torch.bool
    )
    split_mask = split_mask.flatten().to(device=base_scores.device, dtype=torch.bool)
    if (
        candidate_mask.numel() != base_scores.numel()
        or split_mask.numel() != base_scores.numel()
    ):
        raise ValueError("candidate and split masks must match base_scores")
    if child_scores.numel() != child_parent_indices.numel():
        raise ValueError("child_scores and child_parent_indices must have equal length")

    child_sum = torch.zeros_like(base_scores)
    child_sum.scatter_add_(0, child_parent_indices, child_scores)
    utilities = torch.zeros_like(base_scores)
    utilities[candidate_mask] = child_sum[candidate_mask]
    utilities[split_mask] = utilities[split_mask] - base_scores[split_mask]
    return utilities


def compute_vanilla_proposal_first_variation(
    cameras,
    gaussians,
    pipe,
    background,
    probes,
    lambda_dssim: float,
    *,
    fused_ssim_fn=None,
    separate_sh=False,
    use_trained_exp=False,
):
    """Score current mass and exact native vanilla proposals in one backward pass."""
    from gaussian_renderer import render_mass_gated_with_probes

    device = gaussians.get_xyz.device
    base_gate = torch.ones_like(
        gaussians.get_opacity.detach(), device=device, requires_grad=True
    )
    probe_gate = torch.zeros_like(
        probes["opacity"].detach(), device=device, requires_grad=True
    )
    num_views = max(1, len(cameras))
    objective_sum = 0.0

    torch.cuda.synchronize(device=device)
    started = time.perf_counter()
    with torch.enable_grad():
        for view in cameras:
            package = render_mass_gated_with_probes(
                view,
                gaussians,
                pipe,
                background,
                base_gate,
                probe_gate,
                probes,
                separate_sh=separate_sh,
                use_trained_exp=use_trained_exp,
            )
            prediction = package["render"]
            if view.alpha_mask is not None:
                prediction = prediction * view.alpha_mask.cuda()
            ground_truth = view.original_image.cuda()
            objective = first_variation_image_objective(
                prediction,
                ground_truth,
                lambda_dssim,
                fused_ssim_fn=fused_ssim_fn,
            )
            objective_sum += float(objective.detach().item())
            (objective / num_views).backward()

    torch.cuda.synchronize(device=device)
    if base_gate.grad is None or probe_gate.grad is None:
        raise RuntimeError("proposal score pass produced no mass gradient")
    base_score = -base_gate.grad.detach().flatten()
    child_score = -probe_gate.grad.detach().flatten()
    proposal_utility = aggregate_vanilla_proposal_utilities(
        base_score,
        child_score,
        probes["parent_indices"],
        probes["candidate_mask"],
        probes["split_mask"],
    )
    elapsed = time.perf_counter() - started
    candidate_utility = proposal_utility[probes["candidate_mask"]]
    return base_score, proposal_utility, {
        "score_compute_sec": elapsed,
        "score_objective": objective_sum / num_views,
        "score_views": len(cameras),
        "score_view_names": [view.image_name for view in cameras],
        "raw_score_mean": float(base_score.mean().item()),
        "raw_score_std": float(base_score.std(unbiased=False).item()),
        "raw_score_min": float(base_score.min().item()),
        "raw_score_max": float(base_score.max().item()),
        "raw_positive_fraction": float((base_score > 0).float().mean().item()),
        "proposal_count": int(probes["candidate_mask"].sum().item()),
        "proposal_probe_count": int(child_score.numel()),
        "proposal_clone_count": int(probes["clone_count"]),
        "proposal_split_count": int(probes["split_count"]),
        "proposal_utility_mean": (
            float(candidate_utility.mean().item()) if candidate_utility.numel() else 0.0
        ),
        "proposal_utility_std": (
            float(candidate_utility.std(unbiased=False).item())
            if candidate_utility.numel()
            else 0.0
        ),
        "proposal_positive_fraction": (
            float((candidate_utility > 0).float().mean().item())
            if candidate_utility.numel()
            else 0.0
        ),
        "proposal_seed": int(probes["proposal_seed"]),
    }


def stabilize_first_variation_score(raw_score, previous_ema, ema_decay: float, clip_quantile: float):
    raw_score = raw_score.detach().flatten()
    state_reset = previous_ema is None or previous_ema.numel() != raw_score.numel()
    if state_reset:
        score_ema = raw_score.clone()
    else:
        score_ema = float(ema_decay) * previous_ema.to(raw_score) + (1.0 - float(ema_decay)) * raw_score

    quantile = min(max(float(clip_quantile), 0.0), 1.0)
    clip_value = 0.0
    stabilized = score_ema
    if score_ema.numel() > 0 and quantile > 0.0:
        clip_tensor = torch.quantile(score_ema.abs(), quantile)
        clip_value = float(clip_tensor.item())
        if clip_value > 0.0:
            stabilized = score_ema.clamp(min=-clip_value, max=clip_value)

    return score_ema, stabilized, {
        "score_ema_reset": state_reset,
        "score_ema_decay": float(ema_decay),
        "score_clip_quantile": quantile,
        "score_clip_value": clip_value,
        "stabilized_mean": float(stabilized.mean().item()),
        "stabilized_std": float(stabilized.std(unbiased=False).item()),
        "stabilized_positive_fraction": float((stabilized > 0).float().mean().item()),
    }
