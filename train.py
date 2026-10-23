#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.adc_utils import (
    rank_normalize_masked,
    restrict_to_topk_shortlist,
    select_low_score_death_mask,
    select_score_threshold_mask,
    select_topk_mask,
)
from utils.first_variation import (
    build_standard_vanilla_probes,
    compute_first_variation_score,
    compute_vanilla_proposal_first_variation,
    load_birth_quota_schedule,
    select_positive_weighted_mask,
    select_positive_weighted_with_ranked_fallback,
    select_threshold_with_growth_limit,
    taming_population_target,
    select_score_views,
    stabilize_first_variation_score,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    matched_birth_schedule = None
    if opt.first_variation_adc and opt.first_variation_birth_quota_log:
        matched_birth_schedule = load_birth_quota_schedule(opt.first_variation_birth_quota_log)
        print(
            f"Loaded {len(matched_birth_schedule)} matched birth quotas from "
            f"{opt.first_variation_birth_quota_log}"
        )

    pure_q_threshold_adc = (
        opt.first_variation_adc
        and opt.first_variation_birth_selection == "q_threshold"
    )
    if pure_q_threshold_adc:
        if matched_birth_schedule is not None:
            raise ValueError("q-threshold ADC does not support a frozen birth quota")
        if opt.max_gaussians is not None and opt.max_gaussians > 0:
            raise ValueError("q-threshold ADC is quota-free and requires max_gaussians <= 0")
        if float(opt.first_variation_score_ema) != 0.0:
            raise ValueError("q-threshold ADC requires score EMA 0 to threshold raw q_i")
        if float(opt.first_variation_score_clip_quantile) != 0.0:
            raise ValueError("q-threshold ADC requires score clipping 0 to threshold raw q_i")
        if not 0.0 <= float(opt.first_variation_max_growth_fraction) <= 1.0:
            raise ValueError("first_variation_max_growth_fraction must be in [0, 1]")

    initial_population = int(gaussians.get_xyz.shape[0])
    population_budget = int(opt.first_variation_population_budget)
    population_budget_enabled = population_budget > 0
    population_budget_end_iter = None
    if population_budget_enabled:
        if not pure_q_threshold_adc:
            raise ValueError("first_variation_population_budget requires q-threshold ADC")
        if population_budget < initial_population:
            raise ValueError(
                "first_variation_population_budget must be at least the initial population"
            )
        last_budgeted_iteration = min(
            int(opt.iterations) - 1,
            int(opt.densify_until_iter) - 1,
        )
        population_budget_end_iter = (
            last_budgeted_iteration // int(opt.densification_interval)
        ) * int(opt.densification_interval)
        if population_budget_end_iter <= int(opt.densify_from_iter):
            raise ValueError("no effective ADC event is available for the population budget")
        print(
            "Using Taming-style cumulative population budget "
            f"{initial_population} -> {population_budget} by iteration "
            f"{population_budget_end_iter}"
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if matched_birth_schedule is None and not pure_q_threshold_adc:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    selection_mask = None
                    scored_prune_mask = None
                    split_sample_table = None
                    first_variation_stats = None
                    reference_birth_quota = None
                    if opt.first_variation_adc:
                        if pure_q_threshold_adc:
                            reference_birth_quota = 0
                            matched_birth_quota = 0
                            remaining_capacity = 0
                            quota_source = "first_variation_absolute_threshold"
                        elif matched_birth_schedule is None:
                            pre_adc_grads = gaussians.xyz_gradient_accum / gaussians.denom
                            pre_adc_grads[pre_adc_grads.isnan()] = 0.0
                            standard_grad_score = torch.norm(pre_adc_grads, dim=-1)
                            standard_eligible = standard_grad_score >= opt.densify_grad_threshold
                            reference_birth_quota = int(standard_eligible.sum().item())
                            quota_source = "online_position_gradient_count"
                        else:
                            if iteration not in matched_birth_schedule:
                                raise ValueError(f"matched birth quota is missing iteration {iteration}")
                            reference_birth_quota = int(matched_birth_schedule[iteration])
                            quota_source = "matched_vanilla_reference_log"

                        if not pure_q_threshold_adc:
                            remaining_capacity = reference_birth_quota
                            if opt.max_gaussians is not None and opt.max_gaussians > 0:
                                remaining_capacity = max(0, int(opt.max_gaussians) - int(gaussians.get_xyz.shape[0]))
                            matched_birth_quota = min(reference_birth_quota, remaining_capacity)

                        proposal_mode = opt.first_variation_birth_selection in {
                            "proposal_greedy_top_r",
                            "proposal_positive_proportional",
                        }
                        proposal_utility = None
                        proposal_probes = None
                        proposal_scoring_active = (
                            proposal_mode
                            and matched_birth_quota > 0
                            and matched_birth_quota < reference_birth_quota
                        )
                        if proposal_mode:
                            if opt.first_variation_candidate_gate != "position_gradient":
                                raise ValueError(
                                    "proposal-conditioned scoring requires the position-gradient candidate gate"
                                )
                            if matched_birth_schedule is not None:
                                raise ValueError(
                                    "proposal-conditioned scoring does not support a frozen quota log"
                                )
                            if opt.first_variation_death_selection != "standard_opacity":
                                raise ValueError(
                                    "proposal-conditioned scoring currently requires standard opacity pruning"
                                )

                        score_seed = int(opt.first_variation_seed) + iteration * 7919
                        score_cameras = select_score_views(
                            scene.getTrainCameras(),
                            opt.first_variation_score_views,
                            score_seed,
                        )
                        if proposal_scoring_active:
                            proposal_candidate_mask = standard_eligible
                            shortlist_factor = int(opt.first_variation_proposal_shortlist_factor)
                            if shortlist_factor > 0:
                                proposal_candidate_mask = restrict_to_topk_shortlist(
                                    standard_grad_score,
                                    proposal_candidate_mask,
                                    matched_birth_quota,
                                    shortlist_factor,
                                )
                            proposal_probes = build_standard_vanilla_probes(
                                gaussians,
                                proposal_candidate_mask,
                                scene.cameras_extent,
                                score_seed + 130363,
                            )
                            raw_score, proposal_utility, first_variation_stats = (
                                compute_vanilla_proposal_first_variation(
                                    score_cameras,
                                    gaussians,
                                    pipe,
                                    background,
                                    proposal_probes,
                                    opt.lambda_dssim,
                                    fused_ssim_fn=(
                                        fused_ssim if FUSED_SSIM_AVAILABLE else None
                                    ),
                                    separate_sh=SPARSE_ADAM_AVAILABLE,
                                    use_trained_exp=dataset.train_test_exp,
                                )
                            )
                            split_sample_table = proposal_probes["split_sample_table"]
                        else:
                            raw_score, first_variation_stats = compute_first_variation_score(
                                score_cameras,
                                gaussians,
                                pipe,
                                background,
                                opt.lambda_dssim,
                                fused_ssim_fn=fused_ssim if FUSED_SSIM_AVAILABLE else None,
                                separate_sh=SPARSE_ADAM_AVAILABLE,
                                use_trained_exp=dataset.train_test_exp,
                            )
                        score_ema, stabilized_score, stabilization_stats = stabilize_first_variation_score(
                            raw_score,
                            gaussians._adc_score_ema,
                            opt.first_variation_score_ema,
                            opt.first_variation_score_clip_quantile,
                        )
                        gaussians._adc_score_ema = score_ema
                        first_variation_stats.update(stabilization_stats)

                        if opt.first_variation_death_selection == "low_q_low_opacity_pool":
                            if opt.first_variation_candidate_gate != "position_gradient":
                                raise ValueError(
                                    "low-q death selection requires vanilla position-gradient birth proposals"
                                )
                            if matched_birth_schedule is not None:
                                raise ValueError("low-q death selection does not support a frozen birth quota")
                            scored_prune_mask = select_low_score_death_mask(
                                stabilized_score,
                                gaussians.get_opacity,
                                0.005,
                                opt.first_variation_donor_pool_opacity,
                            )
                            standard_eligible = torch.logical_and(standard_eligible, ~scored_prune_mask)
                            reference_birth_quota = int(standard_eligible.sum().item())
                            remaining_capacity = reference_birth_quota
                            if opt.max_gaussians is not None and opt.max_gaussians > 0:
                                remaining_capacity = max(
                                    0,
                                    int(opt.max_gaussians) - int(gaussians.get_xyz.shape[0]),
                                )
                            matched_birth_quota = min(reference_birth_quota, remaining_capacity)
                            quota_source = "online_position_gradient_count_excluding_scored_donors"
                            death_stats = {
                                "death_selection": "lowest_q_in_low_opacity_pool",
                                "death_count": int(scored_prune_mask.sum().item()),
                                "donor_pool_opacity": float(opt.first_variation_donor_pool_opacity),
                                "first_variation_death_role": "donor_selection",
                            }
                        elif opt.first_variation_death_selection == "standard_opacity":
                            death_stats = {
                                "death_selection": "standard_opacity",
                                "death_count": 0,
                                "first_variation_death_role": "none",
                            }
                        else:
                            raise ValueError(
                                "first_variation_death_selection must be standard_opacity or "
                                "low_q_low_opacity_pool"
                            )
                        first_variation_stats.update(death_stats)

                        finite_score = torch.isfinite(stabilized_score)
                        if pure_q_threshold_adc:
                            live_parent_pool = finite_score
                            candidate_gate_stats = {
                                "candidate_gate": "first_variation_absolute_threshold",
                                "parent_opacity_floor_applied": False,
                                "position_gradient_role": "none",
                                "first_variation_role": (
                                    "adc_trigger_with_population_relative_growth_limit"
                                    if opt.first_variation_max_growth_fraction > 0.0
                                    else "quota_free_adc_trigger"
                                ),
                            }
                        elif opt.first_variation_candidate_gate == "all_live":
                            live_parent_pool = torch.logical_and(
                                finite_score,
                                gaussians.get_opacity.detach().flatten()
                                > opt.first_variation_parent_opacity_floor,
                            )
                            candidate_gate_stats = {
                                "candidate_gate": "all_live_above_opacity_floor",
                                "parent_opacity_floor_applied": True,
                                "position_gradient_role": "quota_only" if matched_birth_schedule is None else "none",
                                "first_variation_role": "parent_selection",
                            }
                        elif opt.first_variation_candidate_gate == "position_gradient":
                            if matched_birth_schedule is not None:
                                raise ValueError(
                                    "position-gradient candidate gating cannot be combined with a frozen quota log"
                                )
                            live_parent_pool = torch.logical_and(finite_score, standard_eligible)
                            shortlist_factor = int(opt.first_variation_proposal_shortlist_factor)
                            shortlist_applied = shortlist_factor > 0 and matched_birth_quota < reference_birth_quota
                            if shortlist_applied:
                                live_parent_pool = restrict_to_topk_shortlist(
                                    standard_grad_score,
                                    live_parent_pool,
                                    matched_birth_quota,
                                    shortlist_factor,
                                )
                            candidate_gate_stats = {
                                "candidate_gate": (
                                    "position_gradient_topk_shortlist"
                                    if shortlist_applied
                                    else "position_gradient_eligible"
                                ),
                                "parent_opacity_floor_applied": False,
                                "position_gradient_role": (
                                    "proposal_gate_quota_and_shortlist"
                                    if shortlist_applied
                                    else "proposal_gate_and_quota"
                                ),
                                "first_variation_role": "shortlist_ranking_when_cap_limited",
                                "proposal_shortlist_factor": shortlist_factor,
                                "proposal_shortlist_applied": shortlist_applied,
                            }
                        else:
                            raise ValueError(
                                "first_variation_candidate_gate must be all_live or position_gradient"
                            )
                        parent_pool_fallback = int(live_parent_pool.sum().item()) < matched_birth_quota
                        if parent_pool_fallback:
                            live_parent_pool = finite_score
                        proposal_selection_score = proposal_utility
                        if proposal_scoring_active:
                            if parent_pool_fallback:
                                raise RuntimeError(
                                    "proposal candidate pool unexpectedly fell below the birth quota"
                                )
                            if not torch.equal(
                                live_parent_pool,
                                proposal_probes["candidate_mask"],
                            ):
                                raise RuntimeError(
                                    "proposal score pool does not match the committed candidate pool"
                                )
                            proposal_clip_quantile = min(
                                max(float(opt.first_variation_score_clip_quantile), 0.0),
                                1.0,
                            )
                            proposal_clip_value = 0.0
                            if proposal_clip_quantile > 0.0:
                                proposal_clip = torch.quantile(
                                    proposal_selection_score[live_parent_pool].abs(),
                                    proposal_clip_quantile,
                                )
                                proposal_clip_value = float(proposal_clip.item())
                                if proposal_clip_value > 0.0:
                                    proposal_selection_score = proposal_selection_score.clamp(
                                        min=-proposal_clip_value,
                                        max=proposal_clip_value,
                                    )
                            first_variation_stats.update({
                                "proposal_utility_clip_quantile": proposal_clip_quantile,
                                "proposal_utility_clip_value": proposal_clip_value,
                                "proposal_scoring_active": True,
                                "proposal_realization": "exact_prescored_standard_clone_split",
                            })
                        elif proposal_mode:
                            first_variation_stats.update({
                                "proposal_utility_clip_quantile": 0.0,
                                "proposal_utility_clip_value": 0.0,
                                "proposal_scoring_active": False,
                                "proposal_realization": "standard_clone_split_no_subset_choice",
                            })
                        if opt.first_variation_birth_selection == "q_threshold":
                            threshold_parent_pool = torch.logical_and(
                                live_parent_pool,
                                select_score_threshold_mask(
                                    stabilized_score,
                                    opt.first_variation_birth_threshold,
                                ),
                            )
                            population_target = None
                            remaining_population_budget = None
                            if population_budget_enabled:
                                population_target = taming_population_target(
                                    initial_population,
                                    population_budget,
                                    iteration,
                                    opt.densify_from_iter,
                                    population_budget_end_iter,
                                )
                                remaining_population_budget = max(
                                    0,
                                    population_target - int(gaussians.get_xyz.shape[0]),
                                )
                            selection_mask, selection_stats = select_threshold_with_growth_limit(
                                stabilized_score,
                                live_parent_pool,
                                opt.first_variation_birth_threshold,
                                opt.first_variation_max_growth_fraction,
                                score_seed + 104729,
                                max_births=remaining_population_budget,
                            )
                            selection_stats.update({
                                "population_budget_schedule": (
                                    "taming_parabolic" if population_budget_enabled else "none"
                                ),
                                "population_budget_initial_count": initial_population,
                                "population_budget_final_count": (
                                    population_budget if population_budget_enabled else -1
                                ),
                                "population_budget_target_count": (
                                    population_target if population_budget_enabled else -1
                                ),
                                "population_budget_remaining_before_birth": (
                                    remaining_population_budget
                                    if population_budget_enabled
                                    else -1
                                ),
                                "population_budget_end_iteration": (
                                    population_budget_end_iter
                                    if population_budget_enabled
                                    else -1
                                ),
                            })
                            if int(threshold_parent_pool.sum().item()) != int(
                                selection_stats["threshold_candidate_count"]
                            ):
                                raise RuntimeError("q-threshold candidate accounting mismatch")
                            reference_birth_quota = int(selection_mask.sum().item())
                            matched_birth_quota = reference_birth_quota
                            remaining_capacity = reference_birth_quota
                            quota_source = (
                                "first_variation_population_relative_growth_limit"
                                if selection_stats["growth_limit_enabled"]
                                else "first_variation_absolute_threshold"
                            )
                        elif opt.first_variation_birth_selection == "greedy_top_r":
                            selection_score = stabilized_score
                            q_rank_weight = float(opt.first_variation_rank_blend)
                            if not 0.0 <= q_rank_weight <= 1.0:
                                raise ValueError("first_variation_rank_blend must be in [0, 1]")
                            rank_blend_applied = (
                                opt.first_variation_candidate_gate == "position_gradient"
                                and q_rank_weight < 1.0
                            )
                            if rank_blend_applied:
                                gradient_rank = rank_normalize_masked(standard_grad_score, live_parent_pool)
                                first_variation_rank = rank_normalize_masked(stabilized_score, live_parent_pool)
                                selection_score = (
                                    (1.0 - q_rank_weight) * gradient_rank
                                    + q_rank_weight * first_variation_rank
                                )
                            selection_mask = select_topk_mask(
                                selection_score,
                                live_parent_pool,
                                matched_birth_quota,
                            )
                            selection_stats = {
                                "birth_selection": (
                                    "greedy_blended_rank"
                                    if rank_blend_applied
                                    else "greedy_top_r"
                                ),
                                "first_variation_rank_weight": q_rank_weight,
                                "rank_blend_applied": rank_blend_applied,
                                "positive_weight_count": int(
                                    torch.logical_and(stabilized_score > 0, live_parent_pool).sum().item()
                                ),
                                "positive_weight_fallback": False,
                            }
                        elif opt.first_variation_birth_selection == "positive_proportional":
                            selection_mask, selection_stats = select_positive_weighted_mask(
                                stabilized_score,
                                live_parent_pool,
                                matched_birth_quota,
                                score_seed + 104729,
                            )
                            selection_stats["birth_selection"] = "positive_proportional_without_replacement"
                        elif opt.first_variation_birth_selection == "position_gradient_top_r":
                            if opt.first_variation_candidate_gate != "position_gradient":
                                raise ValueError(
                                    "position-gradient birth selection requires the position-gradient candidate gate"
                                )
                            selection_mask = select_topk_mask(
                                standard_grad_score,
                                live_parent_pool,
                                matched_birth_quota,
                            )
                            selection_stats = {
                                "birth_selection": "vanilla_position_gradient_top_r",
                                "positive_weight_count": int(
                                    torch.logical_and(stabilized_score > 0, live_parent_pool).sum().item()
                                ),
                                "positive_weight_fallback": False,
                            }
                            candidate_gate_stats["first_variation_role"] = "death_only"
                        elif opt.first_variation_birth_selection == "proposal_greedy_top_r":
                            if proposal_scoring_active:
                                selection_mask = select_topk_mask(
                                    proposal_selection_score,
                                    live_parent_pool,
                                    matched_birth_quota,
                                )
                            elif matched_birth_quota == 0:
                                selection_mask = torch.zeros_like(live_parent_pool)
                            else:
                                selection_mask = live_parent_pool.clone()
                            selection_stats = {
                                "birth_selection": "proposal_utility_greedy_top_r",
                                "positive_weight_count": (
                                    int(torch.logical_and(
                                        proposal_selection_score > 0,
                                        live_parent_pool,
                                    ).sum().item())
                                    if proposal_scoring_active
                                    else 0
                                ),
                                "positive_weight_fallback": False,
                            }
                            candidate_gate_stats["first_variation_role"] = (
                                "native_proposal_scoring_when_cap_limited"
                            )
                        elif opt.first_variation_birth_selection == "proposal_positive_proportional":
                            if proposal_scoring_active:
                                selection_mask, selection_stats = (
                                    select_positive_weighted_with_ranked_fallback(
                                        proposal_selection_score,
                                        live_parent_pool,
                                        standard_grad_score,
                                        matched_birth_quota,
                                        score_seed + 104729,
                                    )
                                )
                            elif matched_birth_quota == 0:
                                selection_mask = torch.zeros_like(live_parent_pool)
                                selection_stats = {
                                    "sampling_seed": int(score_seed + 104729),
                                    "positive_weight_count": 0,
                                    "positive_weight_fallback": False,
                                }
                            else:
                                selection_mask = live_parent_pool.clone()
                                selection_stats = {
                                    "sampling_seed": int(score_seed + 104729),
                                    "positive_weight_count": 0,
                                    "positive_weight_fallback": False,
                                }
                            selection_stats["birth_selection"] = (
                                "proposal_utility_positive_proportional_with_native_ranked_fallback"
                            )
                            candidate_gate_stats["first_variation_role"] = (
                                "native_proposal_scoring_when_cap_limited"
                            )
                        else:
                            raise ValueError(
                                "first_variation_birth_selection must be q_threshold, greedy_top_r, positive_proportional, "
                                "position_gradient_top_r, proposal_greedy_top_r, or "
                                "proposal_positive_proportional"
                            )
                        selected_ranking_score = (
                            proposal_selection_score
                            if proposal_scoring_active
                            else stabilized_score
                        )
                        selected_scores = selected_ranking_score[selection_mask]
                        first_variation_stats.update(candidate_gate_stats)
                        first_variation_stats.update(selection_stats)
                        first_variation_stats.update({
                            "score_seed": score_seed,
                            "quota_source": quota_source,
                            "reference_birth_quota": reference_birth_quota,
                            "matched_birth_quota": matched_birth_quota,
                            "position_gradient_used_for_decision": (
                                matched_birth_schedule is None and not pure_q_threshold_adc
                            ),
                            "parent_pool_size": int(live_parent_pool.sum().item()),
                            "parent_pool_fallback": parent_pool_fallback,
                            "parent_opacity_floor": float(opt.first_variation_parent_opacity_floor),
                            "selected_positive": int((selected_scores > 0).sum().item()),
                            "selected_score_mean": float(selected_scores.mean().item()) if selected_scores.numel() else 0.0,
                            "selected_score_min": float(selected_scores.min().item()) if selected_scores.numel() else 0.0,
                            "selected_parent_q_mean": (
                                float(stabilized_score[selection_mask].mean().item())
                                if selection_mask.any()
                                else 0.0
                            ),
                        })

                    adc_stats = gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        max_gaussians=opt.max_gaussians,
                        selection_mask=selection_mask,
                        selection_source="first_variation" if opt.first_variation_adc else "position_gradient",
                        report_standard_eligible=(
                            matched_birth_schedule is None and not pure_q_threshold_adc
                        ),
                        requested_birth_quota=reference_birth_quota,
                        position_gradient_used_for_decision=(
                            (
                                matched_birth_schedule is None and not pure_q_threshold_adc
                            ) if opt.first_variation_adc else True
                        ),
                        scored_prune_mask=scored_prune_mask,
                        split_sample_table=split_sample_table,
                    )
                    adc_stats["iteration"] = iteration
                    if first_variation_stats is not None:
                        adc_stats["first_variation"] = first_variation_stats
                    adc_line = json.dumps(adc_stats, sort_keys=True)
                    print(f"[ADC] {adc_line}")
                    with open(os.path.join(scene.model_path, "adc_events.jsonl"), "a", encoding="utf-8") as adc_log:
                        adc_log.write(adc_line + "\n")
                    if tb_writer:
                        for key in ("standard_eligible", "birth_quota", "clone_count", "split_count", "prune_count", "count_after"):
                            tb_writer.add_scalar(f"adc/{key}", adc_stats[key], iteration)
                        if first_variation_stats is not None:
                            for key in ("score_compute_sec", "score_objective", "raw_score_mean", "raw_score_std", "raw_positive_fraction", "score_clip_value", "selected_positive"):
                                tb_writer.add_scalar(f"first_variation/{key}", first_variation_stats[key], iteration)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
