#!/usr/bin/env python3
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_mass_gated
from scene import GaussianModel, Scene
from utils.first_variation import compute_first_variation_score, select_score_views

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SEPARATE_SH = True
except ImportError:
    SEPARATE_SH = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM = fused_ssim
except ImportError:
    FUSED_SSIM = None


def main():
    parser = ArgumentParser(description="Validate the detached first-variation score path")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    cameras = select_score_views(scene.getTrainCameras(), args.num_views, args.seed)

    equality = []
    for camera in cameras:
        with torch.no_grad():
            normal = render(
                camera,
                gaussians,
                pipe,
                background,
                separate_sh=SEPARATE_SH,
                use_trained_exp=dataset.train_test_exp,
            )["render"]
        with torch.enable_grad():
            gate = torch.ones_like(gaussians.get_opacity.detach(), requires_grad=True)
            gated = render_mass_gated(
                camera,
                gaussians,
                pipe,
                background,
                gate,
                separate_sh=SEPARATE_SH,
                use_trained_exp=dataset.train_test_exp,
            )["render"]
        difference = (normal - gated.detach()).abs()
        equality.append({
            "view": camera.image_name,
            "max_abs_difference": float(difference.max().item()),
            "mean_abs_difference": float(difference.mean().item()),
        })

    score, score_metadata = compute_first_variation_score(
        cameras,
        gaussians,
        pipe,
        background,
        lambda_dssim=0.2,
        fused_ssim_fn=FUSED_SSIM,
        separate_sh=SEPARATE_SH,
        use_trained_exp=dataset.train_test_exp,
    )
    scene_tensors = (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    )
    summary = {
        "status": "passed",
        "model_path": dataset.model_path,
        "iteration": scene.loaded_iter,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "forward_equality": equality,
        "max_forward_difference": max(item["max_abs_difference"] for item in equality),
        "score_finite": bool(torch.isfinite(score).all().item()),
        "score_nonzero": bool((score != 0).any().item()),
        "scene_parameter_grads_all_none": all(tensor.grad is None for tensor in scene_tensors),
        "score_metadata": score_metadata,
    }
    if summary["max_forward_difference"] > 1e-6:
        summary["status"] = "failed"
    if not summary["score_finite"] or not summary["score_nonzero"]:
        summary["status"] = "failed"
    if not summary["scene_parameter_grads_all_none"]:
        summary["status"] = "failed"

    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(summary_json + "\n", encoding="utf-8")
    print(summary_json)
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
