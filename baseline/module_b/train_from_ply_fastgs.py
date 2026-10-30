"""Module B (FastGS variant): FastGS continuation training from ReSplat Gaussians.

Replaces the classic-3DGS trainer with FastGS (multiview-consistent
densification/pruning + clipped-tile renderer), mirroring the
benchmarks/gs_compare_20261020 FastGS recipe:
  * standard 30k LR schedule kept verbatim (position_lr_max_steps=30000),
    training stops early at --iterations (10000 default)
  * FastGS defaults: mult=0.5, densification_interval=100,
    grad_abs_thresh=0.0012, dense=0.001, percent_dense=0.001
Initialization comes from ReSplat's world-space PLY (--init_ply).

Usage (fastgs conda env, from baseline/):
  python module_b/train_from_ply_fastgs.py \
      -s work/<group>/<scene> \
      -m results/<group>/<scene>/module_b \
      --init_ply results/<group>/<scene>/resplat/resplat_world.ply \
      --iterations 10000
"""

import json
import hashlib
import os
import sys
import time
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

BASELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINE_ROOT / "fastgs"))

from argparse import ArgumentParser, Namespace  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402

try:
    from fused_ssim import fused_ssim as fast_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


def random_frustum_init(scene, num_points, sh_degree):
    """Uniformly sample points inside train-camera frustums."""
    cams = scene.getTrainCameras()
    r = max(scene.cameras_extent, 1e-6)
    per_cam = max(num_points // len(cams), 1)
    points, colors = [], []
    for cam in cams:
        c2w = cam.world_view_transform.inverse().cpu().numpy()
        R = c2w[:3, :3]
        C = c2w[:3, 3]
        per = per_cam
        us = np.random.rand(per) * 2 - 1
        vs = np.random.rand(per) * 2 - 1
        depths = np.exp(np.random.uniform(np.log(0.05 * r), np.log(1.5 * r), per))
        tanx, tany = np.tan(cam.FoVx / 2), np.tan(cam.FoVy / 2)
        dirs_cam = np.stack([us * tanx * depths, vs * tany * depths, depths], axis=1)
        pts = (R @ dirs_cam.T).T + C
        points.append(pts)
        colors.append(np.random.rand(per, 3))
    from scene.dataset_readers import BasicPointCloud
    xyz = np.concatenate(points, 0).astype(np.float32)
    rgb = np.concatenate(colors, 0).astype(np.float32)
    return BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))


def budgeted_growth_mask(gaussians, importance_score, opt, extent, max_points):
    """Keep the highest-gradient eligible growth sites within the point budget."""
    n = gaussians.get_xyz.shape[0]
    room = max(0, max_points - n)
    grad = gaussians.xyz_gradient_accum / gaussians.denom
    grad_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
    grad = torch.nan_to_num(grad.squeeze(-1))
    grad_abs = torch.nan_to_num(grad_abs.squeeze(-1))
    scale = gaussians.get_scaling.max(dim=1).values
    eligible = (importance_score > 5) & (
        ((scale <= opt.dense * extent) & (grad >= opt.grad_thresh)) |
        ((scale > opt.dense * extent) & (grad_abs >= opt.grad_abs_thresh)))
    eligible_count = int(eligible.sum().item())
    selected_count = min(room, eligible_count)
    masked = torch.zeros_like(importance_score)
    if selected_count:
        priority = torch.where(eligible, torch.maximum(grad, grad_abs),
                               torch.full_like(grad, -1.0))
        selected = torch.topk(priority, selected_count).indices
        masked[selected] = importance_score[selected]
    return masked, eligible_count, selected_count


def training(dataset, opt, pipe, args):
    t_total0 = time.time()

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.long_axis_split = args.long_axis_split
    gaussians.preserve_unclamped_opacity_state = args.preserve_unclamped_opacity_state

    # output dir + config dump (mirrors classic prepare_output_and_logger)
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(dataset))))

    t0 = time.time()
    scene = Scene(dataset, gaussians,
                  resolution_scales=[1.0, 2.0] if args.coarse_to_fine else [1.0])
    scene_load_s = time.time() - t0

    t0 = time.time()
    if args.init_ply:
        gaussians.load_ply(args.init_ply)
        n_init = gaussians.get_xyz.shape[0]
        init_mode = f"ply:{os.path.basename(args.init_ply)}"
    elif args.init_random:
        pcd = random_frustum_init(scene, args.init_random_num, dataset.sh_degree)
        gaussians.create_from_pcd(pcd, scene.cameras_extent)
        n_init = gaussians.get_xyz.shape[0]
        init_mode = f"random:{n_init}"
    else:
        raise SystemExit("either --init_ply or --init_random is required")
    gaussians.spatial_lr_scale = scene.cameras_extent
    gaussians.training_setup(opt)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    init_s = time.time() - t0
    max_points = n_init if args.topk_growth else None
    growth_eligible_total = 0
    growth_selected_total = 0
    peak_gaussians = n_init
    coarse_seconds = None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    perceptual = None
    perceptual_calls = 0
    if args.perceptual_loss:
        from lpipsPyTorch.modules.lpips import LPIPS
        perceptual = LPIPS(net_type="vgg").cuda().eval()
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)

    coarse_until = round(opt.iterations * 0.4) if args.coarse_to_fine else 0
    active_scale = 2.0 if args.coarse_to_fine else 1.0
    train_cameras = scene.getTrainCameras(active_scale)
    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    t_train0 = time.time()
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="FastGS training")
    for iteration in range(1, opt.iterations + 1):
        if args.coarse_to_fine and iteration == coarse_until + 1:
            coarse_seconds = time.time() - t_train0
            active_scale = 1.0
            train_cameras = scene.getTrainCameras(active_scale)
            viewpoint_stack = train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe,
                                   background, opt.mult, 1.0)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"])

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = 1.0
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if perceptual is not None and iteration % 16 == 0:
            pred_small = F.interpolate(image.unsqueeze(0), size=(256, 256),
                                       mode="bilinear", align_corners=False).clamp(0, 1)
            gt_small = F.interpolate(gt_image.unsqueeze(0), size=(256, 256),
                                     mode="bilinear", align_corners=False)
            loss = loss + 0.1 * perceptual(pred_small * 2 - 1,
                                           gt_small * 2 - 1).mean()
            perceptual_calls += 1
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "N": gaussians.get_xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification (FastGS multiview-consistent)
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor,
                                                  visibility_filter)
                if iteration > opt.densify_from_iter and \
                        iteration % opt.densification_interval == 0 and \
                        not (args.skip_final_densification and iteration == opt.iterations):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    camlist = sampling_cameras(train_cameras.copy())
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, background, opt, DENSIFY=True)
                    if args.topk_growth:
                        importance_score, eligible_count, selected_count = budgeted_growth_mask(
                            gaussians, importance_score, opt, scene.cameras_extent, max_points)
                        growth_eligible_total += eligible_count
                        growth_selected_total += selected_count
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold, min_opacity=0.005,
                        extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=importance_score,
                        pruning_score=pruning_score)
                    if args.topk_growth and gaussians.get_xyz.shape[0] > max_points:
                        overflow = gaussians.get_xyz.shape[0] - max_points
                        lowest = torch.topk(gaussians.get_opacity.squeeze(-1), overflow,
                                            largest=False).indices
                        prune = torch.zeros(gaussians.get_xyz.shape[0],
                                            dtype=torch.bool, device="cuda")
                        prune[lowest] = True
                        gaussians.prune_points(prune)
                    peak_gaussians = max(peak_gaussians, int(gaussians.get_xyz.shape[0]))
                reset_due = (iteration % opt.opacity_reset_interval == 0 or
                             (dataset.white_background and iteration == opt.densify_from_iter))
                if (reset_due and not args.disable_opacity_reset
                        and not (args.skip_final_opacity_reset and iteration == opt.iterations)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer_step(iteration)

    n_before_final_prune = int(gaussians.get_xyz.shape[0])
    if args.final_prune_fastgs:
        with torch.no_grad():
            camlist = sampling_cameras(scene.getTrainCameras().copy())
            _, pruning_score = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, background, opt, DENSIFY=False)
            gaussians.final_prune_fastgs(min_opacity=0.01,
                                        pruning_score=pruning_score)

    train_s = time.time() - t_train0
    if args.coarse_to_fine and coarse_seconds is None:
        coarse_seconds = train_s

    scene.save(opt.iterations)
    total_s = time.time() - t_total0

    # quick test PSNR at mult=1.0 (faithful full-tile rendering)
    test_psnr = None
    try:
        psnrs = []
        for cam in scene.getTestCameras():
            out = render_fastgs(cam, gaussians, pipe, background, 1.0, 1.0)
            img = torch.clamp(out["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            psnrs.append(psnr(img, gt).mean().item())
        if psnrs:
            test_psnr = float(np.mean(psnrs))
    except Exception as e:  # noqa: BLE001
        print(f"[B] test psnr failed: {e}")

    timing = {
        "trainer": "fastgs",
        "init_mode": init_mode,
        "num_init_gaussians": int(n_init),
        "num_final_gaussians": int(gaussians.get_xyz.shape[0]),
        "num_before_final_prune": n_before_final_prune,
        "final_prune_fastgs": bool(args.final_prune_fastgs),
        "iterations": int(opt.iterations),
        "skip_final_opacity_reset": bool(args.skip_final_opacity_reset),
        "disable_opacity_reset": bool(args.disable_opacity_reset),
        "preserve_unclamped_opacity_state": bool(args.preserve_unclamped_opacity_state),
        "opacity_clamped_total": gaussians.opacity_clamped_total,
        "skip_final_densification": bool(args.skip_final_densification),
        "densify_until_iter": int(opt.densify_until_iter),
        "seed": 0,
        "init_ply_sha256": hashlib.sha256(Path(args.init_ply).read_bytes()).hexdigest() if args.init_ply else None,
        "optimization_args": vars(opt),
        "topk_growth": bool(args.topk_growth),
        "growth_point_budget": max_points,
        "growth_eligible_total": growth_eligible_total if args.topk_growth else None,
        "growth_selected_total": growth_selected_total if args.topk_growth else None,
        "peak_gaussians": peak_gaussians if args.topk_growth else None,
        "coarse_to_fine": bool(args.coarse_to_fine),
        "long_axis_split": bool(args.long_axis_split),
        "perceptual_loss": bool(args.perceptual_loss),
        "perceptual_interval": 16 if args.perceptual_loss else None,
        "perceptual_weight": 0.1 if args.perceptual_loss else None,
        "perceptual_calls": perceptual_calls if args.perceptual_loss else None,
        "coarse_until_iteration": coarse_until if args.coarse_to_fine else None,
        "coarse_seconds": round(coarse_seconds, 2) if coarse_seconds is not None else None,
        "fine_seconds": round(train_s - coarse_seconds, 2) if coarse_seconds is not None else None,
        "mult": opt.mult,
        "num_train_views": len(scene.getTrainCameras()),
        "num_test_views": len(scene.getTestCameras()),
        "scene_load_seconds": round(scene_load_s, 2),
        "init_seconds": round(init_s, 2),
        "train_seconds": round(train_s, 2),
        "total_seconds": round(total_s, 2),
        "test_psnr_quick": round(test_psnr, 3) if test_psnr is not None else None,
    }
    with open(os.path.join(scene.model_path, "module_b.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"[B-fastgs] done: {timing}")


if __name__ == "__main__":
    parser = ArgumentParser(description="FastGS continuation from ReSplat PLY")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--init_ply", type=str, default=None)
    parser.add_argument("--init_random", action="store_true")
    parser.add_argument("--init_random_num", type=int, default=100_000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_final_opacity_reset", action="store_true",
                        help="avoid saving immediately after an opacity reset at the last step")
    parser.add_argument("--disable_opacity_reset", action="store_true",
                        help="skip all scheduled opacity resets during short training")
    parser.add_argument("--skip_final_densification", action="store_true",
                        help="do not change Gaussian topology on the final training iteration")
    parser.add_argument("--preserve_unclamped_opacity_state", action="store_true",
                        help="during 0.8 clamp, keep Adam moments for opacity entries below the limit")
    parser.add_argument("--final_prune_fastgs", action="store_true",
                        help="apply FastGS multiview final prune before saving")
    parser.add_argument("--topk_growth", action="store_true",
                        help="limit densification to highest-gradient eligible sites under input point count")
    parser.add_argument("--coarse_to_fine", action="store_true",
                        help="train first 40 percent of steps at half image width and height")
    parser.add_argument("--long_axis_split", action="store_true",
                        help="sample FastGS split children along the longest local scale axis")
    parser.add_argument("--perceptual_loss", action="store_true",
                        help="add frozen VGG LPIPS on one training view every 16 steps")
    args = parser.parse_args()
    args.save_iterations = [args.iterations]
    args.test_iterations = []

    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nTraining complete.")
