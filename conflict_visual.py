#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import torch
import numpy as np
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_conflict_set(model_path, name, iteration, views, gaussians, pipeline, background,
                        train_test_exp, separate_sh, num_samples=10):
    conflict_path = os.path.join(model_path, name, "ours_{}".format(iteration), "conflict_vis")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(conflict_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Select evenly spaced indices from training views
    n_views = len(views)
    sample_indices = np.linspace(0, n_views - 1, num_samples, dtype=int)

    # Build per-Gaussian conflict colors
    conflict = gaussians.conflict_score.squeeze()
    conflict_clamped = torch.clamp(conflict, 0.0, 1.0)
    conflict_colors = torch.zeros((conflict.shape[0], 3), device="cuda")
    conflict_colors[:, 0] = conflict_clamped
    conflict_colors[:, 2] = 1.0 - conflict_clamped

    for idx in sample_indices:
        view = views[idx]

        # RGB render
        rgb = render(view, gaussians, pipeline, background,
                     use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        # Conflict-colored render
        conflict_render = render(view, gaussians, pipeline, background,
                                 override_color=conflict_colors,
                                 use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

        if train_test_exp:
            rgb = rgb[..., rgb.shape[-1] // 2:]
            conflict_render = conflict_render[..., conflict_render.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rgb, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(conflict_render,
                                     os.path.join(conflict_path, 'conflict_{0:05d}'.format(idx) + ".png"))

    # Also save a composite: conflict overlay on GT
    overlay_path = os.path.join(model_path, name, "ours_{}".format(iteration), "conflict_overlay")
    makedirs(overlay_path, exist_ok=True)

    for idx in sample_indices:
        view = views[idx]
        rgb = render(view, gaussians, pipeline, background,
                     use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        conflict_render = render(view, gaussians, pipeline, background,
                                 override_color=conflict_colors,
                                 use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

        if train_test_exp:
            rgb = rgb[..., rgb.shape[-1] // 2:]
            conflict_render = conflict_render[..., conflict_render.shape[-1] // 2:]

        overlay = 0.5 * rgb + 0.5 * conflict_render
        torchvision.utils.save_image(overlay,
                                     os.path.join(overlay_path, 'overlay_{0:05d}'.format(idx) + ".png"))


def render_conflict_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                         skip_train: bool, skip_test: bool, separate_sh: bool, num_samples: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Load saved conflict scores if available
        point_cloud_path = os.path.join(dataset.model_path,
                                        "point_cloud/iteration_{}".format(scene.loaded_iter))
        conflict_pt_path = os.path.join(point_cloud_path, "conflict_score.pt")
        if os.path.exists(conflict_pt_path):
            gaussians.conflict_score = torch.load(conflict_pt_path, map_location="cuda")
            print("Loaded conflict scores from {}".format(conflict_pt_path))
        else:
            print("Warning: No conflict_score.pt found at {}. All scores default to 0.5.".format(
                conflict_pt_path))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_conflict_set(dataset.model_path, "train", scene.loaded_iter,
                                scene.getTrainCameras(), gaussians, pipeline, background,
                                dataset.train_test_exp, separate_sh, num_samples)

        if not skip_test:
            render_conflict_set(dataset.model_path, "test", scene.loaded_iter,
                                scene.getTestCameras(), gaussians, pipeline, background,
                                dataset.train_test_exp, separate_sh, num_samples)


if __name__ == "__main__":
    parser = ArgumentParser(description="Conflict visualization script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering conflict visualization for " + args.model_path)

    safe_state(args.quiet)

    render_conflict_sets(model.extract(args), args.iteration, pipeline.extract(args),
                         args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.num_samples)
