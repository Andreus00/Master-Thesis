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
import scipy.spatial
import torch
from random import randint

import sys
sys.path.append("./ext/gaussian-splatting/")
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene_4ddress import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, ParamGroup, PipelineParams, OptimizationParams
from arguments_4d_dress import ModelParams4dDress
from functools import partial

from pycpd import RigidRegistration, DeformableRegistration

import matplotlib.pyplot as plt
import numpy as np

from scene_4ddress.dataset_readers import orbit_camera, CameraInfo, focal2fov, fov2focal

from scene_4ddress.cameras import Camera

import pickle

from utils.sh_utils import RGB2SH

from PIL import Image

import scipy

from swap_utils import *


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def run_rigid_registration(source_gaussians, target_gaussians, src_camera, trg_camera, pipe, bg):

    num_samples = 300
    
    sampled_indexes = np.random.randint(0, source_gaussians._xyz.shape[0], num_samples)

    source_clothes_xyz = source_gaussians._xyz.detach().cpu().numpy().astype(np.float16)
    downsampled_source_clothes_xyz = source_clothes_xyz[sampled_indexes]

    target_body_xyz = target_gaussians._xyz.detach().cpu().numpy().astype(np.float16)
    downsampled_target_body_xyz = target_body_xyz[sampled_indexes]

    s, r, t = None, None, None
    try:
        reg = RigidRegistration(X=downsampled_source_clothes_xyz.copy(), Y=downsampled_target_body_xyz.copy())
        TY, (s, r, t) = reg.register()
        
        # Rotate the source clothes to align with the target body
        print("Rotation matrix: ", r)
        print("Translation: ", t)
        print("Scale: ", s)
        t = torch.tensor(t, dtype=torch.float32, device="cpu")
        s = torch.tensor(s, dtype=torch.float32, device="cpu")
        r = torch.tensor(r, dtype=torch.float32, device="cpu")

        rotate_gaussians(source_gaussians, r)
        translate_gaussians(source_gaussians, t)
        rescale_gaussians(source_gaussians, 1/s)

        with torch.no_grad():
            render_tgt_pkg = render(trg_camera, target_gaussians, pipe, bg)
            tgt_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = render_tgt_pkg["render"], render_tgt_pkg["viewspace_points"], render_tgt_pkg["visibility_filter"], render_tgt_pkg["radii"]

        with torch.no_grad():
            render_src_pkg = render(src_camera, source_gaussians, pipe, bg)
            src_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = render_src_pkg["render"], render_src_pkg["viewspace_points"], render_src_pkg["visibility_filter"], render_src_pkg["radii"]

        # show the image
        full_image = torch.cat([src_image, tgt_image], dim=2)
        plt.imshow(full_image.cpu().numpy().transpose(1, 2, 0))
        plt.show()
    
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print("Error in rigid registration: ", e)
        return False
    

N_pts_include = 61
marker_size = 100
IDs, IDs_Y, IDs_X = None, None, None

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')

    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def run_deformable_registration(source_gaussians, target_gaussians, src_camera, trg_camera, pipe, bg):

    num_samples = 350
    
    sampled_indexes = np.random.randint(0, source_gaussians._xyz.shape[0], num_samples)

    source_clothes_xyz = source_gaussians._xyz.detach().cpu().numpy().astype(np.float32)
    downsampled_source_clothes_xyz = source_clothes_xyz[sampled_indexes]

    target_body_xyz = target_gaussians._xyz.detach().cpu().numpy().astype(np.float32)
    downsampled_target_body_xyz = target_body_xyz[sampled_indexes]

    try:
        reg = DeformableRegistration(**{'X': downsampled_target_body_xyz, 'Y': downsampled_source_clothes_xyz}, alpha=100, beta=0.5, low_rank=False, num_eig=100)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        callback = partial(visualize, ax=ax)
        TY, (G, W) = reg.register(callback=callback)
        plt.show()
        plt.close()

        with torch.no_grad():
            before_pkg = render(src_camera, source_gaussians, pipe, bg)
            before_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = before_pkg["render"], before_pkg["viewspace_points"], before_pkg["visibility_filter"], before_pkg["radii"]

        print("AAAAAAAAAAAAAAAA")
        YT = reg.transform_point_cloud(Y=source_clothes_xyz)
        source_gaussians._xyz = torch.tensor(YT, dtype=torch.float32, device="cuda")
        print("BBBBBBBBBBBBBBB")

        # Rotate the source clothes to align with the target body

        with torch.no_grad():
            render_tgt_pkg = render(trg_camera, target_gaussians, pipe, bg)
            tgt_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = render_tgt_pkg["render"], render_tgt_pkg["viewspace_points"], render_tgt_pkg["visibility_filter"], render_tgt_pkg["radii"]

        with torch.no_grad():
            render_src_pkg = render(src_camera, source_gaussians, pipe, bg)
            src_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = render_src_pkg["render"], render_src_pkg["viewspace_points"], render_src_pkg["visibility_filter"], render_src_pkg["radii"]

        # show the image
        src_image = src_image.detach().cpu().permute(1, 2, 0)
        tgt_image = tgt_image.detach().cpu().permute(1, 2, 0)
        before_image = before_image.detach().cpu().permute(1, 2, 0)
        src_x_before = 0.5 * src_image + 0.5 * before_image
        full_image = torch.cat([before_image, src_image, tgt_image], dim=1)
        plt.imshow(full_image)
        plt.show()
    
    except KeyboardInterrupt as e:
        raise e
    




def run_swap_clothes(cfg, pipe):

    source_scan_labels = load_pickle(cfg.source_clothes_label)["scan_labels"]
    target_scan_labels = load_pickle(cfg.target_clothes_label)["scan_labels"]

    source_gaussians = GaussianModel(3)
    source_gaussians.load_ply(os.path.join(cfg.source_model))
    
    target_gaussians = GaussianModel(3)
    target_gaussians.load_ply(os.path.join(cfg.target_model))


    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    src_camera = create_camera(azimuth=0)
    trg_camera = create_camera(azimuth=0)

    bg = background

    rotate_gaussians_along_y(source_gaussians, -100)
    rotate_gaussians_along_y(target_gaussians, 20)


    with torch.no_grad():
        render_tgt_pkg = render(trg_camera, target_gaussians, pipe, bg)
        tgt_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = render_tgt_pkg["render"], render_tgt_pkg["viewspace_points"], render_tgt_pkg["visibility_filter"], render_tgt_pkg["radii"]

    with torch.no_grad():
        render_src_pkg = render(src_camera, source_gaussians, pipe, bg)
        src_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = render_src_pkg["render"], render_src_pkg["viewspace_points"], render_src_pkg["visibility_filter"], render_src_pkg["radii"]

    # show the image
    full_image = torch.cat([src_image, tgt_image], dim=2)
    plt.imshow(full_image.cpu().numpy().transpose(1, 2, 0))
    plt.show()

    target_clothes = target_scan_labels != 3
    source_clothes = source_scan_labels == 3

    # # Use cpd to align the source and target clothes

    global IDs, IDs_X, IDs_Y
    IDs = [1,10,20,30]
    IDs_Y = IDs + [source_gaussians._xyz.shape[0] + i for i in IDs]
    IDs_X = IDs + [N_pts_include + i for i in IDs]
    # rigid registration
    run_rigid_registration(source_gaussians, target_gaussians, src_camera, trg_camera, pipe, bg)

    # deformable registration

    run_deformable_registration(source_gaussians, target_gaussians, src_camera, trg_camera, pipe, bg)

    # transform the source clothes based on G and W

    remove_samples_from_gaussians(target_gaussians, target_clothes)
    copy_gaussians_to_target(source_gaussians, target_gaussians, source_clothes)

    with torch.no_grad():
        render_tgt_pkg = render(trg_camera, target_gaussians, pipe, bg)
        tgt_image, tgt_viewspace_point_tensor, tgt_visibility_filter, tgt_radii = render_tgt_pkg["render"], render_tgt_pkg["viewspace_points"], render_tgt_pkg["visibility_filter"], render_tgt_pkg["radii"]

    # show the image
    plt.imsave("output.png", tgt_image.cpu().numpy().transpose(1, 2, 0))
    plt.imshow(tgt_image.cpu().numpy().transpose(1, 2, 0))
    plt.show()

    exit()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    parser.add_argument("--source_model", type=str, required=True, help="Path to the model from which we have to get clothes")
    parser.add_argument("--target_model", type=str, required=True, help="Path to the model to which we have to put clothes")
    parser.add_argument("--source_clothes_label", type=str, required=True, help="Label of the source clothes to swap")
    parser.add_argument("--target_clothes_label", type=str, required=True, help="Label of the target clothes to replace")
    args = parser.parse_args(sys.argv[1:])
    
    print("Source model: " + args.source_model)
    print("Target model: " + args.target_model)

    run_swap_clothes(args, pp.extract(args))

    # All done
    print("\nTraining complete.")
